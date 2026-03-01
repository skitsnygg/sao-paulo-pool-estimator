#!/usr/bin/env python3
"""
Convert QGIS-drawn pool polygons (GeoJSON in EPSG:3857) to YOLOv8 segmentation labels
for georeferenced GeoTIFF chips.

Inputs:
- --chips-dir: directory of GeoTIFF chips (e.g., 1024x1024) in EPSG:3857
- --labels-geojson: GeoJSON with Polygon/MultiPolygon features in EPSG:3857
  and an attribute field (default "class") containing the class name (default "pool").

Outputs:
- YOLO dataset layout:
  <out-dir>/
    images/train/*.tif
    images/val/*.tif
    labels/train/*.txt
    labels/val/*.txt
    dataset.yaml

Label format (YOLOv8-seg):
  class_id x1 y1 x2 y2 ... (normalized [0,1] image coordinates)

Deterministic split:
- hash of image filename -> train/val (default 80/20)

Dependencies:
- rasterio, shapely
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import rasterio
from shapely.geometry import shape, Polygon, MultiPolygon, box


def stable_hash01(s: str) -> float:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # use first 8 hex digits -> 32-bit int
    v = int(h[:8], 16)
    return v / 0xFFFFFFFF


def iter_geojson_geoms(labels_path: Path) -> Iterable[Tuple[dict, dict]]:
    """Yield (geometry_dict, properties_dict) from GeoJSON FeatureCollection."""
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") == "FeatureCollection":
        feats = data.get("features", [])
    elif data.get("type") == "Feature":
        feats = [data]
    else:
        raise ValueError("Expected FeatureCollection/Feature GeoJSON")

    for feat in feats:
        geom = feat.get("geometry")
        props = feat.get("properties") or {}
        if not geom:
            continue
        yield geom, props


def geom_to_polygons(geom_obj) -> List[Polygon]:
    g = shape(geom_obj)
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return []


def polygon_to_yolo_seg(poly: Polygon, ds: rasterio.io.DatasetReader) -> Optional[List[Tuple[float, float]]]:
    """
    Convert polygon in map coords (EPSG:3857) to normalized pixel coords in [0,1].
    Returns list of (x_norm, y_norm) points, or None if invalid/outside.
    """
    if poly.is_empty:
        return None

    # clip to raster bounds in map coords
    left, bottom, right, top = ds.bounds
    clip_box = box(left, bottom, right, top)
    clipped = poly.intersection(clip_box)
    if clipped.is_empty:
        return None

    # use exterior ring only for YOLO seg (holes ignored for now)
    if isinstance(clipped, MultiPolygon):
        # take largest part
        clipped = max(clipped.geoms, key=lambda p: p.area)

    if not isinstance(clipped, Polygon) or clipped.is_empty:
        return None

    coords = list(clipped.exterior.coords)
    if len(coords) < 4:
        return None

    # map coords -> pixel coords
    pts_norm: List[Tuple[float, float]] = []
    for x_map, y_map in coords:
        col, row = ds.index(x_map, y_map)  # integer pixel index
        # convert to continuous-ish coords by adding 0.5 center
        x = (col + 0.5) / ds.width
        y = (row + 0.5) / ds.height
        # clamp
        if x < -0.1 or x > 1.1 or y < -0.1 or y > 1.1:
            # still allow slight overshoot from rounding; will clamp
            pass
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        pts_norm.append((x, y))

    # remove duplicate last point if equals first (YOLO allows either way; keep simple)
    if len(pts_norm) >= 2 and pts_norm[0] == pts_norm[-1]:
        pts_norm = pts_norm[:-1]

    if len(pts_norm) < 3:
        return None
    return pts_norm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips-dir", required=True)
    ap.add_argument("--labels-geojson", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--class-field", default="class")
    ap.add_argument("--class-name", default="pool")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-area-m2", type=float, default=2.0, help="drop polygons smaller than this in m^2")
    args = ap.parse_args()

    chips_dir = Path(args.chips_dir)
    labels_path = Path(args.labels_geojson)
    out_dir = Path(args.out_dir)

    img_train = out_dir / "images" / "train"
    img_val = out_dir / "images" / "val"
    lab_train = out_dir / "labels" / "train"
    lab_val = out_dir / "labels" / "val"
    for p in (img_train, img_val, lab_train, lab_val):
        p.mkdir(parents=True, exist_ok=True)

    # Load all polygons from GeoJSON
    polys_all: List[Polygon] = []
    for geom, props in iter_geojson_geoms(labels_path):
        if (props.get(args.class_field) or "").lower() != args.class_name.lower():
            continue
        polys_all.extend(geom_to_polygons(geom))

    if not polys_all:
        raise SystemExit("No matching pool polygons found in GeoJSON")

    # Index polygons by raster intersection on the fly (small dataset; brute-force OK)
    chips = sorted(list(chips_dir.glob("*.tif"))) + sorted(list(chips_dir.glob("*.tiff")))
    if not chips:
        raise SystemExit(f"No GeoTIFF chips found in {chips_dir}")

    written = 0
    for chip in chips:
        split_val = stable_hash01(chip.name) < args.val_frac
        img_out_dir = img_val if split_val else img_train
        lab_out_dir = lab_val if split_val else lab_train

        # Copy image as-is (hardlink if possible)
        dst_img = img_out_dir / chip.name
        if not dst_img.exists():
            try:
                dst_img.hardlink_to(chip)
            except Exception:
                # fallback copy
                dst_img.write_bytes(chip.read_bytes())

        with rasterio.open(chip) as ds:
            # Collect polygons that intersect this chip
            left, bottom, right, top = ds.bounds
            chip_box = box(left, bottom, right, top)

            seg_lines: List[str] = []
            for poly in polys_all:
                if poly.area < args.min_area_m2:
                    continue
                if not poly.intersects(chip_box):
                    continue
                pts = polygon_to_yolo_seg(poly, ds)
                if not pts:
                    continue
                # YOLOv8 seg line: class_id then x y pairs
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                seg_lines.append(f"0 {flat}")

        # Write label file (even if empty, YOLO expects missing file means no labels; we'll only write if non-empty)
        if seg_lines:
            (lab_out_dir / (chip.stem + ".txt")).write_text("\n".join(seg_lines) + "\n", encoding="utf-8")
            written += 1

    # dataset.yaml
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: pool\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Polygons total (input): {len(polys_all)}")
    print(f"Chips scanned: {len(chips)}")
    print(f"Chips with labels written: {written}")
    print(f"Wrote: {(out_dir / 'dataset.yaml').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())