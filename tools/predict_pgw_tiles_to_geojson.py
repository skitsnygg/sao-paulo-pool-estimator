#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
from pyproj import Transformer
from shapely.geometry import Polygon, mapping, shape
from ultralytics import YOLO


@dataclass
class Stats:
    tiles_seen: int = 0
    tiles_with_masks: int = 0
    features_written: int = 0
    features_dropped_invalid: int = 0
    features_dropped_too_small: int = 0


def iter_pngs(tiles_dir: Path) -> List[Path]:
    return sorted([p for p in tiles_dir.glob("*.png") if p.is_file()])


def read_pgw(pgw_path: Path) -> Tuple[float, float, float, float, float, float]:
    """
    Worldfile (PGW) for north-up images, 6 lines:
      A: pixel size in x-direction (map units / pixel)
      D: rotation about y-axis
      B: rotation about x-axis
      E: pixel size in y-direction (typically negative)
      C: x-coordinate of center of upper-left pixel
      F: y-coordinate of center of upper-left pixel
    """
    lines = pgw_path.read_text().strip().splitlines()
    if len(lines) < 6:
        raise ValueError(f"Invalid PGW (expected 6 lines): {pgw_path}")
    A = float(lines[0].strip())
    D = float(lines[1].strip())
    B = float(lines[2].strip())
    E = float(lines[3].strip())
    C = float(lines[4].strip())
    F = float(lines[5].strip())
    return A, D, B, E, C, F


def pixel_to_projected(A: float, D: float, B: float, E: float, C: float, F: float, px: float, py: float) -> Tuple[float, float]:
    X = A * px + B * py + C
    Y = D * px + E * py + F
    return X, Y


def run_dedupe(
    *,
    python_exe: str,
    dedupe_script: Path,
    in_geojson: Path,
    out_geojson: Path,
    iou: float,
    buffer: float,
) -> None:
    cmd = [
        python_exe,
        str(dedupe_script),
        "--in-geojson",
        str(in_geojson),
        "--out-geojson",
        str(out_geojson),
        "--iou",
        str(iou),
        "--buffer",
        str(buffer),
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def clean_geojson(in_path: Path, out_path: Path) -> Tuple[int, int]:
    d = json.loads(in_path.read_text())
    feats = d.get("features", [])
    kept = []
    dropped = 0
    for f in feats:
        g = shape(f["geometry"])
        if g.is_empty or (not g.is_valid):
            dropped += 1
            continue
        f["geometry"] = mapping(g)
        kept.append(f)
    out_fc = {"type": "FeatureCollection", "features": kept}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_fc))
    return len(kept), dropped


def main() -> int:
    ap = argparse.ArgumentParser(description="Run YOLO-seg on GeoSampa chips (PNG+PGW) and write WGS84 GeoJSON polygons.")
    ap.add_argument("--model", required=True, help="Path to YOLO segmentation .pt")
    ap.add_argument("--tiles-dir", required=True, help="Directory containing .png chips and matching .pgw worldfiles")
    ap.add_argument("--out-dir", required=True, help="Output directory (will be created)")
    ap.add_argument("--name", required=True, help="Run name used to name outputs (e.g. vila_mariana_2020_conf015)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-tiles", type=int, default=0, help="If >0, limit tiles processed (debug)")
    ap.add_argument("--min-poly-area-deg2", type=float, default=0.0, help="Drop polygons smaller than this area (in degrees^2)")

    ap.add_argument("--src-crs", default="EPSG:31983", help="CRS of PGW coordinates (default: EPSG:31983)")
    ap.add_argument("--dst-crs", default="EPSG:4326", help="CRS for GeoJSON output (default: EPSG:4326)")

    ap.add_argument("--dedupe", action="store_true", help="Run dedupe pass using tools/dedupe_geojson_polygons.py")
    ap.add_argument("--dedupe-iou", type=float, default=0.35)
    ap.add_argument("--dedupe-buffer", type=float, default=0.0)
    ap.add_argument("--dedupe-script", default="tools/dedupe_geojson_polygons.py")

    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_geojson = out_dir / "pools.geojson"
    dedup_geojson = out_dir / f"pools_dedup_iou{str(args.dedupe_iou).replace('.', '')}.geojson"
    clean_geojson_path = out_dir / f"pools_dedup_iou{str(args.dedupe_iou).replace('.', '')}_clean.geojson"

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    tiles = iter_pngs(tiles_dir)
    if args.max_tiles and args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]

    if not tiles:
        raise SystemExit(f"No .png tiles found in: {tiles_dir}")

    model = YOLO(str(model_path))
    transformer = Transformer.from_crs(args.src_crs, args.dst_crs, always_xy=True)

    stats = Stats()
    features = []

    for img_path in tiles:
        stats.tiles_seen += 1
        pgw_path = img_path.with_suffix(".pgw")
        if not pgw_path.exists():
            continue

        try:
            A, D, B, E, C, F = read_pgw(pgw_path)
        except Exception:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        r = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]

        if r.masks is None or r.masks.xy is None:
            continue

        stats.tiles_with_masks += 1

        # One confidence per mask; fallback if indexing mismatches
        confs: List[float] = []
        try:
            if r.boxes is not None and getattr(r.boxes, "conf", None) is not None:
                confs = [float(x) for x in r.boxes.conf.tolist()]
        except Exception:
            confs = []

        for i, mask_xy in enumerate(r.masks.xy):
            coords_lonlat = []
            for px, py in mask_xy:
                X, Y = pixel_to_projected(A, D, B, E, C, F, float(px), float(py))
                lon, lat = transformer.transform(X, Y)
                coords_lonlat.append((float(lon), float(lat)))

            if len(coords_lonlat) < 3:
                continue

            poly = Polygon(coords_lonlat)
            if not poly.is_valid or poly.area == 0.0:
                stats.features_dropped_invalid += 1
                continue

            if args.min_poly_area_deg2 > 0.0 and poly.area < args.min_poly_area_deg2:
                stats.features_dropped_too_small += 1
                continue

            confidence = confs[i] if i < len(confs) else float(args.conf)

            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "confidence": confidence,
                        "tile": img_path.name,
                        "model": str(model_path),
                        "conf": float(args.conf),
                        "iou": float(args.iou),
                    },
                }
            )

    fc = {"type": "FeatureCollection", "features": features}
    raw_geojson.write_text(json.dumps(fc))
    stats.features_written = len(features)

    print("Wrote:", raw_geojson)
    print("Tiles processed:", stats.tiles_seen)
    print("Tiles with masks:", stats.tiles_with_masks)
    print("Features:", stats.features_written)
    if stats.features_dropped_invalid:
        print("Dropped invalid polys:", stats.features_dropped_invalid)
    if stats.features_dropped_too_small:
        print("Dropped too-small polys:", stats.features_dropped_too_small)

    final_in = raw_geojson

    if args.dedupe:
        dedupe_script = Path(args.dedupe_script)
        if not dedupe_script.exists():
            raise SystemExit(f"Dedupe script not found: {dedupe_script}")

        run_dedupe(
            python_exe=sys.executable,
            dedupe_script=dedupe_script,
            in_geojson=raw_geojson,
            out_geojson=dedup_geojson,
            iou=float(args.dedupe_iou),
            buffer=float(args.dedupe_buffer),
        )
        final_in = dedup_geojson

    kept, dropped = clean_geojson(final_in, clean_geojson_path)
    print("Clean kept:", kept)
    print("Clean dropped:", dropped)
    print("Wrote:", clean_geojson_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
