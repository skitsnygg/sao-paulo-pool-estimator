#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from pyproj import Transformer

GeoTransform = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ImageMeta:
    image_id: int
    unique_name: str
    tile_rel: str
    tile_path_abs: Path
    width: int
    height: int
    transform: GeoTransform


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Convert geospatial prediction GeoJSON (tile_rel-based) to CVAT-importable COCO "
            "polygons in image pixel coordinates using tile worldfiles."
        )
    )
    ap.add_argument("--geojson", required=True, type=Path, help="Prediction GeoJSON with tile_rel properties.")
    ap.add_argument(
        "--manifest-csv",
        required=True,
        type=Path,
        help="Manifest from tools/prepare_cvat_tiles.py (must include unique_name,tile_rel,tile_path_abs).",
    )
    ap.add_argument("--images-dir", required=True, type=Path, help="Directory containing unique packaged CVAT images.")
    ap.add_argument("--out", required=True, type=Path, help="Output COCO JSON file.")
    ap.add_argument("--geojson-crs", default="", help="Override GeoJSON CRS, e.g. EPSG:31983.")
    ap.add_argument("--worldfile-crs", default="EPSG:31983", help="CRS of tile worldfiles.")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="Skip features with confidence below this.")
    ap.add_argument("--category-id", type=int, default=1)
    ap.add_argument("--category-name", default="pool")
    return ap.parse_args()


def normalize_crs(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    return v.upper()


def find_worldfile(img: Path) -> Optional[Path]:
    suffix = img.suffix.lower()
    if suffix == ".png":
        candidates = [img.with_suffix(".pgw"), img.with_suffix(".pngw"), img.with_suffix(".wld")]
    elif suffix in (".jpg", ".jpeg"):
        candidates = [img.with_suffix(".jgw"), img.with_suffix(".jpgw"), img.with_suffix(".wld")]
    else:
        candidates = [img.with_suffix(".wld")]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_worldfile(path: Path) -> GeoTransform:
    vals = [float(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(vals) != 6:
        raise ValueError(f"Worldfile must have 6 lines, got {len(vals)}: {path}")
    a, d, b, e, c, f = vals
    return (c, a, b, f, d, e)


def world_to_pixel(transform: GeoTransform, x: float, y: float) -> Tuple[float, float]:
    c, a, b, f, d, e = transform
    det = (a * e) - (b * d)
    if abs(det) < 1e-18:
        raise ValueError("Non-invertible geotransform")
    inv00 = e / det
    inv01 = -b / det
    inv10 = -d / det
    inv11 = a / det
    dx = float(x) - c
    dy = float(y) - f
    px = (inv00 * dx) + (inv01 * dy)
    py = (inv10 * dx) + (inv11 * dy)
    return px, py


def parse_geojson_crs(payload: dict, override: str) -> str:
    if normalize_crs(override):
        return normalize_crs(override)
    crs_obj = payload.get("crs")
    if isinstance(crs_obj, dict):
        props = crs_obj.get("properties")
        if isinstance(props, dict):
            name = props.get("name")
            if isinstance(name, str) and name.strip():
                return normalize_crs(name)
    # predict_tiles_to_geojson output is typically EPSG:31983.
    return "EPSG:31983"


def read_manifest_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_unique_name_from_tile_rel(tile_rel: str) -> str:
    p = Path(tile_rel)
    parts = list(p.with_suffix("").parts)
    return "__".join(parts) + p.suffix.lower()


def load_image_metas(
    manifest_rows: Sequence[dict],
    images_dir: Path,
) -> Dict[str, ImageMeta]:
    metas: Dict[str, ImageMeta] = {}
    next_id = 1
    for row in manifest_rows:
        tile_rel = (row.get("tile_rel") or "").strip().replace("\\", "/")
        if not tile_rel:
            continue
        unique_name = (row.get("unique_name") or "").strip()
        if not unique_name:
            unique_name = infer_unique_name_from_tile_rel(tile_rel)

        tile_path_abs_text = (row.get("tile_path_abs") or "").strip()
        if not tile_path_abs_text:
            continue
        tile_path_abs = Path(tile_path_abs_text)
        if not tile_path_abs.exists():
            continue

        out_img = images_dir / unique_name
        if not out_img.exists():
            continue

        worldfile = find_worldfile(tile_path_abs)
        if worldfile is None:
            continue

        transform = read_worldfile(worldfile)
        with Image.open(out_img) as im:
            width, height = im.size

        metas[tile_rel] = ImageMeta(
            image_id=next_id,
            unique_name=unique_name,
            tile_rel=tile_rel,
            tile_path_abs=tile_path_abs.resolve(),
            width=int(width),
            height=int(height),
            transform=transform,
        )
        next_id += 1
    return metas


def iter_polygons(geometry: dict) -> Iterable[List[List[float]]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not isinstance(coords, list):
        return []
    if gtype == "Polygon":
        if coords and isinstance(coords[0], list):
            return [coords[0]]
        return []
    if gtype == "MultiPolygon":
        out: list[List[List[float]]] = []
        for poly in coords:
            if not isinstance(poly, list) or not poly:
                continue
            ring = poly[0]
            if isinstance(ring, list):
                out.append(ring)
        return out
    return []


def poly_area_xy(coords: Sequence[Tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    s = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        s += (x1 * y2) - (x2 * y1)
    return abs(s) * 0.5


def to_bbox(coords: Sequence[Tuple[float, float]]) -> Optional[List[float]]:
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return [float(x_min), float(y_min), float(w), float(h)]


def clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def main() -> None:
    args = parse_args()
    geojson_path = args.geojson.expanduser().resolve()
    manifest_csv = args.manifest_csv.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    if not geojson_path.exists():
        raise SystemExit(f"--geojson not found: {geojson_path}")
    if not manifest_csv.exists():
        raise SystemExit(f"--manifest-csv not found: {manifest_csv}")
    if not images_dir.exists():
        raise SystemExit(f"--images-dir not found: {images_dir}")

    fc = json.loads(geojson_path.read_text(encoding="utf-8"))
    if fc.get("type") != "FeatureCollection":
        raise SystemExit("GeoJSON must be a FeatureCollection.")
    features = fc.get("features", [])
    if not isinstance(features, list):
        raise SystemExit("GeoJSON FeatureCollection missing 'features' list.")

    src_crs = parse_geojson_crs(fc, args.geojson_crs)
    worldfile_crs = normalize_crs(args.worldfile_crs) or "EPSG:31983"
    transformer: Optional[Transformer] = None
    if src_crs != worldfile_crs:
        transformer = Transformer.from_crs(src_crs, worldfile_crs, always_xy=True)

    manifest_rows = read_manifest_rows(manifest_csv)
    image_metas = load_image_metas(manifest_rows, images_dir=images_dir)

    images = [
        {
            "id": m.image_id,
            "file_name": m.unique_name,
            "width": m.width,
            "height": m.height,
        }
        for m in sorted(image_metas.values(), key=lambda x: x.image_id)
    ]

    annotations: list[dict] = []
    ann_id = 1
    skipped_no_tile_rel = 0
    skipped_no_image_meta = 0
    skipped_conf = 0
    skipped_invalid_geom = 0

    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        tile_rel = str(props.get("tile_rel") or "").strip().replace("\\", "/")
        if not tile_rel:
            skipped_no_tile_rel += 1
            continue

        conf = props.get("confidence")
        if conf is not None:
            try:
                conf_v = float(conf)
            except Exception:
                conf_v = None
            if conf_v is not None and conf_v < float(args.min_confidence):
                skipped_conf += 1
                continue

        meta = image_metas.get(tile_rel)
        if meta is None:
            skipped_no_image_meta += 1
            continue

        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            skipped_invalid_geom += 1
            continue

        for ring in iter_polygons(geom):
            if len(ring) < 4:
                skipped_invalid_geom += 1
                continue
            px_coords: list[Tuple[float, float]] = []
            for xy in ring:
                if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                    continue
                xw = float(xy[0])
                yw = float(xy[1])
                if transformer is not None:
                    xw, yw = transformer.transform(xw, yw)
                try:
                    px, py = world_to_pixel(meta.transform, xw, yw)
                except ValueError:
                    continue
                # Keep polygons bounded to image extent so CVAT accepts them.
                px = clamp(px, 0.0, float(meta.width - 1))
                py = clamp(py, 0.0, float(meta.height - 1))
                px_coords.append((px, py))

            if len(px_coords) < 3:
                skipped_invalid_geom += 1
                continue

            # Remove duplicated closing point if present.
            if len(px_coords) >= 2 and px_coords[0] == px_coords[-1]:
                px_coords = px_coords[:-1]
            if len(px_coords) < 3:
                skipped_invalid_geom += 1
                continue

            area = poly_area_xy(px_coords)
            bbox = to_bbox(px_coords)
            if area <= 0 or bbox is None:
                skipped_invalid_geom += 1
                continue

            segmentation: list[float] = []
            for x, y in px_coords:
                segmentation.append(float(x))
                segmentation.append(float(y))
            if len(segmentation) < 6:
                skipped_invalid_geom += 1
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": meta.image_id,
                    "category_id": int(args.category_id),
                    "segmentation": [segmentation],
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "info": {
            "description": "Pool predictions converted from geospatial GeoJSON to image-pixel COCO",
            "source_geojson": str(geojson_path),
            "source_crs": src_crs,
            "worldfile_crs": worldfile_crs,
        },
        "licenses": [],
        "categories": [
            {
                "id": int(args.category_id),
                "name": str(args.category_name),
                "supercategory": str(args.category_name),
            }
        ],
        "images": images,
        "annotations": annotations,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    print("Wrote:", out_path)
    print("images:", len(images))
    print("annotations:", len(annotations))
    print("source_crs:", src_crs)
    print("worldfile_crs:", worldfile_crs)
    print("skipped_no_tile_rel:", skipped_no_tile_rel)
    print("skipped_no_image_meta:", skipped_no_image_meta)
    print("skipped_conf:", skipped_conf)
    print("skipped_invalid_geom:", skipped_invalid_geom)


if __name__ == "__main__":
    main()
