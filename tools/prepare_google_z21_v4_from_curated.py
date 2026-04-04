#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from tile_id_guard import (
    canonical_rel_from_tile_id,
    extract_tile_id,
    extract_tile_id_from_row,
)

try:
    from pyproj import Transformer
except Exception:
    Transformer = None  # type: ignore


GeoTransform = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class FlatTile:
    tile_id: str
    bucket: str
    source_image: Path
    flat_image: Path
    cvat_image: Path


@dataclass(frozen=True)
class TileMeta:
    tile_id: str
    tile_rel: str
    tile_path_abs: Path


@dataclass(frozen=True)
class ImageMeta:
    image_id: int
    tile_id: str
    file_name: str
    path: Path
    width: int
    height: int
    transform: Optional[GeoTransform]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Create google_z21_v4 dataset skeleton, flatten curated validation images into "
            "canonical tile IDs, and export CVAT-ready COCO preannotations."
        )
    )
    ap.add_argument(
        "--curated-images-root",
        type=Path,
        default=Path("data/validation/google_z21_v2_curated_better/images"),
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/segment/z21_ft_v4_targeted_fix_google_z21_c10_20260330_012450"),
        help="Prediction run directory containing pools.geojson and pools_tiles.csv",
    )
    ap.add_argument(
        "--raw-tiles-root",
        type=Path,
        default=Path("data/raw/google/sp_city_2020_rebuild_google_z21"),
    )
    ap.add_argument(
        "--src-dataset-root",
        type=Path,
        default=Path("data/datasets/google_z21_v2"),
        help="Source dataset for dataset.yaml copy",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/datasets/google_z21_v4"),
    )
    ap.add_argument(
        "--flat-root",
        type=Path,
        default=Path("/tmp/google_z21_v4_flat"),
    )
    ap.add_argument(
        "--cvat-root",
        type=Path,
        default=Path("/tmp/google_z21_v4_cvat_prep"),
    )
    ap.add_argument(
        "--worldfile-crs",
        default="EPSG:31983",
        help="CRS of tile worldfiles",
    )
    ap.add_argument(
        "--overwrite-tmp",
        action="store_true",
        help="Overwrite /tmp output folders if they already exist",
    )
    return ap.parse_args()


def norm_crs(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return s.upper()


def parse_geojson_crs(fc: dict) -> str:
    crs_obj = fc.get("crs")
    if isinstance(crs_obj, dict):
        props = crs_obj.get("properties")
        if isinstance(props, dict):
            name = props.get("name")
            if isinstance(name, str) and name.strip():
                return norm_crs(name)
    return "EPSG:31983"


def parse_float(v: object) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def read_worldfile(path: Path) -> GeoTransform:
    vals = [float(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(vals) != 6:
        raise ValueError(f"Worldfile must have 6 lines: {path}")
    a, d, b, e, c, f = vals
    return (c, a, b, f, d, e)


def find_worldfile(img: Path) -> Optional[Path]:
    s = img.suffix.lower()
    if s == ".png":
        candidates = [img.with_suffix(".pgw"), img.with_suffix(".pngw"), img.with_suffix(".wld")]
    elif s in (".jpg", ".jpeg"):
        candidates = [img.with_suffix(".jgw"), img.with_suffix(".jpgw"), img.with_suffix(".wld")]
    else:
        candidates = [img.with_suffix(".wld")]
    for p in candidates:
        if p.exists():
            return p
    return None


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


def clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


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
        out: List[List[List[float]]] = []
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
    acc = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        acc += (x1 * y2) - (x2 * y1)
    return abs(acc) * 0.5


def bbox_xy(coords: Sequence[Tuple[float, float]]) -> Optional[List[float]]:
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return [float(x0), float(y0), float(w), float(h)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_dataset_skeleton(dataset_root: Path, src_dataset_root: Path) -> None:
    ensure_dir(dataset_root / "images" / "train")
    ensure_dir(dataset_root / "images" / "val")
    ensure_dir(dataset_root / "labels" / "train")
    ensure_dir(dataset_root / "labels" / "val")
    src_yaml = src_dataset_root / "dataset.yaml"
    if not src_yaml.exists():
        raise SystemExit(f"Missing source dataset.yaml: {src_yaml}")
    shutil.copy2(src_yaml, dataset_root / "dataset.yaml")


def flatten_curated_images(
    curated_images_root: Path,
    flat_jpegimages_dir: Path,
    cvat_images_dir: Path,
) -> Tuple[List[FlatTile], Dict[str, int]]:
    if not curated_images_root.exists():
        raise SystemExit(f"Curated images root not found: {curated_images_root}")
    ensure_dir(flat_jpegimages_dir)
    ensure_dir(cvat_images_dir)

    flat: List[FlatTile] = []
    seen_tile_ids: Dict[str, Path] = {}
    counters = {
        "source_files_scanned": 0,
        "flattened_copied": 0,
        "skipped_non_png": 0,
    }

    for src in sorted(curated_images_root.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix.lower() != ".png":
            counters["skipped_non_png"] += 1
            continue
        counters["source_files_scanned"] += 1
        rel = src.relative_to(curated_images_root)
        parts = rel.parts
        if len(parts) < 3:
            raise SystemExit(f"Unexpected curated image layout: {src}")

        bucket = parts[0]
        cell = parts[1]
        stem = src.stem
        tile_id = f"{cell}__{stem}"

        if extract_tile_id(tile_id) != tile_id:
            raise SystemExit(f"Invalid canonical tile_id parsed from path: {src} -> {tile_id}")
        if tile_id in seen_tile_ids:
            raise SystemExit(
                f"Duplicate canonical tile_id detected: {tile_id}\n"
                f"  first: {seen_tile_ids[tile_id]}\n"
                f"  second: {src}"
            )
        seen_tile_ids[tile_id] = src

        file_name = f"{tile_id}.png"
        flat_dst = flat_jpegimages_dir / file_name
        cvat_dst = cvat_images_dir / file_name
        shutil.copy2(src, flat_dst)
        shutil.copy2(src, cvat_dst)
        counters["flattened_copied"] += 1

        flat.append(
            FlatTile(
                tile_id=tile_id,
                bucket=bucket,
                source_image=src.resolve(),
                flat_image=flat_dst.resolve(),
                cvat_image=cvat_dst.resolve(),
            )
        )

    return flat, counters


def load_tile_meta_from_csv(tiles_csv: Path, selected_tile_ids: set[str]) -> Tuple[Dict[str, TileMeta], Dict[str, int]]:
    if not tiles_csv.exists():
        raise SystemExit(f"Missing pools_tiles.csv: {tiles_csv}")
    out: Dict[str, TileMeta] = {}
    counters = {
        "tiles_csv_rows_scanned": 0,
        "tiles_csv_missing_tile_id_rows": 0,
        "tiles_csv_duplicate_selected_tile_id_rows": 0,
        "tiles_csv_selected_rows": 0,
    }
    with tiles_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counters["tiles_csv_rows_scanned"] += 1
            tid = extract_tile_id_from_row(row) or ""
            if not tid:
                counters["tiles_csv_missing_tile_id_rows"] += 1
                continue
            if tid not in selected_tile_ids:
                continue
            tile_rel = str(row.get("tile_rel") or "").strip()
            tile_path_abs = Path(str(row.get("tile_path_abs") or "").strip())
            if tid in out:
                counters["tiles_csv_duplicate_selected_tile_id_rows"] += 1
                continue
            out[tid] = TileMeta(tile_id=tid, tile_rel=tile_rel, tile_path_abs=tile_path_abs)
            counters["tiles_csv_selected_rows"] += 1
    return out, counters


def build_image_metas(
    flat_tiles: Sequence[FlatTile],
    tile_meta: Dict[str, TileMeta],
    raw_tiles_root: Path,
) -> Tuple[Dict[str, ImageMeta], Dict[str, int]]:
    out: Dict[str, ImageMeta] = {}
    counters = {
        "missing_tiles_csv_mapping": 0,
        "missing_raw_tile_path": 0,
        "missing_worldfile": 0,
        "images_with_transform": 0,
    }

    for idx, ft in enumerate(sorted(flat_tiles, key=lambda x: x.tile_id), start=1):
        with Image.open(ft.cvat_image) as im:
            width, height = im.size

        meta = tile_meta.get(ft.tile_id)
        transform: Optional[GeoTransform] = None
        if meta is None:
            counters["missing_tiles_csv_mapping"] += 1
            source_tile = raw_tiles_root / (canonical_rel_from_tile_id(ft.tile_id) or "")
        else:
            source_tile = meta.tile_path_abs
            if not source_tile.exists():
                source_tile = raw_tiles_root / (canonical_rel_from_tile_id(ft.tile_id) or "")

        if not source_tile.exists():
            counters["missing_raw_tile_path"] += 1
        else:
            wf = find_worldfile(source_tile)
            if wf is None:
                counters["missing_worldfile"] += 1
            else:
                transform = read_worldfile(wf)
                counters["images_with_transform"] += 1

        out[ft.tile_id] = ImageMeta(
            image_id=idx,
            tile_id=ft.tile_id,
            file_name=ft.cvat_image.name,
            path=ft.cvat_image,
            width=int(width),
            height=int(height),
            transform=transform,
        )

    return out, counters


def build_coco(
    pools_geojson: Path,
    image_metas: Dict[str, ImageMeta],
    selected_tile_ids: set[str],
    *,
    worldfile_crs: str,
) -> Tuple[dict, Dict[str, int]]:
    if not pools_geojson.exists():
        raise SystemExit(f"Missing pools.geojson: {pools_geojson}")

    fc = json.loads(pools_geojson.read_text(encoding="utf-8"))
    features = fc.get("features", [])
    if not isinstance(features, list):
        raise SystemExit("pools.geojson has no valid features list.")

    src_crs = parse_geojson_crs(fc)
    wf_crs = norm_crs(worldfile_crs) or "EPSG:31983"
    transformer = None
    if src_crs != wf_crs:
        if Transformer is None:
            raise SystemExit("pyproj unavailable, cannot transform CRS.")
        transformer = Transformer.from_crs(src_crs, wf_crs, always_xy=True)

    images = [
        {
            "id": m.image_id,
            "file_name": m.file_name,
            "width": m.width,
            "height": m.height,
        }
        for m in sorted(image_metas.values(), key=lambda x: x.image_id)
        if m.path.exists()
    ]
    valid_image_ids = {int(i["id"]) for i in images}

    annotations: List[dict] = []
    ann_id = 1
    counters = {
        "geojson_features_scanned": 0,
        "geojson_features_not_selected_tile": 0,
        "geojson_features_missing_image_meta": 0,
        "geojson_features_missing_transform": 0,
        "geojson_features_missing_geometry": 0,
        "geojson_polygons_invalid": 0,
        "geojson_annotations_built_raw": 0,
    }

    for feat in features:
        counters["geojson_features_scanned"] += 1
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        tile_id = extract_tile_id_from_row(props) or ""
        if not tile_id or tile_id not in selected_tile_ids:
            counters["geojson_features_not_selected_tile"] += 1
            continue
        im = image_metas.get(tile_id)
        if im is None:
            counters["geojson_features_missing_image_meta"] += 1
            continue
        if im.transform is None:
            counters["geojson_features_missing_transform"] += 1
            continue
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            counters["geojson_features_missing_geometry"] += 1
            continue

        conf = parse_float(props.get("confidence"))
        for ring in iter_polygons(geom):
            if len(ring) < 4:
                counters["geojson_polygons_invalid"] += 1
                continue
            px_coords: List[Tuple[float, float]] = []
            for xy in ring:
                if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                    continue
                xw, yw = float(xy[0]), float(xy[1])
                if transformer is not None:
                    xw, yw = transformer.transform(xw, yw)
                try:
                    px, py = world_to_pixel(im.transform, xw, yw)
                except Exception:
                    continue
                px = clamp(px, 0.0, float(im.width - 1))
                py = clamp(py, 0.0, float(im.height - 1))
                px_coords.append((px, py))

            if len(px_coords) >= 2 and px_coords[0] == px_coords[-1]:
                px_coords = px_coords[:-1]
            if len(px_coords) < 3:
                counters["geojson_polygons_invalid"] += 1
                continue

            area = poly_area_xy(px_coords)
            bbox = bbox_xy(px_coords)
            if area <= 0 or bbox is None:
                counters["geojson_polygons_invalid"] += 1
                continue

            seg: List[float] = []
            for x, y in px_coords:
                seg.extend([float(x), float(y)])
            if len(seg) < 6:
                counters["geojson_polygons_invalid"] += 1
                continue

            ann = {
                "id": ann_id,
                "image_id": im.image_id,
                "category_id": 0,
                "segmentation": [seg],
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0,
            }
            if conf is not None:
                ann["score"] = float(conf)
            annotations.append(ann)
            ann_id += 1
            counters["geojson_annotations_built_raw"] += 1

    # COCO consistency gate:
    # 1) only images that exist in images/ dir
    # 2) only annotations with existing image_id
    annotations = [a for a in annotations if int(a.get("image_id", -1)) in valid_image_ids]

    coco = {
        "info": {
            "description": "google_z21_v4 curated predictions (CVAT COCO)",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_geojson": str(pools_geojson.resolve()),
            "source_geojson_crs": src_crs,
            "worldfile_crs": wf_crs,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "pool", "supercategory": "pool"}],
    }
    return coco, counters


def main() -> None:
    args = parse_args()

    curated_images_root = args.curated_images_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    raw_tiles_root = args.raw_tiles_root.expanduser().resolve()
    src_dataset_root = args.src_dataset_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    flat_root = args.flat_root.expanduser().resolve()
    cvat_root = args.cvat_root.expanduser().resolve()
    flat_jpegimages_dir = flat_root / "JPEGImages"
    cvat_images_dir = cvat_root / "images"

    pools_geojson = run_dir / "pools.geojson"
    pools_tiles_csv = run_dir / "pools_tiles.csv"
    coco_json = cvat_root / "google_z21_v4_predictions_coco.json"
    coco_zip = cvat_root / "google_z21_v4_cvat_coco.zip"

    if flat_root.exists():
        if not args.overwrite_tmp:
            raise SystemExit(f"Flat output exists: {flat_root} (use --overwrite-tmp)")
        shutil.rmtree(flat_root)
    if cvat_root.exists():
        if not args.overwrite_tmp:
            raise SystemExit(f"CVAT output exists: {cvat_root} (use --overwrite-tmp)")
        shutil.rmtree(cvat_root)
    ensure_dir(flat_root)
    ensure_dir(cvat_root)

    # STEP 1
    build_dataset_skeleton(dataset_root=dataset_root, src_dataset_root=src_dataset_root)

    # STEP 2
    flat_tiles, flatten_stats = flatten_curated_images(
        curated_images_root=curated_images_root,
        flat_jpegimages_dir=flat_jpegimages_dir,
        cvat_images_dir=cvat_images_dir,
    )
    tile_ids = {ft.tile_id for ft in flat_tiles}

    # STEP 3 + STEP 4
    tile_meta, tile_meta_stats = load_tile_meta_from_csv(pools_tiles_csv, tile_ids)

    # STEP 5
    image_metas, image_meta_stats = build_image_metas(
        flat_tiles=flat_tiles,
        tile_meta=tile_meta,
        raw_tiles_root=raw_tiles_root,
    )
    coco, coco_stats = build_coco(
        pools_geojson=pools_geojson,
        image_metas=image_metas,
        selected_tile_ids=tile_ids,
        worldfile_crs=args.worldfile_crs,
    )

    # STEP 7
    coco_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    # STEP 9
    with zipfile.ZipFile(coco_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(coco_json, arcname=coco_json.name)
        for p in sorted(cvat_images_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path("images") / p.name))

    # Summary
    skipped_or_missing = {
        **flatten_stats,
        **tile_meta_stats,
        **image_meta_stats,
        **coco_stats,
    }
    print("total curated images:", len(flat_tiles))
    print("total annotations:", len(coco.get("annotations", [])))
    print("unique tile_ids:", len(tile_ids))
    print("output dataset root:", dataset_root)
    print("output flat images:", flat_jpegimages_dir)
    print("output cvat images:", cvat_images_dir)
    print("output coco json:", coco_json)
    print("output cvat zip:", coco_zip)
    print("skipped/missing mappings:", json.dumps(skipped_or_missing, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
