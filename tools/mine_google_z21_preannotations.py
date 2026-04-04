#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

from tile_id_guard import (
    canonical_rel_from_tile_id,
    collect_tile_ids_from_roots,
    extract_tile_id_from_row,
)

try:
    from pyproj import Transformer
except Exception:
    Transformer = None  # type: ignore

GeoTransform = Tuple[float, float, float, float, float, float]


@dataclass
class SelectedTile:
    tile_id: str
    source_path: Path
    category: str
    preds_per_tile: int
    max_conf: Optional[float]
    mean_conf: Optional[float]
    image_id: int
    file_name: str
    width: int
    height: int
    transform: GeoTransform


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Mine high-value Google z21 tiles from prediction outputs and export CVAT-ready "
            "COCO preannotations."
        )
    )
    ap.add_argument("--tiles-csv", type=Path, required=True)
    ap.add_argument("--pools-geojson", type=Path, required=True)
    ap.add_argument("--tiles-dir", type=Path, required=True)
    ap.add_argument("--exclude-dataset-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def parse_float(v: object) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def parse_int(v: object, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


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


def pick_rows(
    bucket_name: str,
    frame: pd.DataFrame,
    limit: int,
    selected_ids: set[str],
    selected_rows: List[dict],
    stats: Dict[str, int],
) -> None:
    count = 0
    for row in frame.itertuples(index=False):
        tid = str(getattr(row, "tile_id", "")).strip()
        if not tid:
            continue
        if tid in selected_ids:
            stats["skipped_duplicate_in_batch"] += 1
            continue
        selected_ids.add(tid)
        selected_rows.append(
            {
                "tile_id": tid,
                "tile_rel": str(getattr(row, "tile_rel", "")),
                "tile_path_abs": str(getattr(row, "tile_path_abs", "")),
                "category": bucket_name,
                "preds_per_tile": parse_int(getattr(row, "num_preds", 0)),
                "max_conf": parse_float(getattr(row, "max_conf", None)),
                "mean_conf": parse_float(getattr(row, "mean_conf", None)),
            }
        )
        count += 1
        if count >= limit:
            break


def build_selected_tiles(
    tiles_csv: Path,
    tiles_dir: Path,
    exclude_dataset_root: Path,
) -> Tuple[List[dict], Dict[str, int]]:
    df = pd.read_csv(tiles_csv)
    total_candidates_scanned = int(len(df))

    tile_ids: List[str] = []
    for row in df.to_dict(orient="records"):
        tid = extract_tile_id_from_row(row) or ""
        tile_ids.append(tid)
    df["tile_id"] = tile_ids
    df = df[df["tile_id"] != ""].copy().reset_index(drop=True)

    existing_tile_ids, existing_files_scanned = collect_tile_ids_from_roots([exclude_dataset_root])
    before_exclude = len(df)
    df = df[~df["tile_id"].isin(existing_tile_ids)].copy().reset_index(drop=True)
    skipped_already_labeled = int(before_exclude - len(df))

    df["num_preds"] = pd.to_numeric(df.get("num_preds"), errors="coerce").fillna(0).astype(int)
    for c in ("max_conf", "mean_conf", "min_conf", "max_area_m2", "sum_area_m2"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.Series([math.nan] * len(df))
    if "blank_white" in df.columns:
        df["blank_white"] = pd.to_numeric(df["blank_white"], errors="coerce").fillna(0).astype(int)
    else:
        df["blank_white"] = 0

    # Keep only non-white tiles by default for annotation value.
    df = df[df["blank_white"] == 0].copy().reset_index(drop=True)

    low_conf = df[(df["num_preds"] > 0) & (df["max_conf"].notna()) & (df["max_conf"] < 0.40)].copy()
    low_conf = low_conf.sort_values(by=["max_conf", "mean_conf", "num_preds", "tile_id"], ascending=[True, True, False, True])

    many_preds = df[df["num_preds"] >= 4].copy()
    many_preds = many_preds.sort_values(by=["num_preds", "sum_area_m2", "tile_id"], ascending=[False, False, True])

    large_masks = df[df["num_preds"] > 0].copy()
    large_masks = large_masks.sort_values(by=["max_area_m2", "sum_area_m2", "tile_id"], ascending=[False, False, True])

    false_positive = df[df["num_preds"] > 0].copy()
    fp_mask = (
        (false_positive["mean_conf"].notna() & (false_positive["mean_conf"] < 0.45))
        | (false_positive["max_conf"].notna() & (false_positive["max_conf"] < 0.55))
        | ((false_positive["num_preds"] <= 2) & (false_positive["max_area_m2"].fillna(0.0) < 20.0))
    )
    false_positive = false_positive[fp_mask].copy()
    false_positive = false_positive.sort_values(
        by=["mean_conf", "max_conf", "num_preds", "max_area_m2", "tile_id"],
        ascending=[True, True, False, True, True],
    )

    hard_empty = df[df["num_preds"] == 0].copy()
    hard_empty = hard_empty.sort_values(by=["tile_id"], ascending=[True])

    selected_ids: set[str] = set()
    selected_rows: List[dict] = []
    stats: Dict[str, int] = {
        "total_candidates_scanned": total_candidates_scanned,
        "existing_files_scanned": int(existing_files_scanned),
        "existing_tile_ids": int(len(existing_tile_ids)),
        "skipped_already_labeled": skipped_already_labeled,
        "skipped_duplicate_in_batch": 0,
        "selected_low_conf": 0,
        "selected_many_preds": 0,
        "selected_large_masks": 0,
        "selected_false_positive": 0,
        "selected_hard_empty": 0,
    }

    before = len(selected_rows)
    pick_rows("low_conf", low_conf, 120, selected_ids, selected_rows, stats)
    stats["selected_low_conf"] = len(selected_rows) - before

    before = len(selected_rows)
    pick_rows("many_preds", many_preds, 50, selected_ids, selected_rows, stats)
    stats["selected_many_preds"] = len(selected_rows) - before

    before = len(selected_rows)
    pick_rows("large_masks", large_masks, 50, selected_ids, selected_rows, stats)
    stats["selected_large_masks"] = len(selected_rows) - before

    before = len(selected_rows)
    pick_rows("false_positive", false_positive, 80, selected_ids, selected_rows, stats)
    stats["selected_false_positive"] = len(selected_rows) - before

    before = len(selected_rows)
    pick_rows("hard_empty", hard_empty, 60, selected_ids, selected_rows, stats)
    stats["selected_hard_empty"] = len(selected_rows) - before

    # Resolve source paths deterministically.
    for row in selected_rows:
        src = Path(str(row.get("tile_path_abs", "")).strip())
        if not src.exists():
            rel = canonical_rel_from_tile_id(row["tile_id"]) or str(row.get("tile_rel", ""))
            src = tiles_dir / rel
        row["source_path"] = str(src.resolve()) if src.exists() else ""

    selected_rows = [r for r in selected_rows if r["source_path"]]
    return selected_rows, stats


def copy_images_and_manifest(
    out_root: Path,
    selected_rows: Sequence[dict],
) -> Tuple[Dict[str, SelectedTile], Path]:
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"

    selected_map: Dict[str, SelectedTile] = {}
    next_image_id = 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tile_id", "source_path", "category", "preds_per_tile", "max_conf", "mean_conf"],
        )
        writer.writeheader()
        for row in selected_rows:
            tile_id = str(row["tile_id"])
            src = Path(str(row["source_path"]))
            if not src.exists():
                continue
            ext = src.suffix.lower() or ".png"
            dst_name = f"{tile_id}{ext}"
            dst = images_dir / dst_name
            shutil.copy2(src, dst)

            worldfile = find_worldfile(src)
            if worldfile is None:
                continue
            transform = read_worldfile(worldfile)
            with Image.open(src) as im:
                w, h = im.size

            selected_map[tile_id] = SelectedTile(
                tile_id=tile_id,
                source_path=src,
                category=str(row["category"]),
                preds_per_tile=parse_int(row.get("preds_per_tile"), 0),
                max_conf=parse_float(row.get("max_conf")),
                mean_conf=parse_float(row.get("mean_conf")),
                image_id=next_image_id,
                file_name=f"images/{dst_name}",
                width=int(w),
                height=int(h),
                transform=transform,
            )
            next_image_id += 1

            writer.writerow(
                {
                    "tile_id": tile_id,
                    "source_path": str(src),
                    "category": row["category"],
                    "preds_per_tile": row["preds_per_tile"],
                    "max_conf": "" if row["max_conf"] is None else f"{float(row['max_conf']):.6f}",
                    "mean_conf": "" if row["mean_conf"] is None else f"{float(row['mean_conf']):.6f}",
                }
            )

    return selected_map, manifest_path


def build_coco_from_geojson(
    pools_geojson: Path,
    selected_map: Dict[str, SelectedTile],
    out_json: Path,
) -> Tuple[int, int]:
    fc = json.loads(pools_geojson.read_text(encoding="utf-8"))
    features = fc.get("features", [])
    if not isinstance(features, list):
        raise SystemExit("pools.geojson has no valid features list.")

    src_crs = parse_geojson_crs(fc)
    worldfile_crs = "EPSG:31983"
    transformer = None
    if Transformer is not None and norm_crs(src_crs) and norm_crs(src_crs) != norm_crs(worldfile_crs):
        transformer = Transformer.from_crs(src_crs, worldfile_crs, always_xy=True)

    images = [
        {
            "id": t.image_id,
            "file_name": t.file_name,
            "width": t.width,
            "height": t.height,
        }
        for t in sorted(selected_map.values(), key=lambda x: x.image_id)
    ]

    annotations: List[dict] = []
    ann_id = 1

    for feat in features:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        tile_id = extract_tile_id_from_row(props) or ""
        if not tile_id or tile_id not in selected_map:
            continue
        meta = selected_map[tile_id]
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            continue
        confidence = parse_float(props.get("confidence"))

        for ring in iter_polygons(geom):
            if len(ring) < 4:
                continue
            px_coords: List[Tuple[float, float]] = []
            for xy in ring:
                if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                    continue
                xw, yw = float(xy[0]), float(xy[1])
                if transformer is not None:
                    xw, yw = transformer.transform(xw, yw)
                try:
                    px, py = world_to_pixel(meta.transform, xw, yw)
                except Exception:
                    continue
                px = clamp(px, 0.0, float(meta.width - 1))
                py = clamp(py, 0.0, float(meta.height - 1))
                px_coords.append((px, py))

            if len(px_coords) >= 2 and px_coords[0] == px_coords[-1]:
                px_coords = px_coords[:-1]
            if len(px_coords) < 3:
                continue

            area = poly_area_xy(px_coords)
            bbox = bbox_xy(px_coords)
            if area <= 0 or bbox is None:
                continue

            seg: List[float] = []
            for x, y in px_coords:
                seg.extend([float(x), float(y)])
            if len(seg) < 6:
                continue

            ann = {
                "id": ann_id,
                "image_id": meta.image_id,
                "category_id": 1,
                "segmentation": [seg],
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0,
            }
            if confidence is not None:
                ann["score"] = float(confidence)
            annotations.append(ann)
            ann_id += 1

    coco = {
        "info": {
            "description": "Google z21 round1 preannotations from prediction run",
            "source_geojson": str(pools_geojson),
            "source_crs": src_crs,
            "worldfile_crs": worldfile_crs,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pool", "supercategory": "pool"}],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return len(images), len(annotations)


def build_zip(out_root: Path, coco_json: Path, zip_path: Path) -> None:
    images_dir = out_root / "images"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(coco_json, arcname=coco_json.name)
        for p in sorted(images_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path("images") / p.name))


def main() -> None:
    args = parse_args()
    tiles_csv = args.tiles_csv.expanduser().resolve()
    pools_geojson = args.pools_geojson.expanduser().resolve()
    tiles_dir = args.tiles_dir.expanduser().resolve()
    exclude_dataset_root = args.exclude_dataset_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    if not tiles_csv.exists():
        raise SystemExit(f"tiles csv not found: {tiles_csv}")
    if not pools_geojson.exists():
        raise SystemExit(f"pools geojson not found: {pools_geojson}")
    if not tiles_dir.exists():
        raise SystemExit(f"tiles dir not found: {tiles_dir}")
    if not exclude_dataset_root.exists():
        raise SystemExit(f"exclude dataset root not found: {exclude_dataset_root}")

    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output root exists: {out_root} (use --overwrite)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_rows, sel_stats = build_selected_tiles(
        tiles_csv=tiles_csv,
        tiles_dir=tiles_dir,
        exclude_dataset_root=exclude_dataset_root,
    )
    selected_map, manifest_path = copy_images_and_manifest(out_root=out_root, selected_rows=selected_rows)

    coco_json = out_root / "google_z21_round1_predictions_coco.json"
    coco_images, coco_annotations = build_coco_from_geojson(
        pools_geojson=pools_geojson,
        selected_map=selected_map,
        out_json=coco_json,
    )

    zip_path = out_root / "google_z21_round1_cvat_coco.zip"
    build_zip(out_root=out_root, coco_json=coco_json, zip_path=zip_path)

    counts_per_category: Dict[str, int] = {}
    for t in selected_map.values():
        counts_per_category[t.category] = counts_per_category.get(t.category, 0) + 1

    stats = {
        **sel_stats,
        "total_selected": len(selected_map),
        "counts_per_category": counts_per_category,
        "unique_tile_ids": len(selected_map),
        "coco_images": coco_images,
        "coco_annotations": coco_annotations,
        "manifest_csv": str(manifest_path),
        "coco_json": str(coco_json),
        "zip_path": str(zip_path),
    }
    stats_path = out_root / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("total selected:", stats["total_selected"])
    print("counts per category:", json.dumps(counts_per_category, ensure_ascii=False, sort_keys=True))
    print("unique tile ids:", stats["unique_tile_ids"])
    print("number of coco images:", coco_images)
    print("number of coco annotations:", coco_annotations)
    print("output json path:", coco_json)
    print("output zip path:", zip_path)


if __name__ == "__main__":
    main()
