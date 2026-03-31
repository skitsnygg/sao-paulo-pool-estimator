from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set

from tile_id_guard import (
    canonical_rel_from_tile_id,
    collect_tile_ids_from_roots,
    default_existing_image_roots,
    extract_tile_id_from_row,
)


DEFAULT_EXIST_TRAIN, DEFAULT_EXIST_VAL = default_existing_image_roots()


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def copy_tile(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def pick_unique(
    rows: List[Dict[str, str]],
    already_used_tile_ids: Set[str],
    limit: int,
    stats: Dict[str, int],
) -> List[Dict[str, str]]:
    picked: List[Dict[str, str]] = []
    for row in rows:
        tile_id = str(row.get("_tile_id", "")).strip()
        if not tile_id:
            continue
        if tile_id in already_used_tile_ids:
            stats["skipped_duplicate_in_batch"] += 1
            continue
        picked.append(row)
        already_used_tile_ids.add(tile_id)
        if len(picked) >= limit:
            break
    return picked


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = [
        "bucket",
        "tile_id",
        "tile_rel",
        "cell",
        "tile",
        "tile_path_abs",
        "blank_white",
        "num_preds",
        "min_conf",
        "mean_conf",
        "max_conf",
        "max_area_m2",
        "sum_area_m2",
        "max_mask_area_px",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tiles-csv",
        type=Path,
        required=True,
        help="Per-tile summary CSV from predict_tiles_to_geojson.py",
    )
    ap.add_argument(
        "--tiles-root",
        type=Path,
        required=True,
        help="Root directory containing source image tiles",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for annotation buckets",
    )
    ap.add_argument("--low-conf-count", type=int, default=80)
    ap.add_argument("--large-mask-count", type=int, default=60)
    ap.add_argument("--many-preds-count", type=int, default=60)
    ap.add_argument("--random-count", type=int, default=40)
    ap.add_argument(
        "--low-conf-max",
        type=float,
        default=0.30,
        help="Only include low-conf rows with min_conf <= this value",
    )
    ap.add_argument(
        "--many-preds-min",
        type=int,
        default=3,
        help="Only include many-preds rows with num_preds >= this value",
    )
    ap.add_argument(
        "--max-white",
        type=int,
        default=0,
        help="Require blank_white <= this value. Keep 0 to reject white tiles.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--existing-images-train",
        type=Path,
        default=DEFAULT_EXIST_TRAIN,
        help=f"Existing dataset train images root (default: {DEFAULT_EXIST_TRAIN}).",
    )
    ap.add_argument(
        "--existing-images-val",
        type=Path,
        default=DEFAULT_EXIST_VAL,
        help=f"Existing dataset val images root (default: {DEFAULT_EXIST_VAL}).",
    )
    args = ap.parse_args()

    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Out dir exists: {args.out_dir} (use --overwrite)")
        shutil.rmtree(args.out_dir)

    rows = read_rows(args.tiles_csv)
    total_candidates_scanned = len(rows)
    existing_roots = [
        args.existing_images_train.expanduser().resolve(),
        args.existing_images_val.expanduser().resolve(),
    ]
    existing_tile_ids, existing_files_scanned = collect_tile_ids_from_roots(existing_roots)
    skipped_already_labeled = 0
    stats = {"skipped_duplicate_in_batch": 0}

    # Filter to non-white positive tiles with real source files.
    usable: List[Dict[str, str]] = []
    for row in rows:
        blank_white = as_int(row.get("blank_white", "0"))
        num_preds = as_int(row.get("num_preds", "0"))

        if blank_white > args.max_white:
            continue
        if num_preds <= 0:
            continue
        tile_id = extract_tile_id_from_row(row)
        if not tile_id:
            continue
        if tile_id in existing_tile_ids:
            skipped_already_labeled += 1
            continue

        canonical_rel = canonical_rel_from_tile_id(tile_id) or row.get("tile_rel", "")
        src = args.tiles_root / canonical_rel
        if not src.exists():
            src = args.tiles_root / str(row.get("tile_rel", ""))
        if not src.exists():
            tile_path_abs = str(row.get("tile_path_abs", "")).strip()
            src = Path(tile_path_abs) if tile_path_abs else src
        if not src.exists():
            continue

        row = dict(row)
        row["_src"] = str(src)
        row["_tile_id"] = tile_id
        row["_tile_rel_canonical"] = canonical_rel
        usable.append(row)

    # Buckets
    low_conf_candidates = [
        r for r in usable
        if r.get("min_conf", "") != "" and as_float(r["min_conf"]) <= args.low_conf_max
    ]
    low_conf_candidates.sort(key=lambda r: as_float(r["min_conf"], 999.0))

    large_mask_candidates = [
        r for r in usable
        if r.get("max_area_m2", "") != ""
    ]
    large_mask_candidates.sort(key=lambda r: as_float(r["max_area_m2"], -1.0), reverse=True)

    many_preds_candidates = [
        r for r in usable
        if as_int(r["num_preds"]) >= args.many_preds_min
    ]
    many_preds_candidates.sort(
        key=lambda r: (as_int(r["num_preds"]), as_float(r.get("sum_area_m2", "0"))),
        reverse=True,
    )

    random_candidates = list(usable)
    rng = random.Random(args.seed)
    rng.shuffle(random_candidates)

    used_tile_ids: Set[str] = set()

    buckets = {
        "low_conf": pick_unique(low_conf_candidates, used_tile_ids, args.low_conf_count, stats),
        "large_masks": pick_unique(large_mask_candidates, used_tile_ids, args.large_mask_count, stats),
        "many_preds": pick_unique(many_preds_candidates, used_tile_ids, args.many_preds_count, stats),
        "random": pick_unique(random_candidates, used_tile_ids, args.random_count, stats),
    }

    manifest_rows: List[Dict[str, str]] = []

    for bucket, picked in buckets.items():
        bucket_dir = args.out_dir / bucket
        for row in picked:
            src = Path(row["_src"])
            rel_for_dst = str(row.get("_tile_rel_canonical") or row["tile_rel"])
            dst = bucket_dir / rel_for_dst
            copy_tile(src, dst)

            manifest_rows.append(
                {
                    "bucket": bucket,
                    "tile_id": row["_tile_id"],
                    "tile_rel": str(row.get("_tile_rel_canonical") or row["tile_rel"]),
                    "cell": row["cell"],
                    "tile": row["tile"],
                    "tile_path_abs": row["tile_path_abs"],
                    "blank_white": row["blank_white"],
                    "num_preds": row["num_preds"],
                    "min_conf": row["min_conf"],
                    "mean_conf": row["mean_conf"],
                    "max_conf": row["max_conf"],
                    "max_area_m2": row["max_area_m2"],
                    "sum_area_m2": row["sum_area_m2"],
                    "max_mask_area_px": row["max_mask_area_px"],
                }
            )

        write_manifest(bucket_dir / "manifest.csv", [
            {
                "bucket": bucket,
                "tile_id": row["_tile_id"],
                "tile_rel": str(row.get("_tile_rel_canonical") or row["tile_rel"]),
                "cell": row["cell"],
                "tile": row["tile"],
                "tile_path_abs": row["tile_path_abs"],
                "blank_white": row["blank_white"],
                "num_preds": row["num_preds"],
                "min_conf": row["min_conf"],
                "mean_conf": row["mean_conf"],
                "max_conf": row["max_conf"],
                "max_area_m2": row["max_area_m2"],
                "sum_area_m2": row["sum_area_m2"],
                "max_mask_area_px": row["max_mask_area_px"],
            }
            for row in picked
        ])

    write_manifest(args.out_dir / "manifest.csv", manifest_rows)

    final_selected_count = sum(len(v) for v in buckets.values())
    print("total candidates scanned:", total_candidates_scanned)
    print("existing tile ids loaded:", len(existing_tile_ids))
    print("existing files scanned:", existing_files_scanned)
    print("skipped (already labeled):", skipped_already_labeled)
    print("skipped (duplicate in batch):", stats["skipped_duplicate_in_batch"])
    print("final selected count:", final_selected_count)
    print("usable positive tiles:", len(usable))
    for bucket, picked in buckets.items():
        print(f"{bucket}: {len(picked)}")
    print("wrote:", args.out_dir)


if __name__ == "__main__":
    main()
