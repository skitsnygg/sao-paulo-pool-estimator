#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

from tile_id_guard import (
    collect_tile_ids_from_roots,
    default_existing_image_roots,
    extract_tile_id_from_row,
)


TRUTHY = {"1", "true", "t", "yes", "y"}
NAN_LIKE = {"", "nan", "na", "none", "null"}
DEFAULT_EXIST_TRAIN, DEFAULT_EXIST_VAL = default_existing_image_roots()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Select uncertain prediction tiles from a per-tile inference CSV "
            "for active-learning annotation."
        )
    )
    ap.add_argument("--tiles-csv", type=Path, required=True, help="Input per-tile stats CSV.")
    ap.add_argument("--min-conf", type=float, required=True, help="Minimum max_conf (inclusive).")
    ap.add_argument("--max-conf", type=float, required=True, help="Maximum max_conf (inclusive).")
    ap.add_argument("--num-tiles", type=int, required=True, help="Number of tiles to sample.")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV for selected tiles.")
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling (default: 42).",
    )
    ap.add_argument("--existing-images-train", type=Path, default=DEFAULT_EXIST_TRAIN)
    ap.add_argument("--existing-images-val", type=Path, default=DEFAULT_EXIST_VAL)
    return ap.parse_args()


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in TRUTHY


def is_nan_like(value: object) -> bool:
    return str(value).strip().lower() in NAN_LIKE


def load_rows(csv_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Input CSV has no header: {csv_path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def filter_candidates(
    rows: Sequence[Dict[str, str]],
    *,
    min_conf: float,
    max_conf: float,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        num_preds = parse_int(row.get("num_preds", 0))
        conf = parse_float(row.get("max_conf"))
        skip_reason = row.get("skip_reason", "")
        blank_white = parse_bool(row.get("blank_white", "0"))

        if num_preds <= 0:
            continue
        if conf is None:
            continue
        if conf < min_conf or conf > max_conf:
            continue
        if not is_nan_like(skip_reason):
            continue
        if blank_white:
            continue
        out.append(row)
    return out


def write_rows(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_conf(rows: Sequence[Dict[str, str]]) -> str:
    confs = [parse_float(r.get("max_conf")) for r in rows]
    vals = [v for v in confs if v is not None]
    if not vals:
        return "selected batch max_conf min/mean/max: n/a"
    return (
        "selected batch max_conf min/mean/max: "
        f"{min(vals):.4f} / {mean(vals):.4f} / {max(vals):.4f}"
    )


def main() -> None:
    args = parse_args()

    if args.num_tiles <= 0:
        raise SystemExit("--num-tiles must be > 0")
    if args.min_conf > args.max_conf:
        raise SystemExit("--min-conf must be <= --max-conf")

    rows, fieldnames = load_rows(args.tiles_csv)
    total_candidates_scanned = len(rows)
    existing_roots = [
        args.existing_images_train.expanduser().resolve(),
        args.existing_images_val.expanduser().resolve(),
    ]
    existing_tile_ids, _ = collect_tile_ids_from_roots(existing_roots)
    candidates = filter_candidates(rows, min_conf=args.min_conf, max_conf=args.max_conf)

    prepared: List[Dict[str, str]] = []
    skipped_already_labeled = 0
    skipped_duplicate_in_batch = 0
    seen_tile_ids: set[str] = set()
    for row in candidates:
        tid = extract_tile_id_from_row(row)
        if not tid:
            continue
        if tid in existing_tile_ids:
            skipped_already_labeled += 1
            continue
        if tid in seen_tile_ids:
            skipped_duplicate_in_batch += 1
            continue
        seen_tile_ids.add(tid)
        row2 = dict(row)
        row2["tile_id"] = tid
        prepared.append(row2)
    if "tile_id" not in fieldnames:
        fieldnames = [*fieldnames, "tile_id"]

    rng = random.Random(args.seed)
    n_select = min(args.num_tiles, len(prepared))
    selected = rng.sample(prepared, n_select) if n_select > 0 else []

    write_rows(args.out, selected, fieldnames)

    print(f"total candidates scanned: {total_candidates_scanned}")
    print(f"skipped (already labeled): {skipped_already_labeled}")
    print(f"skipped (duplicate in batch): {skipped_duplicate_in_batch}")
    print(f"final selected count: {len(selected)}")
    print(f"total candidate tiles: {len(prepared)}")
    print(f"tiles selected: {len(selected)}")
    print(summarize_conf(selected))
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
