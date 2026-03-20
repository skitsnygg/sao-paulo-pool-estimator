#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence


TRUTHY = {"1", "true", "t", "yes", "y"}
NAN_LIKE = {"", "nan", "na", "none", "null"}


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
    candidates = filter_candidates(rows, min_conf=args.min_conf, max_conf=args.max_conf)

    rng = random.Random(args.seed)
    n_select = min(args.num_tiles, len(candidates))
    selected = rng.sample(candidates, n_select) if n_select > 0 else []

    write_rows(args.out, selected, fieldnames)

    print(f"total candidate tiles: {len(candidates)}")
    print(f"tiles selected: {len(selected)}")
    print(summarize_conf(selected))
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
