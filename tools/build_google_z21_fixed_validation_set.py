#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from tile_id_guard import canonical_rel_from_tile_id, collect_tile_ids_from_roots, extract_tile_id_from_row


@dataclass(frozen=True)
class TileRow:
    tile_id: str
    tile_rel: str
    tile_path_abs: str
    cell: str
    num_preds: int
    min_conf: Optional[float]
    mean_conf: Optional[float]
    max_conf: Optional[float]
    max_area_m2: Optional[float]
    sum_area_m2: Optional[float]
    max_mask_area_px: Optional[float]


@dataclass(frozen=True)
class TileFeatures:
    brightness: float
    dark_ratio: float
    blue_ratio: float
    green_ratio: float
    gray_ratio: float


@dataclass(frozen=True)
class Candidate:
    row: TileRow
    src: Path
    features: TileFeatures
    origins: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a fixed curated validation set for google_z21 using outputs from existing "
            "selection scripts, with strict canonical tile-id dedupe."
        )
    )
    ap.add_argument("--tiles-csv", type=Path, required=True)
    ap.add_argument("--tiles-root", type=Path, required=True)
    ap.add_argument(
        "--candidate-csv",
        type=Path,
        action="append",
        required=True,
        help="CSV manifest/candidate file produced by existing selection scripts.",
    )
    ap.add_argument(
        "--exclude-val-root",
        type=Path,
        action="append",
        default=[],
        help="Validation image root(s) to exclude from selection.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def as_int(v: object, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def as_float(v: object) -> Optional[float]:
    try:
        x = float(str(v).strip())
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def load_tiles(csv_path: Path) -> Dict[str, TileRow]:
    out: Dict[str, TileRow] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            tid = extract_tile_id_from_row(raw) or ""
            if not tid:
                continue
            if tid in out:
                continue
            out[tid] = TileRow(
                tile_id=tid,
                tile_rel=str(raw.get("tile_rel") or "").strip(),
                tile_path_abs=str(raw.get("tile_path_abs") or "").strip(),
                cell=str(raw.get("cell") or "").strip(),
                num_preds=as_int(raw.get("num_preds", 0)),
                min_conf=as_float(raw.get("min_conf")),
                mean_conf=as_float(raw.get("mean_conf")),
                max_conf=as_float(raw.get("max_conf")),
                max_area_m2=as_float(raw.get("max_area_m2")),
                sum_area_m2=as_float(raw.get("sum_area_m2")),
                max_mask_area_px=as_float(raw.get("max_mask_area_px")),
            )
    return out


def load_candidate_ids(csv_paths: Sequence[Path]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    tile_ids: Set[str] = set()
    origins: Dict[str, Set[str]] = {}
    for p in csv_paths:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = extract_tile_id_from_row(row) or ""
                if not tid:
                    continue
                base = p.stem
                bucket = str(row.get("bucket") or "").strip()
                tag = f"{base}:{bucket}" if bucket else base
                tile_ids.add(tid)
                origins.setdefault(tid, set()).add(tag)
    return tile_ids, origins


def source_path_for_row(row: TileRow, tiles_root: Path) -> Optional[Path]:
    rel = canonical_rel_from_tile_id(row.tile_id) or row.tile_rel
    c1 = tiles_root / rel
    if c1.exists():
        return c1
    c2 = tiles_root / row.tile_rel
    if c2.exists():
        return c2
    if row.tile_path_abs:
        c3 = Path(row.tile_path_abs)
        if c3.exists():
            return c3
    return None


def compute_features(path: Path) -> TileFeatures:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    v = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    sat = np.where(v > 1e-6, (v - min_rgb) / np.maximum(v, 1e-6), 0.0)
    brightness = float(arr.mean())
    dark_ratio = float(np.mean(v < 0.25))
    blue_ratio = float(np.mean((b > 0.35) & (b > r * 1.15) & (b > g * 1.05)))
    green_ratio = float(np.mean((g > 0.30) & (g > r * 1.08) & (g > b * 1.06)))
    gray_ratio = float(np.mean(sat < 0.12))
    return TileFeatures(
        brightness=brightness,
        dark_ratio=dark_ratio,
        blue_ratio=blue_ratio,
        green_ratio=green_ratio,
        gray_ratio=gray_ratio,
    )


def stable_unique(cands: Iterable[Candidate]) -> List[Candidate]:
    out: List[Candidate] = []
    seen: Set[str] = set()
    for c in cands:
        if c.row.tile_id in seen:
            continue
        seen.add(c.row.tile_id)
        out.append(c)
    return out


def sorted_candidates(cands: Iterable[Candidate], key_fn) -> List[Candidate]:
    arr = list(cands)
    arr.sort(key=key_fn)
    return arr


def note_for(bucket: str, c: Candidate) -> str:
    r = c.row
    f = c.features
    base = (
        f"origins={';'.join(c.origins)} "
        f"preds={r.num_preds} "
        f"max_conf={'' if r.max_conf is None else f'{r.max_conf:.3f}'} "
        f"max_area_m2={'' if r.max_area_m2 is None else f'{r.max_area_m2:.2f}'} "
        f"blue={f.blue_ratio:.3f} green={f.green_ratio:.3f} dark={f.dark_ratio:.3f}"
    )
    return f"{bucket}: {base}"


def pick_bucket(
    bucket: str,
    ordered: Sequence[Candidate],
    target: int,
    used_tile_ids: Set[str],
    duplicate_counter: Dict[str, int],
) -> List[Tuple[Candidate, str]]:
    picked: List[Tuple[Candidate, str]] = []
    for c in ordered:
        tid = c.row.tile_id
        if tid in used_tile_ids:
            duplicate_counter["skipped_duplicate_in_batch"] += 1
            continue
        used_tile_ids.add(tid)
        picked.append((c, note_for(bucket, c)))
        if len(picked) >= target:
            break
    return picked


def ensure_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"out-dir exists: {path} (use --overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    tiles = load_tiles(args.tiles_csv)
    candidate_ids, origin_map = load_candidate_ids(args.candidate_csv)

    exclude_roots: List[Path] = [p.expanduser().resolve() for p in args.exclude_val_root]
    auto_val_roots = sorted(Path("data/datasets").glob("*/images/val"))
    for p in auto_val_roots:
        rp = p.resolve()
        if rp not in exclude_roots:
            exclude_roots.append(rp)
    excluded_tile_ids, excluded_files_scanned = collect_tile_ids_from_roots(exclude_roots)

    missing_from_tiles_csv = 0
    missing_source = 0
    excluded_existing_val = 0

    features_cache: Dict[Path, TileFeatures] = {}
    candidates: List[Candidate] = []
    for tid in sorted(candidate_ids):
        row = tiles.get(tid)
        if row is None:
            missing_from_tiles_csv += 1
            continue
        if tid in excluded_tile_ids:
            excluded_existing_val += 1
            continue
        src = source_path_for_row(row, args.tiles_root)
        if src is None:
            missing_source += 1
            continue
        if src not in features_cache:
            features_cache[src] = compute_features(src)
        c = Candidate(
            row=row,
            src=src,
            features=features_cache[src],
            origins=tuple(sorted(origin_map.get(tid, {"unknown"}))),
        )
        candidates.append(c)

    # Deterministic shuffle to avoid bias when metric ties are common, followed by stable sorting.
    rng.shuffle(candidates)

    positives = [c for c in candidates if c.row.num_preds > 0]
    empties = [c for c in candidates if c.row.num_preds == 0]

    obvious_primary = sorted_candidates(
        (
            c
            for c in positives
            if (c.row.max_conf or 0.0) >= 0.82
            and (c.row.max_area_m2 or 0.0) >= 16.0
            and c.row.num_preds <= 3
        ),
        lambda c: (
            -(c.row.max_conf or 0.0),
            -(c.row.max_area_m2 or 0.0),
            c.row.num_preds,
            c.row.tile_id,
        ),
    )
    obvious_fallback = sorted_candidates(
        (
            c
            for c in positives
            if (c.row.max_conf or 0.0) >= 0.76 and (c.row.max_area_m2 or 0.0) >= 12.0
        ),
        lambda c: (-(c.row.max_conf or 0.0), -(c.row.max_area_m2 or 0.0), c.row.tile_id),
    )

    small_primary = sorted_candidates(
        (
            c
            for c in positives
            if 4.0 <= (c.row.max_area_m2 or 0.0) <= 18.0 and (c.row.max_conf or 0.0) >= 0.55
        ),
        lambda c: (
            abs((c.row.max_area_m2 or 0.0) - 10.0),
            -(c.row.max_conf or 0.0),
            c.row.tile_id,
        ),
    )
    small_fallback = sorted_candidates(
        (
            c
            for c in positives
            if 3.0 <= (c.row.max_area_m2 or 0.0) <= 22.0 and (c.row.max_conf or 0.0) >= 0.45
        ),
        lambda c: (
            abs((c.row.max_area_m2 or 0.0) - 11.0),
            -(c.row.max_conf or 0.0),
            c.row.tile_id,
        ),
    )

    covered_primary = sorted_candidates(
        (
            c
            for c in positives
            if c.features.dark_ratio >= 0.30
            and (c.row.max_conf or 0.0) <= 0.82
            and (c.row.max_area_m2 or 0.0) <= 60.0
        ),
        lambda c: (
            -c.features.dark_ratio,
            (c.row.max_conf or 1.0),
            -(c.row.max_area_m2 or 0.0),
            c.row.tile_id,
        ),
    )
    covered_fallback = sorted_candidates(
        (
            c
            for c in positives
            if c.features.dark_ratio >= 0.24 and (c.row.max_conf or 0.0) <= 0.86
        ),
        lambda c: (-c.features.dark_ratio, (c.row.max_conf or 1.0), c.row.tile_id),
    )

    dense_primary = sorted_candidates(
        (c for c in positives if c.row.num_preds >= 5),
        lambda c: (
            -c.row.num_preds,
            -(c.row.sum_area_m2 or 0.0),
            -(c.row.max_conf or 0.0),
            c.row.tile_id,
        ),
    )
    dense_fallback = sorted_candidates(
        (c for c in positives if c.row.num_preds >= 4),
        lambda c: (
            -c.row.num_preds,
            -(c.row.sum_area_m2 or 0.0),
            c.row.tile_id,
        ),
    )

    hard_neg_primary = sorted_candidates(
        (
            c
            for c in positives
            if c.row.num_preds <= 3
            and (c.row.max_conf or 1.0) <= 0.65
            and (c.row.mean_conf or 1.0) <= 0.58
            and (c.row.max_area_m2 or 0.0) <= 70.0
            and c.features.blue_ratio >= 0.17
            and c.features.green_ratio <= 0.28
        ),
        lambda c: (
            -c.features.blue_ratio,
            (c.row.mean_conf or 1.0),
            (c.row.max_conf or 1.0),
            c.row.tile_id,
        ),
    )
    hard_neg_fallback = sorted_candidates(
        (
            c
            for c in positives
            if c.row.num_preds <= 3
            and (c.row.max_conf or 1.0) <= 0.70
            and c.features.blue_ratio >= 0.13
            and c.features.green_ratio <= 0.30
        ),
        lambda c: (-c.features.blue_ratio, (c.row.max_conf or 1.0), c.row.tile_id),
    )

    empty_primary = sorted_candidates(
        (
            c
            for c in empties
            if c.features.green_ratio <= 0.22
            and c.features.blue_ratio <= 0.18
            and c.features.dark_ratio <= 0.45
        ),
        lambda c: (
            c.features.green_ratio,
            abs(c.features.brightness - 0.55),
            c.row.tile_id,
        ),
    )
    empty_fallback = sorted_candidates(
        (
            c
            for c in empties
            if c.features.green_ratio <= 0.26 and c.features.blue_ratio <= 0.22
        ),
        lambda c: (
            c.features.green_ratio,
            abs(c.features.brightness - 0.55),
            c.row.tile_id,
        ),
    )

    low_conf_primary = sorted_candidates(
        (
            c
            for c in positives
            if 0.32 <= (c.row.max_conf or 0.0) <= 0.58
            and (c.row.mean_conf or 0.0) <= 0.62
        ),
        lambda c: (
            abs((c.row.max_conf or 0.0) - 0.45),
            -c.row.num_preds,
            -(c.row.max_area_m2 or 0.0),
            c.row.tile_id,
        ),
    )
    low_conf_fallback = sorted_candidates(
        (c for c in positives if 0.30 <= (c.row.max_conf or 0.0) <= 0.62),
        lambda c: (
            abs((c.row.max_conf or 0.0) - 0.46),
            -c.row.num_preds,
            c.row.tile_id,
        ),
    )

    buckets_plan = [
        ("obvious_real_pools", 35, stable_unique([*obvious_primary, *obvious_fallback])),
        ("small_pools", 30, stable_unique([*small_primary, *small_fallback])),
        ("covered_dark_pools", 30, stable_unique([*covered_primary, *covered_fallback])),
        ("dense_many_pools", 30, stable_unique([*dense_primary, *dense_fallback])),
        ("hard_negatives_lookalikes", 40, stable_unique([*hard_neg_primary, *hard_neg_fallback])),
        ("true_empty_negatives", 35, stable_unique([*empty_primary, *empty_fallback])),
        ("low_conf_borderline", 30, stable_unique([*low_conf_primary, *low_conf_fallback])),
    ]

    used_tile_ids: Set[str] = set()
    dup_stats = {"skipped_duplicate_in_batch": 0}
    selected_rows: List[Dict[str, object]] = []
    counts_per_bucket: Dict[str, int] = {}

    for bucket, target, ordered in buckets_plan:
        picked = pick_bucket(bucket, ordered, target, used_tile_ids, dup_stats)
        counts_per_bucket[bucket] = len(picked)
        for c, note in picked:
            selected_rows.append(
                {
                    "tile_id": c.row.tile_id,
                    "source_path": str(c.src.resolve()),
                    "category_bucket": bucket,
                    "preds_per_tile": c.row.num_preds,
                    "max_conf": "" if c.row.max_conf is None else f"{c.row.max_conf:.6f}",
                    "mean_conf": "" if c.row.mean_conf is None else f"{c.row.mean_conf:.6f}",
                    "notes": note,
                }
            )

    ensure_out_dir(args.out_dir, overwrite=args.overwrite)
    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for row in selected_rows:
        tid = str(row["tile_id"])
        src = Path(str(row["source_path"]))
        bucket = str(row["category_bucket"])
        rel = canonical_rel_from_tile_id(tid) or f"{tid}.png"
        rel_path = Path(rel)
        dst = images_dir / bucket / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tile_id",
            "source_path",
            "category_bucket",
            "preds_per_tile",
            "max_conf",
            "mean_conf",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "tiles_csv": str(args.tiles_csv),
        "tiles_root": str(args.tiles_root),
        "candidate_csvs": [str(p) for p in args.candidate_csv],
        "exclude_val_roots": [str(p) for p in exclude_roots],
        "excluded_files_scanned": int(excluded_files_scanned),
        "excluded_existing_val_tile_ids": int(excluded_existing_val),
        "candidate_universe_tile_ids": int(len(candidate_ids)),
        "candidate_rows_after_exclusions": int(len(candidates)),
        "missing_from_tiles_csv": int(missing_from_tiles_csv),
        "missing_source_path": int(missing_source),
        "total_selected": int(len(selected_rows)),
        "unique_tile_ids": int(len({r["tile_id"] for r in selected_rows})),
        "counts_per_bucket": counts_per_bucket,
        "skipped_duplicate_in_batch": int(dup_stats["skipped_duplicate_in_batch"]),
        "tree_heavy_guard": {
            "hard_negatives_green_ratio_max": 0.30,
            "true_empty_green_ratio_max": 0.26,
        },
    }
    stats_path = args.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"total selected: {len(selected_rows)}")
    print("counts per bucket:")
    for k, v in counts_per_bucket.items():
        print(f"  {k}: {v}")
    print(f"unique tile ids: {len({r['tile_id'] for r in selected_rows})}")
    print(f"duplicates prevented: {dup_stats['skipped_duplicate_in_batch']}")
    print(f"output images: {images_dir}")
    print(f"output manifest: {manifest_path}")
    print(f"output stats: {stats_path}")


if __name__ == "__main__":
    main()
