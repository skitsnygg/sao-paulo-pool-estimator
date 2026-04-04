#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from tile_id_guard import canonical_rel_from_tile_id, collect_tile_ids_from_roots, extract_tile_id_from_row


CELL_RE = re.compile(r"^cell_(\d+)_(\d+)$")
RC_RE = re.compile(r"^r(\d+)_c(\d+)$")

DEFAULT_BUCKET_TARGETS = {
    "obvious_real_pools": 50,
    "large_pools": 40,
    "small_pools": 25,
    "covered_dark_pools": 30,
    "dense_many_pools": 30,
    "hard_negatives_lookalikes": 25,
    "true_empty_negatives": 10,
    "low_conf_borderline": 10,
}


@dataclass(frozen=True)
class TileRow:
    tile_id: str
    tile_rel: str
    tile: str
    tile_stem: str
    cell: str
    tile_path_abs: str
    blank_white: bool
    num_preds: int
    min_conf: Optional[float]
    mean_conf: Optional[float]
    max_conf: Optional[float]
    max_area_m2: Optional[float]
    sum_area_m2: float
    max_mask_area_px: Optional[float]
    cell_xy: Optional[Tuple[int, int]]
    rc: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class TileFeatures:
    brightness: float
    dark_ratio: float
    blue_ratio: float
    green_ratio: float
    gray_ratio: float


@dataclass(frozen=True)
class ScoredTile:
    row: TileRow
    src: Path
    features: TileFeatures
    score: float
    note: str


@dataclass(frozen=True)
class MissedContext:
    neighbor_preds_sum: float
    neighbor_sum_area_m2: float
    neighbor_positive_tiles: int
    score: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a geographically diverse, urban-representative curated Google z21 "
            "validation set with strict per-cell caps and high-value pool prioritization."
        )
    )
    ap.add_argument("--tiles-csv", type=Path, required=True)
    ap.add_argument("--tiles-root", type=Path, required=True)
    ap.add_argument(
        "--pools-geojson",
        type=Path,
        default=None,
        help="Optional source geojson path for provenance in stats.",
    )
    ap.add_argument(
        "--exclude-root",
        type=Path,
        action="append",
        default=[],
        help="Additional image root(s) scanned for tile IDs to exclude.",
    )
    ap.add_argument(
        "--existing-dataset-root",
        type=Path,
        default=Path("data/datasets/google_z21_v2"),
        help="Dataset root whose train/val image tile IDs will be excluded.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-val-per-cell", type=int, default=5)
    ap.add_argument("--max-annotation-per-cell", type=int, default=8)
    ap.add_argument(
        "--max-val-per-region",
        type=int,
        default=12,
        help="Soft cap for 2x2 cell regions during first-pass validation selection.",
    )
    ap.add_argument(
        "--max-annotation-per-region",
        type=int,
        default=20,
        help="Soft cap for 2x2 cell regions during first-pass annotation candidate selection.",
    )
    ap.add_argument(
        "--annotation-candidate-target",
        type=int,
        default=1800,
        help="Global target size for intermediate annotation candidates.",
    )

    ap.add_argument("--target-obvious-real-pools", type=int, default=DEFAULT_BUCKET_TARGETS["obvious_real_pools"])
    ap.add_argument("--target-large-pools", type=int, default=DEFAULT_BUCKET_TARGETS["large_pools"])
    ap.add_argument("--target-small-pools", type=int, default=DEFAULT_BUCKET_TARGETS["small_pools"])
    ap.add_argument("--target-covered-dark-pools", type=int, default=DEFAULT_BUCKET_TARGETS["covered_dark_pools"])
    ap.add_argument("--target-dense-many-pools", type=int, default=DEFAULT_BUCKET_TARGETS["dense_many_pools"])
    ap.add_argument(
        "--target-hard-negatives-lookalikes",
        type=int,
        default=DEFAULT_BUCKET_TARGETS["hard_negatives_lookalikes"],
    )
    ap.add_argument(
        "--target-true-empty-negatives",
        type=int,
        default=DEFAULT_BUCKET_TARGETS["true_empty_negatives"],
    )
    ap.add_argument("--target-low-conf-borderline", type=int, default=DEFAULT_BUCKET_TARGETS["low_conf_borderline"])
    return ap.parse_args()


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def scale(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp01((float(v) - lo) / (hi - lo))


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_float(value: object) -> Optional[float]:
    try:
        x = float(str(value).strip())
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def parse_cell_xy(cell: str) -> Optional[Tuple[int, int]]:
    m = CELL_RE.match(str(cell).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_rc(tile_stem: str) -> Optional[Tuple[int, int]]:
    m = RC_RE.match(str(tile_stem).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def region_key(cell_xy: Optional[Tuple[int, int]], div: int = 2) -> str:
    if cell_xy is None:
        return "region_unknown"
    return f"region_{cell_xy[0] // div:04d}_{cell_xy[1] // div:04d}"


def normalize_dict(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    xs = [float(v) for v in values.values()]
    lo = min(xs)
    hi = max(xs)
    if hi <= lo:
        return {k: 0.0 for k in values}
    denom = hi - lo
    return {k: (float(v) - lo) / denom for k, v in values.items()}


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
    return TileFeatures(
        brightness=float(arr.mean()),
        dark_ratio=float(np.mean(v < 0.25)),
        blue_ratio=float(np.mean((b > 0.35) & (b > r * 1.15) & (b > g * 1.05))),
        green_ratio=float(np.mean((g > 0.30) & (g > r * 1.08) & (g > b * 1.06))),
        gray_ratio=float(np.mean(sat < 0.12)),
    )


def load_tiles(csv_path: Path) -> Tuple[Dict[str, TileRow], Dict[str, int]]:
    rows: Dict[str, TileRow] = {}
    counters = {
        "rows_total": 0,
        "rows_missing_tile_id": 0,
        "rows_duplicate_tile_id": 0,
    }
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            counters["rows_total"] += 1
            tile_id = extract_tile_id_from_row(raw) or ""
            if not tile_id:
                counters["rows_missing_tile_id"] += 1
                continue
            if tile_id in rows:
                counters["rows_duplicate_tile_id"] += 1
                continue
            tile_stem = str(raw.get("tile_stem") or "").strip()
            rows[tile_id] = TileRow(
                tile_id=tile_id,
                tile_rel=str(raw.get("tile_rel") or "").strip(),
                tile=str(raw.get("tile") or "").strip(),
                tile_stem=tile_stem,
                cell=str(raw.get("cell") or "").strip(),
                tile_path_abs=str(raw.get("tile_path_abs") or "").strip(),
                blank_white=parse_bool(raw.get("blank_white", "0")),
                num_preds=parse_int(raw.get("num_preds", 0)),
                min_conf=parse_float(raw.get("min_conf")),
                mean_conf=parse_float(raw.get("mean_conf")),
                max_conf=parse_float(raw.get("max_conf")),
                max_area_m2=parse_float(raw.get("max_area_m2")),
                sum_area_m2=parse_float(raw.get("sum_area_m2")) or 0.0,
                max_mask_area_px=parse_float(raw.get("max_mask_area_px")),
                cell_xy=parse_cell_xy(str(raw.get("cell") or "").strip()),
                rc=parse_rc(tile_stem),
            )
    return rows, counters


def build_neighbor_context(rows: Iterable[TileRow]) -> Dict[str, MissedContext]:
    by_cell_rc: Dict[str, Dict[Tuple[int, int], TileRow]] = defaultdict(dict)
    for row in rows:
        if row.rc is None:
            continue
        by_cell_rc[row.cell][row.rc] = row

    raw_rows: List[Tuple[str, float, float, int]] = []
    for row in rows:
        if row.num_preds != 0 or row.rc is None:
            continue
        rr, cc = row.rc
        grid = by_cell_rc.get(row.cell, {})
        nb_preds = 0.0
        nb_area = 0.0
        nb_pos_tiles = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = grid.get((rr + dr, cc + dc))
                if nb is None:
                    continue
                nb_preds += float(nb.num_preds)
                nb_area += float(nb.sum_area_m2)
                if nb.num_preds > 0:
                    nb_pos_tiles += 1
        raw_rows.append((row.tile_id, nb_preds, nb_area, nb_pos_tiles))

    preds_norm = normalize_dict({tid: p for tid, p, _, _ in raw_rows})
    pos_tiles_norm = normalize_dict({tid: float(n) for tid, _, _, n in raw_rows})
    area_norm = normalize_dict({tid: a for tid, _, a, _ in raw_rows})

    out: Dict[str, MissedContext] = {}
    for tid, nb_preds, nb_area, nb_pos_tiles in raw_rows:
        score = (0.52 * preds_norm.get(tid, 0.0)) + (0.30 * pos_tiles_norm.get(tid, 0.0)) + (0.18 * area_norm.get(tid, 0.0))
        out[tid] = MissedContext(
            neighbor_preds_sum=nb_preds,
            neighbor_sum_area_m2=nb_area,
            neighbor_positive_tiles=nb_pos_tiles,
            score=float(score),
        )
    return out


def interleave_by_cell(cands: Sequence[ScoredTile]) -> List[ScoredTile]:
    if not cands:
        return []
    ranked = sorted(cands, key=lambda c: (-c.score, c.row.tile_id))
    by_cell: Dict[str, deque[ScoredTile]] = defaultdict(deque)
    for c in ranked:
        by_cell[c.row.cell].append(c)
    cells = sorted(by_cell.keys(), key=lambda cell: (-by_cell[cell][0].score, cell))
    out: List[ScoredTile] = []
    while True:
        added = False
        for cell in cells:
            q = by_cell[cell]
            if not q:
                continue
            out.append(q.popleft())
            added = True
        if not added:
            break
    return out


def select_diverse(
    ranked: Sequence[ScoredTile],
    *,
    target: int,
    used_tile_ids: set[str],
    used_per_cell: Counter[str],
    used_per_region: Counter[str],
    max_per_cell: int,
    max_per_region: int,
) -> List[ScoredTile]:
    if target <= 0 or not ranked:
        return []

    ordered = interleave_by_cell(ranked)
    keep: List[ScoredTile] = []
    selected_ids: set[str] = set()

    def try_pass(use_region_cap: bool) -> None:
        for c in ordered:
            if len(keep) >= target:
                return
            tid = c.row.tile_id
            if tid in used_tile_ids or tid in selected_ids:
                continue
            cell = c.row.cell
            if max_per_cell > 0 and used_per_cell[cell] >= max_per_cell:
                continue
            reg = region_key(c.row.cell_xy)
            if use_region_cap and max_per_region > 0 and used_per_region[reg] >= max_per_region:
                continue
            selected_ids.add(tid)
            keep.append(c)
            used_tile_ids.add(tid)
            used_per_cell[cell] += 1
            used_per_region[reg] += 1

    # First pass is strict on regional clustering; second pass relaxes region cap only.
    try_pass(use_region_cap=True)
    if len(keep) < target:
        try_pass(use_region_cap=False)
    return keep


def ensure_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"out-dir exists: {path} (use --overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tiles_csv = args.tiles_csv.expanduser().resolve()
    tiles_root = args.tiles_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    existing_dataset_root = args.existing_dataset_root.expanduser().resolve()

    if not tiles_csv.exists():
        raise SystemExit(f"--tiles-csv not found: {tiles_csv}")
    if not tiles_root.exists():
        raise SystemExit(f"--tiles-root not found: {tiles_root}")
    if not existing_dataset_root.exists():
        raise SystemExit(f"--existing-dataset-root not found: {existing_dataset_root}")

    targets = {
        "obvious_real_pools": int(args.target_obvious_real_pools),
        "large_pools": int(args.target_large_pools),
        "small_pools": int(args.target_small_pools),
        "covered_dark_pools": int(args.target_covered_dark_pools),
        "dense_many_pools": int(args.target_dense_many_pools),
        "hard_negatives_lookalikes": int(args.target_hard_negatives_lookalikes),
        "true_empty_negatives": int(args.target_true_empty_negatives),
        "low_conf_borderline": int(args.target_low_conf_borderline),
    }
    for k, v in targets.items():
        if v < 0:
            raise SystemExit(f"Target count cannot be negative: {k}={v}")

    loaded_rows, load_stats = load_tiles(tiles_csv)

    exclude_roots: List[Path] = [
        (existing_dataset_root / "images" / "train").resolve(),
        (existing_dataset_root / "images" / "val").resolve(),
    ]
    for p in args.exclude_root:
        exclude_roots.append(p.expanduser().resolve())
    excluded_tile_ids, excluded_files_scanned = collect_tile_ids_from_roots(exclude_roots)

    eligible_rows: List[TileRow] = []
    source_by_id: Dict[str, Path] = {}
    excluded_already_labeled = 0
    missing_source = 0
    skipped_blank = 0

    for row in loaded_rows.values():
        if row.tile_id in excluded_tile_ids:
            excluded_already_labeled += 1
            continue
        if row.blank_white:
            skipped_blank += 1
            continue
        src = source_path_for_row(row, tiles_root)
        if src is None:
            missing_source += 1
            continue
        source_by_id[row.tile_id] = src
        eligible_rows.append(row)

    if not eligible_rows:
        raise SystemExit("No eligible rows after exclusions.")

    positives = [r for r in eligible_rows if r.num_preds > 0]
    empties = [r for r in eligible_rows if r.num_preds == 0]

    cell_total = Counter(r.cell for r in eligible_rows)
    cell_pos = Counter(r.cell for r in positives)
    cell_sum_preds: Dict[str, float] = defaultdict(float)
    cell_sum_area: Dict[str, float] = defaultdict(float)
    for r in eligible_rows:
        cell_sum_preds[r.cell] += float(r.num_preds)
        cell_sum_area[r.cell] += float(r.sum_area_m2)

    cell_pos_rate_raw = {cell: (float(cell_pos[cell]) / float(max(1, cell_total[cell]))) for cell in cell_total}
    cell_pred_intensity_raw = {cell: (cell_sum_preds[cell] / float(max(1, cell_total[cell]))) for cell in cell_total}
    cell_area_intensity_raw = {cell: (cell_sum_area[cell] / float(max(1, cell_total[cell]))) for cell in cell_total}
    cell_pos_rate = normalize_dict(cell_pos_rate_raw)
    cell_pred_intensity = normalize_dict(cell_pred_intensity_raw)
    cell_area_intensity = normalize_dict(cell_area_intensity_raw)
    cell_proxy = {
        cell: (0.50 * cell_pred_intensity.get(cell, 0.0))
        + (0.25 * cell_pos_rate.get(cell, 0.0))
        + (0.25 * cell_area_intensity.get(cell, 0.0))
        for cell in cell_total
    }

    missed_context = build_neighbor_context(eligible_rows)

    def pre_sort(rows: Iterable[TileRow], score_fn: Callable[[TileRow], float], limit: int) -> List[TileRow]:
        ranked = sorted(rows, key=lambda r: (-score_fn(r), r.tile_id))
        return ranked[:limit]

    obvious_pre = pre_sort(
        (
            r
            for r in positives
            if (r.max_conf or 0.0) >= 0.70 and (r.max_area_m2 or 0.0) >= 10.0
        ),
        lambda r: (0.62 * scale((r.max_conf or 0.0), 0.70, 0.92))
        + (0.30 * scale(math.log1p(r.max_area_m2 or 0.0), math.log1p(10.0), math.log1p(320.0)))
        + (0.08 * cell_proxy.get(r.cell, 0.0)),
        limit=1500,
    )
    large_pre = pre_sort(
        (
            r
            for r in positives
            if (r.max_area_m2 or 0.0) >= 38.0 and (r.max_conf or 0.0) >= 0.35
        ),
        lambda r: (0.66 * scale(math.log1p(r.max_area_m2 or 0.0), math.log1p(35.0), math.log1p(700.0)))
        + (0.22 * scale((r.max_conf or 0.0), 0.35, 0.92))
        + (0.12 * cell_proxy.get(r.cell, 0.0)),
        limit=1300,
    )
    small_pre = pre_sort(
        (
            r
            for r in positives
            if 2.5 <= (r.max_area_m2 or 0.0) <= 24.0 and (r.max_conf or 0.0) >= 0.40
        ),
        lambda r: (0.45 * (1.0 - scale(abs((r.max_area_m2 or 0.0) - 10.0), 0.0, 11.0)))
        + (0.35 * scale((r.max_conf or 0.0), 0.40, 0.90))
        + (0.20 * cell_proxy.get(r.cell, 0.0)),
        limit=1200,
    )
    covered_pre = pre_sort(
        (
            r
            for r in positives
            if 4.0 <= (r.max_area_m2 or 0.0) <= 180.0 and 0.28 <= (r.max_conf or 0.0) <= 0.92
        ),
        lambda r: (0.50 * (1.0 - scale((r.max_conf or 0.0), 0.28, 0.92)))
        + (0.30 * scale(math.log1p(r.max_area_m2 or 0.0), math.log1p(4.0), math.log1p(180.0)))
        + (0.20 * cell_proxy.get(r.cell, 0.0)),
        limit=1400,
    )
    dense_pre = pre_sort(
        (r for r in positives if r.num_preds >= 3),
        lambda r: (0.50 * scale(float(r.num_preds), 3.0, 12.0))
        + (0.35 * scale(math.log1p(r.sum_area_m2), math.log1p(25.0), math.log1p(900.0)))
        + (0.15 * cell_proxy.get(r.cell, 0.0)),
        limit=1300,
    )
    hard_neg_pre = pre_sort(
        (
            r
            for r in positives
            if r.num_preds <= 4
            and (r.max_conf or 1.0) <= 0.78
            and (r.mean_conf or 1.0) <= 0.70
            and (r.max_area_m2 or 0.0) <= 180.0
        ),
        lambda r: (0.42 * (1.0 - scale((r.max_conf or 0.0), 0.20, 0.78)))
        + (0.30 * (1.0 - scale((r.mean_conf or 0.0), 0.20, 0.70)))
        + (0.18 * (1.0 - scale(r.max_area_m2 or 0.0, 12.0, 180.0)))
        + (0.10 * cell_proxy.get(r.cell, 0.0)),
        limit=1600,
    )
    true_empty_pre = pre_sort(
        (r for r in empties if cell_proxy.get(r.cell, 0.0) >= 0.10),
        lambda r: (0.80 * cell_proxy.get(r.cell, 0.0)) + (0.20 * cell_pos_rate.get(r.cell, 0.0)),
        limit=1200,
    )
    low_conf_pre = pre_sort(
        (
            r
            for r in positives
            if 0.25 <= (r.max_conf or 0.0) <= 0.66 and (r.mean_conf or 0.0) <= 0.68
        ),
        lambda r: (0.48 * (1.0 - scale(abs((r.max_conf or 0.0) - 0.46), 0.0, 0.20)))
        + (0.22 * scale(float(r.num_preds), 1.0, 8.0))
        + (0.20 * scale(math.log1p(r.max_area_m2 or 0.0), math.log1p(3.0), math.log1p(120.0)))
        + (0.10 * cell_proxy.get(r.cell, 0.0)),
        limit=1300,
    )
    missed_pre = pre_sort(
        (
            r
            for r in empties
            if r.tile_id in missed_context
            and (
                missed_context[r.tile_id].neighbor_positive_tiles >= 3
                or missed_context[r.tile_id].neighbor_preds_sum >= 5.0
                or missed_context[r.tile_id].score >= 0.52
            )
        ),
        lambda r: missed_context.get(r.tile_id, MissedContext(0.0, 0.0, 0, 0.0)).score,
        limit=1200,
    )

    union_ids = {
        r.tile_id
        for arr in (
            obvious_pre,
            large_pre,
            small_pre,
            covered_pre,
            dense_pre,
            hard_neg_pre,
            true_empty_pre,
            low_conf_pre,
            missed_pre,
        )
        for r in arr
    }

    features_cache: Dict[Path, TileFeatures] = {}
    missing_feature_tiles = 0

    def get_features_for_row(row: TileRow) -> Optional[TileFeatures]:
        nonlocal missing_feature_tiles
        src = source_by_id.get(row.tile_id)
        if src is None:
            missing_feature_tiles += 1
            return None
        if src not in features_cache:
            try:
                features_cache[src] = compute_features(src)
            except Exception:
                missing_feature_tiles += 1
                return None
        return features_cache[src]

    def urban_score(f: TileFeatures) -> float:
        return clamp01((f.gray_ratio - f.green_ratio + 0.06) / 0.34)

    def wooded_score(f: TileFeatures) -> float:
        return scale(f.green_ratio, 0.12, 0.45)

    def blue_score(f: TileFeatures) -> float:
        return scale(f.blue_ratio, 0.18, 0.72)

    def dark_score(f: TileFeatures) -> float:
        return scale(f.dark_ratio, 0.12, 0.68)

    def mid_brightness(f: TileFeatures) -> float:
        return 1.0 - scale(abs(f.brightness - 0.53), 0.0, 0.36)

    def build_scored(
        pre_rows: Sequence[TileRow],
        scorer: Callable[[TileRow, TileFeatures], Optional[Tuple[float, str]]],
    ) -> List[ScoredTile]:
        out: List[ScoredTile] = []
        for row in pre_rows:
            if row.tile_id not in union_ids:
                continue
            feat = get_features_for_row(row)
            if feat is None:
                continue
            scored = scorer(row, feat)
            if scored is None:
                continue
            score, note = scored
            src = source_by_id[row.tile_id]
            out.append(ScoredTile(row=row, src=src, features=feat, score=float(score), note=note))
        out.sort(key=lambda c: (-c.score, c.row.tile_id))
        return out

    def score_obvious(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        conf = row.max_conf or 0.0
        area = row.max_area_m2 or 0.0
        if conf < 0.74 or area < 12.0:
            return None
        u = urban_score(f)
        w = wooded_score(f)
        b = blue_score(f)
        score = (
            0.44 * scale(conf, 0.72, 0.92)
            + 0.30 * scale(math.log1p(area), math.log1p(12.0), math.log1p(320.0))
            + 0.12 * u
            + 0.09 * b
            + 0.05 * cell_proxy.get(row.cell, 0.0)
            - 0.06 * w
        )
        note = (
            f"conf={conf:.3f} area={area:.1f} urban={u:.3f} blue={b:.3f} "
            f"green={f.green_ratio:.3f} gray={f.gray_ratio:.3f}"
        )
        return score, note

    def score_large(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        conf = row.max_conf or 0.0
        area = row.max_area_m2 or 0.0
        if area < 48.0 or conf < 0.32:
            return None
        u = urban_score(f)
        w = wooded_score(f)
        score = (
            0.58 * scale(math.log1p(area), math.log1p(45.0), math.log1p(900.0))
            + 0.20 * scale(conf, 0.30, 0.92)
            + 0.12 * u
            + 0.06 * scale(float(row.num_preds), 1.0, 10.0)
            + 0.04 * cell_proxy.get(row.cell, 0.0)
            - 0.05 * w
        )
        note = f"conf={conf:.3f} area={area:.1f} preds={row.num_preds} urban={u:.3f}"
        return score, note

    def score_small(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        conf = row.max_conf or 0.0
        area = row.max_area_m2 or 0.0
        if conf < 0.48 or area < 3.0 or area > 22.0:
            return None
        u = urban_score(f)
        w = wooded_score(f)
        area_mid = 1.0 - scale(abs(area - 10.0), 0.0, 11.0)
        score = (
            0.40 * area_mid
            + 0.30 * scale(conf, 0.45, 0.90)
            + 0.18 * u
            + 0.07 * blue_score(f)
            + 0.05 * scale(float(row.num_preds), 1.0, 4.0)
            - 0.05 * w
        )
        note = f"conf={conf:.3f} area={area:.1f} area_mid={area_mid:.3f} urban={u:.3f}"
        return score, note

    def score_covered(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        conf = row.max_conf or 0.0
        area = row.max_area_m2 or 0.0
        d = dark_score(f)
        if area < 5.0 or area > 140.0 or conf < 0.30 or d < 0.22:
            return None
        u = urban_score(f)
        score = (
            0.42 * d
            + 0.18 * (1.0 - scale(conf, 0.32, 0.90))
            + 0.14 * u
            + 0.10 * scale(area, 8.0, 90.0)
            + 0.10 * blue_score(f)
            + 0.06 * scale(float(row.num_preds), 1.0, 6.0)
        )
        note = f"dark={f.dark_ratio:.3f} conf={conf:.3f} area={area:.1f} urban={u:.3f}"
        return score, note

    def score_dense(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        if row.num_preds < 4:
            return None
        conf = row.max_conf or 0.0
        u = urban_score(f)
        w = wooded_score(f)
        score = (
            0.37 * scale(float(row.num_preds), 4.0, 12.0)
            + 0.28 * scale(math.log1p(row.sum_area_m2), math.log1p(30.0), math.log1p(900.0))
            + 0.20 * u
            + 0.08 * scale(conf, 0.35, 0.90)
            + 0.07 * blue_score(f)
            - 0.12 * w
        )
        note = f"preds={row.num_preds} sum_area={row.sum_area_m2:.1f} urban={u:.3f} green={f.green_ratio:.3f}"
        return score, note

    def score_hard_neg(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        conf = row.max_conf or 0.0
        mean_conf = row.mean_conf or 0.0
        area = row.max_area_m2 or 0.0
        if row.num_preds < 1 or row.num_preds > 3:
            return None
        if conf > 0.72 or mean_conf > 0.62 or area > 120.0:
            return None
        u = urban_score(f)
        if u < 0.22:
            return None
        b = blue_score(f)
        gray_n = scale(f.gray_ratio, 0.12, 0.46)
        area_pref = 1.0 - scale(abs(area - 24.0), 0.0, 36.0)
        score = (
            0.30 * u
            + 0.23 * (1.0 - scale(conf, 0.25, 0.72))
            + 0.19 * b
            + 0.14 * gray_n
            + 0.08 * area_pref
            + 0.06 * (1.0 - wooded_score(f))
        )
        note = (
            f"conf={conf:.3f} mean_conf={mean_conf:.3f} area={area:.1f} "
            f"urban={u:.3f} blue={f.blue_ratio:.3f} green={f.green_ratio:.3f}"
        )
        return score, note

    def score_true_empty(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        if row.num_preds != 0:
            return None
        u = urban_score(f)
        if u < 0.28:
            return None
        if f.green_ratio > 0.25 or f.dark_ratio > 0.60:
            return None
        score = (
            0.40 * u
            + 0.22 * scale(f.gray_ratio, 0.15, 0.48)
            + 0.20 * mid_brightness(f)
            + 0.10 * (1.0 - scale(f.blue_ratio, 0.15, 0.58))
            + 0.08 * cell_proxy.get(row.cell, 0.0)
        )
        note = (
            f"urban={u:.3f} gray={f.gray_ratio:.3f} green={f.green_ratio:.3f} "
            f"blue={f.blue_ratio:.3f} dark={f.dark_ratio:.3f}"
        )
        return score, note

    def score_low_conf(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        conf = row.max_conf or 0.0
        if row.num_preds <= 0 or conf < 0.30 or conf > 0.62:
            return None
        if (row.mean_conf or 0.0) > 0.66:
            return None
        border = 1.0 - scale(abs(conf - 0.46), 0.0, 0.16)
        score = (
            0.35 * border
            + 0.18 * scale(float(row.num_preds), 1.0, 8.0)
            + 0.15 * scale(math.log1p(row.max_area_m2 or 0.0), math.log1p(3.0), math.log1p(120.0))
            + 0.20 * urban_score(f)
            + 0.07 * blue_score(f)
            + 0.05 * cell_proxy.get(row.cell, 0.0)
        )
        note = f"conf={conf:.3f} border={border:.3f} preds={row.num_preds} area={row.max_area_m2 or 0.0:.1f}"
        return score, note

    def score_missed(row: TileRow, f: TileFeatures) -> Optional[Tuple[float, str]]:
        if row.num_preds != 0:
            return None
        ctx = missed_context.get(row.tile_id)
        if ctx is None or ctx.score < 0.45:
            return None
        if f.green_ratio > 0.34:
            return None
        u = urban_score(f)
        score = (0.58 * ctx.score) + (0.18 * blue_score(f)) + (0.14 * u) + (0.10 * dark_score(f))
        note = (
            f"context={ctx.score:.3f} nb_preds={ctx.neighbor_preds_sum:.1f} "
            f"nb_pos={ctx.neighbor_positive_tiles} blue={f.blue_ratio:.3f}"
        )
        return score, note

    ranked = {
        "obvious_real_pools": build_scored(obvious_pre, score_obvious),
        "large_pools": build_scored(large_pre, score_large),
        "small_pools": build_scored(small_pre, score_small),
        "covered_dark_pools": build_scored(covered_pre, score_covered),
        "dense_many_pools": build_scored(dense_pre, score_dense),
        "hard_negatives_lookalikes": build_scored(hard_neg_pre, score_hard_neg),
        "true_empty_negatives": build_scored(true_empty_pre, score_true_empty),
        "low_conf_borderline": build_scored(low_conf_pre, score_low_conf),
        "missed_obvious_pools": build_scored(missed_pre, score_missed),
    }

    annotation_bucket_targets = {
        "large_pools": 200,
        "obvious_real_pools": 230,
        "dense_many_pools": 170,
        "covered_dark_pools": 170,
        "missed_obvious_pools": 170,
        "low_conf_borderline": 130,
        "small_pools": 120,
        "hard_negatives_lookalikes": 140,
        "true_empty_negatives": 90,
    }

    ann_used_tile_ids: set[str] = set()
    ann_per_cell: Counter[str] = Counter()
    ann_per_region: Counter[str] = Counter()
    annotation_rows: List[Dict[str, object]] = []
    annotation_origin_map: Dict[str, List[str]] = defaultdict(list)
    for bucket_name, bucket_limit in annotation_bucket_targets.items():
        picked = select_diverse(
            ranked.get(bucket_name, []),
            target=bucket_limit,
            used_tile_ids=ann_used_tile_ids,
            used_per_cell=ann_per_cell,
            used_per_region=ann_per_region,
            max_per_cell=max(1, int(args.max_annotation_per_cell)),
            max_per_region=max(0, int(args.max_annotation_per_region)),
        )
        for idx, c in enumerate(picked, start=1):
            annotation_rows.append(
                {
                    "tile_id": c.row.tile_id,
                    "bucket": bucket_name,
                    "annotation_rank": idx,
                    "annotation_score": f"{c.score:.6f}",
                    "cell": c.row.cell,
                    "tile_rel": c.row.tile_rel,
                    "source_path": str(c.src.resolve()),
                    "num_preds": c.row.num_preds,
                    "max_conf": "" if c.row.max_conf is None else f"{c.row.max_conf:.6f}",
                    "max_area_m2": "" if c.row.max_area_m2 is None else f"{c.row.max_area_m2:.3f}",
                    "note": c.note,
                }
            )
            annotation_origin_map[c.row.tile_id].append(bucket_name)

    # Truncate to global annotation target with round-robin fairness across buckets.
    target_ann = max(0, int(args.annotation_candidate_target))
    if target_ann and len(annotation_rows) > target_ann:
        by_bucket_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in annotation_rows:
            by_bucket_rows[str(row["bucket"])].append(row)
        bucket_order = list(annotation_bucket_targets.keys())
        trimmed: List[Dict[str, object]] = []
        idx_by_bucket = {b: 0 for b in bucket_order}
        while len(trimmed) < target_ann:
            advanced = False
            for b in bucket_order:
                i = idx_by_bucket[b]
                rows_b = by_bucket_rows.get(b, [])
                if i >= len(rows_b):
                    continue
                trimmed.append(rows_b[i])
                idx_by_bucket[b] = i + 1
                advanced = True
                if len(trimmed) >= target_ann:
                    break
            if not advanced:
                break
        annotation_rows = trimmed
    annotation_ids: set[str] = {str(r["tile_id"]) for r in annotation_rows}

    validation_used_ids: set[str] = set()
    validation_per_cell: Counter[str] = Counter()
    validation_per_region: Counter[str] = Counter()
    selected_by_bucket: Dict[str, List[ScoredTile]] = {}

    for bucket_name in (
        "obvious_real_pools",
        "large_pools",
        "small_pools",
        "covered_dark_pools",
        "dense_many_pools",
        "hard_negatives_lookalikes",
        "true_empty_negatives",
        "low_conf_borderline",
    ):
        bucket_ranked = [c for c in ranked.get(bucket_name, []) if c.row.tile_id in annotation_ids]
        picked = select_diverse(
            bucket_ranked,
            target=targets[bucket_name],
            used_tile_ids=validation_used_ids,
            used_per_cell=validation_per_cell,
            used_per_region=validation_per_region,
            max_per_cell=max(1, int(args.max_val_per_cell)),
            max_per_region=max(0, int(args.max_val_per_region)),
        )
        selected_by_bucket[bucket_name] = picked

    selected_rows: List[Dict[str, object]] = []
    for bucket_name, picked in selected_by_bucket.items():
        for c in picked:
            selected_rows.append(
                {
                    "tile_id": c.row.tile_id,
                    "tile_rel": c.row.tile_rel,
                    "cell": c.row.cell,
                    "category_bucket": bucket_name,
                    "source_path": str(c.src.resolve()),
                    "preds_per_tile": c.row.num_preds,
                    "min_conf": "" if c.row.min_conf is None else f"{c.row.min_conf:.6f}",
                    "mean_conf": "" if c.row.mean_conf is None else f"{c.row.mean_conf:.6f}",
                    "max_conf": "" if c.row.max_conf is None else f"{c.row.max_conf:.6f}",
                    "max_area_m2": "" if c.row.max_area_m2 is None else f"{c.row.max_area_m2:.3f}",
                    "sum_area_m2": f"{c.row.sum_area_m2:.3f}",
                    "brightness": f"{c.features.brightness:.6f}",
                    "dark_ratio": f"{c.features.dark_ratio:.6f}",
                    "blue_ratio": f"{c.features.blue_ratio:.6f}",
                    "green_ratio": f"{c.features.green_ratio:.6f}",
                    "gray_ratio": f"{c.features.gray_ratio:.6f}",
                    "selection_score": f"{c.score:.6f}",
                    "annotation_origins": ";".join(annotation_origin_map.get(c.row.tile_id, [])),
                    "notes": c.note,
                }
            )

    ensure_out_dir(out_dir, overwrite=args.overwrite)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for row in selected_rows:
        tile_id = str(row["tile_id"])
        src = Path(str(row["source_path"]))
        bucket = str(row["category_bucket"])
        rel = canonical_rel_from_tile_id(tile_id) or f"{tile_id}.png"
        dst = images_dir / bucket / Path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tile_id",
            "tile_rel",
            "cell",
            "category_bucket",
            "source_path",
            "preds_per_tile",
            "min_conf",
            "mean_conf",
            "max_conf",
            "max_area_m2",
            "sum_area_m2",
            "brightness",
            "dark_ratio",
            "blue_ratio",
            "green_ratio",
            "gray_ratio",
            "selection_score",
            "annotation_origins",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    annotation_manifest = out_dir / "annotation_candidates.csv"
    with annotation_manifest.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tile_id",
            "bucket",
            "annotation_rank",
            "annotation_score",
            "cell",
            "tile_rel",
            "source_path",
            "num_preds",
            "max_conf",
            "max_area_m2",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in annotation_rows:
            writer.writerow(row)

    counts_per_bucket = {k: len(v) for k, v in selected_by_bucket.items()}
    selected_cell_counts = Counter(str(r["cell"]) for r in selected_rows)
    annotation_cell_counts = Counter(str(r["cell"]) for r in annotation_rows)
    top_cells = [{"cell": cell, "count": int(count)} for cell, count in selected_cell_counts.most_common(20)]

    warnings: List[str] = []
    val_cap = int(args.max_val_per_cell)
    ann_cap = int(args.max_annotation_per_cell)
    over_val = [(cell, cnt) for cell, cnt in selected_cell_counts.items() if cnt > val_cap]
    over_ann = [(cell, cnt) for cell, cnt in annotation_cell_counts.items() if cnt > ann_cap]
    underfilled = [(b, counts_per_bucket.get(b, 0), targets[b]) for b in targets if counts_per_bucket.get(b, 0) < targets[b]]

    if over_val:
        for cell, cnt in sorted(over_val):
            warnings.append(f"WARNING: validation cell cap exceeded ({cell}: {cnt} > {val_cap})")
    if over_ann:
        for cell, cnt in sorted(over_ann):
            warnings.append(f"WARNING: annotation cell cap exceeded ({cell}: {cnt} > {ann_cap})")
    for b, got, want in underfilled:
        warnings.append(f"WARNING: bucket underfilled ({b}: selected={got}, target={want})")

    total_selected = len(selected_rows)
    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "source_inputs": {
            "tiles_csv": str(tiles_csv),
            "tiles_root": str(tiles_root),
            "pools_geojson": None if args.pools_geojson is None else str(args.pools_geojson.expanduser().resolve()),
            "existing_dataset_root": str(existing_dataset_root),
        },
        "selection_caps": {
            "max_validation_per_cell": val_cap,
            "max_annotation_per_cell": ann_cap,
            "max_validation_per_region_2x2_first_pass": int(args.max_val_per_region),
            "max_annotation_per_region_2x2_first_pass": int(args.max_annotation_per_region),
        },
        "targets_per_bucket": targets,
        "total_tiles": int(total_selected),
        "counts_per_bucket": counts_per_bucket,
        "unique_cells": int(len(selected_cell_counts)),
        "top_cells_by_frequency": top_cells,
        "max_tiles_single_cell": int(max(selected_cell_counts.values()) if selected_cell_counts else 0),
        "annotation_candidates": {
            "total": int(len(annotation_rows)),
            "unique_cells": int(len(annotation_cell_counts)),
            "max_tiles_single_cell": int(max(annotation_cell_counts.values()) if annotation_cell_counts else 0),
            "counts_per_bucket": dict(Counter(str(r["bucket"]) for r in annotation_rows)),
        },
        "input_filtering": {
            **load_stats,
            "excluded_files_scanned": int(excluded_files_scanned),
            "excluded_already_labeled": int(excluded_already_labeled),
            "excluded_tile_id_universe": int(len(excluded_tile_ids)),
            "skipped_blank_white": int(skipped_blank),
            "missing_source_path": int(missing_source),
            "eligible_rows": int(len(eligible_rows)),
            "eligible_positive_rows": int(len(positives)),
            "eligible_empty_rows": int(len(empties)),
            "feature_tiles_loaded": int(len(features_cache)),
            "missing_feature_tiles": int(missing_feature_tiles),
        },
        "warnings": warnings,
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"total selected: {total_selected}")
    print("counts per bucket:")
    for b in targets:
        print(f"  {b}: {counts_per_bucket.get(b, 0)} / target {targets[b]}")
    print(f"unique cells: {len(selected_cell_counts)}")
    print(f"max tiles in a single validation cell: {max(selected_cell_counts.values()) if selected_cell_counts else 0}")
    print(f"annotation candidates: {len(annotation_rows)}")
    print(f"max tiles in a single annotation cell: {max(annotation_cell_counts.values()) if annotation_cell_counts else 0}")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  {w}")
    print(f"output images: {images_dir}")
    print(f"output manifest: {manifest_path}")
    print(f"output annotation candidates: {annotation_manifest}")
    print(f"output stats: {stats_path}")


if __name__ == "__main__":
    main()
