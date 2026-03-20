#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TRUTHY = {"1", "true", "t", "yes", "y"}
NAN_LIKE = {"", "nan", "na", "none", "null"}
RC_PATTERN = re.compile(r"^r(?P<row>\d+)_c(?P<col>\d+)$")


@dataclass
class TileRow:
    raw: Dict[str, str]
    tile_rel: str
    cell: str
    tile_stem: str
    num_preds: int
    sum_area_m2: float
    blank_white: bool
    skip_reason: str
    rc: Optional[Tuple[int, int]]


@dataclass
class RankedCandidate:
    tile: TileRow
    cell_total_preds: float
    cell_total_sum_area_m2: float
    neighbor_preds_sum: float
    neighbor_sum_area_m2: float
    neighbor_positive_tiles: float
    cell_score: float
    neighborhood_score: float
    candidate_score: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Select high-likelihood missed-pool tiles from a per-tile inference CSV. "
            "Candidates are restricted to low-prediction tiles and ranked by "
            "cell/neighborhood prediction intensity."
        )
    )
    ap.add_argument("--tiles-csv", type=Path, required=True, help="Input per-tile CSV.")
    ap.add_argument("--num-tiles", type=int, required=True, help="Number of ranked tiles to select.")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV with selected candidates.")
    return ap.parse_args()


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in TRUTHY


def is_nan_like(value: object) -> bool:
    return str(value).strip().lower() in NAN_LIKE


def parse_rc(tile_stem: str) -> Optional[Tuple[int, int]]:
    m = RC_PATTERN.match((tile_stem or "").strip())
    if not m:
        return None
    return int(m.group("row")), int(m.group("col"))


def load_rows(csv_path: Path) -> Tuple[List[TileRow], List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Input CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        out: List[TileRow] = []
        for row in reader:
            tile_stem = str(row.get("tile_stem") or "").strip()
            out.append(
                TileRow(
                    raw=row,
                    tile_rel=str(row.get("tile_rel") or "").strip(),
                    cell=str(row.get("cell") or "").strip(),
                    tile_stem=tile_stem,
                    num_preds=parse_int(row.get("num_preds", 0)),
                    sum_area_m2=parse_float(row.get("sum_area_m2", 0.0), default=0.0),
                    blank_white=parse_bool(row.get("blank_white", "0")),
                    skip_reason=str(row.get("skip_reason") or "").strip(),
                    rc=parse_rc(tile_stem),
                )
            )
    return out, fieldnames


def minmax_norm(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [0.0 for _ in values]
    denom = vmax - vmin
    return [(v - vmin) / denom for v in values]


def build_ranked_candidates(rows: Sequence[TileRow]) -> List[RankedCandidate]:
    cell_total_preds: Dict[str, float] = {}
    cell_total_sum_area_m2: Dict[str, float] = {}
    grid: Dict[str, Dict[Tuple[int, int], TileRow]] = {}

    for row in rows:
        cell_total_preds[row.cell] = cell_total_preds.get(row.cell, 0.0) + float(row.num_preds)
        cell_total_sum_area_m2[row.cell] = cell_total_sum_area_m2.get(row.cell, 0.0) + float(row.sum_area_m2)
        if row.rc is not None:
            g = grid.setdefault(row.cell, {})
            g[row.rc] = row

    candidates: List[TileRow] = []
    for row in rows:
        if row.num_preds > 1:
            continue
        if row.blank_white:
            continue
        if not is_nan_like(row.skip_reason):
            continue
        candidates.append(row)

    raw_cell_preds: List[float] = []
    raw_cell_area: List[float] = []
    raw_neighbor_preds: List[float] = []
    raw_neighbor_area: List[float] = []
    raw_neighbor_pos_tiles: List[float] = []
    own_low_pred_pref: List[float] = []

    neighbor_cache: Dict[str, Tuple[float, float, float]] = {}

    for row in candidates:
        cp = float(cell_total_preds.get(row.cell, 0.0))
        ca = float(cell_total_sum_area_m2.get(row.cell, 0.0))

        np_sum = 0.0
        na_sum = 0.0
        np_tiles = 0.0
        if row.rc is not None and row.cell in grid:
            rr, cc = row.rc
            g = grid[row.cell]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = g.get((rr + dr, cc + dc))
                    if nb is None:
                        continue
                    np_sum += float(nb.num_preds)
                    na_sum += float(nb.sum_area_m2)
                    if nb.num_preds > 0:
                        np_tiles += 1.0
        neighbor_cache[row.tile_rel] = (np_sum, na_sum, np_tiles)

        raw_cell_preds.append(cp)
        raw_cell_area.append(ca)
        raw_neighbor_preds.append(np_sum)
        raw_neighbor_area.append(na_sum)
        raw_neighbor_pos_tiles.append(np_tiles)
        own_low_pred_pref.append(1.0 if row.num_preds == 0 else 0.65)

    norm_cell_preds = minmax_norm(raw_cell_preds)
    norm_cell_area = minmax_norm(raw_cell_area)
    norm_neighbor_preds = minmax_norm(raw_neighbor_preds)
    norm_neighbor_area = minmax_norm(raw_neighbor_area)
    norm_neighbor_pos_tiles = minmax_norm(raw_neighbor_pos_tiles)

    ranked: List[RankedCandidate] = []
    for i, row in enumerate(candidates):
        np_sum, na_sum, np_tiles = neighbor_cache[row.tile_rel]

        cell_score = (0.65 * norm_cell_preds[i]) + (0.35 * norm_cell_area[i])
        neighborhood_score = (
            (0.50 * norm_neighbor_preds[i])
            + (0.35 * norm_neighbor_area[i])
            + (0.15 * norm_neighbor_pos_tiles[i])
        )
        candidate_score = (0.45 * cell_score) + (0.45 * neighborhood_score) + (0.10 * own_low_pred_pref[i])

        ranked.append(
            RankedCandidate(
                tile=row,
                cell_total_preds=raw_cell_preds[i],
                cell_total_sum_area_m2=raw_cell_area[i],
                neighbor_preds_sum=np_sum,
                neighbor_sum_area_m2=na_sum,
                neighbor_positive_tiles=np_tiles,
                cell_score=cell_score,
                neighborhood_score=neighborhood_score,
                candidate_score=candidate_score,
            )
        )

    ranked.sort(
        key=lambda c: (
            -c.candidate_score,
            -c.neighborhood_score,
            -c.cell_score,
            c.tile.num_preds,
            c.tile.tile_rel,
        )
    )
    return ranked


def add_fieldnames(base: Sequence[str], extras: Sequence[str]) -> List[str]:
    out = list(base)
    for x in extras:
        if x not in out:
            out.append(x)
    return out


def write_selected_csv(
    out_path: Path,
    selected: Sequence[RankedCandidate],
    input_fieldnames: Sequence[str],
) -> None:
    extra_fields = [
        "candidate_score",
        "cell_score",
        "neighborhood_score",
        "cell_total_preds",
        "cell_total_sum_area_m2",
        "neighbor_preds_sum",
        "neighbor_sum_area_m2",
        "neighbor_positive_tiles",
    ]
    fieldnames = add_fieldnames(input_fieldnames, extra_fields)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in selected:
            row = dict(cand.tile.raw)
            row.update(
                {
                    "candidate_score": f"{cand.candidate_score:.6f}",
                    "cell_score": f"{cand.cell_score:.6f}",
                    "neighborhood_score": f"{cand.neighborhood_score:.6f}",
                    "cell_total_preds": f"{cand.cell_total_preds:.1f}",
                    "cell_total_sum_area_m2": f"{cand.cell_total_sum_area_m2:.3f}",
                    "neighbor_preds_sum": f"{cand.neighbor_preds_sum:.1f}",
                    "neighbor_sum_area_m2": f"{cand.neighbor_sum_area_m2:.3f}",
                    "neighbor_positive_tiles": f"{cand.neighbor_positive_tiles:.1f}",
                }
            )
            writer.writerow(row)


def print_top_rows(rows: Sequence[RankedCandidate], limit: int = 8) -> None:
    print("top ranked candidates:")
    print("tile_rel,cell,num_preds,neighborhood_score,candidate_score")
    for cand in rows[:limit]:
        print(
            f"{cand.tile.tile_rel},"
            f"{cand.tile.cell},"
            f"{cand.tile.num_preds},"
            f"{cand.neighborhood_score:.6f},"
            f"{cand.candidate_score:.6f}"
        )


def main() -> None:
    args = parse_args()
    if args.num_tiles <= 0:
        raise SystemExit("--num-tiles must be > 0")
    if not args.tiles_csv.exists():
        raise SystemExit(f"--tiles-csv not found: {args.tiles_csv}")

    rows, input_fieldnames = load_rows(args.tiles_csv)
    ranked = build_ranked_candidates(rows)
    selected = ranked[: min(args.num_tiles, len(ranked))]

    write_selected_csv(args.out, selected, input_fieldnames)

    print(f"total candidate tiles: {len(ranked)}")
    print(f"selected tiles: {len(selected)}")
    print_top_rows(selected)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
