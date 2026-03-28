
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


@dataclass
class TileRow:
    tile_rel: str
    tile: str
    tile_stem: str
    cell: str
    tile_path_abs: str
    blank_white: int
    num_preds: int
    min_conf: Optional[float]
    mean_conf: Optional[float]
    max_conf: Optional[float]
    max_area_m2: Optional[float]
    sum_area_m2: float
    max_mask_area_px: Optional[float]

    @property
    def is_positive(self) -> bool:
        return self.num_preds > 0

    @property
    def is_empty(self) -> bool:
        return self.num_preds == 0


def canonical_tile_id(cell: str, tile_stem: str) -> str:
    cell_clean = (cell or "").strip()
    stem_clean = (tile_stem or "").strip()
    if cell_clean:
        return f"{cell_clean}__{stem_clean}"
    return stem_clean


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def read_rows(csv_path: Path) -> List[TileRow]:
    rows: List[TileRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                TileRow(
                    tile_rel=r["tile_rel"],
                    tile=r["tile"],
                    tile_stem=r["tile_stem"],
                    cell=r["cell"],
                    tile_path_abs=r["tile_path_abs"],
                    blank_white=parse_int(r.get("blank_white", "0")),
                    num_preds=parse_int(r.get("num_preds", "0")),
                    min_conf=parse_float(r.get("min_conf", "")),
                    mean_conf=parse_float(r.get("mean_conf", "")),
                    max_conf=parse_float(r.get("max_conf", "")),
                    max_area_m2=parse_float(r.get("max_area_m2", "")),
                    sum_area_m2=parse_float(r.get("sum_area_m2", "")) or 0.0,
                    max_mask_area_px=parse_float(r.get("max_mask_area_px", "")),
                )
            )
    return rows


def build_existing_canonical_ids(label_roots: Iterable[Path]) -> Set[str]:
    existing: Set[str] = set()
    for root in label_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.txt"):
            if not p.is_file():
                continue
            stem = p.stem
            if "__" in stem:
                existing.add(stem)
                continue
            # Fallback for nested cell/<labels>/<tile>.txt style layouts.
            cell = ""
            for part in p.parts:
                if part.startswith("cell_"):
                    cell = part
                    break
            existing.add(canonical_tile_id(cell, stem))
    return existing


def mirror_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def manifest_row(bucket: str, row: TileRow) -> Dict[str, object]:
    return {
        "bucket": bucket,
        "tile_rel": row.tile_rel,
        "cell": row.cell,
        "tile": row.tile,
        "tile_stem": row.tile_stem,
        "tile_path_abs": row.tile_path_abs,
        "blank_white": row.blank_white,
        "num_preds": row.num_preds,
        "min_conf": row.min_conf if row.min_conf is not None else "",
        "mean_conf": row.mean_conf if row.mean_conf is not None else "",
        "max_conf": row.max_conf if row.max_conf is not None else "",
        "max_area_m2": row.max_area_m2 if row.max_area_m2 is not None else "",
        "sum_area_m2": row.sum_area_m2,
        "max_mask_area_px": row.max_mask_area_px if row.max_mask_area_px is not None else "",
    }


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket",
        "tile_rel",
        "cell",
        "tile",
        "tile_stem",
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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_source_path(tiles_root: Path, row: TileRow) -> Optional[Path]:
    candidate = tiles_root / row.tile_rel
    if candidate.exists():
        return candidate
    p = Path(row.tile_path_abs)
    return p if p.exists() else None


def cell_positive_counts(rows: Iterable[TileRow]) -> Counter:
    counts: Counter = Counter()
    for r in rows:
        if r.is_positive:
            counts[r.cell] += 1
    return counts


def pick_rows(
    ranked: List[TileRow],
    *,
    limit: int,
    used_tiles: Set[str],
    per_cell_limit: int,
    used_per_cell: Counter,
) -> List[TileRow]:
    picked: List[TileRow] = []
    for row in ranked:
        if row.tile_rel in used_tiles:
            continue
        if per_cell_limit > 0 and used_per_cell[row.cell] >= per_cell_limit:
            continue
        picked.append(row)
        used_tiles.add(row.tile_rel)
        used_per_cell[row.cell] += 1
        if len(picked) >= limit:
            break
    return picked


def sort_low_conf(rows: List[TileRow]) -> List[TileRow]:
    return sorted(
        rows,
        key=lambda r: (
            999.0 if r.min_conf is None else r.min_conf,
            -(r.num_preds),
            -(r.sum_area_m2),
            r.tile_rel,
        ),
    )


def sort_large_masks(rows: List[TileRow]) -> List[TileRow]:
    return sorted(
        rows,
        key=lambda r: (
            -1.0 if r.max_area_m2 is None else -r.max_area_m2,
            -(r.sum_area_m2),
            -(r.num_preds),
            r.tile_rel,
        ),
    )


def sort_many_preds(rows: List[TileRow]) -> List[TileRow]:
    return sorted(
        rows,
        key=lambda r: (
            -r.num_preds,
            -(r.sum_area_m2),
            999.0 if r.min_conf is None else r.min_conf,
            r.tile_rel,
        ),
    )


def sort_false_positive_candidates(rows: List[TileRow]) -> List[TileRow]:
    return sorted(
        rows,
        key=lambda r: (
            999.0 if r.max_area_m2 is None else r.max_area_m2,
            999.0 if r.mean_conf is None else abs(r.mean_conf - 0.35),
            r.num_preds,
            r.tile_rel,
        ),
    )


def sort_hard_empty(rows: List[TileRow], pos_per_cell: Counter, seed: int) -> List[TileRow]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return sorted(
        shuffled,
        key=lambda r: (
            -pos_per_cell[r.cell],
            r.blank_white,
            r.tile_rel,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-csv", type=Path, required=True)
    ap.add_argument("--tiles-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--exclude-label-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Optional YOLO label roots used to exclude already-labeled canonical tile ids.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Compute selection and summary without copying files.")

    ap.add_argument("--low-conf-count", type=int, default=60)
    ap.add_argument("--large-mask-count", type=int, default=40)
    ap.add_argument("--many-preds-count", type=int, default=40)
    ap.add_argument("--false-positive-count", type=int, default=30)
    ap.add_argument("--hard-empty-count", type=int, default=20)

    ap.add_argument("--low-conf-max", type=float, default=0.30)
    ap.add_argument("--many-preds-min", type=int, default=3)
    ap.add_argument("--false-pos-max-area-m2", type=float, default=30.0)
    ap.add_argument("--false-pos-max-preds", type=int, default=2)
    ap.add_argument("--false-pos-min-conf", type=float, default=0.18)
    ap.add_argument("--false-pos-max-conf", type=float, default=0.65)
    ap.add_argument("--hard-empty-min-positive-tiles-in-cell", type=int, default=5)
    ap.add_argument("--per-cell-limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if not args.dry_run and args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"out dir exists: {args.out_dir} (use --overwrite)")
        shutil.rmtree(args.out_dir)

    rows = read_rows(args.tiles_csv)
    tiles_root = args.tiles_root.resolve()
    exclude_label_roots = [p.expanduser().resolve() for p in args.exclude_label_roots]
    existing_canonical_ids = build_existing_canonical_ids(exclude_label_roots)

    usable: List[TileRow] = []
    missing_sources = 0
    excluded_existing_label = 0
    for row in rows:
        src = safe_source_path(tiles_root, row)
        if src is None:
            missing_sources += 1
            continue
        if row.blank_white == 1:
            continue
        if existing_canonical_ids:
            cid = canonical_tile_id(row.cell, row.tile_stem)
            if cid in existing_canonical_ids:
                excluded_existing_label += 1
                continue
        usable.append(row)

    pos_per_cell = cell_positive_counts(usable)

    positives = [r for r in usable if r.is_positive]
    empties = [r for r in usable if r.is_empty]

    low_conf_candidates = [
        r for r in positives
        if r.min_conf is not None and r.min_conf <= args.low_conf_max
    ]
    large_mask_candidates = [
        r for r in positives
        if r.max_area_m2 is not None
    ]
    many_preds_candidates = [
        r for r in positives
        if r.num_preds >= args.many_preds_min
    ]
    false_positive_candidates = [
        r for r in positives
        if r.num_preds <= args.false_pos_max_preds
        and r.max_area_m2 is not None
        and r.max_area_m2 <= args.false_pos_max_area_m2
        and r.mean_conf is not None
        and args.false_pos_min_conf <= r.mean_conf <= args.false_pos_max_conf
    ]
    hard_empty_candidates = [
        r for r in empties
        if pos_per_cell[r.cell] >= args.hard_empty_min_positive_tiles_in_cell
    ]

    low_conf_ranked = sort_low_conf(low_conf_candidates)
    large_mask_ranked = sort_large_masks(large_mask_candidates)
    many_preds_ranked = sort_many_preds(many_preds_candidates)
    false_pos_ranked = sort_false_positive_candidates(false_positive_candidates)
    hard_empty_ranked = sort_hard_empty(hard_empty_candidates, pos_per_cell, args.seed)

    used_tiles: Set[str] = set()
    used_per_cell: Counter = Counter()

    buckets: Dict[str, List[TileRow]] = {
        "low_conf": pick_rows(
            low_conf_ranked,
            limit=args.low_conf_count,
            used_tiles=used_tiles,
            per_cell_limit=args.per_cell_limit,
            used_per_cell=used_per_cell,
        ),
        "large_masks": pick_rows(
            large_mask_ranked,
            limit=args.large_mask_count,
            used_tiles=used_tiles,
            per_cell_limit=args.per_cell_limit,
            used_per_cell=used_per_cell,
        ),
        "many_preds": pick_rows(
            many_preds_ranked,
            limit=args.many_preds_count,
            used_tiles=used_tiles,
            per_cell_limit=args.per_cell_limit,
            used_per_cell=used_per_cell,
        ),
        "false_positive_candidates": pick_rows(
            false_pos_ranked,
            limit=args.false_positive_count,
            used_tiles=used_tiles,
            per_cell_limit=args.per_cell_limit,
            used_per_cell=used_per_cell,
        ),
        "hard_empty": pick_rows(
            hard_empty_ranked,
            limit=args.hard_empty_count,
            used_tiles=used_tiles,
            per_cell_limit=args.per_cell_limit,
            used_per_cell=used_per_cell,
        ),
    }

    global_manifest: List[Dict[str, object]] = []

    for bucket, picked in buckets.items():
        bucket_manifest: List[Dict[str, object]] = []
        for row in picked:
            src = safe_source_path(tiles_root, row)
            if src is None:
                continue
            mr = manifest_row(bucket, row)
            bucket_manifest.append(mr)
            global_manifest.append(mr)
            if args.dry_run:
                continue
            bucket_dir = args.out_dir / bucket
            dst = bucket_dir / row.tile_rel
            mirror_copy(src, dst)
        if not args.dry_run:
            write_manifest((args.out_dir / bucket) / "manifest.csv", bucket_manifest)

    if not args.dry_run:
        write_manifest(args.out_dir / "manifest.csv", global_manifest)

    summary = {
        "tiles_csv": str(args.tiles_csv),
        "tiles_root": str(tiles_root),
        "out_dir": str(args.out_dir),
        "rows_total": len(rows),
        "rows_usable_nonwhite_with_sources": len(usable),
        "missing_sources": missing_sources,
        "excluded_existing_label": excluded_existing_label,
        "exclude_label_roots": [str(p) for p in exclude_label_roots],
        "existing_canonical_ids_loaded": len(existing_canonical_ids),
        "positive_tiles_usable": len(positives),
        "empty_tiles_usable": len(empties),
        "candidate_counts": {
            "low_conf": len(low_conf_candidates),
            "large_masks": len(large_mask_candidates),
            "many_preds": len(many_preds_candidates),
            "false_positive_candidates": len(false_positive_candidates),
            "hard_empty": len(hard_empty_candidates),
        },
        "selected_counts": {k: len(v) for k, v in buckets.items()},
        "selected_cells_per_bucket": {k: len({r.cell for r in v}) for k, v in buckets.items()},
        "params": {
            "low_conf_max": args.low_conf_max,
            "many_preds_min": args.many_preds_min,
            "false_pos_max_area_m2": args.false_pos_max_area_m2,
            "false_pos_max_preds": args.false_pos_max_preds,
            "false_pos_min_conf": args.false_pos_min_conf,
            "false_pos_max_conf": args.false_pos_max_conf,
            "hard_empty_min_positive_tiles_in_cell": args.hard_empty_min_positive_tiles_in_cell,
            "per_cell_limit": args.per_cell_limit,
            "seed": args.seed,
            "dry_run": bool(args.dry_run),
        },
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("usable non-white tiles with sources:", len(usable))
    print("usable positives:", len(positives))
    print("usable empties:", len(empties))
    for bucket, picked in buckets.items():
        print(f"{bucket}: {len(picked)}")
    print("dry_run:", bool(args.dry_run))
    if not args.dry_run:
        print("wrote:", args.out_dir)


if __name__ == "__main__":
    main()
