#!/usr/bin/env python3
from __future__ import annotations

"""
Build a deterministic next-annotation batch from inference outputs.

This version tightens candidate generation so the selection pool is focused on
informative correction tiles (hard negatives/hard positives), while preserving:
  - canonical tile matching
  - exclusion of existing dataset train/val tiles
  - optional extra excluded tile-id list
  - deterministic seeded selection
  - per-cell cap and overlap/spacing suppression
  - manifest/GeoJSON/COCO outputs
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from shapely.geometry import shape

from tile_id_guard import extract_tile_id_from_row


TILE_ID_RE = r"^cell_(?P<cx>\d+)_(?P<cy>\d+)__r(?P<row>\d+)_c(?P<col>\d+)$"
CELL_RE = r"^cell_(?P<cx>\d+)_(?P<cy>\d+)$"
RC_RE = r"^r(?P<row>\d+)_c(?P<col>\d+)$"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

DEFAULT_EXCLUDE_DATASET = Path("data/datasets/google_z21_v7")
CURRENT_RUN_EXAMPLE = "/Users/admin/sao-paulo-pool-estimator/runs/segment/z21_v7_infer_20260410_054950"

BUCKET_TREE = "hard_negatives_tree_shadows"
BUCKET_EDGE = "hard_negatives_pool_edge_shadows"
BUCKET_BLUE = "hard_negatives_blue_roofs"
BUCKET_SHORE = "hard_negatives_shoreline_water_edge"
BUCKET_HARD_POS = "hard_positives"
BUCKET_MIXED = "mixed_review"
BUCKET_FALLBACK = "fallback_fill"
BUCKET_ORDER = (
    BUCKET_TREE,
    BUCKET_EDGE,
    BUCKET_BLUE,
    BUCKET_SHORE,
    BUCKET_HARD_POS,
    BUCKET_MIXED,
    BUCKET_FALLBACK,
)
NEGATIVE_BUCKETS = (BUCKET_TREE, BUCKET_EDGE, BUCKET_BLUE, BUCKET_SHORE)
DEFAULT_BUCKET_WEIGHTS = {
    BUCKET_TREE: 0.0875,
    BUCKET_EDGE: 0.0875,
    BUCKET_BLUE: 0.0875,
    BUCKET_SHORE: 0.0875,
    BUCKET_HARD_POS: 0.35,
    BUCKET_MIXED: 0.20,
    BUCKET_FALLBACK: 0.10,
}


@dataclass
class TileSummary:
    tile_id: str
    tile_rel: str
    cell: str
    tile_stem: str
    row: Optional[int]
    col: Optional[int]
    tile_path_abs: str
    num_preds: int
    min_conf: Optional[float]
    mean_conf: Optional[float]
    max_conf: Optional[float]
    max_area_m2: Optional[float]
    sum_area_m2: float
    blank_white: int
    skip_reason: str


@dataclass
class TileFeatureStats:
    num_features: int = 0
    conf_sum: float = 0.0
    conf_min: float = float("inf")
    conf_max: float = 0.0
    area_sum: float = 0.0
    area_max: float = 0.0

    low_conf_count: int = 0
    medium_conf_count: int = 0
    medium_area_count: int = 0
    small_count: int = 0
    large_count: int = 0
    irregular_count: int = 0
    irregular_large_count: int = 0
    rectangular_count: int = 0
    fill_low_count: int = 0

    @property
    def mean_conf(self) -> float:
        if self.num_features <= 0:
            return 0.0
        return self.conf_sum / float(self.num_features)

    def add(
        self,
        *,
        confidence: float,
        area_m2: float,
        compactness: float,
        aspect: float,
        fill_ratio: float,
    ) -> None:
        self.num_features += 1
        self.conf_sum += confidence
        self.conf_min = min(self.conf_min, confidence)
        self.conf_max = max(self.conf_max, confidence)
        self.area_sum += area_m2
        self.area_max = max(self.area_max, area_m2)

        if confidence <= 0.45:
            self.low_conf_count += 1
        if 0.35 <= confidence <= 0.72:
            self.medium_conf_count += 1
        if 20.0 <= area_m2 <= 450.0:
            self.medium_area_count += 1
        if area_m2 <= 85.0:
            self.small_count += 1
        if area_m2 >= 250.0:
            self.large_count += 1

        is_irregular = (compactness < 0.50) or (aspect > 2.25) or (fill_ratio < 0.45)
        if is_irregular:
            self.irregular_count += 1
        if is_irregular and area_m2 >= 180.0:
            self.irregular_large_count += 1

        is_rectangular = (
            (0.48 <= compactness <= 0.96)
            and (1.08 <= aspect <= 4.80)
            and (fill_ratio >= 0.50)
            and (20.0 <= area_m2 <= 1300.0)
        )
        if is_rectangular:
            self.rectangular_count += 1
        if fill_ratio < 0.50:
            self.fill_low_count += 1


@dataclass
class BucketCandidate:
    tile_id: str
    bucket: str
    score: float
    reason: str


@dataclass
class SelectedTile:
    tile_id: str
    bucket: str
    score: float
    reason: str
    order: int


@dataclass
class SelectionStats:
    tile_rows_scanned: int = 0
    geojson_features_scanned: int = 0

    tiles_seen_total: int = 0
    excluded_existing_dataset: int = 0
    excluded_tile_ids_file: int = 0

    tiles_with_predictions: int = 0
    excluded_zero_predictions: int = 0
    excluded_too_few_predictions: int = 0
    excluded_too_many_predictions: int = 0
    excluded_too_confident: int = 0
    excluded_low_confidence: int = 0
    excluded_area_cluster: int = 0
    excluded_small_total_area: int = 0
    remaining_candidate_tiles: int = 0

    candidate_tiles_considered: int = 0
    rejected_overlap: int = 0
    rejected_per_cell_cap: int = 0
    rejected_duplicate_tile: int = 0
    rejected_missing_tile_meta: int = 0

    selected_prediction_features: int = 0
    selected_total: int = 0

    existing_dataset_files_scanned: int = 0
    existing_dataset_tile_ids_loaded: int = 0
    exclude_file_tile_ids_loaded: int = 0

    selected_by_bucket: Counter = field(default_factory=Counter)
    selected_by_cell: Counter = field(default_factory=Counter)
    bucket_candidate_counts: Dict[str, int] = field(default_factory=dict)
    bucket_targets: Dict[str, int] = field(default_factory=dict)


@dataclass
class SelectionState:
    selected_ids: set[str] = field(default_factory=set)
    selected_list: List[SelectedTile] = field(default_factory=list)
    selected_cell_counts: Counter = field(default_factory=Counter)
    selected_cell_rcs: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    selected_rc_groups: set[Tuple[str, int, int]] = field(default_factory=set)
    selected_centers: List[Tuple[float, float]] = field(default_factory=list)
    center_cache: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    worldfile_cache: Dict[str, Optional[Tuple[float, float, float, float, float, float]]] = field(default_factory=dict)
    image_size_cache: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    epilog = dedent(
        f"""
        Example:
          .venv/bin/python tools/build_next_annotation_batch.py \\
            --run-dir {CURRENT_RUN_EXAMPLE} \\
            --in-geojson {CURRENT_RUN_EXAMPLE}/pools_dedup.geojson \\
            --tiles-csv {CURRENT_RUN_EXAMPLE}/pools_tiles.csv \\
            --tiles-jsonl {CURRENT_RUN_EXAMPLE}/pools_tiles.jsonl \\
            --tiles-root data/raw/google/sp_city_2020_rebuild_google_z21 \\
            --exclude-existing-dataset data/datasets/google_z21_v7 \\
            --seed 42 \\
            --max-tiles-per-cell 4 \\
            --target-batch-size 200 \\
            --out-tile-list runs/annotation_batches/z21_v7_next_batch/tile_ids.txt \\
            --out-manifest-csv runs/annotation_batches/z21_v7_next_batch/manifest.csv \\
            --out-geojson runs/annotation_batches/z21_v7_next_batch/selected_predictions.geojson \\
            --out-coco-json runs/annotation_batches/z21_v7_next_batch/predictions_coco.json
        """
    ).strip()
    ap = argparse.ArgumentParser(
        description=(
            "Build deterministic next-annotation batches from inference outputs using "
            "tightened high-value candidate filtering and bucketed selection."
        ),
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--in-geojson", type=Path, default=None)
    ap.add_argument("--tiles-csv", type=Path, default=None)
    ap.add_argument("--tiles-jsonl", type=Path, default=None)
    ap.add_argument(
        "--tiles-root",
        type=Path,
        default=Path("data/raw/google/sp_city_2020_rebuild_google_z21"),
    )
    ap.add_argument(
        "--exclude-existing-dataset",
        type=Path,
        default=DEFAULT_EXCLUDE_DATASET,
        help="Dataset root. Scans images/train and images/val for canonical tile IDs to exclude.",
    )
    ap.add_argument(
        "--exclude-tile-ids",
        type=Path,
        default=None,
        help="Optional text file with one canonical tile id per line to exclude.",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tiles-per-cell", type=int, default=4)
    ap.add_argument("--local-neighbor-radius", type=int, default=1)
    ap.add_argument("--rc-group-size", type=int, default=0)
    ap.add_argument("--min-spacing-m", type=float, default=0.0)
    ap.add_argument(
        "--neighbor-radius",
        type=int,
        default=1,
        help="Neighborhood radius in tile row/col units for contextual scoring.",
    )

    ap.add_argument("--target-batch-size", type=int, default=200)
    ap.add_argument("--min-preds-per-tile", type=int, default=1)
    ap.add_argument("--max-preds-per-tile", type=int, default=6)
    ap.add_argument("--min-max-conf", type=float, default=0.0)
    ap.add_argument("--max-max-conf", type=float, default=0.75)
    ap.add_argument("--min-mean-conf", type=float, default=0.0)
    ap.add_argument("--max-mean-conf", type=float, default=0.60)
    ap.add_argument("--min-total-area-m2", type=float, default=8.0)
    ap.add_argument("--max-total-area-m2", type=float, default=1600.0)
    ap.add_argument(
        "--prefer-low-confidence",
        type=float,
        default=1.0,
        help="Weight multiplier for low-to-mid confidence ranking preference.",
    )
    ap.add_argument(
        "--prefer-medium-density",
        type=float,
        default=1.0,
        help="Weight multiplier for medium detections-per-tile preference.",
    )
    ap.add_argument(
        "--prefer-hard-negatives",
        type=float,
        default=1.0,
        help="Weight multiplier for hard-negative buckets.",
    )
    ap.add_argument(
        "--prefer-hard-positives",
        type=float,
        default=1.0,
        help="Weight multiplier for hard-positive bucket.",
    )
    ap.add_argument(
        "--bucket-targets-json",
        default="",
        help=(
            "JSON string or JSON file path to override bucket targets. "
            "Supports weights (0-1) or absolute counts by bucket/group."
        ),
    )

    # Legacy count overrides (kept for CLI compatibility).
    ap.add_argument("--tree-shadows-count", type=int, default=None)
    ap.add_argument("--pool-edge-shadows-count", type=int, default=None)
    ap.add_argument("--blue-roofs-count", type=int, default=None)
    ap.add_argument("--shoreline-count", type=int, default=None)
    ap.add_argument("--hard-positives-count", type=int, default=None)
    ap.add_argument("--mixed-review-count", type=int, default=None)
    ap.add_argument("--fallback-fill-count", type=int, default=None)

    ap.add_argument("--out-tile-list", type=Path, required=True)
    ap.add_argument("--out-manifest-csv", type=Path, required=True)
    ap.add_argument("--out-geojson", type=Path, required=True)
    ap.add_argument("--out-coco-json", type=Path, required=True)
    ap.add_argument(
        "--out-images-dir",
        type=Path,
        default=None,
        help="Destination for selected images. Default: <out-manifest dir>/JPEGImages",
    )
    ap.add_argument(
        "--image-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to materialize selected images into --out-images-dir.",
    )

    ap.add_argument("--geojson-crs", default="")
    ap.add_argument("--worldfile-crs", default="EPSG:31983")
    ap.add_argument("--coco-min-confidence", type=float, default=0.0)
    ap.add_argument("--coco-category-id", type=int, default=1)
    ap.add_argument("--coco-category-name", default="pool")
    ap.add_argument(
        "--converter-script",
        type=Path,
        default=Path("tools/geojson_to_coco_cvat.py"),
    )
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def normalize_tile_stem(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return Path(raw).with_suffix("").name


def canonical_tile_id_from_parts(cell: str, tile_stem: str) -> Optional[str]:
    import re

    cm = re.match(CELL_RE, str(cell or "").strip())
    rm = re.match(RC_RE, normalize_tile_stem(tile_stem))
    if cm is None or rm is None:
        return None
    cx = int(cm.group("cx"))
    cy = int(cm.group("cy"))
    row = int(rm.group("row"))
    col = int(rm.group("col"))
    return f"cell_{cx:04d}_{cy:04d}__r{row:04d}_c{col:04d}"


def canonicalize_tile_id(tile_id: str) -> Optional[str]:
    import re

    tm = re.match(TILE_ID_RE, str(tile_id or "").strip())
    if tm is None:
        return None
    cx = int(tm.group("cx"))
    cy = int(tm.group("cy"))
    row = int(tm.group("row"))
    col = int(tm.group("col"))
    return f"cell_{cx:04d}_{cy:04d}__r{row:04d}_c{col:04d}"


def canonical_rel_from_tile_id(tile_id: str, ext: str = ".png") -> Optional[str]:
    tid = canonicalize_tile_id(tile_id)
    if tid is None:
        return None
    cell, rc = tid.split("__", 1)
    suffix = ext.lower() if ext else ".png"
    if not suffix.startswith("."):
        suffix = "." + suffix
    return f"{cell}/{rc}{suffix}"


def parse_tile_id_parts(tile_id: str) -> Optional[Tuple[str, int, int]]:
    tid = canonicalize_tile_id(tile_id)
    if tid is None:
        return None
    cell, rc = tid.split("__", 1)
    stem = normalize_tile_stem(rc)
    import re

    rm = re.match(RC_RE, stem)
    if rm is None:
        return None
    return cell, int(rm.group("row")), int(rm.group("col"))


def extract_canonical_tile_id(row: Mapping[str, Any]) -> Optional[str]:
    raw = extract_tile_id_from_row(row)
    if raw:
        tid = canonicalize_tile_id(raw)
        if tid:
            return tid
    return canonical_tile_id_from_parts(str(row.get("cell") or ""), str(row.get("tile_stem") or row.get("tile") or ""))


def stable_hash_int(seed: int, *parts: str) -> int:
    joined = f"{seed}|" + "|".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_unit(seed: int, *parts: str) -> float:
    return float(stable_hash_int(seed, *parts) % 10_000_000) / 10_000_000.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def norm01(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))


def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"--run-dir not found: {run_dir}")
    args.run_dir = run_dir

    in_geojson = args.in_geojson.expanduser().resolve() if args.in_geojson is not None else None
    if in_geojson is None:
        p_dedup = run_dir / "pools_dedup.geojson"
        p_raw = run_dir / "pools.geojson"
        if p_dedup.exists():
            in_geojson = p_dedup
        elif p_raw.exists():
            in_geojson = p_raw
        else:
            raise SystemExit(f"Missing GeoJSON in run dir: expected {p_dedup} or {p_raw}")
    if not in_geojson.exists():
        raise SystemExit(f"--in-geojson not found: {in_geojson}")
    args.in_geojson = in_geojson

    tiles_csv = args.tiles_csv.expanduser().resolve() if args.tiles_csv is not None else (run_dir / "pools_tiles.csv")
    tiles_jsonl = args.tiles_jsonl.expanduser().resolve() if args.tiles_jsonl is not None else (run_dir / "pools_tiles.jsonl")
    if not tiles_csv.exists() and not tiles_jsonl.exists():
        raise SystemExit(f"Missing tile manifest: {tiles_csv} and {tiles_jsonl} were both not found.")
    args.tiles_csv = tiles_csv
    args.tiles_jsonl = tiles_jsonl

    args.tiles_root = args.tiles_root.expanduser().resolve()
    if not args.tiles_root.exists():
        raise SystemExit(f"--tiles-root not found: {args.tiles_root}")

    if args.exclude_existing_dataset is not None:
        args.exclude_existing_dataset = args.exclude_existing_dataset.expanduser().resolve()
        if not args.exclude_existing_dataset.exists():
            raise SystemExit(f"--exclude-existing-dataset not found: {args.exclude_existing_dataset}")

    if args.exclude_tile_ids is not None:
        args.exclude_tile_ids = args.exclude_tile_ids.expanduser().resolve()
        if not args.exclude_tile_ids.exists():
            raise SystemExit(f"--exclude-tile-ids not found: {args.exclude_tile_ids}")

    args.out_tile_list = args.out_tile_list.expanduser().resolve()
    args.out_manifest_csv = args.out_manifest_csv.expanduser().resolve()
    args.out_geojson = args.out_geojson.expanduser().resolve()
    args.out_coco_json = args.out_coco_json.expanduser().resolve()
    args.out_images_dir = (
        args.out_images_dir.expanduser().resolve()
        if args.out_images_dir is not None
        else (args.out_manifest_csv.parent / "JPEGImages").resolve()
    )
    args.converter_script = args.converter_script.expanduser().resolve()
    return args


def iter_tile_rows(csv_path: Path, jsonl_path: Path) -> Tuple[str, Iterator[Mapping[str, Any]], int]:
    if csv_path.exists():

        def csv_iter() -> Iterator[Mapping[str, Any]]:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row

        row_count = 0
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for _ in reader:
                row_count += 1
        return "csv", csv_iter(), row_count

    def jsonl_iter() -> Iterator[Mapping[str, Any]]:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                yield json.loads(text)

    row_count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row_count += 1
    return "jsonl", jsonl_iter(), row_count


def build_tile_summary(row: Mapping[str, Any], tiles_root: Path) -> Optional[TileSummary]:
    tile_id = extract_canonical_tile_id(row)
    if tile_id is None:
        return None
    parts = parse_tile_id_parts(tile_id)
    if parts is None:
        return None
    cell, row_idx, col_idx = parts

    tile_stem = normalize_tile_stem(row.get("tile_stem") or row.get("tile"))
    if not tile_stem:
        tile_stem = tile_id.split("__", 1)[1]

    raw_rel = str(row.get("tile_rel") or "").strip().replace("\\", "/")
    ext = Path(raw_rel).suffix.lower() if raw_rel else Path(str(row.get("tile_path_abs") or "")).suffix.lower()
    if not ext:
        ext = ".png"
    canonical_rel = canonical_rel_from_tile_id(tile_id, ext=ext) or raw_rel

    tile_path_abs_text = str(row.get("tile_path_abs") or "").strip()
    tile_path = Path(tile_path_abs_text) if tile_path_abs_text else (tiles_root / canonical_rel)
    if not tile_path.is_absolute():
        tile_path = (tiles_root / tile_path).resolve()
    tile_path_abs = str(tile_path)

    num_preds = parse_int(row.get("num_preds"), default=0)
    min_conf = parse_float(row.get("min_conf"))
    mean_conf = parse_float(row.get("mean_conf"))
    max_conf = parse_float(row.get("max_conf"))
    max_area_m2 = parse_float(row.get("max_area_m2"))
    sum_area_m2 = parse_float(row.get("sum_area_m2")) or 0.0
    blank_white = parse_int(row.get("blank_white"), default=0)
    skip_reason = str(row.get("skip_reason") or "").strip()

    return TileSummary(
        tile_id=tile_id,
        tile_rel=canonical_rel,
        cell=cell,
        tile_stem=tile_stem,
        row=row_idx,
        col=col_idx,
        tile_path_abs=tile_path_abs,
        num_preds=num_preds,
        min_conf=min_conf,
        mean_conf=mean_conf,
        max_conf=max_conf,
        max_area_m2=max_area_m2,
        sum_area_m2=float(sum_area_m2),
        blank_white=blank_white,
        skip_reason=skip_reason,
    )


def choose_better_tile_summary(current: TileSummary, incoming: TileSummary) -> TileSummary:
    key_current = (
        current.num_preds,
        current.sum_area_m2,
        current.max_conf if current.max_conf is not None else -1.0,
        current.tile_path_abs,
    )
    key_incoming = (
        incoming.num_preds,
        incoming.sum_area_m2,
        incoming.max_conf if incoming.max_conf is not None else -1.0,
        incoming.tile_path_abs,
    )
    if key_incoming > key_current:
        return incoming
    return current


def load_tile_summaries(
    *,
    tiles_csv: Path,
    tiles_jsonl: Path,
    tiles_root: Path,
) -> Tuple[Dict[str, TileSummary], str, int]:
    source, rows_iter, rows_scanned = iter_tile_rows(tiles_csv, tiles_jsonl)
    out: Dict[str, TileSummary] = {}
    for row in rows_iter:
        ts = build_tile_summary(row, tiles_root)
        if ts is None:
            continue
        prev = out.get(ts.tile_id)
        if prev is None:
            out[ts.tile_id] = ts
        else:
            out[ts.tile_id] = choose_better_tile_summary(prev, ts)
    return out, source, rows_scanned


def collect_existing_dataset_tile_ids(dataset_root: Path) -> Tuple[set[str], int]:
    ids: set[str] = set()
    files_scanned = 0
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            continue
        for p in image_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            files_scanned += 1
            tid = canonicalize_tile_id(p.stem.strip())
            if tid:
                ids.add(tid)
    return ids, files_scanned


def load_excluded_tile_ids_file(path: Optional[Path]) -> set[str]:
    if path is None:
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tid = canonicalize_tile_id(line.split()[0])
        if tid:
            out.add(tid)
    return out


def apply_initial_exclusions(
    tiles: Mapping[str, TileSummary],
    *,
    existing_dataset_ids: set[str],
    excluded_tile_ids: set[str],
) -> Tuple[Dict[str, TileSummary], Dict[str, int]]:
    all_ids = list(tiles.keys())
    all_set = set(all_ids)

    excluded_existing = all_set & existing_dataset_ids
    remaining_after_existing = all_set - excluded_existing
    excluded_from_file = remaining_after_existing & excluded_tile_ids
    remaining_ids = remaining_after_existing - excluded_from_file

    filtered: Dict[str, TileSummary] = {}
    for tid in all_ids:
        if tid in remaining_ids:
            filtered[tid] = tiles[tid]

    return filtered, {
        "tiles_seen_total": int(len(all_set)),
        "excluded_existing_dataset": int(len(excluded_existing)),
        "excluded_tile_ids_file": int(len(excluded_from_file)),
        "remaining_after_initial_exclusions": int(len(remaining_ids)),
    }


def geometry_metrics(geom_payload: Any) -> Tuple[float, float, float, float]:
    if not isinstance(geom_payload, dict):
        return 0.0, 1.0, 1.0, 0.0
    try:
        geom = shape(geom_payload)
    except Exception:
        return 0.0, 1.0, 1.0, 0.0
    if geom.is_empty:
        return 0.0, 1.0, 1.0, 0.0

    area = float(geom.area)
    perimeter = float(geom.length)
    compactness = 0.0 if perimeter <= 0.0 else float((4.0 * math.pi * area) / max(perimeter * perimeter, 1e-9))
    minx, miny, maxx, maxy = geom.bounds
    w = max(0.0, float(maxx - minx))
    h = max(0.0, float(maxy - miny))
    mn = max(min(w, h), 1e-9)
    mx = max(w, h)
    aspect = mx / mn
    bbox_area = max(w * h, 1e-9)
    fill = area / bbox_area
    return compactness, aspect, fill, area


def load_geojson_and_tile_stats(
    geojson_path: Path,
) -> Tuple[Dict[str, Any], List[str], Dict[str, TileFeatureStats], int]:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise SystemExit(f"GeoJSON must be FeatureCollection: {geojson_path}")
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise SystemExit(f"GeoJSON features is not a list: {geojson_path}")

    feature_tile_ids: List[str] = []
    stats: Dict[str, TileFeatureStats] = {}
    for feat in features:
        if not isinstance(feat, dict):
            feature_tile_ids.append("")
            continue

        props = feat.get("properties")
        if not isinstance(props, dict):
            props = {}
            feat["properties"] = props

        tile_id = extract_canonical_tile_id(props) or ""
        if tile_id:
            props["tile_id"] = tile_id
            canonical_rel = canonical_rel_from_tile_id(tile_id, ext=Path(str(props.get("tile_rel") or "")).suffix or ".png")
            if canonical_rel:
                props["tile_rel"] = canonical_rel
            parts = parse_tile_id_parts(tile_id)
            if parts is not None:
                props["cell"] = parts[0]
                props["tile_stem"] = tile_id.split("__", 1)[1]
        feature_tile_ids.append(tile_id)
        if not tile_id:
            continue

        confidence = parse_float(props.get("confidence")) or 0.0
        compactness, aspect, fill, geom_area = geometry_metrics(feat.get("geometry"))
        area_m2 = parse_float(props.get("area_m2"))
        if area_m2 is None or area_m2 <= 0:
            area_m2 = geom_area

        tile_stats = stats.get(tile_id)
        if tile_stats is None:
            tile_stats = TileFeatureStats()
            stats[tile_id] = tile_stats
        tile_stats.add(
            confidence=float(confidence),
            area_m2=float(max(area_m2, 0.0)),
            compactness=float(compactness),
            aspect=float(aspect),
            fill_ratio=float(fill),
        )

    return payload, feature_tile_ids, stats, len(features)


def resolved_tile_metrics(tile: TileSummary, feature: Optional[TileFeatureStats]) -> Tuple[float, float, float]:
    mean_conf = tile.mean_conf if tile.mean_conf is not None else 0.0
    max_conf = tile.max_conf if tile.max_conf is not None else mean_conf
    area_sum = float(tile.sum_area_m2)
    if feature is not None and feature.num_features > 0:
        mean_conf = float(feature.mean_conf)
        max_conf = float(feature.conf_max)
        area_sum = float(feature.area_sum)
    return mean_conf, max_conf, area_sum


def filter_informative_candidates(
    *,
    tiles: Mapping[str, TileSummary],
    feature_stats: Mapping[str, TileFeatureStats],
    args: argparse.Namespace,
    stats: SelectionStats,
) -> Dict[str, TileSummary]:
    filtered: Dict[str, TileSummary] = {}
    for tile_id, tile in tiles.items():
        if tile.num_preds <= 0:
            stats.excluded_zero_predictions += 1
            continue
        stats.tiles_with_predictions += 1

        if tile.num_preds < int(args.min_preds_per_tile):
            stats.excluded_too_few_predictions += 1
            continue
        if int(args.max_preds_per_tile) > 0 and tile.num_preds > int(args.max_preds_per_tile):
            stats.excluded_too_many_predictions += 1
            continue

        feature = feature_stats.get(tile_id)
        mean_conf, max_conf, area_sum = resolved_tile_metrics(tile, feature)

        if max_conf < float(args.min_max_conf) or mean_conf < float(args.min_mean_conf):
            stats.excluded_low_confidence += 1
            continue
        if max_conf > float(args.max_max_conf) or mean_conf > float(args.max_mean_conf):
            stats.excluded_too_confident += 1
            continue

        if area_sum < float(args.min_total_area_m2):
            stats.excluded_small_total_area += 1
            continue
        if float(args.max_total_area_m2) > 0.0 and area_sum > float(args.max_total_area_m2):
            stats.excluded_area_cluster += 1
            continue

        filtered[tile_id] = tile

    stats.remaining_candidate_tiles = len(filtered)
    return filtered


def build_neighbor_context(
    tiles: Mapping[str, TileSummary],
    *,
    radius: int,
) -> Dict[str, Tuple[float, float, float, float]]:
    cell_total_preds: Dict[str, float] = defaultdict(float)
    cell_grid: Dict[str, Dict[Tuple[int, int], TileSummary]] = defaultdict(dict)
    for tile in tiles.values():
        cell_total_preds[tile.cell] += float(tile.num_preds)
        if tile.row is not None and tile.col is not None:
            cell_grid[tile.cell][(tile.row, tile.col)] = tile

    out: Dict[str, Tuple[float, float, float, float]] = {}
    rr = max(0, int(radius))
    for tile in tiles.values():
        neighbor_preds = 0.0
        neighbor_tiles = 0.0
        neighbor_area = 0.0
        if tile.row is not None and tile.col is not None and rr > 0:
            grid = cell_grid.get(tile.cell, {})
            for dr in range(-rr, rr + 1):
                for dc in range(-rr, rr + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = grid.get((tile.row + dr, tile.col + dc))
                    if nb is None:
                        continue
                    neighbor_preds += float(nb.num_preds)
                    neighbor_area += float(nb.sum_area_m2)
                    if nb.num_preds > 0:
                        neighbor_tiles += 1.0
        out[tile.tile_id] = (
            neighbor_preds,
            neighbor_tiles,
            neighbor_area,
            float(cell_total_preds.get(tile.cell, 0.0)),
        )
    return out


def medium_density_score(num_preds: int, args: argparse.Namespace) -> float:
    lo = float(max(1, int(args.min_preds_per_tile)))
    hi = float(max(lo, int(args.max_preds_per_tile)))
    center = lo + ((hi - lo) * 0.5)
    spread = max(1.0, (hi - lo) * 0.55)
    return clamp01(1.0 - abs(float(num_preds) - center) / spread)


def low_confidence_priority(mean_conf: float, args: argparse.Namespace) -> float:
    return 1.0 - norm01(mean_conf, float(args.min_mean_conf), float(args.max_mean_conf))


def uncertainty_score(mean_conf: float, args: argparse.Namespace) -> float:
    mid = (float(args.min_mean_conf) + float(args.max_mean_conf)) * 0.5
    span = max(0.20, (float(args.max_mean_conf) - float(args.min_mean_conf)) * 0.65)
    return clamp01(1.0 - (abs(mean_conf - mid) / span))


def build_bucket_candidates(
    *,
    tiles: Mapping[str, TileSummary],
    feature_stats: Mapping[str, TileFeatureStats],
    neighbor_ctx: Mapping[str, Tuple[float, float, float, float]],
    args: argparse.Namespace,
) -> Dict[str, List[BucketCandidate]]:
    out: Dict[str, List[BucketCandidate]] = {b: [] for b in BUCKET_ORDER}

    for tile_id, tile in tiles.items():
        ts = feature_stats.get(tile_id)
        feature_n = float(ts.num_features) if (ts is not None and ts.num_features > 0) else float(max(tile.num_preds, 1))
        mean_conf, _max_conf, area_sum = resolved_tile_metrics(tile, ts)

        low_conf_ratio = safe_ratio(float(ts.low_conf_count if ts else 0), feature_n)
        medium_area_ratio = safe_ratio(float(ts.medium_area_count if ts else 0), feature_n)
        small_ratio = safe_ratio(float(ts.small_count if ts else 0), feature_n)
        irregular_ratio = safe_ratio(float(ts.irregular_count if ts else 0), feature_n)
        irregular_large_ratio = safe_ratio(float(ts.irregular_large_count if ts else 0), feature_n)
        rect_ratio = safe_ratio(float(ts.rectangular_count if ts else 0), feature_n)
        fill_low_ratio = safe_ratio(float(ts.fill_low_count if ts else 0), feature_n)
        large_ratio = safe_ratio(float(ts.large_count if ts else 0), feature_n)

        neighbor_preds, neighbor_tiles, neighbor_area, cell_preds = neighbor_ctx.get(tile_id, (0.0, 0.0, 0.0, 0.0))
        density_s = medium_density_score(tile.num_preds, args) * float(args.prefer_medium_density)
        low_conf_s = low_confidence_priority(mean_conf, args) * float(args.prefer_low_confidence)
        uncertain_s = uncertainty_score(mean_conf, args)
        area_mid_s = 1.0 - abs(norm01(area_sum, float(args.min_total_area_m2), float(args.max_total_area_m2)) - 0.45)
        area_mid_s = clamp01(area_mid_s)

        if low_conf_ratio >= 0.25 and medium_area_ratio >= 0.20:
            score = float(args.prefer_hard_negatives) * (
                0.40 * low_conf_ratio
                + 0.22 * medium_area_ratio
                + 0.18 * density_s
                + 0.12 * low_conf_s
                + 0.08 * uncertain_s
            )
            reason = (
                f"tree_shadow_proxy low_conf_ratio={low_conf_ratio:.3f} medium_area_ratio={medium_area_ratio:.3f} "
                f"num_preds={tile.num_preds} mean_conf={mean_conf:.3f}"
            )
            out[BUCKET_TREE].append(BucketCandidate(tile_id=tile_id, bucket=BUCKET_TREE, score=score, reason=reason))

        if tile.num_preds >= 2 and fill_low_ratio >= 0.20 and small_ratio >= 0.20:
            score = float(args.prefer_hard_negatives) * (
                0.36 * fill_low_ratio
                + 0.22 * small_ratio
                + 0.20 * density_s
                + 0.14 * low_conf_s
                + 0.08 * norm01(neighbor_preds, 0.0, 10.0)
            )
            reason = (
                f"pool_edge_shadow_proxy fill_low_ratio={fill_low_ratio:.3f} small_ratio={small_ratio:.3f} "
                f"num_preds={tile.num_preds} neighbor_preds={neighbor_preds:.1f}"
            )
            out[BUCKET_EDGE].append(BucketCandidate(tile_id=tile_id, bucket=BUCKET_EDGE, score=score, reason=reason))

        if rect_ratio >= 0.35 and mean_conf >= 0.15 and mean_conf <= float(args.max_mean_conf):
            score = float(args.prefer_hard_negatives) * (
                0.42 * rect_ratio
                + 0.18 * density_s
                + 0.15 * low_conf_s
                + 0.15 * norm01(mean_conf, 0.15, float(args.max_mean_conf))
                + 0.10 * area_mid_s
            )
            reason = (
                f"blue_roof_proxy rect_ratio={rect_ratio:.3f} mean_conf={mean_conf:.3f} "
                f"area_sum={area_sum:.1f}"
            )
            out[BUCKET_BLUE].append(BucketCandidate(tile_id=tile_id, bucket=BUCKET_BLUE, score=score, reason=reason))

        if irregular_ratio >= 0.30 and (irregular_large_ratio >= 0.10 or area_sum >= 220.0):
            score = float(args.prefer_hard_negatives) * (
                0.35 * irregular_ratio
                + 0.20 * irregular_large_ratio
                + 0.18 * norm01(neighbor_area, 0.0, 1500.0)
                + 0.15 * low_conf_s
                + 0.12 * density_s
            )
            reason = (
                f"shoreline_proxy irregular_ratio={irregular_ratio:.3f} irregular_large_ratio={irregular_large_ratio:.3f} "
                f"area_sum={area_sum:.1f}"
            )
            out[BUCKET_SHORE].append(BucketCandidate(tile_id=tile_id, bucket=BUCKET_SHORE, score=score, reason=reason))

        positive_signal = (
            0.33 * small_ratio
            + 0.28 * irregular_ratio
            + 0.20 * fill_low_ratio
            + 0.11 * low_conf_s
            + 0.08 * norm01(neighbor_tiles, 0.0, 6.0)
        )
        if positive_signal >= 0.18:
            score = float(args.prefer_hard_positives) * (
                0.55 * positive_signal
                + 0.20 * density_s
                + 0.15 * uncertain_s
                + 0.10 * norm01(cell_preds, 0.0, 200.0)
            )
            reason = (
                f"hard_positive_proxy small={small_ratio:.3f} irregular={irregular_ratio:.3f} "
                f"fill_low={fill_low_ratio:.3f} mean_conf={mean_conf:.3f}"
            )
            out[BUCKET_HARD_POS].append(BucketCandidate(tile_id=tile_id, bucket=BUCKET_HARD_POS, score=score, reason=reason))

        mixed_score = (
            0.40 * uncertain_s
            + 0.26 * density_s
            + 0.18 * low_conf_s
            + 0.08 * norm01(neighbor_preds, 0.0, 10.0)
            + 0.08 * stable_unit(int(args.seed), "mixed", tile_id)
        )
        mixed_reason = (
            f"mixed_review uncertainty={uncertain_s:.3f} density={density_s:.3f} "
            f"low_conf={low_conf_s:.3f} mean_conf={mean_conf:.3f}"
        )
        out[BUCKET_MIXED].append(BucketCandidate(tile_id=tile_id, bucket=BUCKET_MIXED, score=mixed_score, reason=mixed_reason))

        fallback_score = (
            0.33 * low_conf_s
            + 0.28 * density_s
            + 0.17 * uncertain_s
            + 0.12 * (1.0 - large_ratio)
            + 0.10 * stable_unit(int(args.seed), "fallback", tile_id)
        )
        fallback_reason = (
            f"fallback low_conf={low_conf_s:.3f} density={density_s:.3f} uncertainty={uncertain_s:.3f} "
            f"num_preds={tile.num_preds}"
        )
        out[BUCKET_FALLBACK].append(
            BucketCandidate(tile_id=tile_id, bucket=BUCKET_FALLBACK, score=fallback_score, reason=fallback_reason)
        )

    return out


def dedupe_bucket_candidates(
    candidates: Sequence[BucketCandidate],
    *,
    seed: int,
    bucket: str,
) -> List[BucketCandidate]:
    by_tile: Dict[str, BucketCandidate] = {}
    for cand in candidates:
        prev = by_tile.get(cand.tile_id)
        if prev is None:
            by_tile[cand.tile_id] = cand
            continue
        if cand.score > prev.score:
            by_tile[cand.tile_id] = cand
            continue
        if cand.score == prev.score:
            if stable_hash_int(seed, bucket, cand.tile_id) < stable_hash_int(seed, bucket, prev.tile_id):
                by_tile[cand.tile_id] = cand
    out = list(by_tile.values())
    out.sort(key=lambda c: (-c.score, stable_hash_int(seed, bucket, c.tile_id), c.tile_id))
    return out


def find_worldfile(img: Path) -> Optional[Path]:
    suffix = img.suffix.lower()
    if suffix == ".png":
        candidates = [img.with_suffix(".pgw"), img.with_suffix(".pngw"), img.with_suffix(".wld")]
    elif suffix in {".jpg", ".jpeg"}:
        candidates = [img.with_suffix(".jgw"), img.with_suffix(".jpgw"), img.with_suffix(".wld")]
    else:
        candidates = [img.with_suffix(".wld")]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_worldfile(path: Path) -> Optional[Tuple[float, float, float, float, float, float]]:
    try:
        vals = [float(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception:
        return None
    if len(vals) != 6:
        return None
    a, d, b, e, c, f = vals
    return (c, a, b, f, d, e)


def pixel_to_world(transform: Tuple[float, float, float, float, float, float], px: float, py: float) -> Tuple[float, float]:
    c, a, b, f, d, e = transform
    x = c + (a * px) + (b * py)
    y = f + (d * px) + (e * py)
    return x, y


def resolve_image_path(tile: TileSummary, tiles_root: Path) -> Optional[Path]:
    p = Path(tile.tile_path_abs)
    if p.exists():
        return p
    rel = tile.tile_rel.replace("\\", "/")
    fallback = tiles_root / rel
    if fallback.exists():
        return fallback
    return None


def resolve_tile_center(
    tile: TileSummary,
    *,
    state: SelectionState,
    tiles_root: Path,
) -> Optional[Tuple[float, float]]:
    if tile.tile_id in state.center_cache:
        return state.center_cache[tile.tile_id]

    img_path = resolve_image_path(tile, tiles_root)
    if img_path is None:
        state.center_cache[tile.tile_id] = None
        return None

    worldfile = find_worldfile(img_path)
    if worldfile is None:
        state.center_cache[tile.tile_id] = None
        return None

    world_key = str(worldfile)
    transform = state.worldfile_cache.get(world_key)
    if world_key not in state.worldfile_cache:
        transform = read_worldfile(worldfile)
        state.worldfile_cache[world_key] = transform
    if transform is None:
        state.center_cache[tile.tile_id] = None
        return None

    img_key = str(img_path)
    size = state.image_size_cache.get(img_key)
    if img_key not in state.image_size_cache:
        try:
            with Image.open(img_path) as im:
                size = (int(im.width), int(im.height))
        except Exception:
            size = None
        state.image_size_cache[img_key] = size
    if size is None:
        state.center_cache[tile.tile_id] = None
        return None
    width, height = size
    cx_pix = (float(width) - 1.0) * 0.5
    cy_pix = (float(height) - 1.0) * 0.5
    center = pixel_to_world(transform, cx_pix, cy_pix)
    state.center_cache[tile.tile_id] = center
    return center


def candidate_rejection_reason(
    candidate: BucketCandidate,
    *,
    tiles: Mapping[str, TileSummary],
    state: SelectionState,
    args: argparse.Namespace,
) -> Optional[str]:
    tile = tiles.get(candidate.tile_id)
    if tile is None:
        return "missing_tile_meta"
    if tile.tile_id in state.selected_ids:
        return "duplicate_tile"

    if int(args.max_tiles_per_cell) > 0 and state.selected_cell_counts[tile.cell] >= int(args.max_tiles_per_cell):
        return "per_cell_cap"

    if int(args.local_neighbor_radius) > 0 and tile.row is not None and tile.col is not None:
        for rr, cc in state.selected_cell_rcs.get(tile.cell, []):
            if abs(tile.row - rr) <= int(args.local_neighbor_radius) and abs(tile.col - cc) <= int(args.local_neighbor_radius):
                return "overlap"

    if int(args.rc_group_size) > 1 and tile.row is not None and tile.col is not None:
        group = (tile.cell, tile.row // int(args.rc_group_size), tile.col // int(args.rc_group_size))
        if group in state.selected_rc_groups:
            return "overlap"

    if float(args.min_spacing_m) > 0.0:
        center = resolve_tile_center(tile, state=state, tiles_root=args.tiles_root)
        if center is not None:
            for ox, oy in state.selected_centers:
                if math.hypot(center[0] - ox, center[1] - oy) < float(args.min_spacing_m):
                    return "overlap"
    return None


def accept_candidate(
    candidate: BucketCandidate,
    *,
    order: int,
    tiles: Mapping[str, TileSummary],
    state: SelectionState,
    stats: SelectionStats,
    args: argparse.Namespace,
) -> None:
    tile = tiles[candidate.tile_id]
    state.selected_ids.add(tile.tile_id)
    state.selected_list.append(
        SelectedTile(
            tile_id=tile.tile_id,
            bucket=candidate.bucket,
            score=float(candidate.score),
            reason=candidate.reason,
            order=order,
        )
    )
    state.selected_cell_counts[tile.cell] += 1
    stats.selected_by_bucket[candidate.bucket] += 1
    stats.selected_by_cell[tile.cell] += 1

    if tile.row is not None and tile.col is not None:
        state.selected_cell_rcs[tile.cell].append((tile.row, tile.col))
        if int(args.rc_group_size) > 1:
            state.selected_rc_groups.add((tile.cell, tile.row // int(args.rc_group_size), tile.col // int(args.rc_group_size)))

    if float(args.min_spacing_m) > 0.0:
        center = resolve_tile_center(tile, state=state, tiles_root=args.tiles_root)
        if center is not None:
            state.selected_centers.append(center)


def select_bucket_round_robin(
    *,
    bucket: str,
    candidates: Sequence[BucketCandidate],
    target_count: int,
    tiles: Mapping[str, TileSummary],
    state: SelectionState,
    stats: SelectionStats,
    args: argparse.Namespace,
) -> None:
    if target_count <= 0:
        return
    deduped = dedupe_bucket_candidates(candidates, seed=int(args.seed), bucket=bucket)
    if not deduped:
        return

    by_cell: Dict[str, List[BucketCandidate]] = defaultdict(list)
    for cand in deduped:
        tile = tiles.get(cand.tile_id)
        if tile is None:
            continue
        by_cell[tile.cell].append(cand)
    if not by_cell:
        return

    for cell in by_cell:
        by_cell[cell].sort(key=lambda c: (-c.score, stable_hash_int(int(args.seed), bucket, c.tile_id), c.tile_id))

    cell_order = list(by_cell.keys())
    cell_order.sort(
        key=lambda cell: (
            -(by_cell[cell][0].score if by_cell[cell] else -1.0),
            stable_hash_int(int(args.seed), bucket, cell),
            cell,
        )
    )
    cursor: Dict[str, int] = {cell: 0 for cell in cell_order}

    bucket_selected = 0
    active_cells = list(cell_order)
    while bucket_selected < target_count and active_cells:
        progress = False
        next_active: List[str] = []
        for cell in active_cells:
            rows = by_cell[cell]
            idx = cursor[cell]
            while idx < len(rows):
                cand = rows[idx]
                idx += 1
                reason = candidate_rejection_reason(cand, tiles=tiles, state=state, args=args)
                if reason is None:
                    order = len(state.selected_list) + 1
                    accept_candidate(cand, order=order, tiles=tiles, state=state, stats=stats, args=args)
                    bucket_selected += 1
                    progress = True
                    break
                if reason == "overlap":
                    stats.rejected_overlap += 1
                elif reason == "per_cell_cap":
                    stats.rejected_per_cell_cap += 1
                elif reason == "duplicate_tile":
                    stats.rejected_duplicate_tile += 1
                else:
                    stats.rejected_missing_tile_meta += 1
            cursor[cell] = idx
            if idx < len(rows):
                next_active.append(cell)
            if bucket_selected >= target_count:
                break
        active_cells = next_active
        if not progress:
            break


def select_all_buckets(
    *,
    bucket_candidates: Mapping[str, Sequence[BucketCandidate]],
    bucket_targets: Mapping[str, int],
    tiles: Mapping[str, TileSummary],
    args: argparse.Namespace,
    stats: SelectionStats,
) -> List[SelectedTile]:
    state = SelectionState()
    target_total = max(0, int(args.target_batch_size))
    for bucket in BUCKET_ORDER:
        select_bucket_round_robin(
            bucket=bucket,
            candidates=bucket_candidates.get(bucket, []),
            target_count=int(bucket_targets.get(bucket, 0)),
            tiles=tiles,
            state=state,
            stats=stats,
            args=args,
        )

    remaining = max(0, target_total - len(state.selected_list))
    if remaining > 0:
        select_bucket_round_robin(
            bucket=BUCKET_FALLBACK,
            candidates=bucket_candidates.get(BUCKET_FALLBACK, []),
            target_count=remaining,
            tiles=tiles,
            state=state,
            stats=stats,
            args=args,
        )

    stats.selected_total = len(state.selected_list)
    return state.selected_list


def copy_or_symlink(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel_target = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel_target)
        return
    raise ValueError(f"Unsupported image mode: {mode}")


def build_manifest_rows(
    *,
    selected: Sequence[SelectedTile],
    tiles: Mapping[str, TileSummary],
    feature_stats: Mapping[str, TileFeatureStats],
    out_images_dir: Path,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sel in selected:
        tile = tiles[sel.tile_id]
        feature = feature_stats.get(sel.tile_id)
        src_path = Path(tile.tile_path_abs)
        ext = src_path.suffix.lower() if src_path.suffix else Path(tile.tile_rel).suffix.lower()
        if not ext:
            ext = ".png"
        unique_name = f"{tile.tile_id}{ext}"
        out_img = out_images_dir / unique_name
        feature_mean_conf = feature.mean_conf if feature is not None and feature.num_features > 0 else 0.0
        feature_max_area = feature.area_max if feature is not None else 0.0
        out.append(
            {
                "selected_order": sel.order,
                "bucket": sel.bucket,
                "tile_id": tile.tile_id,
                "unique_name": unique_name,
                "tile_rel": tile.tile_rel,
                "cell": tile.cell,
                "tile_stem": tile.tile_stem,
                "row": tile.row if tile.row is not None else "",
                "col": tile.col if tile.col is not None else "",
                "tile_path_abs": tile.tile_path_abs,
                "out_image_abs": str(out_img),
                "num_preds": tile.num_preds,
                "min_conf": "" if tile.min_conf is None else f"{tile.min_conf:.6f}",
                "mean_conf": "" if tile.mean_conf is None else f"{tile.mean_conf:.6f}",
                "max_conf": "" if tile.max_conf is None else f"{tile.max_conf:.6f}",
                "max_area_m2": "" if tile.max_area_m2 is None else f"{tile.max_area_m2:.6f}",
                "sum_area_m2": f"{tile.sum_area_m2:.6f}",
                "feature_count": 0 if feature is None else feature.num_features,
                "feature_mean_conf": f"{feature_mean_conf:.6f}",
                "feature_max_area_m2": f"{feature_max_area:.6f}",
                "selection_score": f"{sel.score:.6f}",
                "selection_reason": sel.reason,
            }
        )
    return out


def write_tile_list(path: Path, selected: Sequence[SelectedTile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [s.tile_id for s in selected]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_manifest_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "selected_order",
        "bucket",
        "tile_id",
        "unique_name",
        "tile_rel",
        "cell",
        "tile_stem",
        "row",
        "col",
        "tile_path_abs",
        "out_image_abs",
        "num_preds",
        "min_conf",
        "mean_conf",
        "max_conf",
        "max_area_m2",
        "sum_area_m2",
        "feature_count",
        "feature_mean_conf",
        "feature_max_area_m2",
        "selection_score",
        "selection_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_selected_images(
    *,
    manifest_rows: Sequence[Dict[str, Any]],
    image_mode: str,
    dry_run: bool,
) -> None:
    if image_mode == "none":
        return
    for row in manifest_rows:
        src = Path(str(row["tile_path_abs"]))
        dst = Path(str(row["out_image_abs"]))
        if not src.exists():
            raise SystemExit(f"Selected source image is missing: {src}")
        if dry_run:
            continue
        copy_or_symlink(src, dst, mode=image_mode)


def write_selected_geojson(
    *,
    src_payload: Mapping[str, Any],
    feature_tile_ids: Sequence[str],
    selected_ids: set[str],
    out_path: Path,
) -> int:
    features = src_payload.get("features", [])
    selected_features: List[Dict[str, Any]] = []
    for feat, tile_id in zip(features, feature_tile_ids):
        if not tile_id or tile_id not in selected_ids:
            continue
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties")
        if not isinstance(props, dict):
            props = {}
        props = dict(props)
        props["tile_id"] = tile_id
        canonical_rel = canonical_rel_from_tile_id(tile_id, ext=Path(str(props.get("tile_rel") or "")).suffix or ".png")
        if canonical_rel:
            props["tile_rel"] = canonical_rel
        out_feat = {
            "type": "Feature",
            "geometry": feat.get("geometry"),
            "properties": props,
        }
        selected_features.append(out_feat)

    out_fc: Dict[str, Any] = {
        "type": "FeatureCollection",
        "features": selected_features,
    }
    if "crs" in src_payload:
        out_fc["crs"] = src_payload["crs"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_fc, ensure_ascii=False), encoding="utf-8")
    return len(selected_features)


def run_geojson_to_coco_converter(args: argparse.Namespace) -> None:
    if args.skip_coco:
        return
    if args.image_mode == "none":
        raise SystemExit("--image-mode none is incompatible with COCO export. Use symlink or copy.")
    if not args.converter_script.exists():
        raise SystemExit(f"--converter-script not found: {args.converter_script}")
    cmd = [
        sys.executable,
        str(args.converter_script),
        "--geojson",
        str(args.out_geojson),
        "--manifest-csv",
        str(args.out_manifest_csv),
        "--images-dir",
        str(args.out_images_dir),
        "--out",
        str(args.out_coco_json),
        "--worldfile-crs",
        str(args.worldfile_crs),
        "--min-confidence",
        str(float(args.coco_min_confidence)),
        "--category-id",
        str(int(args.coco_category_id)),
        "--category-name",
        str(args.coco_category_name),
    ]
    geojson_crs = str(args.geojson_crs or "").strip()
    if geojson_crs:
        cmd.extend(["--geojson-crs", geojson_crs])
    subprocess.run(cmd, check=True)


def normalize_bucket_key(key: str) -> str:
    return str(key or "").strip().lower().replace("-", "_")


def canonical_bucket_key(key: str) -> Optional[str]:
    k = normalize_bucket_key(key)
    aliases = {
        BUCKET_TREE: BUCKET_TREE,
        "tree_shadows": BUCKET_TREE,
        "hard_negatives_tree": BUCKET_TREE,
        BUCKET_EDGE: BUCKET_EDGE,
        "pool_edge_shadows": BUCKET_EDGE,
        "edge_shadows": BUCKET_EDGE,
        BUCKET_BLUE: BUCKET_BLUE,
        "blue_roofs": BUCKET_BLUE,
        BUCKET_SHORE: BUCKET_SHORE,
        "shoreline": BUCKET_SHORE,
        "shoreline_water_edge": BUCKET_SHORE,
        BUCKET_HARD_POS: BUCKET_HARD_POS,
        "hard_positives_misses": BUCKET_HARD_POS,
        "hard_positive": BUCKET_HARD_POS,
        BUCKET_MIXED: BUCKET_MIXED,
        "mixed": BUCKET_MIXED,
        BUCKET_FALLBACK: BUCKET_FALLBACK,
        "fallback": BUCKET_FALLBACK,
    }
    return aliases.get(k)


def parse_bucket_targets_json_arg(raw: str) -> Dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}
    maybe_path = Path(text)
    if maybe_path.exists() and maybe_path.is_file():
        text = maybe_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise SystemExit(f"Invalid --bucket-targets-json content: {exc}")
    if not isinstance(payload, dict):
        raise SystemExit("--bucket-targets-json must parse to an object/dict.")
    out: Dict[str, float] = {}
    for key, value in payload.items():
        v = parse_float(value)
        if v is None:
            raise SystemExit(f"--bucket-targets-json value for key '{key}' is not numeric: {value!r}")
        out[str(key)] = float(v)
    return out


def expand_bucket_values(raw_values: Mapping[str, float]) -> Dict[str, float]:
    expanded: Dict[str, float] = {b: 0.0 for b in BUCKET_ORDER}
    for key, value in raw_values.items():
        v = max(0.0, float(value))
        norm = normalize_bucket_key(key)
        direct = canonical_bucket_key(norm)
        if direct is not None:
            expanded[direct] += v
            continue
        if norm in {"hard_negatives", "negative", "negatives"}:
            share = v / float(len(NEGATIVE_BUCKETS))
            for b in NEGATIVE_BUCKETS:
                expanded[b] += share
            continue
        if norm in {"hard_positives"}:
            expanded[BUCKET_HARD_POS] += v
            continue
        if norm in {"mixed_review_group", "mixed_review_bucket"}:
            expanded[BUCKET_MIXED] += v
            continue
        if norm in {"fallback_fill_group"}:
            expanded[BUCKET_FALLBACK] += v
            continue
        raise SystemExit(f"Unrecognized bucket target key: {key!r}")
    return expanded


def allocate_counts_from_weights(weights: Mapping[str, float], target: int) -> Dict[str, int]:
    target = max(0, int(target))
    if target <= 0:
        return {b: 0 for b in BUCKET_ORDER}
    cleaned = {b: max(0.0, float(weights.get(b, 0.0))) for b in BUCKET_ORDER}
    total_w = sum(cleaned.values())
    if total_w <= 0:
        cleaned = dict(DEFAULT_BUCKET_WEIGHTS)
        total_w = sum(cleaned.values())

    raw_counts = {b: (cleaned[b] / total_w) * float(target) for b in BUCKET_ORDER}
    counts = {b: int(math.floor(raw_counts[b])) for b in BUCKET_ORDER}
    assigned = sum(counts.values())
    remainder = target - assigned
    if remainder > 0:
        order = sorted(BUCKET_ORDER, key=lambda b: (-(raw_counts[b] - counts[b]), b))
        for b in order:
            if remainder <= 0:
                break
            counts[b] += 1
            remainder -= 1
    return counts


def normalize_counts_to_target(counts: Mapping[str, int], target: int) -> Dict[str, int]:
    target = max(0, int(target))
    out = {b: max(0, int(counts.get(b, 0))) for b in BUCKET_ORDER}
    total = sum(out.values())
    if total == target:
        return out
    if total < target:
        out[BUCKET_FALLBACK] += target - total
        return out
    # Too many requested count slots: scale proportionally.
    return allocate_counts_from_weights({b: float(out[b]) for b in BUCKET_ORDER}, target)


def build_bucket_targets(args: argparse.Namespace) -> Dict[str, int]:
    target = max(0, int(args.target_batch_size))

    # Highest precedence: explicit JSON override.
    json_raw = parse_bucket_targets_json_arg(str(args.bucket_targets_json or ""))
    if json_raw:
        expanded = expand_bucket_values(json_raw)
        values = [v for v in expanded.values() if v > 0.0]
        if values and all(v <= 1.0 for v in values):
            return allocate_counts_from_weights(expanded, target)
        counts = {b: int(round(expanded.get(b, 0.0))) for b in BUCKET_ORDER}
        return normalize_counts_to_target(counts, target)

    # Next: legacy explicit count overrides.
    legacy_map = {
        BUCKET_TREE: args.tree_shadows_count,
        BUCKET_EDGE: args.pool_edge_shadows_count,
        BUCKET_BLUE: args.blue_roofs_count,
        BUCKET_SHORE: args.shoreline_count,
        BUCKET_HARD_POS: args.hard_positives_count,
        BUCKET_MIXED: args.mixed_review_count,
        BUCKET_FALLBACK: args.fallback_fill_count,
    }
    if any(v is not None for v in legacy_map.values()):
        counts = {b: int(v) if v is not None else 0 for b, v in legacy_map.items()}
        return normalize_counts_to_target(counts, target)

    return allocate_counts_from_weights(DEFAULT_BUCKET_WEIGHTS, target)


def summarize_to_stdout(
    *,
    stats: SelectionStats,
    bucket_targets: Mapping[str, int],
    selected: Sequence[SelectedTile],
    manifest_rows: Sequence[Dict[str, Any]],
    dry_run: bool,
) -> None:
    print("dry_run:", bool(dry_run))
    print("tiles_seen_total:", int(stats.tiles_seen_total))
    print("tiles_with_predictions:", int(stats.tiles_with_predictions))
    print("excluded_existing_dataset:", int(stats.excluded_existing_dataset))
    print("excluded_tile_ids_file:", int(stats.excluded_tile_ids_file))
    print("excluded_zero_predictions:", int(stats.excluded_zero_predictions))
    print("excluded_too_many_predictions:", int(stats.excluded_too_many_predictions))
    print("excluded_too_confident:", int(stats.excluded_too_confident))
    print("excluded_area_cluster:", int(stats.excluded_area_cluster))
    print("remaining_candidate_tiles:", int(stats.remaining_candidate_tiles))
    print("selected_total:", int(stats.selected_total))
    print("rejected_for_overlap:", int(stats.rejected_overlap))
    print("rejected_for_per_cell_cap:", int(stats.rejected_per_cell_cap))

    print("tile_rows_scanned:", int(stats.tile_rows_scanned))
    print("geojson_features_scanned:", int(stats.geojson_features_scanned))
    print("exclude_existing_dataset_files_scanned:", int(stats.existing_dataset_files_scanned))
    print("exclude_existing_dataset_tile_ids_loaded:", int(stats.existing_dataset_tile_ids_loaded))
    print("exclude_tile_ids_file_loaded:", int(stats.exclude_file_tile_ids_loaded))
    print("selected_prediction_features:", int(stats.selected_prediction_features))
    print("candidate_tiles_considered:", int(stats.candidate_tiles_considered))
    print("rejected_duplicate_tile_id:", int(stats.rejected_duplicate_tile))
    print("rejected_missing_tile_metadata:", int(stats.rejected_missing_tile_meta))
    print("excluded_too_few_predictions:", int(stats.excluded_too_few_predictions))
    print("excluded_low_confidence:", int(stats.excluded_low_confidence))
    print("excluded_small_total_area:", int(stats.excluded_small_total_area))

    print("selected by bucket:")
    for bucket in BUCKET_ORDER:
        print(
            f"  {bucket}: {int(stats.selected_by_bucket.get(bucket, 0))} "
            f"(target={int(bucket_targets.get(bucket, 0))}, candidates={int(stats.bucket_candidate_counts.get(bucket, 0))})"
        )

    print("selected by cell:")
    for cell, count in sorted(stats.selected_by_cell.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cell}: {int(count)}")

    if selected:
        print("sample selected tile_ids:", ", ".join(s.tile_id for s in selected[:10]))
    else:
        print("sample selected tile_ids: (none)")
    print("manifest rows:", len(manifest_rows))


def write_summary_json(
    *,
    path: Path,
    args: argparse.Namespace,
    stats: SelectionStats,
    bucket_targets: Mapping[str, int],
    manifest_rows: Sequence[Dict[str, Any]],
) -> None:
    summary = {
        "run_dir": str(args.run_dir),
        "in_geojson": str(args.in_geojson),
        "tiles_csv": str(args.tiles_csv),
        "tiles_jsonl": str(args.tiles_jsonl),
        "tiles_root": str(args.tiles_root),
        "exclude_existing_dataset": str(args.exclude_existing_dataset) if args.exclude_existing_dataset is not None else "",
        "exclude_tile_ids": str(args.exclude_tile_ids) if args.exclude_tile_ids is not None else "",
        "seed": int(args.seed),
        "target_batch_size": int(args.target_batch_size),
        "max_tiles_per_cell": int(args.max_tiles_per_cell),
        "local_neighbor_radius": int(args.local_neighbor_radius),
        "rc_group_size": int(args.rc_group_size),
        "min_spacing_m": float(args.min_spacing_m),
        "neighbor_radius": int(args.neighbor_radius),
        "min_preds_per_tile": int(args.min_preds_per_tile),
        "max_preds_per_tile": int(args.max_preds_per_tile),
        "min_max_conf": float(args.min_max_conf),
        "max_max_conf": float(args.max_max_conf),
        "min_mean_conf": float(args.min_mean_conf),
        "max_mean_conf": float(args.max_mean_conf),
        "min_total_area_m2": float(args.min_total_area_m2),
        "max_total_area_m2": float(args.max_total_area_m2),
        "prefer_low_confidence": float(args.prefer_low_confidence),
        "prefer_medium_density": float(args.prefer_medium_density),
        "prefer_hard_negatives": float(args.prefer_hard_negatives),
        "prefer_hard_positives": float(args.prefer_hard_positives),
        "bucket_targets": {k: int(v) for k, v in bucket_targets.items()},
        "bucket_candidate_counts": {k: int(v) for k, v in stats.bucket_candidate_counts.items()},
        "tile_rows_scanned": int(stats.tile_rows_scanned),
        "geojson_features_scanned": int(stats.geojson_features_scanned),
        "tiles_seen_total": int(stats.tiles_seen_total),
        "tiles_with_predictions": int(stats.tiles_with_predictions),
        "excluded_existing_dataset": int(stats.excluded_existing_dataset),
        "excluded_tile_ids_file": int(stats.excluded_tile_ids_file),
        "excluded_zero_predictions": int(stats.excluded_zero_predictions),
        "excluded_too_few_predictions": int(stats.excluded_too_few_predictions),
        "excluded_too_many_predictions": int(stats.excluded_too_many_predictions),
        "excluded_too_confident": int(stats.excluded_too_confident),
        "excluded_low_confidence": int(stats.excluded_low_confidence),
        "excluded_area_cluster": int(stats.excluded_area_cluster),
        "excluded_small_total_area": int(stats.excluded_small_total_area),
        "remaining_candidate_tiles": int(stats.remaining_candidate_tiles),
        "candidate_tiles_considered": int(stats.candidate_tiles_considered),
        "selected_total": int(stats.selected_total),
        "selected_prediction_features": int(stats.selected_prediction_features),
        "selected_by_bucket": {k: int(v) for k, v in stats.selected_by_bucket.items()},
        "selected_by_cell": {k: int(v) for k, v in stats.selected_by_cell.items()},
        "rejected_for_overlap": int(stats.rejected_overlap),
        "rejected_for_per_cell_cap": int(stats.rejected_per_cell_cap),
        "rejected_duplicate_tile": int(stats.rejected_duplicate_tile),
        "rejected_missing_tile_metadata": int(stats.rejected_missing_tile_meta),
        "existing_dataset_files_scanned": int(stats.existing_dataset_files_scanned),
        "existing_dataset_tile_ids_loaded": int(stats.existing_dataset_tile_ids_loaded),
        "exclude_file_tile_ids_loaded": int(stats.exclude_file_tile_ids_loaded),
        "manifest_rows": int(len(manifest_rows)),
        "out_tile_list": str(args.out_tile_list),
        "out_manifest_csv": str(args.out_manifest_csv),
        "out_geojson": str(args.out_geojson),
        "out_coco_json": str(args.out_coco_json),
        "out_images_dir": str(args.out_images_dir),
        "dry_run": bool(args.dry_run),
        "skip_coco": bool(args.skip_coco),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = resolve_paths(parse_args())

    tile_summaries_all, tile_source, tile_rows_scanned = load_tile_summaries(
        tiles_csv=args.tiles_csv,
        tiles_jsonl=args.tiles_jsonl,
        tiles_root=args.tiles_root,
    )
    if not tile_summaries_all:
        raise SystemExit("No valid tile summaries parsed from tile manifest.")

    existing_dataset_ids: set[str] = set()
    existing_files_scanned = 0
    if args.exclude_existing_dataset is not None:
        existing_dataset_ids, existing_files_scanned = collect_existing_dataset_tile_ids(args.exclude_existing_dataset)
    exclude_file_ids = load_excluded_tile_ids_file(args.exclude_tile_ids)

    after_initial, initial_counts = apply_initial_exclusions(
        tile_summaries_all,
        existing_dataset_ids=existing_dataset_ids,
        excluded_tile_ids=exclude_file_ids,
    )
    if not after_initial:
        raise SystemExit(
            "All candidates were excluded before threshold filtering. "
            "Check --exclude-existing-dataset / --exclude-tile-ids inputs."
        )

    src_geojson_payload, feature_tile_ids, feature_stats, geojson_feature_count = load_geojson_and_tile_stats(args.in_geojson)

    stats = SelectionStats()
    stats.tile_rows_scanned = int(tile_rows_scanned)
    stats.geojson_features_scanned = int(geojson_feature_count)
    stats.tiles_seen_total = int(initial_counts["tiles_seen_total"])
    stats.excluded_existing_dataset = int(initial_counts["excluded_existing_dataset"])
    stats.excluded_tile_ids_file = int(initial_counts["excluded_tile_ids_file"])
    stats.existing_dataset_files_scanned = int(existing_files_scanned)
    stats.existing_dataset_tile_ids_loaded = int(len(existing_dataset_ids))
    stats.exclude_file_tile_ids_loaded = int(len(exclude_file_ids))

    informative_tiles = filter_informative_candidates(
        tiles=after_initial,
        feature_stats=feature_stats,
        args=args,
        stats=stats,
    )
    if not informative_tiles:
        raise SystemExit(
            "No informative candidates remained after tightened filtering. "
            "Relax thresholds (max conf/mean conf/area/preds) and retry."
        )

    neighbor_ctx = build_neighbor_context(informative_tiles, radius=max(0, int(args.neighbor_radius)))
    bucket_candidates = build_bucket_candidates(
        tiles=informative_tiles,
        feature_stats=feature_stats,
        neighbor_ctx=neighbor_ctx,
        args=args,
    )
    bucket_targets = build_bucket_targets(args)

    stats.bucket_targets = dict(bucket_targets)
    stats.bucket_candidate_counts = {
        b: len(dedupe_bucket_candidates(bucket_candidates.get(b, []), seed=int(args.seed), bucket=b))
        for b in BUCKET_ORDER
    }
    stats.candidate_tiles_considered = len(
        {
            cand.tile_id
            for bucket in BUCKET_ORDER
            for cand in dedupe_bucket_candidates(bucket_candidates.get(bucket, []), seed=int(args.seed), bucket=bucket)
        }
    )

    selected = select_all_buckets(
        bucket_candidates=bucket_candidates,
        bucket_targets=bucket_targets,
        tiles=informative_tiles,
        args=args,
        stats=stats,
    )
    selected_ids = {s.tile_id for s in selected}

    manifest_rows = build_manifest_rows(
        selected=selected,
        tiles=informative_tiles,
        feature_stats=feature_stats,
        out_images_dir=args.out_images_dir,
    )

    selected_prediction_features = 0
    if not args.dry_run:
        write_tile_list(args.out_tile_list, selected)
        write_manifest_csv(args.out_manifest_csv, manifest_rows)
        materialize_selected_images(manifest_rows=manifest_rows, image_mode=args.image_mode, dry_run=args.dry_run)
        selected_prediction_features = write_selected_geojson(
            src_payload=src_geojson_payload,
            feature_tile_ids=feature_tile_ids,
            selected_ids=selected_ids,
            out_path=args.out_geojson,
        )
    else:
        for tile_id in feature_tile_ids:
            if tile_id and tile_id in selected_ids:
                selected_prediction_features += 1

    stats.selected_prediction_features = int(selected_prediction_features)

    summary_path = args.out_manifest_csv.with_name(f"{args.out_manifest_csv.stem}_summary.json")
    if not args.dry_run:
        write_summary_json(
            path=summary_path,
            args=args,
            stats=stats,
            bucket_targets=bucket_targets,
            manifest_rows=manifest_rows,
        )
        if not args.skip_coco:
            run_geojson_to_coco_converter(args)

    print("tile summary source:", tile_source)
    summarize_to_stdout(
        stats=stats,
        bucket_targets=bucket_targets,
        selected=selected,
        manifest_rows=manifest_rows,
        dry_run=args.dry_run,
    )
    print("wrote tile list:", args.out_tile_list)
    print("wrote manifest:", args.out_manifest_csv)
    print("wrote selected geojson:", args.out_geojson)
    if args.skip_coco:
        print("wrote coco: skipped (--skip-coco)")
    else:
        print("wrote coco:", args.out_coco_json if not args.dry_run else f"{args.out_coco_json} (dry-run planned)")
    print("wrote summary:", summary_path if not args.dry_run else f"{summary_path} (dry-run planned)")


if __name__ == "__main__":
    main()
