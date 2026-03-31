#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import shutil
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd


RC_RE = re.compile(r"^r(?P<row>\d+)_c(?P<col>\d+)$")
PROFILE_ROUND_NEXT = "round_next"
PROFILE_V5_CORRECTIVE = "v5_corrective"
PROFILE_V6_CORRECTIVE = "v6_corrective"
TILE_ID_PATTERN = r"cell_\d+_\d+__r\d+_c\d+"
TILE_ID_RE = re.compile(rf"({TILE_ID_PATTERN})")
CELL_RE = re.compile(r"^cell_\d+_\d+$")
CELL_RC_SUFFIX_RE = re.compile(r"(cell_\d+_\d+)__(r\d+_c\d+)$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

ROUND_NEXT_BUCKETS = (
    "hard_negatives_shorelines",
    "hard_negatives_courts",
    "hard_negatives_blue_roofs",
    "duplicate_fragmented_real_pools",
    "missed_obvious_pools",
    "review_mixed_random",
)
V5_BUCKETS = (
    "hard_negatives_tree_shadows",
    "hard_negatives_pool_edge_shadows",
    "clean_positive_anchors",
    "review_mixed_random_small",
)
V6_BUCKETS = (
    "hard_negatives_tree_shadows",
    "missed_obvious_pools",
    "duplicate_fragmented_real_pools",
    "hard_negatives_blue_roofs",
    "review_mixed_small",
)


def bucket_order(profile: str) -> Tuple[str, ...]:
    if profile == PROFILE_V6_CORRECTIVE:
        return V6_BUCKETS
    if profile == PROFILE_V5_CORRECTIVE:
        return V5_BUCKETS
    return ROUND_NEXT_BUCKETS


@dataclass
class WorldTransform:
    a: float
    e: float
    c: float
    f: float


@dataclass
class CacheBundle:
    image: "OrderedDict[str, np.ndarray]"
    world: Dict[str, WorldTransform]
    missed_pool_signal: Dict[str, Dict[str, float]]
    max_image_cache: int = 320


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a targeted annotation batch for Google z21 pool inference outputs with "
            "failure-mode buckets (round_next, v5_corrective, or v6_corrective profiles)."
        )
    )
    ap.add_argument(
        "--profile",
        choices=(PROFILE_ROUND_NEXT, PROFILE_V5_CORRECTIVE, PROFILE_V6_CORRECTIVE),
        default=PROFILE_ROUND_NEXT,
        help="Batch logic profile to run.",
    )
    ap.add_argument("--run-dir", type=Path, required=True, help="Inference run dir containing pools.geojson + tiles.csv.")
    ap.add_argument(
        "--tiles-root",
        type=Path,
        required=True,
        help="Root directory of source tiles (e.g. data/raw/google/sp_city_2020_rebuild_google_z21).",
    )
    ap.add_argument(
        "--out-base",
        type=Path,
        default=Path("runs/annotation_batches"),
        help="Base output directory where the timestamped batch folder is created.",
    )
    ap.add_argument(
        "--batch-prefix",
        default="google_z21_round_next",
        help="Prefix for output batch folder name.",
    )
    ap.add_argument(
        "--timestamp",
        default="",
        help="Optional fixed timestamp token for deterministic reruns. Default: local datetime YYYYmmdd_HHMMSS.",
    )
    ap.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    ap.add_argument("--seed", type=int, default=42)

    # round_next profile counts (kept for compatibility)
    ap.add_argument("--shorelines-count", type=int, default=80)
    ap.add_argument("--courts-count", type=int, default=70)
    ap.add_argument("--blue-roofs-count", type=int, default=60)
    ap.add_argument("--duplicate-count", type=int, default=80)
    ap.add_argument("--missed-count", type=int, default=50)
    ap.add_argument("--mixed-random-count", type=int, default=30)

    # v5 corrective profile counts
    ap.add_argument("--tree-shadows-count", type=int, default=72)
    ap.add_argument("--pool-edge-shadows-count", type=int, default=40)
    ap.add_argument("--clean-positive-anchors-count", type=int, default=24)
    ap.add_argument("--mixed-random-small-count", type=int, default=15)
    ap.add_argument(
        "--exclude-dataset-roots",
        nargs="*",
        default=["data/datasets/geosampa_z21_v1"],
        help=(
            "Optional dataset roots to exclude from candidate selection. "
            "Each root contributes images/train and images/val (if present)."
        ),
    )
    ap.add_argument(
        "--exclude-image-roots",
        nargs="*",
        default=[],
        help="Optional explicit image roots to exclude from candidate selection.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow removing an existing output directory if it already exists.",
    )
    return ap.parse_args()


def tile_rel_noext(tile_rel: object) -> str:
    return Path(str(tile_rel)).with_suffix("").as_posix().lstrip("./")


def infer_exclusion_image_roots(dataset_roots: Sequence[Path], explicit_image_roots: Sequence[Path]) -> List[Path]:
    roots: List[Path] = []
    seen: set[str] = set()

    def add_root(p: Path) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        roots.append(p)

    for raw in dataset_roots:
        d = raw.expanduser()
        imgs = d / "images"
        if (imgs / "train").exists() or (imgs / "val").exists():
            if (imgs / "train").exists():
                add_root((imgs / "train").resolve())
            if (imgs / "val").exists():
                add_root((imgs / "val").resolve())
            continue
        if imgs.exists():
            add_root(imgs.resolve())
            continue
        if d.exists():
            add_root(d.resolve())

    for raw in explicit_image_roots:
        p = raw.expanduser()
        if p.exists():
            add_root(p.resolve())

    return roots


def collect_excluded_tile_keys(image_roots: Sequence[Path]) -> Tuple[set[str], set[str], Dict[str, int]]:
    excluded_tile_rel_noext: set[str] = set()
    excluded_tile_stems: set[str] = set()
    ambiguous_rc_only_stems: set[str] = set()
    images_scanned = 0
    roots_scanned = 0

    for root in image_roots:
        if not root.exists():
            continue
        roots_scanned += 1
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            images_scanned += 1
            stem = p.stem.strip()
            if not stem:
                continue
            if RC_RE.match(stem):
                parent = p.parent.name.strip()
                if CELL_RE.match(parent):
                    excluded_tile_rel_noext.add(f"{parent}/{stem}")
                else:
                    ambiguous_rc_only_stems.add(stem)
            m = CELL_RC_SUFFIX_RE.search(stem)
            if m:
                cell = m.group(1)
                rc = m.group(2)
                excluded_tile_rel_noext.add(f"{cell}/{rc}")
            tid = extract_tile_id_from_text(p.as_posix())
            if tid:
                cell, rc = tid.split("__", 1)
                excluded_tile_rel_noext.add(f"{cell}/{rc}")

    stats = {
        "roots_scanned": roots_scanned,
        "images_scanned": images_scanned,
        "excluded_tile_rel_noext": len(excluded_tile_rel_noext),
        "excluded_tile_stems": len(excluded_tile_stems),
        "ambiguous_rc_only_stems": len(ambiguous_rc_only_stems),
    }
    return excluded_tile_rel_noext, excluded_tile_stems, stats


def apply_exclusions(df: pd.DataFrame, *, excluded_tile_rel_noext: set[str], excluded_tile_stems: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    rel_key = df["tile_rel"].map(tile_rel_noext)
    if "tile_stem" in df.columns:
        stem_key = df["tile_stem"].astype(str)
    else:
        stem_key = df["tile_rel"].map(lambda x: Path(str(x)).stem)
    keep = ~(rel_key.isin(excluded_tile_rel_noext) | stem_key.isin(excluded_tile_stems))
    return df.loc[keep].copy().reset_index(drop=True)


def resolve_tile_csv_path(run_dir: Path) -> Path:
    candidates = (
        run_dir / "tiles.csv",
        run_dir / "pools_tiles.csv",
    )
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"Missing tile summary CSV in run dir: expected one of {[str(p) for p in candidates]}")


def canonical_tile_rel_from_id(tile_id: str, *, ext: str = ".png") -> str:
    tid = str(tile_id).strip()
    if not re.fullmatch(TILE_ID_PATTERN, tid):
        return ""
    cell, rc = tid.split("__", 1)
    return f"{cell}/{rc}{ext}"


def tile_stem_from_id(tile_id: str) -> str:
    tid = str(tile_id).strip()
    if not re.fullmatch(TILE_ID_PATTERN, tid):
        return ""
    return tid.split("__", 1)[1]


def extract_tile_id_from_text(value: object) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    noext = Path(raw).with_suffix("").as_posix()
    for cand in (Path(noext).name, noext):
        m = TILE_ID_RE.search(cand)
        if m:
            return m.group(1)
        pm = re.search(r"(cell_\d+_\d+)[/\\](r\d+_c\d+)", cand)
        if pm:
            return f"{pm.group(1)}__{pm.group(2)}"
    return None


def extract_tile_id_from_series(col: pd.Series) -> pd.Series:
    s = col.fillna("").astype(str).str.strip()
    noext = s.str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    direct = noext.str.extract(rf"({TILE_ID_PATTERN})", expand=False)
    path_parts = noext.str.extract(r"(cell_\d+_\d+)[/\\](r\d+_c\d+)", expand=True)
    path_based = (path_parts[0].fillna("") + "__" + path_parts[1].fillna("")).where(
        path_parts[0].notna() & path_parts[1].notna(),
        np.nan,
    )
    out = direct.where(direct.notna(), path_based)
    return out.where(out.str.fullmatch(TILE_ID_PATTERN, na=False), np.nan)


def attach_tile_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tile_id = pd.Series(np.nan, index=out.index, dtype=object)
    for col in ("tile_id", "tile_rel", "tile_path_abs", "tile", "tile_stem", "filename", "path"):
        if col in out.columns:
            candidate = extract_tile_id_from_series(out[col])
            tile_id = tile_id.where(tile_id.notna(), candidate)

    if ("cell" in out.columns) and ("tile_stem" in out.columns):
        stem = out["tile_stem"].fillna("").astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        combo = out["cell"].fillna("").astype(str).str.strip() + "__" + stem.str.strip()
        combo = combo.where(combo.str.fullmatch(TILE_ID_PATTERN, na=False), np.nan)
        tile_id = tile_id.where(tile_id.notna(), combo)

    out["tile_id"] = tile_id
    if "tile_rel" in out.columns:
        canonical = out["tile_id"].map(lambda x: canonical_tile_rel_from_id(str(x)))
        out["tile_rel_canonical"] = canonical.where(canonical != "", np.nan)
    return out


def detect_anomaly_column(columns: Sequence[str]) -> Optional[str]:
    prefer = (
        "anomaly_score",
        "tile_anomaly_score",
        "anomaly",
    )
    col_map = {c.lower(): c for c in columns}
    for p in prefer:
        if p in col_map:
            return col_map[p]
    for c in columns:
        cl = c.lower()
        if "anomaly" in cl and "score" in cl:
            return c
    return None


def dedupe_rows_by_tile_id(
    df: pd.DataFrame,
    *,
    conf_col: str,
    anomaly_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    work = df.copy()
    before = int(len(work))
    work = work.loc[work["tile_id"].notna()].copy().reset_index(drop=True)
    work["_row_order"] = np.arange(len(work))

    conf = pd.to_numeric(work[conf_col], errors="coerce") if conf_col in work.columns else pd.Series(np.nan, index=work.index)
    work["_conf_sort"] = conf.fillna(np.inf)

    chosen_anomaly = anomaly_col or detect_anomaly_column(work.columns)
    if chosen_anomaly and chosen_anomaly in work.columns:
        anomaly = pd.to_numeric(work[chosen_anomaly], errors="coerce")
        work["_anomaly_sort"] = anomaly.fillna(-np.inf)
    else:
        chosen_anomaly = None
        work["_anomaly_sort"] = -np.inf

    work = work.sort_values(
        by=["tile_id", "_conf_sort", "_anomaly_sort", "_row_order"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    deduped = work.drop_duplicates(subset=["tile_id"], keep="first").copy()
    duplicates_prevented = int(len(work) - len(deduped))
    dropped_missing_tile_id = int(before - len(work))

    deduped = deduped.drop(columns=["_row_order", "_conf_sort", "_anomaly_sort"], errors="ignore").reset_index(drop=True)
    stats = {
        "rows_in": before,
        "rows_with_tile_id": int(len(work)),
        "dropped_missing_tile_id": dropped_missing_tile_id,
        "duplicates_prevented": duplicates_prevented,
        "dedupe_conf_col": conf_col if conf_col in df.columns else None,
        "dedupe_anomaly_col": chosen_anomaly,
    }
    return deduped, stats


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def parse_int(v: object, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def parse_rc(tile_stem: str) -> Optional[Tuple[int, int]]:
    m = RC_RE.match((tile_stem or "").strip())
    if not m:
        return None
    return int(m.group("row")), int(m.group("col"))


def read_world_transform(pgw_path: Path) -> Optional[WorldTransform]:
    try:
        vals = [float(x.strip()) for x in pgw_path.read_text(encoding="utf-8").splitlines()[:6]]
    except Exception:
        return None
    if len(vals) < 6:
        return None
    a, d, b, e, c, f = vals
    if abs(a) < 1e-12 or abs(e) < 1e-12:
        return None
    if abs(b) > 1e-9 or abs(d) > 1e-9:
        # Project uses axis-aligned worldfiles; skip rotated transforms for safety.
        return None
    return WorldTransform(a=a, e=e, c=c, f=f)


def world_to_pixel(x: float, y: float, tr: WorldTransform) -> Tuple[float, float]:
    col = (float(x) - tr.c) / tr.a
    row = (float(y) - tr.f) / tr.e
    return col, row


def load_tile_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(path)
    expected = {
        "tile_rel",
        "tile_stem",
        "cell",
        "tile_path_abs",
        "blank_white",
        "num_preds",
        "min_conf",
        "mean_conf",
        "max_conf",
        "max_area_m2",
        "sum_area_m2",
    }
    missing = sorted(expected - set(df.columns))
    if missing:
        raise SystemExit(f"tiles.csv missing columns: {missing}")

    out = attach_tile_ids(df.copy())
    missing_tile_id = int(out["tile_id"].isna().sum()) if "tile_id" in out.columns else int(len(out))
    if missing_tile_id > 0:
        raise SystemExit(
            f"tiles.csv contains {missing_tile_id} rows without valid tile_id "
            f"(required pattern: {TILE_ID_PATTERN})."
        )

    deduped, dedupe_stats = dedupe_rows_by_tile_id(out, conf_col="min_conf")
    out = deduped
    out["blank_white"] = out["blank_white"].fillna(0).astype(int)
    out["num_preds"] = out["num_preds"].fillna(0).astype(int)
    for col in ("min_conf", "mean_conf", "max_conf", "max_area_m2", "sum_area_m2"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sum_area_m2"] = out["sum_area_m2"].fillna(0.0)
    out["tile_stem"] = out["tile_stem"].fillna("").astype(str)
    empty_stem = out["tile_stem"].str.strip() == ""
    if empty_stem.any():
        out.loc[empty_stem, "tile_stem"] = out.loc[empty_stem, "tile_id"].map(lambda x: tile_stem_from_id(str(x)))
    out["tile_rel"] = out["tile_id"].map(lambda x: canonical_tile_rel_from_id(str(x))).where(
        out["tile_id"].notna(),
        out["tile_rel"],
    )
    out["tile_rel_canonical"] = out["tile_rel"]
    out["rc"] = out["tile_stem"].map(parse_rc)
    out["row_idx"] = out["rc"].map(lambda x: x[0] if x else np.nan)
    out["col_idx"] = out["rc"].map(lambda x: x[1] if x else np.nan)
    return out.reset_index(drop=True), dedupe_stats


def load_feature_table(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise SystemExit(f"No features found: {path}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:31983", allow_override=True)
    elif gdf.crs.to_string() != "EPSG:31983":
        gdf = gdf.to_crs("EPSG:31983")

    need_cols = ("tile_rel", "tile_stem", "cell", "tile_path_abs", "area_m2", "confidence", "mask_area_px", "mask_idx")
    for col in need_cols:
        if col not in gdf.columns:
            raise SystemExit(f"pools.geojson missing column: {col}")

    gdf = attach_tile_ids(gdf.copy())
    missing_tile_id = int(gdf["tile_id"].isna().sum()) if "tile_id" in gdf.columns else int(len(gdf))
    if missing_tile_id > 0:
        raise SystemExit(
            f"pools.geojson contains {missing_tile_id} rows without valid tile_id "
            f"(required pattern: {TILE_ID_PATTERN})."
        )
    gdf["tile_rel_canonical"] = gdf["tile_id"].map(lambda x: canonical_tile_rel_from_id(str(x)))
    gdf["tile_rel"] = gdf["tile_rel_canonical"].where(gdf["tile_rel_canonical"].notna(), gdf["tile_rel"])
    gdf["area_m2"] = pd.to_numeric(gdf["area_m2"], errors="coerce")
    gdf["confidence"] = pd.to_numeric(gdf["confidence"], errors="coerce")
    gdf["mask_area_px"] = pd.to_numeric(gdf["mask_area_px"], errors="coerce")
    gdf["geom_area_m2"] = gdf.geometry.area
    gdf["geom_perim_m"] = gdf.geometry.length
    gdf["compactness"] = (4.0 * math.pi * gdf["geom_area_m2"]) / np.maximum(np.square(gdf["geom_perim_m"]), 1e-9)

    bounds = gdf.geometry.bounds
    gdf["bbox_w_m"] = bounds["maxx"] - bounds["minx"]
    gdf["bbox_h_m"] = bounds["maxy"] - bounds["miny"]
    gdf["bbox_aspect"] = np.maximum(gdf["bbox_w_m"], gdf["bbox_h_m"]) / np.maximum(np.minimum(gdf["bbox_w_m"], gdf["bbox_h_m"]), 1e-6)
    gdf["bbox_fill"] = gdf["geom_area_m2"] / np.maximum(gdf["bbox_w_m"] * gdf["bbox_h_m"], 1e-6)
    cent = gdf.geometry.centroid
    gdf["centroid_x"] = cent.x
    gdf["centroid_y"] = cent.y
    return gdf


def get_image(path: str, cache: CacheBundle) -> Optional[np.ndarray]:
    if path in cache.image:
        cache.image.move_to_end(path)
        return cache.image[path]
    p = Path(path)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if cache.max_image_cache > 0 and len(cache.image) >= cache.max_image_cache:
        cache.image.popitem(last=False)
    cache.image[path] = img
    return img


def get_world(path: str, cache: CacheBundle) -> Optional[WorldTransform]:
    if path in cache.world:
        return cache.world[path]
    p = Path(path)
    pgw = p.with_suffix(".pgw")
    if not pgw.exists():
        return None
    tr = read_world_transform(pgw)
    if tr is None:
        return None
    cache.world[path] = tr
    return tr


def compute_local_patch_metrics(
    tile_path_abs: str,
    cx: float,
    cy: float,
    cache: CacheBundle,
    *,
    patch_half: int = 56,
    inner_half: int = 18,
    outer_half: int = 48,
) -> Optional[Dict[str, float]]:
    img = get_image(tile_path_abs, cache)
    tr = get_world(tile_path_abs, cache)
    if img is None or tr is None:
        return None

    col, row = world_to_pixel(cx, cy, tr)
    h, w = img.shape[:2]
    c = int(round(col))
    r = int(round(row))
    x0 = max(0, c - patch_half)
    x1 = min(w, c + patch_half)
    y0 = max(0, r - patch_half)
    y1 = min(h, r + patch_half)
    patch = img[y0:y1, x0:x1]
    if patch.shape[0] < 36 or patch.shape[1] < 36:
        return None

    ph, pw = patch.shape[:2]
    yy, xx = np.ogrid[:ph, :pw]
    cyi = ph // 2
    cxi = pw // 2
    dist = np.maximum(np.abs(yy - cyi), np.abs(xx - cxi))
    inner = dist <= inner_half
    ring = (dist <= outer_half) & (~inner)
    if inner.sum() < 64 or ring.sum() < 128:
        return None

    b = patch[:, :, 0].astype(np.int16)
    g = patch[:, :, 1].astype(np.int16)
    rch = patch[:, :, 2].astype(np.int16)
    blue = (b > (rch + 20)) & (b > (g + 12)) & (b > 70)
    dark_blue = (b > (rch + 15)) & (b > (g + 8)) & (b > 50) & (b < 150)
    bright_blue = (b > (rch + 20)) & (b > (g + 12)) & (b >= 150)
    green = (g > (b + 8)) & (g > (rch + 8)) & (g > 60)
    gray = (np.abs(b - g) <= 12) & (np.abs(b - rch) <= 12) & (np.abs(g - rch) <= 12) & (b > 60) & (b < 235)

    gray8 = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray8, 80, 160) > 0
    dark = gray8 < 92

    def ratio(mask: np.ndarray, zone: np.ndarray) -> float:
        z = int(zone.sum())
        if z <= 0:
            return 0.0
        return float((mask & zone).sum()) / float(z)

    inner_blue = ratio(blue, inner)
    ring_blue = ratio(blue, ring)
    inner_dark_blue = ratio(dark_blue, inner)
    ring_dark_blue = ratio(dark_blue, ring)
    inner_bright_blue = ratio(bright_blue, inner)
    ring_bright_blue = ratio(bright_blue, ring)
    inner_green = ratio(green, inner)
    ring_green = ratio(green, ring)
    inner_gray = ratio(gray, inner)
    ring_gray = ratio(gray, ring)
    inner_dark = ratio(dark, inner)
    ring_dark = ratio(dark, ring)
    inner_edge = ratio(edges, inner)
    ring_edge = ratio(edges, ring)
    inner_luma_std = float(np.std(gray8[inner])) if int(inner.sum()) > 0 else 0.0
    ring_luma_std = float(np.std(gray8[ring])) if int(ring.sum()) > 0 else 0.0

    return {
        "inner_blue": inner_blue,
        "ring_blue": ring_blue,
        "inner_dark_blue": inner_dark_blue,
        "ring_dark_blue": ring_dark_blue,
        "inner_bright_blue": inner_bright_blue,
        "ring_bright_blue": ring_bright_blue,
        "inner_green": inner_green,
        "ring_green": ring_green,
        "inner_gray": inner_gray,
        "ring_gray": ring_gray,
        "inner_dark": inner_dark,
        "ring_dark": ring_dark,
        "inner_edge": inner_edge,
        "ring_edge": ring_edge,
        "inner_luma_std": inner_luma_std,
        "ring_luma_std": ring_luma_std,
        "blue_contrast": abs(inner_dark_blue - ring_dark_blue),
    }


def attach_patch_metrics(df: pd.DataFrame, cache: CacheBundle) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    rows: List[Dict[str, float]] = []
    valid_mask: List[bool] = []
    for row in out.itertuples(index=False):
        m = compute_local_patch_metrics(
            tile_path_abs=str(row.tile_path_abs),
            cx=float(row.centroid_x),
            cy=float(row.centroid_y),
            cache=cache,
        )
        if m is None:
            valid_mask.append(False)
            rows.append({})
            continue
        valid_mask.append(True)
        rows.append(m)

    metrics = pd.DataFrame(rows)
    for c in metrics.columns:
        out[c] = metrics[c].to_numpy()
    out = out.loc[np.array(valid_mask, dtype=bool)].copy().reset_index(drop=True)
    return out


def normalize_col(col: pd.Series) -> pd.Series:
    x = pd.to_numeric(col, errors="coerce")
    lo = float(x.min())
    hi = float(x.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - lo) / (hi - lo)


def dedupe_best_per_tile(df: pd.DataFrame, score_col: str = "selection_score") -> pd.DataFrame:
    if df.empty:
        return df
    subset_col = "tile_id" if "tile_id" in df.columns else "tile_rel"
    out = (
        df.sort_values(score_col, ascending=False)
        .drop_duplicates(subset=[subset_col], keep="first")
        .reset_index(drop=True)
    )
    return out


def select_with_cell_limits(
    df: pd.DataFrame,
    *,
    count: int,
    used_tile_ids: set[str],
    per_cell_limit: int,
    dedupe_stats: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    if count <= 0 or df.empty:
        return df.iloc[:0].copy()
    ranked = df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    keep_rows: List[int] = []
    keep_rows_set: set[int] = set()
    cell_count: Counter[str] = Counter()

    for idx, row in ranked.iterrows():
        tile_id = str(row.get("tile_id", "")).strip()
        if not tile_id:
            tile_id = extract_tile_id_from_text(row.get("tile_rel")) or ""
        if not tile_id:
            if dedupe_stats is not None:
                dedupe_stats["missing_tile_id_rows_skipped"] = dedupe_stats.get("missing_tile_id_rows_skipped", 0) + 1
            continue
        if tile_id in used_tile_ids:
            if dedupe_stats is not None:
                dedupe_stats["cross_bucket_collisions_prevented"] = dedupe_stats.get("cross_bucket_collisions_prevented", 0) + 1
            continue
        cell = str(row.get("cell", ""))
        if per_cell_limit > 0 and cell_count[cell] >= per_cell_limit:
            continue
        keep_rows.append(idx)
        keep_rows_set.add(idx)
        used_tile_ids.add(tile_id)
        cell_count[cell] += 1
        if len(keep_rows) >= count:
            break

    # Backfill while preserving tile uniqueness if per-cell cap was too strict.
    if len(keep_rows) < count:
        for idx, row in ranked.iterrows():
            if idx in keep_rows_set:
                continue
            tile_id = str(row.get("tile_id", "")).strip()
            if not tile_id:
                tile_id = extract_tile_id_from_text(row.get("tile_rel")) or ""
            if not tile_id:
                if dedupe_stats is not None:
                    dedupe_stats["missing_tile_id_rows_skipped"] = dedupe_stats.get("missing_tile_id_rows_skipped", 0) + 1
                continue
            if tile_id in used_tile_ids:
                if dedupe_stats is not None:
                    dedupe_stats["cross_bucket_collisions_prevented"] = dedupe_stats.get("cross_bucket_collisions_prevented", 0) + 1
                continue
            keep_rows.append(idx)
            keep_rows_set.add(idx)
            used_tile_ids.add(tile_id)
            if len(keep_rows) >= count:
                break

    out = ranked.iloc[keep_rows].copy().reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def build_shoreline_candidates(features: gpd.GeoDataFrame, cache: CacheBundle) -> pd.DataFrame:
    base = features[
        (features["area_m2"] >= 80.0)
        & (
            (features["compactness"] < 0.62)
            | (features["bbox_aspect"] > 2.2)
            | (features["bbox_fill"] < 0.55)
        )
        & (features["confidence"] <= 0.88)
    ].copy()
    if base.empty:
        return pd.DataFrame()
    base = attach_patch_metrics(base, cache)
    if base.empty:
        return pd.DataFrame()

    base = base[(base["ring_dark_blue"] >= 0.03) & (base["ring_edge"] >= 0.04)].copy()
    if base.empty:
        return pd.DataFrame()

    area_n = normalize_col(base["area_m2"].clip(upper=1200.0))
    irreg = (1.0 - base["compactness"].clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
    aspect_n = ((base["bbox_aspect"] - 1.2) / 3.8).clip(lower=0.0, upper=1.0)
    ring_edge_n = (base["ring_edge"] / 0.30).clip(lower=0.0, upper=1.0)
    mix_n = (base["blue_contrast"] / 0.55).clip(lower=0.0, upper=1.0)
    conf_low = (1.0 - base["confidence"].clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)

    base["selection_score"] = (
        0.28 * area_n
        + 0.20 * irreg
        + 0.14 * aspect_n
        + 0.17 * base["ring_dark_blue"].clip(lower=0.0, upper=1.0)
        + 0.13 * ring_edge_n
        + 0.05 * mix_n
        + 0.03 * conf_low
    )
    base["bucket"] = "hard_negatives_shorelines"
    base["selection_reason"] = base.apply(
        lambda r: (
            f"shore_proxy area={safe_float(r['area_m2']):.1f} compact={safe_float(r['compactness']):.3f} "
            f"aspect={safe_float(r['bbox_aspect']):.2f} ring_dark_blue={safe_float(r['ring_dark_blue']):.3f} "
            f"ring_edge={safe_float(r['ring_edge']):.3f}"
        ),
        axis=1,
    )
    return dedupe_best_per_tile(base)


def build_court_candidates(features: gpd.GeoDataFrame, cache: CacheBundle) -> pd.DataFrame:
    base = features[
        (features["area_m2"] >= 110.0)
        & (features["area_m2"] <= 1400.0)
        & (features["bbox_aspect"] >= 1.2)
        & (features["bbox_aspect"] <= 5.0)
        & (features["bbox_fill"] >= 0.58)
        & (features["compactness"] >= 0.42)
        & (features["confidence"] >= 0.20)
        & (features["confidence"] <= 0.93)
    ].copy()
    if base.empty:
        return pd.DataFrame()
    base = attach_patch_metrics(base, cache)
    if base.empty:
        return pd.DataFrame()

    base = base[(base["ring_dark_blue"] <= 0.85) & (base["ring_edge"] >= 0.02)].copy()
    if base.empty:
        return pd.DataFrame()

    area_n = ((base["area_m2"] - 110.0) / 980.0).clip(lower=0.0, upper=1.0)
    rect_score = (
        0.5 * ((base["bbox_fill"] - 0.55) / 0.45).clip(lower=0.0, upper=1.0)
        + 0.5 * ((base["compactness"] - 0.40) / 0.55).clip(lower=0.0, upper=1.0)
    )
    aspect_pref = (1.0 - (np.abs(base["bbox_aspect"] - 2.0) / 2.6).clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
    edge_n = (base["ring_edge"] / 0.24).clip(lower=0.0, upper=1.0)
    green_n = (base["ring_green"] / 0.18).clip(lower=0.0, upper=1.0)
    non_water = (1.0 - (base["ring_dark_blue"] / 0.80).clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)

    base["selection_score"] = (
        0.25 * area_n
        + 0.24 * rect_score
        + 0.18 * aspect_pref
        + 0.16 * edge_n
        + 0.10 * green_n
        + 0.07 * non_water
    )
    base["bucket"] = "hard_negatives_courts"
    base["selection_reason"] = base.apply(
        lambda r: (
            f"court_proxy area={safe_float(r['area_m2']):.1f} aspect={safe_float(r['bbox_aspect']):.2f} "
            f"fill={safe_float(r['bbox_fill']):.3f} ring_edge={safe_float(r['ring_edge']):.3f} "
            f"ring_green={safe_float(r['ring_green']):.3f}"
        ),
        axis=1,
    )
    return dedupe_best_per_tile(base)


def build_blue_roof_candidates(features: gpd.GeoDataFrame, cache: CacheBundle) -> pd.DataFrame:
    base = features[
        (features["area_m2"] >= 18.0)
        & (features["area_m2"] <= 320.0)
        & (features["bbox_aspect"] <= 3.2)
        & (features["bbox_fill"] >= 0.56)
        & (features["compactness"] >= 0.50)
        & (features["confidence"] >= 0.20)
        & (features["confidence"] <= 0.96)
    ].copy()
    if base.empty:
        return pd.DataFrame()

    # Keep the top geometry-priority subset to bound image I/O.
    base["geom_priority"] = (
        0.50 * ((base["bbox_fill"] - 0.5) / 0.5).clip(lower=0.0, upper=1.0)
        + 0.30 * ((base["compactness"] - 0.4) / 0.6).clip(lower=0.0, upper=1.0)
        + 0.20 * (1.0 - (np.abs(np.log(np.maximum(base["area_m2"], 1.0)) - math.log(90.0)) / 1.5).clip(lower=0.0, upper=1.0))
    )
    base = base.sort_values("geom_priority", ascending=False).head(3500).copy()

    base = attach_patch_metrics(base, cache)
    if base.empty:
        return pd.DataFrame()

    base = base[(base["inner_bright_blue"] >= 0.05) & (base["ring_edge"] >= 0.05)].copy()
    if base.empty:
        return pd.DataFrame()

    area_pref = (1.0 - (np.abs(np.log(np.maximum(base["area_m2"], 1.0)) - math.log(90.0)) / 1.3).clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
    rect_score = (
        0.5 * ((base["bbox_fill"] - 0.55) / 0.45).clip(lower=0.0, upper=1.0)
        + 0.5 * ((base["compactness"] - 0.45) / 0.55).clip(lower=0.0, upper=1.0)
    )
    blue_core = (base["inner_bright_blue"] / 0.90).clip(lower=0.0, upper=1.0)
    urban_ctx = (base["ring_gray"] / 0.24).clip(lower=0.0, upper=1.0)
    low_veg = (1.0 - (base["ring_green"] / 0.20).clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
    not_dark_water = (1.0 - (base["ring_dark_blue"] / 0.75).clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
    edge_n = (base["ring_edge"] / 0.24).clip(lower=0.0, upper=1.0)

    base["selection_score"] = (
        0.30 * blue_core
        + 0.20 * urban_ctx
        + 0.14 * low_veg
        + 0.14 * not_dark_water
        + 0.12 * rect_score
        + 0.06 * edge_n
        + 0.04 * area_pref
    )
    base["bucket"] = "hard_negatives_blue_roofs"
    base["selection_reason"] = base.apply(
        lambda r: (
            f"blue_roof_proxy area={safe_float(r['area_m2']):.1f} inner_bright_blue={safe_float(r['inner_bright_blue']):.3f} "
            f"ring_gray={safe_float(r['ring_gray']):.3f} ring_green={safe_float(r['ring_green']):.3f} "
            f"ring_dark_blue={safe_float(r['ring_dark_blue']):.3f}"
        ),
        axis=1,
    )
    return dedupe_best_per_tile(base)


def build_duplicate_fragment_candidates(features: gpd.GeoDataFrame, tiles_df: pd.DataFrame) -> pd.DataFrame:
    by_tile: Dict[str, List[int]] = defaultdict(list)
    by_col = "tile_id" if "tile_id" in features.columns else "tile_rel"
    for idx, tile_key in enumerate(features[by_col].tolist()):
        by_tile[str(tile_key)].append(idx)

    rows: List[Dict[str, object]] = []
    lookup_key = "tile_id" if "tile_id" in tiles_df.columns else "tile_rel"
    tile_lookup = tiles_df.set_index(lookup_key, drop=False)

    for tile_key, idxs in by_tile.items():
        if len(idxs) < 2:
            continue
        sub = features.iloc[idxs]
        geoms = list(sub.geometry.values)
        confs = [safe_float(v, 0.0) for v in sub["confidence"].tolist()]
        areas = [safe_float(v, 0.0) for v in sub["area_m2"].tolist()]

        overlap_pairs = 0
        near_pairs = 0
        touching_pairs = 0
        max_iou = 0.0
        max_overlap_small = 0.0
        min_centroid_dist = 1e9

        n = len(geoms)
        for i in range(n):
            gi = geoms[i]
            ci = gi.centroid
            ai = max(1e-6, float(gi.area))
            for j in range(i + 1, n):
                gj = geoms[j]
                cj = gj.centroid
                aj = max(1e-6, float(gj.area))
                dist = float(ci.distance(cj))
                min_centroid_dist = min(min_centroid_dist, dist)

                inter = 0.0
                if gi.intersects(gj):
                    inter = float(gi.intersection(gj).area)
                if inter > 0.0:
                    union = ai + aj - inter
                    iou = inter / max(union, 1e-6)
                    overlap_small = inter / max(min(ai, aj), 1e-6)
                    max_iou = max(max_iou, iou)
                    max_overlap_small = max(max_overlap_small, overlap_small)
                    if (0.02 <= iou <= 0.82) or (overlap_small >= 0.35):
                        overlap_pairs += 1
                    if dist <= 5.0:
                        touching_pairs += 1
                elif dist <= 8.0:
                    near_pairs += 1

        medium_count = int(sum(1 for a in areas if 12.0 <= a <= 320.0))
        if medium_count < 2:
            continue

        mean_conf = float(np.mean(confs)) if confs else 0.0
        frag_score = (1.9 * overlap_pairs) + (0.8 * near_pairs) + (0.6 * touching_pairs)
        density_score = max(0.0, float(n - 2)) * 0.18
        conf_score = clamp01((mean_conf - 0.30) / 0.55)
        shape_score = clamp01((max_overlap_small - 0.18) / 0.65)
        selection_score = frag_score + density_score + (0.9 * conf_score) + (0.8 * shape_score)

        tile_row = tile_lookup.loc[tile_key] if tile_key in tile_lookup.index else None
        if isinstance(tile_row, pd.DataFrame):
            tile_row = tile_row.iloc[0]
        tile_id = str(tile_row["tile_id"]) if tile_row is not None and "tile_id" in tile_row else str(tile_key)
        tile_rel = (
            str(tile_row["tile_rel"]) if tile_row is not None and "tile_rel" in tile_row else canonical_tile_rel_from_id(tile_id)
        )
        tile_path_abs = str(tile_row["tile_path_abs"]) if tile_row is not None else str(sub.iloc[0]["tile_path_abs"])
        tile_stem = (
            str(tile_row["tile_stem"])
            if tile_row is not None and "tile_stem" in tile_row
            else tile_stem_from_id(tile_id)
        )
        cell = (
            str(tile_row["cell"])
            if tile_row is not None and "cell" in tile_row
            else tile_id.split("__", 1)[0]
        )
        num_preds = int(tile_row["num_preds"]) if tile_row is not None and "num_preds" in tile_row else int(len(sub))
        min_conf = safe_float(tile_row["min_conf"]) if tile_row is not None and "min_conf" in tile_row else float(np.min(confs))
        mean_conf_tile = safe_float(tile_row["mean_conf"]) if tile_row is not None and "mean_conf" in tile_row else mean_conf
        max_conf = safe_float(tile_row["max_conf"]) if tile_row is not None and "max_conf" in tile_row else float(np.max(confs))
        sum_area_m2 = safe_float(tile_row["sum_area_m2"]) if tile_row is not None and "sum_area_m2" in tile_row else float(np.sum(areas))
        max_area_m2 = safe_float(tile_row["max_area_m2"]) if tile_row is not None and "max_area_m2" in tile_row else float(np.max(areas))

        rows.append(
            {
                "bucket": "duplicate_fragmented_real_pools",
                "tile_id": tile_id,
                "tile_rel": tile_rel,
                "tile_path_abs": tile_path_abs,
                "tile_stem": tile_stem,
                "cell": cell,
                "num_preds": num_preds,
                "min_conf": min_conf,
                "mean_conf": mean_conf_tile,
                "max_conf": max_conf,
                "sum_area_m2": sum_area_m2,
                "max_area_m2": max_area_m2,
                "feature_count": int(n),
                "candidate_feature_conf": mean_conf,
                "candidate_feature_area_m2": float(np.mean(areas)) if areas else 0.0,
                "candidate_feature_compactness": "",
                "candidate_feature_aspect_ratio": "",
                "selection_score": selection_score,
                "selection_reason": (
                    f"duplicate_fragment_proxy overlap_pairs={overlap_pairs} near_pairs={near_pairs} "
                    f"touching_pairs={touching_pairs} max_iou={max_iou:.3f} min_centroid_dist={min_centroid_dist:.2f}"
                ),
                "overlap_pairs": overlap_pairs,
                "near_pairs": near_pairs,
                "ring_dark_blue": "",
                "ring_bright_blue": "",
                "ring_gray": "",
                "ring_green": "",
                "ring_edge": "",
                "pool_component_score": "",
                "neighbor_preds_sum": "",
                "neighbor_positive_tiles": "",
                "context_score": "",
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("selection_score", ascending=False).reset_index(drop=True)
    return out


def compute_neighbor_context(tiles_df: pd.DataFrame) -> pd.DataFrame:
    cell_pred_sum = tiles_df.groupby("cell")["num_preds"].sum().to_dict()
    cell_area_sum = tiles_df.groupby("cell")["sum_area_m2"].sum().to_dict()
    grid: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(dict)

    for idx, row in tiles_df.iterrows():
        rr = row["row_idx"]
        cc = row["col_idx"]
        if math.isnan(rr) or math.isnan(cc):
            continue
        grid[str(row["cell"])][(int(rr), int(cc))] = int(idx)

    out = tiles_df.copy()
    neighbor_preds = np.zeros(len(out), dtype=float)
    neighbor_area = np.zeros(len(out), dtype=float)
    neighbor_pos_tiles = np.zeros(len(out), dtype=float)
    cell_total_preds = np.zeros(len(out), dtype=float)
    cell_total_area = np.zeros(len(out), dtype=float)

    for idx, row in out.iterrows():
        cell = str(row["cell"])
        cell_total_preds[idx] = float(cell_pred_sum.get(cell, 0.0))
        cell_total_area[idx] = float(cell_area_sum.get(cell, 0.0))

        rr = row["row_idx"]
        cc = row["col_idx"]
        if math.isnan(rr) or math.isnan(cc):
            continue
        r0 = int(rr)
        c0 = int(cc)
        lookup = grid.get(cell, {})
        ps = 0.0
        aas = 0.0
        pos = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb_idx = lookup.get((r0 + dr, c0 + dc))
                if nb_idx is None:
                    continue
                nrow = out.iloc[nb_idx]
                npreds = float(nrow["num_preds"])
                ps += npreds
                aas += float(nrow["sum_area_m2"])
                if npreds > 0:
                    pos += 1.0
        neighbor_preds[idx] = ps
        neighbor_area[idx] = aas
        neighbor_pos_tiles[idx] = pos

    out["cell_total_preds"] = cell_total_preds
    out["cell_total_area_m2"] = cell_total_area
    out["neighbor_preds_sum"] = neighbor_preds
    out["neighbor_sum_area_m2"] = neighbor_area
    out["neighbor_positive_tiles"] = neighbor_pos_tiles

    # Context score adapted from select_missed_pool_tiles.py.
    cp = normalize_col(out["cell_total_preds"])
    ca = normalize_col(out["cell_total_area_m2"])
    npred = normalize_col(out["neighbor_preds_sum"])
    narea = normalize_col(out["neighbor_sum_area_m2"])
    npos = normalize_col(out["neighbor_positive_tiles"])
    own_low_pref = np.where(out["num_preds"].values == 0, 1.0, 0.65)
    out["context_score"] = (0.42 * ((0.65 * cp) + (0.35 * ca))) + (0.48 * ((0.50 * npred) + (0.35 * narea) + (0.15 * npos))) + (0.10 * own_low_pref)
    return out


def compute_pool_component_signal(tile_path_abs: str, cache: CacheBundle) -> Dict[str, float]:
    if tile_path_abs in cache.missed_pool_signal:
        return cache.missed_pool_signal[tile_path_abs]

    img = get_image(tile_path_abs, cache)
    if img is None:
        res = {
            "pool_component_score": 0.0,
            "pool_component_count": 0.0,
            "pool_component_best_area_px": 0.0,
            "pool_component_best_aspect": 0.0,
            "pool_component_best_fill": 0.0,
            "pool_component_best_solidity": 0.0,
        }
        cache.missed_pool_signal[tile_path_abs] = res
        return res

    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    mask = ((b > (r + 20)) & (b > (g + 12)) & (b > 65)).astype(np.uint8) * 255
    kern = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    best_area = 0.0
    best_aspect = 0.0
    best_fill = 0.0
    best_solidity = 0.0
    kept = 0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 120.0 or area > 7000.0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        if aspect > 4.8:
            continue
        rect_area = float(w * h)
        fill = area / max(rect_area, 1.0)
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if fill < 0.25 or solidity < 0.45:
            continue

        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [cnt], -1, 255, thickness=-1)
        pix = component_mask > 0
        if int(pix.sum()) == 0:
            continue
        blue_diff = float((b[pix] - r[pix]).mean())

        area_score = clamp01((area - 120.0) / (2200.0 - 120.0))
        fill_score = clamp01((fill - 0.30) / 0.65)
        solidity_score = clamp01((solidity - 0.45) / 0.50)
        color_score = clamp01((blue_diff - 16.0) / 70.0)
        comp_score = (0.28 * area_score) + (0.28 * fill_score) + (0.24 * solidity_score) + (0.20 * color_score)

        kept += 1
        if comp_score > best_score:
            best_score = comp_score
            best_area = area
            best_aspect = aspect
            best_fill = fill
            best_solidity = solidity

    res = {
        "pool_component_score": float(best_score),
        "pool_component_count": float(kept),
        "pool_component_best_area_px": float(best_area),
        "pool_component_best_aspect": float(best_aspect),
        "pool_component_best_fill": float(best_fill),
        "pool_component_best_solidity": float(best_solidity),
    }
    cache.missed_pool_signal[tile_path_abs] = res
    return res


def build_missed_candidates(tiles_df: pd.DataFrame, cache: CacheBundle, *, preselect_limit: int) -> pd.DataFrame:
    ctx = compute_neighbor_context(tiles_df)
    base = ctx[(ctx["blank_white"] == 0) & (ctx["num_preds"] == 0)].copy()
    if base.empty:
        return pd.DataFrame()

    base = base.sort_values("context_score", ascending=False).head(preselect_limit).reset_index(drop=True).copy()
    pool_rows: List[Dict[str, float]] = []
    for row in base.itertuples(index=False):
        pool_rows.append(compute_pool_component_signal(str(row.tile_path_abs), cache))

    pool_df = pd.DataFrame(pool_rows)
    for col in pool_df.columns:
        base[col] = pool_df[col].to_numpy()

    base["selection_score"] = (0.62 * base["context_score"]) + (0.38 * base["pool_component_score"])
    base = base[(base["pool_component_score"] >= 0.14) | (base["context_score"] >= 0.78)].copy()
    if base.empty:
        return pd.DataFrame()

    base["bucket"] = "missed_obvious_pools"
    base["selection_reason"] = base.apply(
        lambda r: (
            f"missed_pool_proxy context={safe_float(r['context_score']):.3f} "
            f"pool_component={safe_float(r['pool_component_score']):.3f} "
            f"neighbor_preds={safe_float(r['neighbor_preds_sum']):.1f}"
        ),
        axis=1,
    )
    base["feature_count"] = 0
    base["candidate_feature_conf"] = ""
    base["candidate_feature_area_m2"] = ""
    base["candidate_feature_compactness"] = ""
    base["candidate_feature_aspect_ratio"] = ""
    base["overlap_pairs"] = ""
    base["near_pairs"] = ""
    base["ring_dark_blue"] = ""
    base["ring_bright_blue"] = ""
    base["ring_gray"] = ""
    base["ring_green"] = ""
    base["ring_edge"] = ""
    return base


def compute_nearest_larger_pool_metrics(features: gpd.GeoDataFrame) -> pd.DataFrame:
    n = len(features)
    out = pd.DataFrame(
        {
            "nearest_larger_gap_m": np.full(n, np.nan, dtype=float),
            "nearest_larger_centroid_m": np.full(n, np.nan, dtype=float),
            "nearest_larger_area_ratio": np.full(n, np.nan, dtype=float),
            "nearest_larger_iou": np.full(n, np.nan, dtype=float),
            "nearest_larger_overlap_small": np.full(n, np.nan, dtype=float),
            "nearest_larger_conf": np.full(n, np.nan, dtype=float),
            "nearest_larger_area_m2": np.full(n, np.nan, dtype=float),
            "larger_neighbor_count": np.zeros(n, dtype=float),
            "tile_feature_count": np.zeros(n, dtype=float),
        }
    )
    if features.empty:
        return out

    by_tile: Dict[str, List[int]] = defaultdict(list)
    for idx, tile_rel in enumerate(features["tile_rel"].tolist()):
        by_tile[str(tile_rel)].append(idx)

    geoms = list(features.geometry.values)
    areas = [max(1e-6, safe_float(v, 1.0)) for v in features["area_m2"].tolist()]
    confs = [safe_float(v, 0.0) for v in features["confidence"].tolist()]
    centroids = [g.centroid for g in geoms]

    for idxs in by_tile.values():
        tcount = float(len(idxs))
        if len(idxs) < 2:
            if idxs:
                out.loc[idxs[0], "tile_feature_count"] = tcount
            continue

        for i in idxs:
            out.loc[i, "tile_feature_count"] = tcount
            ai = areas[i]
            gi = geoms[i]
            ci = centroids[i]
            best_key = (1e9, 1e9, 1e9)
            best_vals: Optional[Tuple[float, float, float, float, float, float, float]] = None
            larger_count = 0

            for j in idxs:
                if i == j:
                    continue
                aj = areas[j]
                if aj < (1.18 * ai):
                    continue
                larger_count += 1

                gj = geoms[j]
                gap = float(gi.distance(gj))
                centroid_d = float(ci.distance(centroids[j]))
                inter = float(gi.intersection(gj).area) if gi.intersects(gj) else 0.0
                iou = inter / max((ai + aj - inter), 1e-6) if inter > 0.0 else 0.0
                overlap_small = inter / max(min(ai, aj), 1e-6) if inter > 0.0 else 0.0
                key = (gap, centroid_d, -aj)

                if key < best_key:
                    best_key = key
                    best_vals = (
                        gap,
                        centroid_d,
                        ai / max(aj, 1e-6),
                        iou,
                        overlap_small,
                        confs[j],
                        aj,
                    )

            out.loc[i, "larger_neighbor_count"] = float(larger_count)
            if best_vals is None:
                continue
            (
                gap,
                centroid_d,
                area_ratio,
                iou,
                overlap_small,
                larger_conf,
                larger_area,
            ) = best_vals
            out.loc[i, "nearest_larger_gap_m"] = gap
            out.loc[i, "nearest_larger_centroid_m"] = centroid_d
            out.loc[i, "nearest_larger_area_ratio"] = area_ratio
            out.loc[i, "nearest_larger_iou"] = iou
            out.loc[i, "nearest_larger_overlap_small"] = overlap_small
            out.loc[i, "nearest_larger_conf"] = larger_conf
            out.loc[i, "nearest_larger_area_m2"] = larger_area

    return out


def build_tree_shadow_candidates(features: gpd.GeoDataFrame, cache: CacheBundle) -> pd.DataFrame:
    base = features[
        (features["area_m2"] >= 18.0)
        & (features["area_m2"] <= 1200.0)
        & (features["confidence"] >= 0.18)
        & (features["confidence"] <= 0.93)
    ].copy()
    if base.empty:
        return pd.DataFrame()

    area_n = ((np.log(np.maximum(base["area_m2"], 1.0)) - math.log(18.0)) / (math.log(1200.0) - math.log(18.0))).clip(0.0, 1.0)
    rectish = (1.0 - (np.abs(np.log(np.maximum(base["bbox_aspect"], 1e-6)) - math.log(1.8)) / 1.55).clip(0.0, 1.0)).clip(0.0, 1.0)
    irregular = (1.0 - base["compactness"].clip(0.0, 1.0)).clip(0.0, 1.0)
    conf_mid = (1.0 - (np.abs(base["confidence"] - 0.56) / 0.44).clip(0.0, 1.0)).clip(0.0, 1.0)
    base["geom_priority"] = (0.44 * area_n) + (0.22 * rectish) + (0.20 * irregular) + (0.14 * conf_mid)
    base = base.sort_values("geom_priority", ascending=False).head(5200).copy()

    base = attach_patch_metrics(base, cache)
    if base.empty:
        return pd.DataFrame()

    base = base[
        (base["inner_dark"] >= 0.12)
        & ((base["ring_green"] >= 0.03) | (base["inner_green"] >= 0.02) | (base["ring_dark_blue"] >= 0.03))
        & (base["inner_edge"] <= 0.18)
        & (base["ring_edge"] <= 0.25)
    ].copy()
    if base.empty:
        return pd.DataFrame()

    area_score = ((np.log(np.maximum(base["area_m2"], 1.0)) - math.log(18.0)) / (math.log(1200.0) - math.log(18.0))).clip(0.0, 1.0)
    shadow_tone = np.maximum(base["inner_dark"], base["ring_dark"]).clip(0.0, 1.0)
    canopy_ctx = np.maximum(base["ring_green"], base["inner_green"]).clip(0.0, 1.0)
    blue_shadow = np.maximum(base["ring_dark_blue"], base["inner_dark_blue"]).clip(0.0, 1.0)
    smooth_n = (
        (0.55 * (1.0 - (base["inner_edge"] / 0.20).clip(0.0, 1.0)))
        + (0.45 * (1.0 - (base["inner_luma_std"] / 44.0).clip(0.0, 1.0)))
    ).clip(0.0, 1.0)
    rectish_n = (1.0 - (np.abs(np.log(np.maximum(base["bbox_aspect"], 1e-6)) - math.log(1.85)) / 1.45).clip(0.0, 1.0)).clip(0.0, 1.0)
    conf_mid = (1.0 - (np.abs(base["confidence"] - 0.55) / 0.45).clip(0.0, 1.0)).clip(0.0, 1.0)

    base["selection_score"] = (
        0.24 * area_score
        + 0.22 * shadow_tone
        + 0.16 * canopy_ctx
        + 0.12 * blue_shadow
        + 0.15 * smooth_n
        + 0.08 * rectish_n
        + 0.03 * conf_mid
    )
    base["bucket"] = "hard_negatives_tree_shadows"
    base["selection_reason"] = base.apply(
        lambda r: (
            f"tree_shadow_proxy area={safe_float(r['area_m2']):.1f} inner_dark={safe_float(r['inner_dark']):.3f} "
            f"ring_green={safe_float(r['ring_green']):.3f} ring_dark_blue={safe_float(r['ring_dark_blue']):.3f} "
            f"inner_edge={safe_float(r['inner_edge']):.3f}"
        ),
        axis=1,
    )
    base["feature_count"] = 1
    base["candidate_feature_conf"] = base["confidence"]
    base["candidate_feature_area_m2"] = base["area_m2"]
    base["candidate_feature_compactness"] = base["compactness"]
    base["candidate_feature_aspect_ratio"] = base["bbox_aspect"]
    return dedupe_best_per_tile(base)


def build_pool_edge_shadow_candidates(features: gpd.GeoDataFrame, cache: CacheBundle) -> pd.DataFrame:
    nbr = compute_nearest_larger_pool_metrics(features)
    base = features.copy().reset_index(drop=True)
    for col in nbr.columns:
        base[col] = nbr[col].to_numpy()

    base = base[
        (base["area_m2"] >= 6.0)
        & (base["area_m2"] <= 240.0)
        & (base["confidence"] >= 0.18)
        & (base["confidence"] <= 0.94)
        & (base["nearest_larger_area_ratio"].notna())
    ].copy()
    if base.empty:
        return pd.DataFrame()

    base = base[
        (base["nearest_larger_area_ratio"] <= 0.72)
        & ((base["nearest_larger_gap_m"] <= 8.0) | (base["nearest_larger_centroid_m"] <= 18.0))
        & (base["nearest_larger_iou"] <= 0.72)
    ].copy()
    if base.empty:
        return pd.DataFrame()

    partial_shape = pd.concat(
        [
            (1.0 - base["compactness"].clip(0.0, 1.0)).clip(0.0, 1.0),
            (1.0 - base["bbox_fill"].clip(0.0, 1.0)).clip(0.0, 1.0),
            ((base["bbox_aspect"] - 1.0) / 2.8).clip(0.0, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    base["quick_score"] = (
        0.38 * (1.0 - (base["nearest_larger_gap_m"] / 8.0).clip(0.0, 1.0))
        + 0.33 * (1.0 - base["nearest_larger_area_ratio"].clip(0.0, 1.0))
        + 0.19 * partial_shape
        + 0.10 * (1.0 - base["confidence"].clip(0.0, 1.0))
    )
    base = base.sort_values("quick_score", ascending=False).head(5200).copy()

    base = attach_patch_metrics(base, cache)
    if base.empty:
        return pd.DataFrame()

    base = base[
        (base["ring_edge"] >= 0.03)
        & ((base["inner_dark"] >= 0.09) | (base["ring_dark"] >= 0.12) | (base["ring_dark_blue"] >= 0.03))
    ].copy()
    if base.empty:
        return pd.DataFrame()

    proximity_n = (
        0.65 * (1.0 - (base["nearest_larger_gap_m"] / 8.0).clip(0.0, 1.0))
        + 0.35 * (1.0 - (base["nearest_larger_centroid_m"] / 18.0).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    size_gap_n = (1.0 - base["nearest_larger_area_ratio"].clip(0.0, 1.0)).clip(0.0, 1.0)
    partial_shape_n = pd.concat(
        [
            (1.0 - base["compactness"].clip(0.0, 1.0)).clip(0.0, 1.0),
            (1.0 - base["bbox_fill"].clip(0.0, 1.0)).clip(0.0, 1.0),
            ((base["bbox_aspect"] - 1.0) / 2.8).clip(0.0, 1.0),
        ],
        axis=1,
    ).max(axis=1)
    edge_shadow_n = (
        0.30 * (base["ring_edge"] / 0.24).clip(0.0, 1.0)
        + 0.28 * base["inner_dark"].clip(0.0, 1.0)
        + 0.20 * base["ring_dark"].clip(0.0, 1.0)
        + 0.12 * (base["inner_dark_blue"] / 0.40).clip(0.0, 1.0)
        + 0.10 * (base["ring_dark_blue"] / 0.40).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    conf_low_n = (1.0 - base["confidence"].clip(0.0, 1.0)).clip(0.0, 1.0)

    base["selection_score"] = (
        0.30 * proximity_n
        + 0.24 * size_gap_n
        + 0.20 * partial_shape_n
        + 0.20 * edge_shadow_n
        + 0.06 * conf_low_n
    )
    base["bucket"] = "hard_negatives_pool_edge_shadows"
    base["selection_reason"] = base.apply(
        lambda r: (
            f"pool_edge_shadow_proxy gap={safe_float(r['nearest_larger_gap_m']):.2f} "
            f"area_ratio={safe_float(r['nearest_larger_area_ratio']):.3f} "
            f"iou={safe_float(r['nearest_larger_iou']):.3f} ring_edge={safe_float(r['ring_edge']):.3f} "
            f"inner_dark={safe_float(r['inner_dark']):.3f}"
        ),
        axis=1,
    )
    base["feature_count"] = base["tile_feature_count"]
    base["candidate_feature_conf"] = base["confidence"]
    base["candidate_feature_area_m2"] = base["area_m2"]
    base["candidate_feature_compactness"] = base["compactness"]
    base["candidate_feature_aspect_ratio"] = base["bbox_aspect"]
    return dedupe_best_per_tile(base)


def build_clean_positive_anchor_candidates(features: gpd.GeoDataFrame, tiles_df: pd.DataFrame, cache: CacheBundle) -> pd.DataFrame:
    num_preds_lookup = tiles_df.set_index("tile_rel")["num_preds"].to_dict()
    base = features[
        (features["area_m2"] >= 16.0)
        & (features["area_m2"] <= 420.0)
        & (features["confidence"] >= 0.74)
        & (features["compactness"] >= 0.52)
        & (features["bbox_fill"] >= 0.56)
        & (features["bbox_aspect"] >= 1.0)
        & (features["bbox_aspect"] <= 3.8)
    ].copy()
    if base.empty:
        return pd.DataFrame()

    base["tile_num_preds"] = base["tile_rel"].map(lambda x: parse_int(num_preds_lookup.get(str(x), 0)))
    base = base[(base["tile_num_preds"] >= 1) & (base["tile_num_preds"] <= 3)].copy()
    if base.empty:
        return pd.DataFrame()

    conf_n = ((base["confidence"] - 0.72) / 0.26).clip(0.0, 1.0)
    geom_n = (
        0.50 * ((base["bbox_fill"] - 0.55) / 0.40).clip(0.0, 1.0)
        + 0.50 * ((base["compactness"] - 0.50) / 0.45).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    aspect_pref = (1.0 - (np.abs(np.log(np.maximum(base["bbox_aspect"], 1e-6)) - math.log(1.6)) / 1.45).clip(0.0, 1.0)).clip(0.0, 1.0)
    base["geom_priority"] = (0.48 * conf_n) + (0.34 * geom_n) + (0.18 * aspect_pref)
    base = base.sort_values("geom_priority", ascending=False).head(5000).copy()

    base = attach_patch_metrics(base, cache)
    if base.empty:
        return pd.DataFrame()

    base = base[
        ((base["inner_dark_blue"] >= 0.08) | (base["inner_blue"] >= 0.12))
        & (base["ring_green"] <= 0.24)
        & (base["inner_dark"] <= 0.76)
        & (base["ring_edge"] >= 0.015)
        & (base["ring_edge"] <= 0.22)
    ].copy()
    if base.empty:
        return pd.DataFrame()

    conf_n = ((base["confidence"] - 0.72) / 0.26).clip(0.0, 1.0)
    rect_n = (
        0.50 * ((base["bbox_fill"] - 0.55) / 0.40).clip(0.0, 1.0)
        + 0.50 * ((base["compactness"] - 0.50) / 0.45).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    water_core = (
        0.55 * (base["inner_dark_blue"] / 0.55).clip(0.0, 1.0)
        + 0.30 * (base["inner_blue"] / 0.72).clip(0.0, 1.0)
        + 0.15 * (1.0 - (base["inner_green"] / 0.18).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    context_clean = (
        0.45 * (1.0 - (base["ring_green"] / 0.24).clip(0.0, 1.0))
        + 0.25 * (base["ring_gray"] / 0.25).clip(0.0, 1.0)
        + 0.30 * (1.0 - (base["inner_dark"] / 0.82).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    edge_mid = (1.0 - (np.abs(base["ring_edge"] - 0.07) / 0.09).clip(0.0, 1.0)).clip(0.0, 1.0)
    size_pref = (1.0 - (np.abs(np.log(np.maximum(base["area_m2"], 1.0)) - math.log(72.0)) / 1.35).clip(0.0, 1.0)).clip(0.0, 1.0)

    base["selection_score"] = (
        0.31 * conf_n
        + 0.24 * rect_n
        + 0.20 * water_core
        + 0.13 * context_clean
        + 0.07 * edge_mid
        + 0.05 * size_pref
    )
    base["bucket"] = "clean_positive_anchors"
    base["selection_reason"] = base.apply(
        lambda r: (
            f"clean_positive_anchor conf={safe_float(r['confidence']):.3f} area={safe_float(r['area_m2']):.1f} "
            f"compact={safe_float(r['compactness']):.3f} fill={safe_float(r['bbox_fill']):.3f} "
            f"inner_dark_blue={safe_float(r['inner_dark_blue']):.3f}"
        ),
        axis=1,
    )
    base["feature_count"] = 1
    base["candidate_feature_conf"] = base["confidence"]
    base["candidate_feature_area_m2"] = base["area_m2"]
    base["candidate_feature_compactness"] = base["compactness"]
    base["candidate_feature_aspect_ratio"] = base["bbox_aspect"]
    return dedupe_best_per_tile(base)


def build_mixed_random_bucket(
    tiles_df: pd.DataFrame,
    used_tile_ids: set[str],
    *,
    count: int,
    seed: int,
    bucket_name: str = "review_mixed_random",
    positive_fraction: float = 0.6,
    excluded_tile_rel_noext: Optional[set[str]] = None,
    excluded_tile_stems: Optional[set[str]] = None,
) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame()
    rng = random.Random(seed)
    base = tiles_df[(tiles_df["blank_white"] == 0)].copy()
    if "tile_id" in base.columns:
        base = base[~base["tile_id"].isin(used_tile_ids)].copy()
    else:
        base = base[~base["tile_rel"].isin(used_tile_ids)].copy()
    if excluded_tile_rel_noext or excluded_tile_stems:
        base = apply_exclusions(
            base,
            excluded_tile_rel_noext=excluded_tile_rel_noext or set(),
            excluded_tile_stems=excluded_tile_stems or set(),
        )
    if base.empty:
        return pd.DataFrame()

    pos = base[(base["num_preds"] >= 1) & (base["num_preds"] <= 4)].copy()
    emp = base[(base["num_preds"] == 0)].copy()
    pos = pos.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    emp = emp.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    target_pos = min(len(pos), int(round(count * positive_fraction)))
    target_emp = min(len(emp), count - target_pos)
    selected = pd.concat([pos.head(target_pos), emp.head(target_emp)], ignore_index=True)
    if len(selected) < count:
        if "tile_id" in selected.columns:
            rest = base[~base["tile_id"].isin(selected["tile_id"].tolist())].copy()
        else:
            rest = base[~base["tile_rel"].isin(selected["tile_rel"].tolist())].copy()
        rest = rest.sample(frac=1.0, random_state=seed + 2)
        selected = pd.concat([selected, rest.head(count - len(selected))], ignore_index=True)

    # Cell-diverse pass.
    by_cell: Dict[str, List[int]] = defaultdict(list)
    for idx, row in selected.iterrows():
        by_cell[str(row["cell"])].append(idx)
    keys = list(by_cell.keys())
    rng.shuffle(keys)
    round_robin_idx: List[int] = []
    while keys and len(round_robin_idx) < min(count, len(selected)):
        next_keys: List[str] = []
        for k in keys:
            arr = by_cell.get(k, [])
            if not arr:
                continue
            round_robin_idx.append(arr.pop(0))
            if arr:
                next_keys.append(k)
            if len(round_robin_idx) >= count:
                break
        keys = next_keys

    selected = selected.iloc[round_robin_idx].copy().reset_index(drop=True)
    selected["selection_score"] = np.linspace(1.0, 0.5, num=len(selected), endpoint=True)
    selected["bucket"] = bucket_name
    selected["selection_reason"] = selected.apply(
        lambda r: (
            f"mixed_random_sanity num_preds={parse_int(r['num_preds'])} "
            f"mean_conf={safe_float(r['mean_conf']):.3f}"
        ),
        axis=1,
    )
    selected["feature_count"] = ""
    selected["candidate_feature_conf"] = ""
    selected["candidate_feature_area_m2"] = ""
    selected["candidate_feature_compactness"] = ""
    selected["candidate_feature_aspect_ratio"] = ""
    selected["overlap_pairs"] = ""
    selected["near_pairs"] = ""
    selected["ring_dark_blue"] = ""
    selected["ring_bright_blue"] = ""
    selected["ring_gray"] = ""
    selected["ring_green"] = ""
    selected["ring_edge"] = ""
    selected["pool_component_score"] = ""
    selected["neighbor_preds_sum"] = ""
    selected["neighbor_positive_tiles"] = ""
    selected["context_score"] = ""
    return selected


def enrich_from_tiles(selected: pd.DataFrame, tiles_df: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    tiles_cols = [
        "tile_id",
        "tile_rel",
        "tile_rel_canonical",
        "tile_path_abs",
        "tile_stem",
        "cell",
        "num_preds",
        "min_conf",
        "mean_conf",
        "max_conf",
        "sum_area_m2",
        "max_area_m2",
    ]
    tiles_small = tiles_df[tiles_cols].copy()
    out = attach_tile_ids(selected.copy())
    join_col = "tile_id" if "tile_id" in out.columns else "tile_rel"
    if "num_preds" not in out.columns:
        out = out.merge(tiles_small, on=join_col, how="left", suffixes=("", "_tile"))
    else:
        merge_extra = tiles_small.rename(columns={c: f"{c}_tile" for c in tiles_cols if c != join_col})
        out = out.merge(merge_extra, on=join_col, how="left")
        for c in ("tile_rel", "tile_rel_canonical", "tile_path_abs", "tile_stem", "cell", "num_preds", "min_conf", "mean_conf", "max_conf", "sum_area_m2", "max_area_m2"):
            tile_c = f"{c}_tile"
            if tile_c in out.columns:
                out[c] = out[c].where(out[c].notna(), out[tile_c])
                out.drop(columns=[tile_c], inplace=True)
    if "tile_rel_canonical" in out.columns:
        out["tile_rel"] = out["tile_rel_canonical"].where(out["tile_rel_canonical"].notna(), out["tile_rel"])
    if "tile_rel" in out.columns and "tile_id" in out.columns:
        fallback_rel = out["tile_id"].map(lambda x: canonical_tile_rel_from_id(str(x)))
        out["tile_rel"] = out["tile_rel"].where(out["tile_rel"].notna(), fallback_rel)
    return out


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bucket",
        "rank",
        "selection_score",
        "selection_reason",
        "tile_id",
        "tile_rel",
        "tile_path_abs",
        "copied_path",
        "cell",
        "tile_stem",
        "num_preds",
        "min_conf",
        "mean_conf",
        "max_conf",
        "sum_area_m2",
        "max_area_m2",
        "feature_count",
        "candidate_feature_conf",
        "candidate_feature_area_m2",
        "candidate_feature_compactness",
        "candidate_feature_aspect_ratio",
        "bbox_fill",
        "overlap_pairs",
        "near_pairs",
        "inner_blue",
        "inner_dark_blue",
        "ring_dark_blue",
        "ring_bright_blue",
        "inner_green",
        "ring_gray",
        "ring_green",
        "inner_dark",
        "ring_dark",
        "inner_edge",
        "ring_edge",
        "inner_luma_std",
        "pool_component_score",
        "neighbor_preds_sum",
        "neighbor_positive_tiles",
        "context_score",
        "nearest_larger_gap_m",
        "nearest_larger_centroid_m",
        "nearest_larger_area_ratio",
        "nearest_larger_iou",
        "nearest_larger_overlap_small",
        "nearest_larger_conf",
        "nearest_larger_area_m2",
        "larger_neighbor_count",
        "tile_feature_count",
    ]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out[cols]


def materialize_tile(
    *,
    tile_path_abs: str,
    tile_id: str,
    tile_rel: str,
    tiles_root: Path,
    bucket_dir: Path,
    mode: str,
) -> str:
    src = Path(tile_path_abs)
    canonical_rel = canonical_tile_rel_from_id(tile_id)
    rel_for_dst = canonical_rel if canonical_rel else str(tile_rel)
    if not src.exists() and canonical_rel:
        fallback_src = tiles_root / canonical_rel
        if fallback_src.exists():
            src = fallback_src
    if not src.exists():
        raise FileNotFoundError(f"Missing tile source for tile_id={tile_id}: {tile_path_abs}")

    dst = bucket_dir / "tiles" / Path(rel_for_dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return str(dst)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src, start=dst.parent)
        os.symlink(rel_src, dst)
    return str(dst)


def write_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def bucket_logic_lines(profile: str) -> List[str]:
    if profile == PROFILE_V6_CORRECTIVE:
        return [
            "- `hard_negatives_tree_shadows`: low-confidence, medium/large shadow-like detections in blue/green textured regions.",
            "- `missed_obvious_pools`: zero-prediction tiles with strong pool-likelihood/context signals.",
            "- `duplicate_fragmented_real_pools`: tiles with clustered overlapping/nearby detections suggesting fragmentation.",
            "- `hard_negatives_blue_roofs`: compact blue rectangular urban false positives.",
            "- `review_mixed_small`: compact deterministic sample from remaining unused tiles.",
        ]
    if profile == PROFILE_V5_CORRECTIVE:
        return [
            "- `hard_negatives_tree_shadows`: dark/smooth canopy-shadow detections with blue-green shadow tone and pool-like geometry.",
            "- `hard_negatives_pool_edge_shadows`: detections near larger true-pool candidates that look like edge-only/deck-shadow fragments.",
            "- `clean_positive_anchors`: high-confidence, high-shape-quality full pools for stable positive anchoring.",
            "- `review_mixed_random_small`: small mixed sanity sample from unseen tiles after targeted selection.",
        ]
    return [
        "- `hard_negatives_shorelines`: large/irregular detections with water-edge style local context (dark-blue ring + strong edge energy).",
        "- `hard_negatives_courts`: medium/large rectangular detections with court-like geometry and structured surroundings.",
        "- `hard_negatives_blue_roofs`: compact rectangular detections with bright-blue core + urban (gray) ring context.",
        "- `duplicate_fragmented_real_pools`: tiles with overlapping/nearby prediction pairs suggesting split/duplicate polygons.",
        "- `missed_obvious_pools`: zero-pred tiles in strong positive neighborhoods plus blue component cues.",
        "- `review_mixed_random`: small sanity bucket from remaining unseen tiles.",
    ]


def write_readme(
    *,
    profile: str,
    buckets: Sequence[str],
    out_dir: Path,
    run_dir: Path,
    tiles_root: Path,
    bucket_counts: Dict[str, int],
    params: Dict[str, object],
) -> None:
    lines = [
        "# Google z21 Targeted Annotation Batch",
        "",
        f"- Created: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Profile: `{profile}`",
        f"- Run source: `{run_dir}`",
        f"- Imagery root: `{tiles_root}`",
        "",
        "## Bucket logic",
        "",
    ]
    lines.extend(bucket_logic_lines(profile))
    lines.extend(["", "## Counts", ""])
    for b in buckets:
        lines.append(f"- `{b}`: {bucket_counts.get(b, 0)}")
    lines.extend(
        [
            "",
            "## Parameters",
            "",
            "```json",
            json.dumps(params, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    active_profile = str(args.profile)
    active_buckets = bucket_order(active_profile)

    run_dir = args.run_dir.expanduser().resolve()
    tiles_root = args.tiles_root.expanduser().resolve()
    out_base = args.out_base.expanduser().resolve()

    pools_geojson = run_dir / "pools.geojson"
    if not run_dir.exists():
        raise SystemExit(f"--run-dir not found: {run_dir}")
    if not tiles_root.exists():
        raise SystemExit(f"--tiles-root not found: {tiles_root}")
    if not pools_geojson.exists():
        raise SystemExit(f"Missing file: {pools_geojson}")
    tiles_csv = resolve_tile_csv_path(run_dir)

    batch_prefix = str(args.batch_prefix)
    if active_profile == PROFILE_V6_CORRECTIVE and batch_prefix == "google_z21_round_next":
        batch_prefix = "z21_v6"
    elif active_profile == PROFILE_V5_CORRECTIVE and batch_prefix == "google_z21_round_next":
        batch_prefix = "google_z21_v5_corrective"

    stamp = args.timestamp.strip() or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"{batch_prefix}_{stamp}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output folder already exists: {out_dir} (use --overwrite or a new --timestamp).")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading run artifacts from: {run_dir}")
    tiles_df, tile_csv_dedupe_stats = load_tile_csv(tiles_csv)
    features = load_feature_table(pools_geojson)

    exclusion_dataset_roots = [Path(p) for p in args.exclude_dataset_roots]
    exclusion_image_roots = [Path(p) for p in args.exclude_image_roots]
    scan_roots = infer_exclusion_image_roots(exclusion_dataset_roots, exclusion_image_roots)
    excluded_tile_rel_noext: set[str] = set()
    excluded_tile_stems: set[str] = set()
    exclusion_stats: Dict[str, object] = {
        "dataset_roots": [str(p.expanduser()) for p in exclusion_dataset_roots],
        "image_roots": [str(p.expanduser()) for p in exclusion_image_roots],
        "resolved_image_roots": [str(p) for p in scan_roots],
        "roots_scanned": 0,
        "images_scanned": 0,
        "excluded_tile_rel_noext": 0,
        "excluded_tile_stems": 0,
        "tiles_removed_from_run": 0,
        "features_removed_from_run": 0,
    }
    if scan_roots:
        excluded_tile_rel_noext, excluded_tile_stems, raw_excl_stats = collect_excluded_tile_keys(scan_roots)
        exclusion_stats.update(raw_excl_stats)
        if excluded_tile_rel_noext or excluded_tile_stems:
            before_tiles = len(tiles_df)
            before_features = len(features)
            tiles_df = apply_exclusions(
                tiles_df,
                excluded_tile_rel_noext=excluded_tile_rel_noext,
                excluded_tile_stems=excluded_tile_stems,
            )
            features = apply_exclusions(
                features,
                excluded_tile_rel_noext=excluded_tile_rel_noext,
                excluded_tile_stems=excluded_tile_stems,
            )
            exclusion_stats["tiles_removed_from_run"] = int(before_tiles - len(tiles_df))
            exclusion_stats["features_removed_from_run"] = int(before_features - len(features))
            print(
                "[1/6] Exclusion filter:"
                f" removed_tiles={exclusion_stats['tiles_removed_from_run']}"
                f" removed_features={exclusion_stats['features_removed_from_run']}"
            )

    cache = CacheBundle(image=OrderedDict(), world={}, missed_pool_signal={})
    used_tile_ids: set[str] = set()
    dedupe_runtime_stats: Dict[str, int] = {
        "tiles_csv_duplicates_prevented": int(tile_csv_dedupe_stats.get("duplicates_prevented", 0)),
        "cross_bucket_collisions_prevented": 0,
        "missing_tile_id_rows_skipped": 0,
    }
    selected_frames: Dict[str, pd.DataFrame] = {}
    if active_profile == PROFILE_V6_CORRECTIVE:
        print("[2/6] Mining v6 tree-shadow hard negatives")
        tree_candidates = build_tree_shadow_candidates(features, cache)
        tree_candidates = enrich_from_tiles(tree_candidates, tiles_df)
        tree_selected = select_with_cell_limits(
            tree_candidates,
            count=args.tree_shadows_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=4,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["hard_negatives_tree_shadows"] = tree_selected

        print("[3/6] Mining v6 missed obvious pools")
        missed_candidates = build_missed_candidates(
            tiles_df,
            cache,
            preselect_limit=max(1200, int(args.missed_count * 25)),
        )
        missed_candidates = enrich_from_tiles(missed_candidates, tiles_df)
        missed_selected = select_with_cell_limits(
            missed_candidates,
            count=args.missed_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=2,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["missed_obvious_pools"] = missed_selected

        print("[4/6] Mining v6 duplicate/fragmented real-pool fixes")
        dup_candidates = build_duplicate_fragment_candidates(features, tiles_df)
        dup_candidates = enrich_from_tiles(dup_candidates, tiles_df)
        dup_selected = select_with_cell_limits(
            dup_candidates,
            count=args.duplicate_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=3,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["duplicate_fragmented_real_pools"] = dup_selected

        print("[5/6] Mining v6 hard negatives (blue roofs)")
        roof_candidates = build_blue_roof_candidates(features, cache)
        roof_candidates = enrich_from_tiles(roof_candidates, tiles_df)
        roof_selected = select_with_cell_limits(
            roof_candidates,
            count=args.blue_roofs_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=3,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["hard_negatives_blue_roofs"] = roof_selected

        print("[5/6] Sampling v6 mixed sanity bucket")
        mixed = build_mixed_random_bucket(
            tiles_df,
            used_tile_ids,
            count=args.mixed_random_small_count,
            seed=args.seed,
            bucket_name="review_mixed_small",
            positive_fraction=0.60,
            excluded_tile_rel_noext=excluded_tile_rel_noext,
            excluded_tile_stems=excluded_tile_stems,
        )
        mixed = enrich_from_tiles(mixed, tiles_df)
        mixed = select_with_cell_limits(
            mixed,
            count=args.mixed_random_small_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=2,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["review_mixed_small"] = mixed
    elif active_profile == PROFILE_V5_CORRECTIVE:
        print("[2/6] Mining v5 tree-shadow hard negatives")
        tree_candidates = build_tree_shadow_candidates(features, cache)
        tree_candidates = enrich_from_tiles(tree_candidates, tiles_df)
        tree_selected = select_with_cell_limits(
            tree_candidates,
            count=args.tree_shadows_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=4,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["hard_negatives_tree_shadows"] = tree_selected

        print("[3/6] Mining v5 pool-edge-shadow hard negatives")
        edge_candidates = build_pool_edge_shadow_candidates(features, cache)
        edge_candidates = enrich_from_tiles(edge_candidates, tiles_df)
        edge_selected = select_with_cell_limits(
            edge_candidates,
            count=args.pool_edge_shadows_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=4,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["hard_negatives_pool_edge_shadows"] = edge_selected

        print("[4/6] Mining v5 clean positive anchors")
        anchor_candidates = build_clean_positive_anchor_candidates(features, tiles_df, cache)
        anchor_candidates = enrich_from_tiles(anchor_candidates, tiles_df)
        anchor_selected = select_with_cell_limits(
            anchor_candidates,
            count=args.clean_positive_anchors_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=3,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["clean_positive_anchors"] = anchor_selected

        print("[5/6] Sampling small mixed sanity bucket")
        mixed = build_mixed_random_bucket(
            tiles_df,
            used_tile_ids,
            count=args.mixed_random_small_count,
            seed=args.seed,
            bucket_name="review_mixed_random_small",
            positive_fraction=0.65,
            excluded_tile_rel_noext=excluded_tile_rel_noext,
            excluded_tile_stems=excluded_tile_stems,
        )
        mixed = enrich_from_tiles(mixed, tiles_df)
        mixed = select_with_cell_limits(
            mixed,
            count=args.mixed_random_small_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=2,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["review_mixed_random_small"] = mixed
    else:
        print("[2/6] Mining targeted hard-negative buckets")
        shoreline_candidates = build_shoreline_candidates(features, cache)
        shoreline_candidates = enrich_from_tiles(shoreline_candidates, tiles_df)
        shoreline_selected = select_with_cell_limits(
            shoreline_candidates,
            count=args.shorelines_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=4,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["hard_negatives_shorelines"] = shoreline_selected

        court_candidates = build_court_candidates(features, cache)
        court_candidates = enrich_from_tiles(court_candidates, tiles_df)
        court_selected = select_with_cell_limits(
            court_candidates,
            count=args.courts_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=4,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["hard_negatives_courts"] = court_selected

        roof_candidates = build_blue_roof_candidates(features, cache)
        roof_candidates = enrich_from_tiles(roof_candidates, tiles_df)
        roof_selected = select_with_cell_limits(
            roof_candidates,
            count=args.blue_roofs_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=4,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["hard_negatives_blue_roofs"] = roof_selected

        print("[3/6] Mining duplicate/fragmented real-pool fixes")
        dup_candidates = build_duplicate_fragment_candidates(features, tiles_df)
        dup_candidates = enrich_from_tiles(dup_candidates, tiles_df)
        dup_selected = select_with_cell_limits(
            dup_candidates,
            count=args.duplicate_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=3,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["duplicate_fragmented_real_pools"] = dup_selected

        print("[4/6] Mining missed obvious pools")
        missed_candidates = build_missed_candidates(
            tiles_df,
            cache,
            preselect_limit=max(1200, int(args.missed_count * 25)),
        )
        missed_candidates = enrich_from_tiles(missed_candidates, tiles_df)
        missed_selected = select_with_cell_limits(
            missed_candidates,
            count=args.missed_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=2,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["missed_obvious_pools"] = missed_selected

        print("[5/6] Sampling mixed random sanity bucket")
        mixed = build_mixed_random_bucket(
            tiles_df,
            used_tile_ids,
            count=args.mixed_random_count,
            seed=args.seed,
            excluded_tile_rel_noext=excluded_tile_rel_noext,
            excluded_tile_stems=excluded_tile_stems,
        )
        mixed = enrich_from_tiles(mixed, tiles_df)
        mixed = select_with_cell_limits(
            mixed,
            count=args.mixed_random_count,
            used_tile_ids=used_tile_ids,
            per_cell_limit=2,
            dedupe_stats=dedupe_runtime_stats,
        )
        selected_frames["review_mixed_random"] = mixed

    print("[6/6] Materializing files and writing manifests")
    combined_rows: List[pd.DataFrame] = []
    bucket_counts: Dict[str, int] = {}

    for bucket in active_buckets:
        frame = selected_frames.get(bucket, pd.DataFrame()).copy()
        if frame.empty:
            frame = pd.DataFrame(columns=["tile_id", "tile_rel", "tile_path_abs"])
        frame = attach_tile_ids(frame)
        if "tile_rel" in frame.columns and "tile_id" in frame.columns:
            fallback_rel = frame["tile_id"].map(lambda x: canonical_tile_rel_from_id(str(x)))
            frame["tile_rel"] = frame["tile_rel"].where(frame["tile_rel"].notna(), fallback_rel)
        frame["bucket"] = bucket
        if "rank" not in frame.columns:
            frame["rank"] = np.arange(1, len(frame) + 1)
        if "selection_score" not in frame.columns:
            frame["selection_score"] = ""
        if "selection_reason" not in frame.columns:
            frame["selection_reason"] = ""

        copied_paths: List[str] = []
        for r in frame.itertuples(index=False):
            rel = str(r.tile_rel)
            tile_id = str(getattr(r, "tile_id", ""))
            src = str(r.tile_path_abs)
            copied_paths.append(
                materialize_tile(
                    tile_path_abs=src,
                    tile_id=tile_id,
                    tile_rel=rel,
                    tiles_root=tiles_root,
                    bucket_dir=out_dir / bucket,
                    mode=args.copy_mode,
                )
            )
        frame["copied_path"] = copied_paths
        frame = ensure_columns(frame)
        frame = frame.sort_values("rank").reset_index(drop=True)

        bucket_manifest = out_dir / bucket / "manifest.csv"
        write_csv(bucket_manifest, frame)
        combined_rows.append(frame)
        bucket_counts[bucket] = int(len(frame))

    combined = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    combined = ensure_columns(combined)
    if "tile_id" in combined.columns:
        dup_mask = combined["tile_id"].astype(str).duplicated(keep=False)
        dup_ids = sorted(set(combined.loc[dup_mask, "tile_id"].astype(str).tolist()))
        if dup_ids:
            preview = ", ".join(dup_ids[:10])
            raise SystemExit(f"Duplicate tile_id detected in final batch (count={len(dup_ids)}): {preview}")
    total_unique_tiles = int(combined["tile_id"].nunique()) if "tile_id" in combined.columns else int(len(combined))
    sample_tile_ids = combined["tile_id"].dropna().astype(str).head(8).tolist() if "tile_id" in combined.columns else []
    write_csv(out_dir / "combined_summary.csv", combined)

    params = {
        "profile": active_profile,
        "run_dir": str(run_dir),
        "tile_summary_csv": str(tiles_csv),
        "tiles_root": str(tiles_root),
        "batch_prefix": batch_prefix,
        "copy_mode": args.copy_mode,
        "seed": args.seed,
        "targets": (
            {
                "hard_negatives_tree_shadows": args.tree_shadows_count,
                "missed_obvious_pools": args.missed_count,
                "duplicate_fragmented_real_pools": args.duplicate_count,
                "hard_negatives_blue_roofs": args.blue_roofs_count,
                "review_mixed_small": args.mixed_random_small_count,
            }
            if active_profile == PROFILE_V6_CORRECTIVE
            else (
            {
                "hard_negatives_tree_shadows": args.tree_shadows_count,
                "hard_negatives_pool_edge_shadows": args.pool_edge_shadows_count,
                "clean_positive_anchors": args.clean_positive_anchors_count,
                "review_mixed_random_small": args.mixed_random_small_count,
            }
            if active_profile == PROFILE_V5_CORRECTIVE
            else {
                "hard_negatives_shorelines": args.shorelines_count,
                "hard_negatives_courts": args.courts_count,
                "hard_negatives_blue_roofs": args.blue_roofs_count,
                "duplicate_fragmented_real_pools": args.duplicate_count,
                "missed_obvious_pools": args.missed_count,
                "review_mixed_random": args.mixed_random_count,
            }
            )
        ),
        "tile_id_dedupe": {
            **tile_csv_dedupe_stats,
            **dedupe_runtime_stats,
            "total_duplicates_prevented": int(
                dedupe_runtime_stats.get("tiles_csv_duplicates_prevented", 0)
                + dedupe_runtime_stats.get("cross_bucket_collisions_prevented", 0)
            ),
        },
        "exclusions": exclusion_stats,
        "selected": bucket_counts,
        "total_unique_tiles_selected": total_unique_tiles,
        "sample_tile_ids": sample_tile_ids,
    }
    (out_dir / "summary.json").write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(
        profile=active_profile,
        buckets=active_buckets,
        out_dir=out_dir,
        run_dir=run_dir,
        tiles_root=tiles_root,
        bucket_counts=bucket_counts,
        params=params,
    )

    print("Wrote batch:", out_dir)
    print(f"total candidates scanned: {int(tile_csv_dedupe_stats.get('rows_in', len(tiles_df)))}")
    print(f"skipped (already labeled): {int(exclusion_stats.get('tiles_removed_from_run', 0))}")
    print(f"total_unique_tiles_selected: {total_unique_tiles}")
    for b in active_buckets:
        print(f"{b}: {bucket_counts.get(b, 0)}")
    print("skipped (duplicate in batch):", int(dedupe_runtime_stats.get("cross_bucket_collisions_prevented", 0)))
    print("final selected count:", total_unique_tiles)
    print("duplicates_prevented:", int(dedupe_runtime_stats.get("tiles_csv_duplicates_prevented", 0)) + int(dedupe_runtime_stats.get("cross_bucket_collisions_prevented", 0)))
    print("sample_tile_ids:", ", ".join(sample_tile_ids) if sample_tile_ids else "(none)")


if __name__ == "__main__":
    main()
