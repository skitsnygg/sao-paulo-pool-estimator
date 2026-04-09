#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from shapely.geometry import shape
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(f"Missing dependency shapely: {exc}")


TILE_ID_PATTERN = r"cell_\d+_\d+__r\d+_c\d+"
TILE_ID_RE = re.compile(rf"({TILE_ID_PATTERN})")
CELL_STEM_RE = re.compile(r"(cell_\d+_\d+)[/\\](r\d+_c\d+)")
CELL_RE = re.compile(r"^cell_\d+_\d+$")
STEM_RE = re.compile(r"^r\d+_c\d+$")
RC_RE = re.compile(r"^r(?P<row>\d+)_c(?P<col>\d+)$")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")

BUCKET_A = "high_fp_density"
BUCKET_B = "borderline_confidence"
BUCKET_C = "large_or_elongated"
BUCKET_D = "random_control"
BUCKET_ORDER = (BUCKET_A, BUCKET_B, BUCKET_C, BUCKET_D)


@dataclass
class TileStats:
    tile_id: str
    tile_rel: str
    tile_stem: str
    cell: str
    tile_path_abs: str
    num_polys: int = 0
    sum_conf: float = 0.0
    avg_conf: float = 0.0
    min_conf: float = 999.0
    max_conf: float = 0.0
    max_area_m2: float = 0.0
    max_elongation: float = 1.0
    borderline_count: int = 0
    neighbor_poly_count: int = 0
    weird_score: float = 0.0


@dataclass
class SelectionRow:
    tile_id: str
    bucket: str
    num_polys: int
    avg_conf: float
    max_area_m2: float
    tile_rel: str
    tile_stem: str
    cell: str
    tile_path_abs: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run z21 inference and build a targeted annotation batch focused on false positives "
            "(density, borderline confidence, large/elongated shapes, plus random control)."
        )
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/Users/admin/sao-paulo-pool-estimator/runs/segment/runs/segment/"
            "z21_v1_reannotated_20260408_071650/weights/best.pt"
        ),
    )
    ap.add_argument(
        "--tiles-dir",
        type=Path,
        default=Path("data/raw/google/sp_city_2020_rebuild_google_z21"),
    )
    ap.add_argument("--run-dir", type=Path, default=None, help="Inference output directory. Default is timestamped.")
    ap.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("data/annotation_batches/z21_v6_targeted"),
        help="Directory for selected tile list/CSV/review folder.",
    )
    ap.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/datasets/google_z21_v5/images/val"),
        help="Validation images directory used for exclusion.",
    )
    ap.add_argument("--target-total", type=int, default=360, help="Final target count (recommended 300-400).")
    ap.add_argument("--seed", type=int, default=20260408)

    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--min-area-m2", type=float, default=6.0)
    ap.add_argument("--worldfile-crs", default="EPSG:31983")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-det", type=int, default=300)

    ap.add_argument("--high-density-min", type=int, default=3)
    ap.add_argument("--borderline-low", type=float, default=0.08)
    ap.add_argument("--borderline-high", type=float, default=0.14)
    ap.add_argument("--large-area-m2", type=float, default=45.0)
    ap.add_argument("--huge-area-m2", type=float, default=120.0)
    ap.add_argument("--elongated-ratio", type=float, default=4.0)
    ap.add_argument("--very-elongated-ratio", type=float, default=7.0)
    ap.add_argument("--neighbor-poly-threshold", type=int, default=8)
    ap.add_argument("--weird-score-threshold", type=float, default=2.0)

    ap.add_argument("--skip-inference", action="store_true", help="Use existing outputs in --run-dir.")
    ap.add_argument("--review-mode", choices=("symlink", "copy", "none"), default="symlink")
    ap.add_argument("--force", action="store_true", help="Overwrite existing batch output artifacts.")
    return ap.parse_args()


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except Exception:
        return default


def normalize_tile_stem(value: Any) -> str:
    if value is None:
        return ""
    stem = str(value).strip()
    if not stem:
        return ""
    if "/" in stem or "\\" in stem:
        stem = Path(stem).name
    return Path(stem).stem


def canonical_tile_id(cell: Any, tile_stem: Any) -> str:
    c = str(cell or "").strip()
    s = normalize_tile_stem(tile_stem)
    if CELL_RE.fullmatch(c) and STEM_RE.fullmatch(s):
        return f"{c}__{s}"
    return ""


def canonical_rel_from_tile_id(tile_id: str, *, ext: str = ".png") -> str:
    tid = str(tile_id).strip()
    if not re.fullmatch(TILE_ID_PATTERN, tid):
        return ""
    cell, stem = tid.split("__", 1)
    return f"{cell}/{stem}{ext}"


def extract_tile_id_from_text(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    raw_noext = Path(raw).with_suffix("").as_posix()
    m = TILE_ID_RE.search(raw_noext)
    if m:
        return m.group(1)
    m2 = CELL_STEM_RE.search(raw_noext)
    if m2:
        return f"{m2.group(1)}__{m2.group(2)}"
    return ""


def extract_tile_id_from_mapping(row: Dict[str, Any]) -> str:
    for key in ("tile_id", "tile_rel", "tile_path_abs", "tile", "filename", "path"):
        if key in row:
            tid = extract_tile_id_from_text(row.get(key))
            if tid:
                return tid
    return canonical_tile_id(row.get("cell"), row.get("tile_stem"))


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path("runs/segment") / f"z21_v6_targeted_precision_{ts}").resolve()


def ensure_exists(path: Path, *, kind: str) -> None:
    if kind == "file" and not path.is_file():
        raise SystemExit(f"Missing required file: {path}")
    if kind == "dir" and not path.is_dir():
        raise SystemExit(f"Missing required directory: {path}")


def run_inference(args: argparse.Namespace, run_dir: Path) -> Dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)

    out_geojson = run_dir / "pools.geojson"
    out_geojson_3857 = run_dir / "pools_3857.geojson"
    out_tiles_csv = run_dir / "pools_tiles.csv"
    out_tiles_jsonl = run_dir / "pools_tiles.jsonl"
    out_stats = run_dir / "pools_stats.json"

    if args.skip_inference:
        ensure_exists(out_geojson, kind="file")
        if not out_tiles_jsonl.exists() and not out_tiles_csv.exists():
            raise SystemExit(
                "Missing tile summary output in run dir for --skip-inference: "
                f"expected {out_tiles_jsonl} or {out_tiles_csv}"
            )
        return {
            "pools_geojson": out_geojson,
            "pools_geojson_3857": out_geojson_3857,
            "pools_tiles_csv": out_tiles_csv,
            "pools_tiles_jsonl": out_tiles_jsonl,
            "pools_stats_json": out_stats,
        }

    repo_root = Path(__file__).resolve().parents[1]
    predict_script = repo_root / "tools" / "predict_tiles_to_geojson.py"
    ensure_exists(predict_script, kind="file")
    ensure_exists(args.model.resolve(), kind="file")
    ensure_exists(args.tiles_dir.resolve(), kind="dir")

    cmd = [
        sys.executable,
        str(predict_script),
        "--model",
        str(args.model.resolve()),
        "--tiles-dir",
        str(args.tiles_dir.resolve()),
        "--out-geojson",
        str(out_geojson),
        "--out-geojson-3857",
        str(out_geojson_3857),
        "--out-tile-summary-csv",
        str(out_tiles_csv),
        "--out-tile-summary-jsonl",
        str(out_tiles_jsonl),
        "--out-stats-json",
        str(out_stats),
        "--imgsz",
        str(int(args.imgsz)),
        "--conf",
        str(float(args.conf)),
        "--iou",
        str(float(args.iou)),
        "--min-area-m2",
        str(float(args.min_area_m2)),
        "--worldfile-crs",
        str(args.worldfile_crs),
        "--device",
        str(args.device),
        "--max-det",
        str(int(args.max_det)),
        "--retina-masks",
    ]
    print("[1/4] Running inference")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Inference failed with exit code {exc.returncode}")

    for p in (out_geojson, out_tiles_jsonl, out_stats):
        ensure_exists(p, kind="file")
    return {
        "pools_geojson": out_geojson,
        "pools_geojson_3857": out_geojson_3857,
        "pools_tiles_csv": out_tiles_csv,
        "pools_tiles_jsonl": out_tiles_jsonl,
        "pools_stats_json": out_stats,
    }


def load_validation_ids(val_dir: Path) -> set[str]:
    ensure_exists(val_dir, kind="dir")
    out: set[str] = set()
    for p in val_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        tile_id = extract_tile_id_from_text(p.stem)
        if not tile_id:
            tile_id = extract_tile_id_from_text(str(p))
        if tile_id:
            out.add(tile_id)
    return out


def load_tile_meta(tile_jsonl: Path, tile_csv: Path) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    if tile_jsonl.exists():
        with tile_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tile_id = extract_tile_id_from_mapping(row)
                if not tile_id:
                    continue
                if tile_id not in meta:
                    tile_rel = str(row.get("tile_rel") or "").strip()
                    tile_stem = normalize_tile_stem(row.get("tile_stem") or row.get("tile"))
                    cell = str(row.get("cell") or "").strip()
                    tile_path_abs = str(row.get("tile_path_abs") or "").strip()
                    if not tile_rel:
                        tile_rel = canonical_rel_from_tile_id(tile_id)
                    if not cell or not tile_stem:
                        if "__" in tile_id:
                            cell2, stem2 = tile_id.split("__", 1)
                            cell = cell or cell2
                            tile_stem = tile_stem or stem2
                    meta[tile_id] = {
                        "tile_rel": tile_rel,
                        "tile_stem": tile_stem,
                        "cell": cell,
                        "tile_path_abs": tile_path_abs,
                    }
        return meta

    if tile_csv.exists():
        with tile_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                tile_id = extract_tile_id_from_mapping(row)
                if not tile_id or tile_id in meta:
                    continue
                tile_rel = str(row.get("tile_rel") or "").strip()
                tile_stem = normalize_tile_stem(row.get("tile_stem") or row.get("tile"))
                cell = str(row.get("cell") or "").strip()
                tile_path_abs = str(row.get("tile_path_abs") or "").strip()
                if not tile_rel:
                    tile_rel = canonical_rel_from_tile_id(tile_id)
                if not cell or not tile_stem:
                    if "__" in tile_id:
                        cell2, stem2 = tile_id.split("__", 1)
                        cell = cell or cell2
                        tile_stem = tile_stem or stem2
                meta[tile_id] = {
                    "tile_rel": tile_rel,
                    "tile_stem": tile_stem,
                    "cell": cell,
                    "tile_path_abs": tile_path_abs,
                }
    if not meta:
        raise SystemExit(
            "Could not read any tile metadata from summary files: "
            f"{tile_jsonl} / {tile_csv}"
        )
    return meta


def iter_geojson_features(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid GeoJSON object: {path}")
    if payload.get("type") != "FeatureCollection":
        raise SystemExit(f"Expected FeatureCollection GeoJSON: {path}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise SystemExit(f"GeoJSON missing features[]: {path}")
    for feat in features:
        if isinstance(feat, dict):
            yield feat


def feature_elongation(feature: Dict[str, Any]) -> float:
    geom = feature.get("geometry")
    if not geom:
        return 1.0
    g = shape(geom)
    if g.is_empty:
        return 1.0
    minx, miny, maxx, maxy = g.bounds
    w = max(0.0, float(maxx) - float(minx))
    h = max(0.0, float(maxy) - float(miny))
    if w == 0.0 and h == 0.0:
        return 1.0
    long_side = max(w, h)
    short_side = max(min(w, h), 1e-6)
    return long_side / short_side


def parse_rc(tile_stem: str) -> Optional[Tuple[int, int]]:
    m = RC_RE.match((tile_stem or "").strip())
    if not m:
        return None
    return int(m.group("row")), int(m.group("col"))


def build_tile_stats(
    pools_geojson: Path,
    tile_meta: Dict[str, Dict[str, str]],
    *,
    borderline_low: float,
    borderline_high: float,
) -> Tuple[Dict[str, TileStats], int]:
    tiles: Dict[str, TileStats] = {}
    skipped_no_tile = 0
    for feat in iter_geojson_features(pools_geojson):
        props = feat.get("properties") if isinstance(feat.get("properties"), dict) else {}
        if not isinstance(props, dict):
            props = {}

        tile_id = extract_tile_id_from_mapping(props)
        if not tile_id:
            skipped_no_tile += 1
            continue

        meta = tile_meta.get(tile_id, {})
        tile_rel = str(props.get("tile_rel") or meta.get("tile_rel") or canonical_rel_from_tile_id(tile_id)).strip()
        tile_stem = normalize_tile_stem(props.get("tile_stem") or meta.get("tile_stem"))
        cell = str(props.get("cell") or meta.get("cell") or "").strip()
        tile_path_abs = str(props.get("tile_path_abs") or meta.get("tile_path_abs") or "").strip()

        if (not cell or not tile_stem) and "__" in tile_id:
            cid, cstem = tile_id.split("__", 1)
            cell = cell or cid
            tile_stem = tile_stem or cstem
        if not tile_rel:
            tile_rel = canonical_rel_from_tile_id(tile_id)

        if tile_id not in tiles:
            tiles[tile_id] = TileStats(
                tile_id=tile_id,
                tile_rel=tile_rel,
                tile_stem=tile_stem,
                cell=cell,
                tile_path_abs=tile_path_abs,
            )
        t = tiles[tile_id]
        if not t.tile_rel and tile_rel:
            t.tile_rel = tile_rel
        if not t.tile_stem and tile_stem:
            t.tile_stem = tile_stem
        if not t.cell and cell:
            t.cell = cell
        if not t.tile_path_abs and tile_path_abs:
            t.tile_path_abs = tile_path_abs

        conf = as_float(props.get("confidence"), default=0.0)
        area_m2 = as_float(props.get("area_m2"), default=0.0)
        elong = feature_elongation(feat)

        t.num_polys += 1
        t.sum_conf += conf
        t.min_conf = min(t.min_conf, conf)
        t.max_conf = max(t.max_conf, conf)
        t.max_area_m2 = max(t.max_area_m2, area_m2)
        t.max_elongation = max(t.max_elongation, elong)
        if borderline_low <= conf <= borderline_high:
            t.borderline_count += 1

    for t in tiles.values():
        t.avg_conf = (t.sum_conf / float(t.num_polys)) if t.num_polys > 0 else 0.0
        if t.min_conf == 999.0:
            t.min_conf = 0.0
    return tiles, skipped_no_tile


def compute_neighbor_density(tiles: Dict[str, TileStats]) -> None:
    by_loc: Dict[Tuple[str, int, int], str] = {}
    for tile_id, t in tiles.items():
        rc = parse_rc(t.tile_stem)
        if rc is None or not t.cell:
            continue
        by_loc[(t.cell, rc[0], rc[1])] = tile_id

    for tile_id, t in tiles.items():
        rc = parse_rc(t.tile_stem)
        if rc is None or not t.cell:
            t.neighbor_poly_count = 0
            continue
        row, col = rc
        neighbor_sum = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                n_id = by_loc.get((t.cell, row + dr, col + dc))
                if n_id is None:
                    continue
                neighbor_sum += int(tiles[n_id].num_polys)
        t.neighbor_poly_count = int(neighbor_sum)


def compute_weird_scores(
    tiles: Dict[str, TileStats],
    *,
    large_area_m2: float,
    huge_area_m2: float,
    elongated_ratio: float,
    very_elongated_ratio: float,
    neighbor_poly_threshold: int,
) -> None:
    for t in tiles.values():
        score = 0.0
        if t.max_area_m2 >= huge_area_m2:
            score += 2.5
        elif t.max_area_m2 >= large_area_m2:
            score += 1.5

        if t.max_elongation >= very_elongated_ratio:
            score += 2.0
        elif t.max_elongation >= elongated_ratio:
            score += 1.0

        if t.neighbor_poly_count >= neighbor_poly_threshold:
            score += 1.0
        if t.num_polys >= 6:
            score += 1.0
        if t.borderline_count >= 2:
            score += 0.5
        t.weird_score = score


def sort_bucket_a(rows: Sequence[TileStats]) -> List[TileStats]:
    return sorted(rows, key=lambda t: (-t.num_polys, t.avg_conf, -t.max_area_m2, t.tile_id))


def sort_bucket_b(rows: Sequence[TileStats]) -> List[TileStats]:
    return sorted(rows, key=lambda t: (-t.borderline_count, t.avg_conf, -t.num_polys, -t.max_area_m2, t.tile_id))


def sort_bucket_c(rows: Sequence[TileStats]) -> List[TileStats]:
    return sorted(
        rows,
        key=lambda t: (-t.weird_score, -t.max_area_m2, -t.max_elongation, -t.neighbor_poly_count, -t.num_polys, t.avg_conf, t.tile_id),
    )


def pick_top(candidates: Sequence[TileStats], quota: int, used: set[str]) -> List[TileStats]:
    out: List[TileStats] = []
    for t in candidates:
        if t.tile_id in used:
            continue
        out.append(t)
        used.add(t.tile_id)
        if len(out) >= quota:
            break
    return out


def resolve_source_tile(t: TileStats, tiles_dir: Path) -> Path:
    if t.tile_path_abs:
        cand = Path(t.tile_path_abs)
        if cand.is_file():
            return cand.resolve()

    if t.cell and t.tile_stem:
        base = (tiles_dir / t.cell / t.tile_stem).resolve()
        for ext in IMAGE_EXTS:
            cand = base.with_suffix(ext)
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"Source tile not found for {t.tile_id} (tile_path_abs='{t.tile_path_abs}')")


def build_selection(
    args: argparse.Namespace,
    tiles: Dict[str, TileStats],
    excluded_val_ids: set[str],
) -> Tuple[List[SelectionRow], Dict[str, int], Dict[str, int]]:
    eligible = [t for t in tiles.values() if t.num_polys > 0 and t.tile_id not in excluded_val_ids]
    if not eligible:
        raise SystemExit("No eligible tiles with detections after validation exclusions.")

    q_a = int(round(args.target_total * 0.40))
    q_b = int(round(args.target_total * 0.30))
    q_c = int(round(args.target_total * 0.20))
    q_d = int(max(0, args.target_total - q_a - q_b - q_c))

    cands_a = sort_bucket_a([t for t in eligible if t.num_polys >= int(args.high_density_min)])
    cands_b = sort_bucket_b([t for t in eligible if t.borderline_count > 0])
    cands_c = sort_bucket_c([t for t in eligible if t.weird_score >= float(args.weird_score_threshold)])

    used_ids: set[str] = set()
    selected: Dict[str, List[TileStats]] = {b: [] for b in BUCKET_ORDER}

    selected[BUCKET_A] = pick_top(cands_a, q_a, used_ids)
    q_b += max(0, q_a - len(selected[BUCKET_A]))

    selected[BUCKET_B] = pick_top(cands_b, q_b, used_ids)
    q_c += max(0, q_b - len(selected[BUCKET_B]))

    selected[BUCKET_C] = pick_top(cands_c, q_c, used_ids)
    q_d += max(0, q_c - len(selected[BUCKET_C]))

    rng = random.Random(int(args.seed))
    rem = sorted([t for t in eligible if t.tile_id not in used_ids], key=lambda t: t.tile_id)
    rng.shuffle(rem)
    selected[BUCKET_D] = rem[:q_d]
    used_ids.update(t.tile_id for t in selected[BUCKET_D])

    total_selected = sum(len(v) for v in selected.values())
    if total_selected < int(args.target_total):
        needed = int(args.target_total) - total_selected
        rem2 = sorted([t for t in eligible if t.tile_id not in used_ids], key=lambda t: (-t.weird_score, -t.num_polys, t.avg_conf, t.tile_id))
        fill = rem2[:needed]
        selected[BUCKET_D].extend(fill)
        used_ids.update(t.tile_id for t in fill)

    rows: List[SelectionRow] = []
    for bucket in BUCKET_ORDER:
        for t in selected[bucket]:
            rows.append(
                SelectionRow(
                    tile_id=t.tile_id,
                    bucket=bucket,
                    num_polys=int(t.num_polys),
                    avg_conf=float(t.avg_conf),
                    max_area_m2=float(t.max_area_m2),
                    tile_rel=t.tile_rel,
                    tile_stem=t.tile_stem,
                    cell=t.cell,
                    tile_path_abs=t.tile_path_abs,
                )
            )

    quotas = {
        BUCKET_A: q_a,
        BUCKET_B: q_b,
        BUCKET_C: q_c,
        BUCKET_D: q_d,
    }
    counts = {bucket: len(selected[bucket]) for bucket in BUCKET_ORDER}
    return rows, counts, quotas


def write_outputs(
    args: argparse.Namespace,
    run_dir: Path,
    batch_dir: Path,
    selected_rows: Sequence[SelectionRow],
    bucket_counts: Dict[str, int],
    quotas: Dict[str, int],
    excluded_val_count: int,
    eligible_count: int,
) -> Dict[str, Path]:
    batch_dir.mkdir(parents=True, exist_ok=True)

    selected_txt = batch_dir / "selected_tiles.txt"
    selected_csv = batch_dir / "selected_tiles_summary.csv"
    manifest_json = batch_dir / "selection_manifest.json"
    review_dir = batch_dir / "review_tiles"

    if (selected_txt.exists() or selected_csv.exists() or manifest_json.exists()) and not args.force:
        raise SystemExit(
            "Batch output files already exist. Use --force to overwrite: "
            f"{selected_txt}, {selected_csv}, {manifest_json}"
        )

    with selected_txt.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(f"{row.tile_id}\n")

    with selected_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_id", "bucket", "num_polys", "avg_conf", "max_area_m2"])
        w.writeheader()
        for row in selected_rows:
            w.writerow(
                {
                    "tile_id": row.tile_id,
                    "bucket": row.bucket,
                    "num_polys": int(row.num_polys),
                    "avg_conf": f"{float(row.avg_conf):.6f}",
                    "max_area_m2": f"{float(row.max_area_m2):.6f}",
                }
            )

    if args.review_mode != "none":
        tiles_dir = args.tiles_dir.resolve()
        if args.force and review_dir.exists():
            for p in review_dir.iterdir():
                if p.is_symlink() or p.is_file():
                    p.unlink()
        review_dir.mkdir(parents=True, exist_ok=True)

        symlink_fallback_to_copy = 0
        for row in selected_rows:
            tile = TileStats(
                tile_id=row.tile_id,
                tile_rel=row.tile_rel,
                tile_stem=row.tile_stem,
                cell=row.cell,
                tile_path_abs=row.tile_path_abs,
            )
            src = resolve_source_tile(tile, tiles_dir)
            out_name = f"{row.tile_id}{src.suffix.lower()}"
            dst = review_dir / out_name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.review_mode == "copy":
                shutil.copy2(src, dst)
            else:
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
                    symlink_fallback_to_copy += 1
        if args.review_mode == "symlink" and symlink_fallback_to_copy > 0:
            print(f"[warn] {symlink_fallback_to_copy} files copied because symlink creation failed.")

    manifest = {
        "run_dir": str(run_dir.resolve()),
        "batch_dir": str(batch_dir.resolve()),
        "selected_count": len(selected_rows),
        "eligible_count": int(eligible_count),
        "excluded_validation_count": int(excluded_val_count),
        "target_total": int(args.target_total),
        "bucket_counts": {b: int(bucket_counts.get(b, 0)) for b in BUCKET_ORDER},
        "bucket_quotas": {b: int(quotas.get(b, 0)) for b in BUCKET_ORDER},
        "review_mode": args.review_mode,
        "inference": {
            "model": str(args.model.resolve()),
            "tiles_dir": str(args.tiles_dir.resolve()),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "imgsz": int(args.imgsz),
            "min_area_m2": float(args.min_area_m2),
            "worldfile_crs": str(args.worldfile_crs),
            "device": str(args.device),
            "retina_masks": True,
        },
        "heuristics": {
            "high_density_min": int(args.high_density_min),
            "borderline_low": float(args.borderline_low),
            "borderline_high": float(args.borderline_high),
            "large_area_m2": float(args.large_area_m2),
            "huge_area_m2": float(args.huge_area_m2),
            "elongated_ratio": float(args.elongated_ratio),
            "very_elongated_ratio": float(args.very_elongated_ratio),
            "neighbor_poly_threshold": int(args.neighbor_poly_threshold),
            "weird_score_threshold": float(args.weird_score_threshold),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "selected_txt": selected_txt,
        "selected_csv": selected_csv,
        "manifest_json": manifest_json,
        "review_dir": review_dir,
    }


def print_summary(
    run_dir: Path,
    paths: Dict[str, Path],
    selected_rows: Sequence[SelectionRow],
    bucket_counts: Dict[str, int],
    quotas: Dict[str, int],
    skipped_no_tile: int,
) -> None:
    print("[4/4] Selection summary")
    print(f"run_dir: {run_dir}")
    print(f"selected_total: {len(selected_rows)}")
    for b in BUCKET_ORDER:
        print(f"  {b}: {bucket_counts.get(b, 0)} (quota={quotas.get(b, 0)})")
    if skipped_no_tile > 0:
        print(f"  skipped_features_without_tile_id: {skipped_no_tile}")
    print(f"selected list: {paths['selected_txt']}")
    print(f"summary csv: {paths['selected_csv']}")
    print(f"manifest: {paths['manifest_json']}")
    if "review_dir" in paths and paths["review_dir"].exists():
        print(f"review folder: {paths['review_dir']}")


def main() -> None:
    args = parse_args()
    if args.target_total <= 0:
        raise SystemExit("--target-total must be > 0")

    run_dir = resolve_run_dir(args)
    if args.skip_inference and args.run_dir is None:
        raise SystemExit("--skip-inference requires --run-dir so outputs can be located.")

    print(f"[0/4] Using run_dir={run_dir}")
    outputs = run_inference(args, run_dir)

    print("[2/4] Loading predictions + exclusions")
    val_ids = load_validation_ids(args.val_dir.resolve())
    tile_meta = load_tile_meta(outputs["pools_tiles_jsonl"], outputs["pools_tiles_csv"])
    tile_stats, skipped_no_tile = build_tile_stats(
        outputs["pools_geojson"],
        tile_meta,
        borderline_low=float(args.borderline_low),
        borderline_high=float(args.borderline_high),
    )
    if not tile_stats:
        raise SystemExit(f"No detection features parsed from {outputs['pools_geojson']}")

    compute_neighbor_density(tile_stats)
    compute_weird_scores(
        tile_stats,
        large_area_m2=float(args.large_area_m2),
        huge_area_m2=float(args.huge_area_m2),
        elongated_ratio=float(args.elongated_ratio),
        very_elongated_ratio=float(args.very_elongated_ratio),
        neighbor_poly_threshold=int(args.neighbor_poly_threshold),
    )

    print("[3/4] Building targeted batch")
    selected_rows, bucket_counts, quotas = build_selection(args, tile_stats, val_ids)
    excluded_candidate_count = len([t for t in tile_stats.values() if t.num_polys > 0 and t.tile_id in val_ids])
    eligible_count = len([t for t in tile_stats.values() if t.num_polys > 0 and t.tile_id not in val_ids])
    paths = write_outputs(
        args,
        run_dir=run_dir,
        batch_dir=args.batch_dir.resolve(),
        selected_rows=selected_rows,
        bucket_counts=bucket_counts,
        quotas=quotas,
        excluded_val_count=excluded_candidate_count,
        eligible_count=eligible_count,
    )
    print_summary(run_dir, paths, selected_rows, bucket_counts, quotas, skipped_no_tile=skipped_no_tile)


if __name__ == "__main__":
    main()
