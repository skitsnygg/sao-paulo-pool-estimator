#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
BUCKETS = ("empty_preds", "low_conf", "large_masks", "many_preds", "random_diverse")


@dataclass
class ConfidenceIndex:
    by_rel: Dict[str, List[float]]
    by_name: Dict[str, Dict[str, List[float]]]
    by_stem: Dict[str, Dict[str, List[float]]]


@dataclass
class TileStats:
    rel_path: Path
    abs_path: Path
    label_path: Optional[Path]
    num_preds: int
    max_conf: Optional[float]
    min_conf: Optional[float]
    mean_conf: Optional[float]
    total_mask_area_fraction: float
    source: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Select active-learning tiles from a YOLO segmentation inference run and "
            "export bucketed image sets with manifests."
        )
    )
    ap.add_argument("--tiles-dir", required=True, type=Path, help="Root directory with source image tiles.")
    ap.add_argument("--labels-dir", required=True, type=Path, help="Root directory with YOLO prediction .txt files.")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for bucketed exports.")
    ap.add_argument(
        "--conf-json",
        default=None,
        type=Path,
        help="Optional JSON/CSV with per-detection confidence scores.",
    )
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking.",
    )
    ap.add_argument("--top-k", required=True, type=int, help="Number of tiles to export per bucket.")

    ap.add_argument(
        "--low-conf-quantile",
        type=float,
        default=0.35,
        help="Quantile used to define low-confidence candidates (default: 0.35).",
    )
    ap.add_argument(
        "--large-mask-min-fraction",
        type=float,
        default=0.20,
        help="Minimum total mask area fraction to qualify as large_masks (default: 0.20).",
    )
    ap.add_argument(
        "--many-preds-min",
        type=int,
        default=3,
        help="Minimum object count to qualify as many_preds (default: 3).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing out-dir before writing.",
    )
    ap.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return ap.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def rel_key(path: Path) -> str:
    return path.as_posix().lstrip("./")


def safe_float(value: object) -> Optional[float]:
    try:
        v = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_conf_value(value: object) -> List[float]:
    out: List[float] = []
    if isinstance(value, (int, float)):
        v = safe_float(value)
        if v is not None:
            out.append(v)
        return out
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return out
        try:
            parsed = json.loads(raw)
            return parse_conf_value(parsed)
        except Exception:
            pass
        for tok in raw.replace("|", ",").replace(";", ",").split(","):
            v = safe_float(tok.strip())
            if v is not None:
                out.append(v)
        return out
    if isinstance(value, Sequence):
        for item in value:
            out.extend(parse_conf_value(item))
        return out
    return out


def init_conf_index() -> ConfidenceIndex:
    return ConfidenceIndex(by_rel={}, by_name=defaultdict(dict), by_stem=defaultdict(dict))


def add_conf(index: ConfidenceIndex, path_str: str, confs: Sequence[float]) -> None:
    vals = [float(v) for v in confs if math.isfinite(float(v))]
    if not vals:
        return
    p = Path(path_str)
    rk = rel_key(p)
    index.by_rel[rk] = vals
    index.by_name[p.name][rk] = vals
    index.by_stem[p.stem][rk] = vals


def load_conf_index(path: Optional[Path]) -> ConfidenceIndex:
    index = init_conf_index()
    if path is None:
        return index
    if not path.exists():
        raise SystemExit(f"--conf-json file not found: {path}")

    logging.info("Loading confidence sidecar: %s", path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        ingest_conf_json(payload, index)
        return index

    if suffix == ".csv":
        ingest_conf_csv(path, index)
        return index

    # Fallback: try JSON first, then CSV
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ingest_conf_json(payload, index)
        return index
    except Exception:
        ingest_conf_csv(path, index)
        return index


def ingest_conf_json(payload: object, index: ConfidenceIndex) -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, dict):
                vals = parse_conf_value(v.get("confidences", v.get("scores", v.get("conf", []))))
            else:
                vals = parse_conf_value(v)
            add_conf(index, str(k), vals)
        return

    if isinstance(payload, list):
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            tile = (
                rec.get("tile_path")
                or rec.get("tile")
                or rec.get("image_path")
                or rec.get("image")
                or rec.get("path")
            )
            if not tile:
                continue
            vals: List[float] = []
            if "confidences" in rec:
                vals.extend(parse_conf_value(rec["confidences"]))
            elif "scores" in rec:
                vals.extend(parse_conf_value(rec["scores"]))
            elif "conf" in rec:
                vals.extend(parse_conf_value(rec["conf"]))
            elif "confidence" in rec:
                vals.extend(parse_conf_value(rec["confidence"]))
            add_conf(index, str(tile), vals)
        return

    raise SystemExit("Unsupported JSON format for --conf-json.")


def ingest_conf_csv(path: Path, index: ConfidenceIndex) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Confidence CSV has no header: {path}")

        tile_cols = ["tile_path", "tile", "image_path", "image", "path", "relative_path", "label_path"]
        conf_cols = ["confidence", "conf", "score", "scores", "confidences"]

        tile_col = next((c for c in tile_cols if c in reader.fieldnames), None)
        if tile_col is None:
            raise SystemExit(f"Confidence CSV missing tile column. Found: {reader.fieldnames}")

        for row in reader:
            tile = (row.get(tile_col) or "").strip()
            if not tile:
                continue
            vals: List[float] = []
            for c in conf_cols:
                if c in row and row.get(c):
                    vals.extend(parse_conf_value(row.get(c)))
            add_conf(index, tile, vals)


def lookup_conf(index: ConfidenceIndex, rel_path: Path) -> Optional[List[float]]:
    rk = rel_key(rel_path)
    if rk in index.by_rel:
        return list(index.by_rel[rk])

    by_name = index.by_name.get(rel_path.name, {})
    if len(by_name) == 1:
        return list(next(iter(by_name.values())))

    by_stem = index.by_stem.get(rel_path.stem, {})
    if len(by_stem) == 1:
        return list(next(iter(by_stem.values())))
    return None


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def build_label_basename_index(labels_dir: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = defaultdict(list)
    for p in labels_dir.rglob("*.txt"):
        if p.is_file():
            out[p.name].append(p)
    return out


def resolve_label_path(
    labels_dir: Path,
    rel_image: Path,
    basename_idx: Dict[str, List[Path]],
) -> Optional[Path]:
    # Mirror layout: labels_dir/<relative_path>.txt
    direct = labels_dir / rel_image.with_suffix(".txt")
    if direct.exists():
        return direct

    # Common run layout: labels_dir/<cell>/labels/<tile>.txt
    cell_labels = labels_dir / rel_image.parent / "labels" / f"{rel_image.stem}.txt"
    if cell_labels.exists():
        return cell_labels

    # Alternate per-cell layout: labels_dir/<cell>/<tile>.txt
    cell_root = labels_dir / rel_image.parent / f"{rel_image.stem}.txt"
    if cell_root.exists():
        return cell_root

    # Flat layout: labels_dir/<tile>.txt
    flat = labels_dir / f"{rel_image.stem}.txt"
    if flat.exists():
        return flat
    matches = basename_idx.get(f"{rel_image.stem}.txt", [])
    if len(matches) == 1:
        return matches[0]
    return None


def shoelace_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area2 += (x1 * y2) - (x2 * y1)
    return abs(area2) * 0.5


def looks_normalized(points: Sequence[Tuple[float, float]]) -> bool:
    if not points:
        return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs) >= -0.5 and min(ys) >= -0.5 and max(xs) <= 1.5 and max(ys) <= 1.5


def read_image_size(path: Path, cache: Dict[Path, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if path in cache:
        return cache[path]
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            size = (int(im.width), int(im.height))
            cache[path] = size
            return size
    except Exception:
        return None


def polygon_area_fraction(
    points: Sequence[Tuple[float, float]],
    image_path: Path,
    size_cache: Dict[Path, Tuple[int, int]],
) -> float:
    if len(points) < 3:
        return 0.0
    if looks_normalized(points):
        return max(0.0, shoelace_area(points))
    size = read_image_size(image_path, size_cache)
    if not size:
        return 0.0
    w, h = size
    if w <= 0 or h <= 0:
        return 0.0
    area_px = shoelace_area(points)
    return max(0.0, area_px / float(w * h))


def parse_yolo_predictions(
    label_path: Optional[Path],
    image_path: Path,
    conf_hint: Optional[List[float]],
    size_cache: Dict[Path, Tuple[int, int]],
) -> Tuple[int, List[float], float]:
    if label_path is None or not label_path.exists():
        return 0, [], 0.0
    try:
        raw = label_path.read_text(encoding="utf-8")
    except Exception as exc:
        logging.warning("Failed to read label file %s: %s", label_path, exc)
        return 0, [], 0.0
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return 0, [], 0.0

    num_preds = 0
    confs: List[float] = []
    total_area_fraction = 0.0

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        has_conf = (len(parts) % 2 == 0) and len(parts) >= 8

        conf_val: Optional[float] = None
        coord_start = 1
        if has_conf:
            conf_val = safe_float(parts[1])
            coord_start = 2

        coords_raw = parts[coord_start:]
        if len(coords_raw) < 6 or len(coords_raw) % 2 != 0:
            continue

        points: List[Tuple[float, float]] = []
        ok = True
        for i in range(0, len(coords_raw), 2):
            x = safe_float(coords_raw[i])
            y = safe_float(coords_raw[i + 1])
            if x is None or y is None:
                ok = False
                break
            points.append((x, y))
        if not ok or len(points) < 3:
            continue

        num_preds += 1
        total_area_fraction += polygon_area_fraction(points, image_path, size_cache)
        if conf_val is not None and 0.0 <= conf_val <= 1.0:
            confs.append(conf_val)

    if not confs and conf_hint:
        confs = [c for c in conf_hint if 0.0 <= c <= 1.0]

    return num_preds, confs, total_area_fraction


def confidence_stats(confs: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not confs:
        return None, None, None
    vals = [float(v) for v in confs if math.isfinite(float(v))]
    if not vals:
        return None, None, None
    return max(vals), min(vals), sum(vals) / len(vals)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, q))
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return arr[idx]


def group_key_for_diversity(rel_path: Path) -> str:
    if rel_path.parts:
        first = rel_path.parts[0]
        if first.startswith("cell_"):
            return first
        return first
    return "."


def diverse_random_sample(records: Sequence[TileStats], k: int, rng: random.Random) -> List[TileStats]:
    if k <= 0 or not records:
        return []
    groups: Dict[str, List[TileStats]] = defaultdict(list)
    for rec in records:
        groups[group_key_for_diversity(rec.rel_path)].append(rec)
    for items in groups.values():
        rng.shuffle(items)
    keys = list(groups.keys())
    rng.shuffle(keys)

    picked: List[TileStats] = []
    while len(picked) < k:
        progressed = False
        for key in keys:
            bucket = groups[key]
            if not bucket:
                continue
            picked.append(bucket.pop())
            progressed = True
            if len(picked) >= k:
                break
        if not progressed:
            break
    return picked


def ensure_unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        cand = parent / f"{stem}__dup{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def materialize_tile(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = ensure_unique_target(dst)
    if copy_images:
        shutil.copy2(src, dst)
        return
    rel_src = os.path.relpath(src, start=dst.parent)
    os.symlink(rel_src, dst)


def maybe_remove_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and overwrite:
        logging.info("Removing existing out-dir: %s", out_dir)
        shutil.rmtree(out_dir)


def write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tile_path",
        "bucket",
        "num_preds",
        "max_conf",
        "min_conf",
        "mean_conf",
        "total_mask_area_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_tiles(records: Sequence[TileStats], args: argparse.Namespace) -> Tuple[Dict[str, List[TileStats]], Dict[str, object]]:
    rng = random.Random(args.seed)
    selected: Dict[str, List[TileStats]] = {b: [] for b in BUCKETS}
    picked_keys: set[str] = set()

    def not_picked(r: TileStats) -> bool:
        return rel_key(r.rel_path) not in picked_keys

    # 1) empty_preds
    empty_candidates = [r for r in records if r.num_preds == 0]
    empty_candidates.sort(key=lambda r: rel_key(r.rel_path))
    selected["empty_preds"] = empty_candidates[: args.top_k]
    picked_keys.update(rel_key(r.rel_path) for r in selected["empty_preds"])

    # 2) low_conf
    low_conf_pool = [r for r in records if not_picked(r) and r.num_preds > 0 and r.min_conf is not None]
    low_conf_cutoff = None
    if low_conf_pool:
        mins = [float(r.min_conf) for r in low_conf_pool if r.min_conf is not None]
        low_conf_cutoff = quantile(mins, args.low_conf_quantile)
        low_conf_candidates = [r for r in low_conf_pool if r.min_conf is not None and r.min_conf <= low_conf_cutoff]
        low_conf_candidates.sort(key=lambda r: (float(r.min_conf), float(r.mean_conf or 1.0), rel_key(r.rel_path)))
        selected["low_conf"] = low_conf_candidates[: args.top_k]
        picked_keys.update(rel_key(r.rel_path) for r in selected["low_conf"])

    # 3) large_masks
    large_pool = [r for r in records if not_picked(r) and r.num_preds > 0]
    large_candidates = [r for r in large_pool if r.total_mask_area_fraction >= args.large_mask_min_fraction]
    if not large_candidates:
        large_candidates = large_pool
    large_candidates.sort(key=lambda r: (r.total_mask_area_fraction, r.num_preds), reverse=True)
    selected["large_masks"] = large_candidates[: args.top_k]
    picked_keys.update(rel_key(r.rel_path) for r in selected["large_masks"])

    # 4) many_preds
    many_pool = [r for r in records if not_picked(r) and r.num_preds >= args.many_preds_min]
    if not many_pool:
        many_pool = [r for r in records if not_picked(r) and r.num_preds > 0]
    many_pool.sort(key=lambda r: (r.num_preds, r.total_mask_area_fraction), reverse=True)
    selected["many_preds"] = many_pool[: args.top_k]
    picked_keys.update(rel_key(r.rel_path) for r in selected["many_preds"])

    # 5) random_diverse
    random_pool = [r for r in records if not_picked(r)]
    selected["random_diverse"] = diverse_random_sample(random_pool, args.top_k, rng)
    picked_keys.update(rel_key(r.rel_path) for r in selected["random_diverse"])

    meta = {
        "low_conf_cutoff": low_conf_cutoff,
        "low_conf_quantile": args.low_conf_quantile,
        "large_mask_min_fraction": args.large_mask_min_fraction,
        "many_preds_min": args.many_preds_min,
    }
    return selected, meta


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0.")
    if not (0.0 <= args.low_conf_quantile <= 1.0):
        raise SystemExit("--low-conf-quantile must be in [0, 1].")

    tiles_dir = args.tiles_dir.expanduser().resolve()
    labels_dir = args.labels_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    conf_path = args.conf_json.expanduser().resolve() if args.conf_json else None

    if not tiles_dir.is_dir():
        raise SystemExit(f"--tiles-dir not found: {tiles_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"--labels-dir not found: {labels_dir}")

    maybe_remove_out_dir(out_dir, args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_index = load_conf_index(conf_path)
    basename_idx = build_label_basename_index(labels_dir)

    images = sorted(iter_images(tiles_dir))
    if not images:
        raise SystemExit(f"No image tiles found under: {tiles_dir}")
    logging.info("Found %d source tiles.", len(images))

    size_cache: Dict[Path, Tuple[int, int]] = {}
    rows_all: List[TileStats] = []

    for img in images:
        rel = img.relative_to(tiles_dir)
        label_path = resolve_label_path(labels_dir, rel, basename_idx)
        conf_hint = lookup_conf(conf_index, rel)
        num_preds, confs, total_area = parse_yolo_predictions(
            label_path=label_path,
            image_path=img,
            conf_hint=conf_hint,
            size_cache=size_cache,
        )
        max_conf, min_conf, mean_conf = confidence_stats(confs)
        source = "missing_label"
        if label_path is not None and label_path.exists():
            source = "label_file"
        elif conf_hint:
            source = "conf_sidecar_only"

        rows_all.append(
            TileStats(
                rel_path=rel,
                abs_path=img,
                label_path=label_path,
                num_preds=num_preds,
                max_conf=max_conf,
                min_conf=min_conf,
                mean_conf=mean_conf,
                total_mask_area_fraction=total_area,
                source=source,
            )
        )

    selected, selection_meta = select_tiles(rows_all, args)

    # Materialize buckets and manifests
    global_manifest_rows: List[Dict[str, object]] = []
    for bucket_name in BUCKETS:
        bucket_rows: List[Dict[str, object]] = []
        for rec in selected[bucket_name]:
            dst = out_dir / bucket_name / rec.rel_path
            materialize_tile(rec.abs_path, dst, copy_images=bool(args.copy_images))
            row = {
                "tile_path": rel_key(rec.rel_path),
                "bucket": bucket_name,
                "num_preds": rec.num_preds,
                "max_conf": "" if rec.max_conf is None else f"{rec.max_conf:.6f}",
                "min_conf": "" if rec.min_conf is None else f"{rec.min_conf:.6f}",
                "mean_conf": "" if rec.mean_conf is None else f"{rec.mean_conf:.6f}",
                "total_mask_area_fraction": f"{rec.total_mask_area_fraction:.6f}",
            }
            bucket_rows.append(row)
            global_manifest_rows.append(row)

        write_manifest(out_dir / bucket_name / "manifest.csv", bucket_rows)
        logging.info("Bucket %-14s -> %d tiles", bucket_name, len(bucket_rows))

    write_manifest(out_dir / "manifest.csv", global_manifest_rows)

    summary = {
        "tiles_dir": str(tiles_dir),
        "labels_dir": str(labels_dir),
        "out_dir": str(out_dir),
        "conf_json": str(conf_path) if conf_path else "",
        "copy_images": bool(args.copy_images),
        "top_k": int(args.top_k),
        "total_tiles_scanned": len(rows_all),
        "tiles_with_preds": sum(1 for r in rows_all if r.num_preds > 0),
        "tiles_empty_preds": sum(1 for r in rows_all if r.num_preds == 0),
        "bucket_counts": {k: len(v) for k, v in selected.items()},
        "selection_meta": selection_meta,
        "global_manifest": str(out_dir / "manifest.csv"),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logging.info("Wrote global manifest: %s", out_dir / "manifest.csv")
    logging.info("Wrote summary report: %s", summary_path)
    logging.info("Done.")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
