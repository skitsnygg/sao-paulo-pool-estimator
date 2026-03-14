#!/usr/bin/env python3
"""
Mine candidate hard-negative tiles from YOLO prediction outputs while excluding anything
already present in existing labeled datasets.

What it does
------------
1. Scans one or more existing label roots and builds a conservative "already done" set.
2. Scans one or more YOLO prediction label roots and keeps only non-empty prediction files.
3. Resolves each prediction back to its source image from one or more tile roots.
4. Excludes candidates whose stem / tile id already appears in existing labeled sets.
5. Copies or symlinks candidates into a review folder with collision-proof filenames.
6. Writes a manifest CSV and summary JSON.

Why the conservative dedupe?
---------------------------
Many GeoSampa tiles reuse names like r0000_c0001 across different cell folders.
If your older labeled datasets flattened those names, we cannot always recover the exact
original cell. To avoid sending you files you've already reviewed, this script excludes on:

- canonical tile id:   cell_XXXX_YYYY__rZZZZ_cWWWW
- filename stem only:  rZZZZ_cWWWW

That means it may skip a few genuinely new tiles that share a stem with old ones, but it
strongly reduces the chance of giving you already-annotated files again.

Typical usage
-------------
python tools/mine_hard_negatives.py \
  --existing-label-roots \
    data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned/labels/train \
    data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned/labels/val \
    data/datasets/YOUR_2024_SET/labels/train \
    data/datasets/YOUR_2024_SET/labels/val \
  --prediction-label-roots \
    runs/segment/predict_higienopolis_moemaft_conf015/hig_cell_0026_0007/labels \
    runs/segment/idesp_full_v26_predict_c10/labels \
  --tile-roots \
    data/raw/geosampa_ortho/sp_city_2020 \
  --out-dir runs/review/hard_negatives_batch1 \
  --max-candidates 150 \
  --seed 42

Notes
-----
- This script prepares candidates for review. It does NOT automatically mark them empty.
- If you want exact cell-safe review filenames, this script writes them as:
    cell_0026_0007__r0000_c0008.png
- If a source image cannot be found, the candidate is skipped and recorded in summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
LABEL_EXT = ".txt"
CELL_RE = re.compile(r"cell_\d{4}_\d{4}$")


@dataclass(frozen=True)
class Candidate:
    pred_label: Path
    source_image: Path
    out_stem: str
    stem_only: str
    cell_name: Optional[str]
    pred_bytes: int
    pred_lines: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine candidate hard negatives from prediction outputs.")
    p.add_argument(
        "--existing-label-roots",
        nargs="+",
        required=True,
        help="One or more roots containing existing labeled .txt files to exclude.",
    )
    p.add_argument(
        "--prediction-label-roots",
        nargs="+",
        required=True,
        help="One or more roots containing YOLO prediction label .txt files.",
    )
    p.add_argument(
        "--tile-roots",
        nargs="+",
        required=True,
        help="One or more roots containing source tile images, typically cell_* folders beneath them.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output review folder.",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=150,
        help="Maximum number of candidates to export.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling after ranking.",
    )
    p.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to place images into the review folder.",
    )
    p.add_argument(
        "--include-empty-predictions",
        action="store_true",
        help="Include empty prediction labels too. Default is off because hard negatives usually come from non-empty predictions.",
    )
    p.add_argument(
        "--shuffle-after-sort",
        action="store_true",
        help="Shuffle candidates after score sorting, before truncating to max-candidates.",
    )
    p.add_argument(
        "--prefer-random",
        action="store_true",
        help="Ignore score sorting and just sample randomly from all eligible candidates.",
    )
    p.add_argument(
        "--write-pred-label-copies",
        action="store_true",
        help="Copy prediction labels alongside images for review reference.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except create files.",
    )
    return p.parse_args()

def find_cell_in_path(path: Path) -> Optional[str]:
    stem = path.stem
    if "__" in stem:
        left = stem.split("__", 1)[0]
        if CELL_RE.match(left):
            return left

    for part in path.parts:
        if CELL_RE.match(part):
            return part

    return None


def canonical_tile_id(path: Path) -> str:
    """
    Build a collision-proof tile id.

    Examples:
      /.../cell_0026_0007/r0000_c0008.png -> cell_0026_0007__r0000_c0008
      /.../cell_0026_0007/labels/r0000_c0008.txt -> cell_0026_0007__r0000_c0008
      /.../cell_0026_0007__r0000_c0008.png -> cell_0026_0007__r0000_c0008
      /.../r0000_c0008.png -> r0000_c0008
    """
    stem = path.stem
    if "__" in stem:
        return stem

    cell_name = find_cell_in_path(path)
    if cell_name:
        return f"{cell_name}__{stem}"

    return stem



def stem_only_from_path(path: Path) -> str:
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def cell_name_from_path(path: Path) -> Optional[str]:
    return find_cell_in_path(path)


def iter_label_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == LABEL_EXT:
        yield root
        return
    if not root.exists():
        return
    for p in root.rglob(f"*{LABEL_EXT}"):
        if p.is_file():
            yield p


def build_existing_sets(label_roots: Sequence[Path]) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    canonical_ids: Set[str] = set()
    stem_only_ids: Set[str] = set()
    stats = {
        "label_files_seen": 0,
        "canonical_ids": 0,
        "stem_only_ids": 0,
    }

    for root in label_roots:
        for p in iter_label_files(root):
            stats["label_files_seen"] += 1
            cid = canonical_tile_id(p)
            sid = stem_only_from_path(p)
            canonical_ids.add(cid)
            stem_only_ids.add(sid)

    stats["canonical_ids"] = len(canonical_ids)
    stats["stem_only_ids"] = len(stem_only_ids)
    return canonical_ids, stem_only_ids, stats


def build_image_index(tile_roots: Sequence[Path]) -> Tuple[Dict[str, Path], Dict[str, List[Path]], Dict[str, int]]:
    """
    Build two lookup tables:
      - by canonical tile id, exact match preferred
      - by stem only, may contain multiple paths from multiple cells
    """
    by_canonical: Dict[str, Path] = {}
    by_stem: Dict[str, List[Path]] = {}
    stats = {
        "images_seen": 0,
        "canonical_keys": 0,
        "stem_keys": 0,
        "canonical_collisions": 0,
    }

    for root in tile_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            stats["images_seen"] += 1
            cid = canonical_tile_id(p)
            sid = stem_only_from_path(p)
            if cid in by_canonical and by_canonical[cid] != p:
                stats["canonical_collisions"] += 1
            else:
                by_canonical[cid] = p
            by_stem.setdefault(sid, []).append(p)

    stats["canonical_keys"] = len(by_canonical)
    stats["stem_keys"] = len(by_stem)
    return by_canonical, by_stem, stats


def count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def resolve_source_image(
    pred_label: Path,
    image_by_canonical: Dict[str, Path],
    images_by_stem: Dict[str, List[Path]],
) -> Tuple[Optional[Path], str, str, Optional[str], str]:
    """
    Returns:
      (source_image, out_stem, stem_only, cell_name, resolution_mode)
    resolution_mode is one of: canonical, unique_stem, ambiguous_stem, not_found
    """
    cid = canonical_tile_id(pred_label)
    sid = stem_only_from_path(pred_label)
    cell_name = cell_name_from_path(pred_label)

    img = image_by_canonical.get(cid)
    if img is not None:
        return img, cid, sid, cell_name, "canonical"

    matches = images_by_stem.get(sid, [])
    if len(matches) == 1:
        img = matches[0]
        resolved_cid = canonical_tile_id(img)
        return img, resolved_cid, sid, cell_name_from_path(img), "unique_stem"
    if len(matches) > 1:
        return None, cid, sid, cell_name, "ambiguous_stem"
    return None, cid, sid, cell_name, "not_found"


def prediction_score(pred_label: Path, pred_lines: int) -> Tuple[int, int, str]:
    """
    Higher first: more predicted objects, then bigger file size, then stable path.
    This tends to surface more obvious false positives for review.
    """
    return (pred_lines, pred_label.stat().st_size, str(pred_label))


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        safe_symlink(src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def main() -> int:
    args = parse_args()

    existing_label_roots = [Path(x).expanduser().resolve() for x in args.existing_label_roots]
    prediction_label_roots = [Path(x).expanduser().resolve() for x in args.prediction_label_roots]
    tile_roots = [Path(x).expanduser().resolve() for x in args.tile_roots]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_images = out_dir / "images"
    out_pred_labels = out_dir / "pred_labels"
    manifest_csv = out_dir / "manifest.csv"
    summary_json = out_dir / "summary.json"

    random.seed(args.seed)

    existing_canonical, existing_stems, existing_stats = build_existing_sets(existing_label_roots)
    image_by_canonical, images_by_stem, image_stats = build_image_index(tile_roots)

    candidates: List[Candidate] = []
    skipped = {
        "prediction_labels_seen": 0,
        "prediction_labels_empty": 0,
        "excluded_existing_canonical": 0,
        "excluded_existing_stem": 0,
        "resolve_not_found": 0,
        "resolve_ambiguous_stem": 0,
        "duplicate_candidate_out_stem": 0,
    }
    resolution_counts: Dict[str, int] = {
        "canonical": 0,
        "unique_stem": 0,
        "ambiguous_stem": 0,
        "not_found": 0,
    }

    seen_candidate_out_stems: Set[str] = set()

    for root in prediction_label_roots:
        for pred_label in iter_label_files(root):
            skipped["prediction_labels_seen"] += 1
            pred_lines = count_nonempty_lines(pred_label)
            if pred_lines == 0 and not args.include_empty_predictions:
                skipped["prediction_labels_empty"] += 1
                continue

            source_image, out_stem, stem_only, cell_name, resolution_mode = resolve_source_image(
                pred_label,
                image_by_canonical,
                images_by_stem,
            )
            resolution_counts[resolution_mode] += 1

            if resolution_mode == "not_found":
                skipped["resolve_not_found"] += 1
                continue
            if resolution_mode == "ambiguous_stem":
                skipped["resolve_ambiguous_stem"] += 1
                continue
            assert source_image is not None

            if out_stem in existing_canonical:
                skipped["excluded_existing_canonical"] += 1
                continue
            if stem_only in existing_stems:
                skipped["excluded_existing_stem"] += 1
                continue
            if out_stem in seen_candidate_out_stems:
                skipped["duplicate_candidate_out_stem"] += 1
                continue
            seen_candidate_out_stems.add(out_stem)

            candidates.append(
                Candidate(
                    pred_label=pred_label,
                    source_image=source_image,
                    out_stem=out_stem,
                    stem_only=stem_only,
                    cell_name=cell_name_from_path(source_image) or cell_name,
                    pred_bytes=pred_label.stat().st_size,
                    pred_lines=pred_lines,
                )
            )

    if args.prefer_random:
        random.shuffle(candidates)
    else:
        candidates.sort(key=lambda c: prediction_score(c.pred_label, c.pred_lines), reverse=True)
        if args.shuffle_after_sort:
            top = candidates[:]
            random.shuffle(top)
            candidates = top

    selected = candidates[: max(0, args.max_candidates)]

    summary = {
        "existing_label_roots": [str(p) for p in existing_label_roots],
        "prediction_label_roots": [str(p) for p in prediction_label_roots],
        "tile_roots": [str(p) for p in tile_roots],
        "out_dir": str(out_dir),
        "existing_stats": existing_stats,
        "image_index_stats": image_stats,
        "scan_stats": skipped,
        "resolution_counts": resolution_counts,
        "eligible_candidates": len(candidates),
        "selected_candidates": len(selected),
        "copy_mode": args.copy_mode,
        "seed": args.seed,
        "max_candidates": args.max_candidates,
        "include_empty_predictions": args.include_empty_predictions,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    out_images.mkdir(parents=True, exist_ok=True)
    if args.write_pred_label_copies:
        out_pred_labels.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "review_image",
                "pred_label_copy",
                "source_image",
                "prediction_label",
                "out_stem",
                "stem_only",
                "cell_name",
                "pred_lines",
                "pred_bytes",
                "source_ext",
            ],
        )
        writer.writeheader()

        for c in selected:
            src = c.source_image
            img_dst = out_images / f"{c.out_stem}{src.suffix.lower()}"
            copy_or_link(src, img_dst, args.copy_mode)

            pred_label_copy = ""
            if args.write_pred_label_copies:
                pred_dst = out_pred_labels / f"{c.out_stem}.pred.txt"
                shutil.copy2(c.pred_label, pred_dst)
                pred_label_copy = str(pred_dst)

            writer.writerow(
                {
                    "review_image": str(img_dst),
                    "pred_label_copy": pred_label_copy,
                    "source_image": str(c.source_image),
                    "prediction_label": str(c.pred_label),
                    "out_stem": c.out_stem,
                    "stem_only": c.stem_only,
                    "cell_name": c.cell_name or "",
                    "pred_lines": c.pred_lines,
                    "pred_bytes": c.pred_bytes,
                    "source_ext": c.source_image.suffix.lower(),
                }
            )

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote review images to: {out_images}")
    print(f"Wrote manifest: {manifest_csv}")
    print(f"Wrote summary: {summary_json}")
    print(f"Selected {len(selected)} candidates from {len(candidates)} eligible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())