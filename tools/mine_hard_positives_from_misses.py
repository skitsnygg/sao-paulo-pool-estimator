#!/usr/bin/env python3
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

from tile_id_guard import collect_tile_ids_from_roots, default_existing_image_roots, extract_tile_id

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
LABEL_EXT = ".txt"
CELL_RE = re.compile(r"cell_\d{4}_\d{4}$")
DEFAULT_EXIST_TRAIN, DEFAULT_EXIST_VAL = default_existing_image_roots()


@dataclass(frozen=True)
class Candidate:
    source_image: Path
    out_stem: str
    stem_only: str
    cell_name: Optional[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine hard-positive candidates from images with no prediction label file."
    )
    p.add_argument("--existing-label-roots", nargs="+", required=True)
    p.add_argument(
        "--existing-image-roots",
        nargs="*",
        default=[],
        help=(
            "Optional roots containing existing train/val images to exclude. "
            "If omitted, image roots are inferred from --existing-label-roots by replacing /labels/ with /images/."
        ),
    )
    p.add_argument("--existing-images-train", default=str(DEFAULT_EXIST_TRAIN))
    p.add_argument("--existing-images-val", default=str(DEFAULT_EXIST_VAL))
    p.add_argument("--prediction-label-roots", nargs="+", required=True)
    p.add_argument("--tile-roots", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-candidates", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    p.add_argument("--dry-run", action="store_true")
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
    return extract_tile_id(path.as_posix()) or ""


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


def iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def infer_image_roots_from_label_roots(label_roots: Sequence[Path]) -> List[Path]:
    inferred: List[Path] = []
    for root in label_roots:
        probe = root.parent if root.suffix.lower() == LABEL_EXT else root
        parts = list(probe.parts)
        idx = -1
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "labels":
                idx = i
                break
        if idx < 0:
            continue
        out_parts = parts[:]
        out_parts[idx] = "images"
        inferred.append(Path(*out_parts))
    return inferred


def unique_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen: Set[str] = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_existing_sets(label_roots: Sequence[Path], image_roots: Sequence[Path]) -> Tuple[Set[str], Dict[str, int]]:
    merged_roots = [*label_roots, *image_roots]
    tile_ids, files_scanned = collect_tile_ids_from_roots(merged_roots)
    stats = {"roots_scanned": len(merged_roots), "files_scanned": files_scanned, "canonical_ids": len(tile_ids)}
    return tile_ids, stats


def build_predicted_set(prediction_label_roots: Sequence[Path]) -> Tuple[Set[str], Dict[str, int]]:
    predicted: Set[str] = set()
    stats = {"prediction_labels_seen": 0, "predicted_canonical_ids": 0}

    for root in prediction_label_roots:
        for p in iter_label_files(root):
            stats["prediction_labels_seen"] += 1
            cid = canonical_tile_id(p)
            if cid:
                predicted.add(cid)

    stats["predicted_canonical_ids"] = len(predicted)
    return predicted, stats


def build_image_index(tile_roots: Sequence[Path]) -> Tuple[List[Path], Dict[str, int]]:
    images: List[Path] = []
    stats = {"images_seen": 0}

    for root in tile_roots:
        for p in iter_image_files(root):
            images.append(p)
            stats["images_seen"] += 1

    return images, stats


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        safe_symlink(src, dst)


def main() -> int:
    args = parse_args()

    existing_label_roots = [Path(x).expanduser().resolve() for x in args.existing_label_roots]
    user_existing_image_roots = [Path(x).expanduser().resolve() for x in args.existing_image_roots]
    default_existing_roots = [
        Path(args.existing_images_train).expanduser().resolve(),
        Path(args.existing_images_val).expanduser().resolve(),
    ]
    inferred_existing_image_roots = [p.expanduser().resolve() for p in infer_image_roots_from_label_roots(existing_label_roots)]
    existing_image_roots = unique_paths([*default_existing_roots, *user_existing_image_roots, *inferred_existing_image_roots])
    prediction_label_roots = [Path(x).expanduser().resolve() for x in args.prediction_label_roots]
    tile_roots = [Path(x).expanduser().resolve() for x in args.tile_roots]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_images = out_dir / "images"
    manifest_csv = out_dir / "manifest.csv"
    summary_json = out_dir / "summary.json"

    random.seed(args.seed)

    existing_canonical, existing_stats = build_existing_sets(existing_label_roots, existing_image_roots)
    predicted_canonical, pred_stats = build_predicted_set(prediction_label_roots)
    images, image_stats = build_image_index(tile_roots)

    candidates: List[Candidate] = []
    skipped = {
        "already_predicted": 0,
        "excluded_existing_canonical": 0,
        "duplicate_candidate_out_stem": 0,
        "invalid_tile_id": 0,
    }
    seen_candidate_out_stems: Set[str] = set()

    # Limit to cells that were actually inferred
    inferred_cells = {find_cell_in_path(Path(p)) for p in prediction_label_roots}
    inferred_cells.discard(None)

    for img in images:
        cell_name = cell_name_from_path(img)
        if not cell_name or cell_name not in inferred_cells:
            continue

        out_stem = canonical_tile_id(img)
        stem_only = stem_only_from_path(img)
        if not out_stem:
            skipped["invalid_tile_id"] += 1
            continue

        if out_stem in predicted_canonical:
            skipped["already_predicted"] += 1
            continue
        if out_stem in existing_canonical:
            skipped["excluded_existing_canonical"] += 1
            continue
        if out_stem in seen_candidate_out_stems:
            skipped["duplicate_candidate_out_stem"] += 1
            continue

        seen_candidate_out_stems.add(out_stem)
        candidates.append(
            Candidate(
                source_image=img,
                out_stem=out_stem,
                stem_only=stem_only,
                cell_name=cell_name,
            )
        )

    random.shuffle(candidates)
    selected = candidates[: max(0, args.max_candidates)]

    summary = {
        "existing_label_roots": [str(p) for p in existing_label_roots],
        "existing_image_roots": [str(p) for p in existing_image_roots],
        "inferred_existing_image_roots": [str(p) for p in inferred_existing_image_roots],
        "prediction_label_roots": [str(p) for p in prediction_label_roots],
        "tile_roots": [str(p) for p in tile_roots],
        "out_dir": str(out_dir),
        "existing_stats": existing_stats,
        "prediction_stats": pred_stats,
        "image_index_stats": image_stats,
        "scan_stats": skipped,
        "eligible_candidates": len(candidates),
        "selected_candidates": len(selected),
        "copy_mode": args.copy_mode,
        "seed": args.seed,
        "max_candidates": args.max_candidates,
        "dry_run": args.dry_run,
        "inferred_cells": sorted(inferred_cells),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    out_images.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "review_image",
                "source_image",
                "out_stem",
                "stem_only",
                "cell_name",
                "source_ext",
            ],
        )
        writer.writeheader()

        for c in selected:
            src = c.source_image
            img_dst = out_images / f"{c.out_stem}{src.suffix.lower()}"
            copy_or_link(src, img_dst, args.copy_mode)

            writer.writerow(
                {
                    "review_image": str(img_dst),
                    "source_image": str(c.source_image),
                    "out_stem": c.out_stem,
                    "stem_only": c.stem_only,
                    "cell_name": c.cell_name or "",
                    "source_ext": c.source_image.suffix.lower(),
                }
            )

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote review images to: {out_images}")
    print(f"Wrote manifest: {manifest_csv}")
    print(f"Wrote summary: {summary_json}")
    print(f"total candidates scanned: {image_stats['images_seen']}")
    print(f"skipped (already labeled): {skipped['excluded_existing_canonical']}")
    print(f"skipped (duplicate in batch): {skipped['duplicate_candidate_out_stem']}")
    print(f"final selected count: {len(selected)}")
    print(f"Selected {len(selected)} candidates from {len(candidates)} eligible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
