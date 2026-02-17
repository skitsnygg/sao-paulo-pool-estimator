#!/usr/bin/env python3
"""
Build a reviewed-images workspace ("chipset") from a chips.csv ordering.

What it does:
- selects first N rows from chips.csv (deterministic ordering)
- copies those images into out_dir/images
- copies positive masks from CVAT export SegmentationClass into out_dir/masks
- creates empty masks for any selected image without a mask
- writes out_dir/manifest.json

Typical usage (your case):
  python tools/build_reviewed_workspace.py \
    --chips-dir /Users/admin/sao-paulo-pool-estimator/data/raw/geosampa_ortho/moema_2020 \
    --export-dir /Users/admin/sao-paulo-pool-estimator/data/annotation/moema/task5_export \
    --out-dir /Users/admin/sao-paulo-pool-estimator/data/annotation/moema/ws_task5_first177 \
    --n 177
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np


FILENAME_KEYS = ("filename", "file", "path", "chip", "image", "img", "name")


@dataclass
class BuildResult:
    selected_count: int
    images_copied: int
    images_missing: int
    masks_copied: int
    empty_masks_created: int
    total_masks: int
    filename_column: str
    out_dir: str


def _find_filename_column(rows: List[Dict[str, str]], chips_csv: Path) -> str:
    if not rows:
        raise SystemExit(f"{chips_csv} is empty (no data rows).")
    cols = list(rows[0].keys())
    for k in FILENAME_KEYS:
        if k in rows[0]:
            return k
    raise SystemExit(
        f"Could not find a filename column in {chips_csv}.\n"
        f"Columns found: {cols}\n"
        f"Expected one of: {list(FILENAME_KEYS)}"
    )


def _read_first_n_filenames(chips_csv: Path, n: int) -> Tuple[str, List[str]]:
    with chips_csv.open(newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    col = _find_filename_column(rows, chips_csv)

    if n > len(rows):
        raise SystemExit(f"Requested n={n}, but chips.csv only has {len(rows)} rows.")

    names = [rows[i][col].strip() for i in range(n)]
    names = [x for x in names if x]  # drop empty just in case
    if len(names) != n:
        raise SystemExit(
            f"Expected {n} filenames from first {n} rows, got {len(names)} after stripping empties."
        )
    return col, names


def _resolve_image_path(chips_dir: Path, name: str) -> Optional[Path]:
    """
    chips.csv entries might be:
      - "something.png"
      - "something" (no extension)
      - "subdir/something.png" (relative)
    We resolve relative paths inside chips_dir.
    """
    p = chips_dir / name
    if p.exists():
        return p

    # try if name missing extension
    if Path(name).suffix == "":
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
            alt = chips_dir / f"{name}{ext}"
            if alt.exists():
                return alt

    # try just basename (in case chips.csv includes paths but files were flattened)
    base = Path(name).name
    p2 = chips_dir / base
    if p2.exists():
        return p2

    if Path(base).suffix == "":
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
            alt = chips_dir / f"{base}{ext}"
            if alt.exists():
                return alt

    return None


def _copy_selected_images(chips_dir: Path, out_images: Path, names: List[str]) -> Tuple[int, int, List[str]]:
    out_images.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    missing_names: List[str] = []

    for name in names:
        src = _resolve_image_path(chips_dir, name)
        if src is None:
            missing += 1
            missing_names.append(name)
            continue
        shutil.copy2(src, out_images / src.name)
        copied += 1

    return copied, missing, missing_names


def _copy_export_masks(export_dir: Path, out_masks: Path) -> int:
    out_masks.mkdir(parents=True, exist_ok=True)

    seg_dir = export_dir / "SegmentationClass"
    if not seg_dir.exists() or not seg_dir.is_dir():
        # not fatal; you might be building reviewed negatives first
        return 0

    copied = 0
    for p in seg_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            shutil.copy2(p, out_masks / p.name)
            copied += 1
    return copied


def _make_empty_masks_for_missing(out_images: Path, out_masks: Path) -> int:
    out_masks.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in out_images.iterdir() if p.is_file()])
    existing = {p.name for p in out_masks.iterdir() if p.is_file() and p.suffix.lower() == ".png"}

    created = 0
    for im_path in img_paths:
        mask_name = im_path.with_suffix(".png").name
        if mask_name in existing:
            continue

        with Image.open(im_path) as im:
            w, h = im.size

        arr = np.zeros((h, w), dtype=np.uint8)
        Image.fromarray(arr).save(out_masks / mask_name)
        created += 1

    return created


def _write_manifest(
    out_dir: Path,
    chips_dir: Path,
    export_dir: Path,
    n: int,
    filename_column: str,
    names: List[str],
    missing_names: List[str],
    result: BuildResult,
) -> None:
    manifest = {
        "inputs": {
            "chips_dir": str(chips_dir),
            "chips_csv": str(chips_dir / "chips.csv"),
            "export_dir": str(export_dir),
            "export_segmentation_class": str(export_dir / "SegmentationClass"),
            "n": n,
            "filename_column": filename_column,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "images_dir": str(out_dir / "images"),
            "masks_dir": str(out_dir / "masks"),
        },
        "selection": {
            "selected_filenames_from_csv_first_n": names,
            "missing_filenames": missing_names,
        },
        "stats": {
            "selected_count": result.selected_count,
            "images_copied": result.images_copied,
            "images_missing": result.images_missing,
            "masks_copied": result.masks_copied,
            "empty_masks_created": result.empty_masks_created,
            "total_masks": result.total_masks,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips-dir", required=True, help="Directory containing chips.csv and chip images.")
    ap.add_argument("--export-dir", required=True, help="CVAT export directory containing SegmentationClass/.")
    ap.add_argument("--out-dir", required=True, help="Workspace output directory.")
    ap.add_argument("--n", type=int, default=177, help="How many chips to include from the start of chips.csv.")
    ap.add_argument("--clean", action="store_true", help="Delete out-dir before building.")
    args = ap.parse_args()

    chips_dir = Path(args.chips_dir).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    chips_csv = chips_dir / "chips.csv"
    if not chips_csv.exists():
        raise SystemExit(f"chips.csv not found at: {chips_csv}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    filename_column, names = _read_first_n_filenames(chips_csv, args.n)

    images_copied, images_missing, missing_names = _copy_selected_images(chips_dir, out_dir / "images", names)
    masks_copied = _copy_export_masks(export_dir, out_dir / "masks")
    empty_masks_created = _make_empty_masks_for_missing(out_dir / "images", out_dir / "masks")

    total_masks = len([p for p in (out_dir / "masks").iterdir() if p.is_file() and p.suffix.lower() == ".png"])

    result = BuildResult(
        selected_count=args.n,
        images_copied=images_copied,
        images_missing=images_missing,
        masks_copied=masks_copied,
        empty_masks_created=empty_masks_created,
        total_masks=total_masks,
        filename_column=filename_column,
        out_dir=str(out_dir),
    )

    _write_manifest(out_dir, chips_dir, export_dir, args.n, filename_column, names, missing_names, result)

    print("OK")
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
