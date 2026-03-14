#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Add reviewed hard-negative images to an existing YOLO segmentation dataset."
    )
    ap.add_argument(
        "--neg-dir",
        type=Path,
        required=True,
        help="Directory containing negative images to add.",
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="YOLO dataset root.",
    )
    ap.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to add negatives to.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be added without writing files.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    neg_dir = args.neg_dir.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()

    img_dst = dataset / "images" / args.split
    lbl_dst = dataset / "labels" / args.split

    if not neg_dir.exists():
        raise SystemExit(f"Negative directory not found: {neg_dir}")
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped_existing = 0
    skipped_non_images = 0

    for img in sorted(neg_dir.iterdir()):
        if not img.is_file():
            continue
        if img.suffix.lower() not in IMG_EXTS:
            skipped_non_images += 1
            continue

        stem = img.stem
        dst_img = img_dst / img.name
        dst_lbl = lbl_dst / f"{stem}.txt"

        if dst_img.exists() or dst_lbl.exists():
            skipped_existing += 1
            continue

        if not args.dry_run:
            shutil.copy2(img, dst_img)
            dst_lbl.write_text("", encoding="utf-8")

        added += 1

    print("neg_dir:", neg_dir)
    print("dataset:", dataset)
    print("split:", args.split)
    print("added:", added)
    print("skipped_existing:", skipped_existing)
    print("skipped_non_images:", skipped_non_images)
    print("dry_run:", args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())