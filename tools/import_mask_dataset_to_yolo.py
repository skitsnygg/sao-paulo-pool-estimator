#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


def mask_to_yolo(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask image: {mask_path}")

    h, w = mask.shape[:2]

    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue

        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue

        coords = []
        for x, y in pts:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")

        lines.append("0 " + " ".join(coords))

    return lines


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in MASK_EXTS:
        cand = mask_dir / f"{stem}{ext}"
        if cand.exists() and cand.is_file() and not cand.name.startswith("._"):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    ds = Path(args.dataset)

    img_src = src / "images"
    mask_src = src / "masks"

    if not img_src.exists():
        raise SystemExit(f"Missing images dir: {img_src}")
    if not mask_src.exists():
        raise SystemExit(f"Missing masks dir: {mask_src}")

    img_dst = ds / "images" / args.split
    lbl_dst = ds / "labels" / args.split

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    imported = 0
    missing_mask = 0
    unreadable_mask = 0
    empty_mask = 0
    skipped_existing = 0

    for img in sorted(img_src.iterdir()):
        if not img.is_file():
            continue
        if img.name.startswith("._"):
            continue
        if img.suffix.lower() not in IMG_EXTS:
            continue

        stem = img.stem
        mask = find_mask(mask_src, stem)

        if mask is None:
            print(f"Missing mask for: {img.name}")
            missing_mask += 1
            continue

        out_img = img_dst / img.name
        out_lbl = lbl_dst / f"{stem}.txt"

        if out_img.exists() or out_lbl.exists():
            print(f"Skipping existing: {stem}")
            skipped_existing += 1
            continue

        try:
            lines = mask_to_yolo(mask)
        except RuntimeError as e:
            print(str(e))
            unreadable_mask += 1
            continue

        if not lines:
            print(f"Mask had no usable contours: {mask}")
            empty_mask += 1
            continue

        if not args.dry_run:
            shutil.copy2(img, out_img)
            out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        imported += 1
        print(f"Imported: {stem}")

    print()
    print("Imported:", imported)
    print("Missing masks:", missing_mask)
    print("Unreadable masks:", unreadable_mask)
    print("Empty masks:", empty_mask)
    print("Skipped existing:", skipped_existing)
    print("Dry run:", args.dry_run)


if __name__ == "__main__":
    main()