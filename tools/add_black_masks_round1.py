#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def main() -> int:
    images_root = Path("runs/annotate_round1_v2")
    masks_root = Path("data/datasets/geosampa_active_learning_round1/SegmentationObject")

    if not images_root.exists():
        raise SystemExit(f"Missing images root: {images_root}")

    images = sorted(
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    scanned = 0
    already_present = 0
    created = 0

    for img_path in images:
        rel = img_path.relative_to(images_root)
        mask_path = masks_root / rel.with_suffix(".png")
        scanned += 1

        if mask_path.exists():
            already_present += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        black = np.zeros((h, w), dtype=np.uint8)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(mask_path), black)
        if ok:
            created += 1

    print(f"images scanned: {scanned}")
    print(f"masks already present: {already_present}")
    print(f"black masks created: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
