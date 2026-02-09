#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


def main() -> None:
    images_dir = Path("data/annotations/jardins_seg/images")
    masks_dir = Path("data/annotations/jardins_seg/masks")

    if not images_dir.exists():
        print(f"Missing images dir: {images_dir}", file=sys.stderr)
        sys.exit(2)
    if not masks_dir.exists():
        print(f"Missing masks dir: {masks_dir}", file=sys.stderr)
        sys.exit(2)

    created = 0
    for img_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / img_path.name
        if mask_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable image: {img_path}", file=sys.stderr)
            continue
        h, w = img.shape[:2]
        blank = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(mask_path), blank)
        created += 1

    print(f"Created {created} empty masks")


if __name__ == "__main__":
    main()
