#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np

# batches you exported from CVAT
batches = [
    Path("data/_cvat_exports_idesp/batch1"),
    Path("data/_cvat_exports_idesp/batch2"),
    Path("data/_cvat_exports_idesp/batch3"),
]

# original images
images_dir = Path("data/annotate_idesp_from2020/idesp_v1_1500/images_idesp")

created_total = 0

for batch in batches:
    masks_dir = batch / "SegmentationObject"

    if not masks_dir.exists():
        raise RuntimeError(f"Missing mask directory: {masks_dir}")

    print(f"\nProcessing {batch}")

    for img_path in images_dir.glob("*.jpg"):
        mask_path = masks_dir / (img_path.stem + ".png")

        if mask_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        blank = np.zeros((h, w), dtype=np.uint8)

        cv2.imwrite(str(mask_path), blank)
        created_total += 1

print("\nCreated", created_total, "black masks")