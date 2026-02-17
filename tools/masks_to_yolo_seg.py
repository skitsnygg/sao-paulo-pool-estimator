#!/usr/bin/env python3
"""
Convert binary mask PNGs to YOLOv8 segmentation label format.

Output format per object:
class_id x1 y1 x2 y2 ... (normalized 0–1 polygon points)

Assumes:
- masks are single-channel (0 background, >0 foreground)
- class_id = 0 (pool)
"""

from pathlib import Path
import argparse
import cv2
import numpy as np


def mask_to_polygons(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # binarize
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    h, w = mask.shape

    for cnt in contours:
        if len(cnt) < 3:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2:
            continue

        # normalize
        pts = pts.astype(np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h

        polygons.append(pts.flatten().tolist())

    return polygons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--out-labels-dir", required=True)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    labels_dir = Path(args.out_labels_dir)

    labels_dir.mkdir(parents=True, exist_ok=True)

    count_images = 0
    count_objects = 0

    for img_path in images_dir.iterdir():
        if not img_path.is_file():
            continue

        mask_path = masks_dir / img_path.with_suffix(".png").name
        label_path = labels_dir / img_path.with_suffix(".txt").name

        polygons = []
        if mask_path.exists():
            polygons = mask_to_polygons(mask_path)

        with label_path.open("w") as f:
            for poly in polygons:
                f.write("0 " + " ".join(f"{x:.6f}" for x in poly) + "\n")

        count_images += 1
        count_objects += len(polygons)

    print("Images processed:", count_images)
    print("Total objects:", count_objects)


if __name__ == "__main__":
    main()
