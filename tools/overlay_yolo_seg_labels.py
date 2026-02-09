#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np


def parse_poly_line(line: str):
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    cls = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    if len(coords) % 2 != 0:
        return None
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    return cls, pts


def main() -> None:
    root = Path("data/processed/yolo_seg")
    images_dir = root / "images" / "val"
    labels_dir = root / "labels" / "val"
    out_dir = Path("reports/label_previews")
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(images_dir.glob("*.png"))
    if not imgs:
        raise SystemExit(f"No images found in {images_dir}")

    random.seed(1)
    sample = random.sample(imgs, k=min(30, len(imgs)))

    for img_path in sample:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        label_path = labels_dir / f"{img_path.stem}.txt"
        overlay = img.copy()

        if label_path.exists():
            txt = label_path.read_text(encoding="utf-8").strip().splitlines()
            for line in txt:
                parsed = parse_poly_line(line)
                if not parsed:
                    continue
                _, pts = parsed
                # pts are normalized [0..1] in YOLO format
                pts_px = np.stack([pts[:, 0] * W, pts[:, 1] * H], axis=1).astype(np.int32)
                if len(pts_px) >= 3:
                    cv2.polylines(overlay, [pts_px], isClosed=True, color=(0, 255, 0), thickness=2)

        out = np.concatenate([img, overlay], axis=1)
        cv2.imwrite(str(out_dir / img_path.name), out)

    print(f"Wrote label previews to {out_dir}")


if __name__ == "__main__":
    main()
