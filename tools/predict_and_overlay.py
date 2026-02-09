#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    model_path = Path("reports/runs/sao-paulo-pools-seg/weights/best.pt")
    images_dir = Path("data/annotations/jardins_seg/images")
    out_dir = Path("reports/pred_previews")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")
    if not images_dir.exists():
        raise SystemExit(f"Missing images dir: {images_dir}")

    model = YOLO(str(model_path))

    imgs = sorted([p for p in images_dir.glob("*.png")])
    if not imgs:
        raise SystemExit(f"No images found in {images_dir}")

    random.seed(1)
    sample = random.sample(imgs, k=min(30, len(imgs)))

    for p in sample:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skip unreadable: {p}")
            continue

        H, W = bgr.shape[:2]
        res = model.predict(source=str(p), conf=0.25, iou=0.5, verbose=False)[0]

        overlay = bgr.copy()

        # Overlay masks (resize to image size if needed)
        if res.masks is not None and res.masks.data is not None:
            masks = res.masks.data.cpu().numpy()  # (N, h, w) in model space
            for m in masks:
                m = (m > 0.5).astype(np.uint8)  # 0/1
                mh, mw = m.shape[:2]
                if (mh, mw) != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                overlay[m == 1] = (0.5 * overlay[m == 1] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

        # Draw boxes
        if res.boxes is not None and res.boxes.xyxy is not None:
            for b in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [int(v) for v in b]
                x1 = max(0, min(W - 1, x1))
                x2 = max(0, min(W - 1, x2))
                y1 = max(0, min(H - 1, y1))
                y2 = max(0, min(H - 1, y2))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        out = np.concatenate([bgr, overlay], axis=1)
        cv2.imwrite(str(out_dir / p.name), out)

    print(f"Wrote previews to {out_dir}")


if __name__ == "__main__":
    main()
