#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def find_images(images_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def find_masks(segclass_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in segclass_dir.glob("*.png"):
        if p.is_file():
            out[p.stem] = p
    return out


def mask_color_to_binary(mask_bgr: np.ndarray, bgr: tuple[int, int, int]) -> np.ndarray:
    # exact color match
    target = np.array(bgr, dtype=np.uint8).reshape(1, 1, 3)
    eq = (mask_bgr == target).all(axis=2)
    return (eq.astype(np.uint8) * 255)


def mask_to_polygons(binary_mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[np.ndarray] = []
    for c in contours:
        if len(c) < 3:
            continue
        polys.append(c.reshape(-1, 2))
    return polys


def poly_to_yolo_line(poly_xy: np.ndarray, w: int, h: int, class_id: int = 0) -> str:
    xs = np.clip(poly_xy[:, 0].astype(np.float32) / float(w), 0.0, 1.0)
    ys = np.clip(poly_xy[:, 1].astype(np.float32) / float(h), 0.0, 1.0)
    coords = []
    for x, y in zip(xs, ys):
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return f"{class_id} " + " ".join(coords)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--segclass-dir", required=True, type=Path)
    ap.add_argument("--out-labels-dir", required=True, type=Path)
    ap.add_argument("--pool-rgb", required=True, help="e.g. 170,60,29 (RGB)")
    ap.add_argument("--min-area-px", type=int, default=100)
    args = ap.parse_args()

    # Convert RGB -> BGR because OpenCV reads PNG as BGR
    r, g, b = [int(x) for x in args.pool_rgb.split(",")]
    pool_bgr = (b, g, r)

    images = find_images(args.images_dir)
    masks = find_masks(args.segclass_dir)

    overlap = sorted(list(set(images.keys()) & set(masks.keys())))
    print(f"images={len(images)} masks={len(masks)} overlap={len(overlap)} pool_bgr={pool_bgr}")

    args.out_labels_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_with_instances = 0

    for stem in overlap:
        img = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        m = cv2.imread(str(masks[stem]), cv2.IMREAD_COLOR)
        if m is None:
            continue

        binary = mask_color_to_binary(m, pool_bgr)

        polys = mask_to_polygons(binary)
        lines = []
        for poly in polys:
            area = cv2.contourArea(poly.astype(np.int32))
            if area < args.min_area_px:
                continue
            if poly.shape[0] < 3:
                continue
            lines.append(poly_to_yolo_line(poly, w, h, class_id=0))

        out_path = args.out_labels_dir / f"{stem}.txt"
        out_path.write_text(("\n".join(lines).strip() + "\n") if lines else "")
        n_written += 1
        if lines:
            n_with_instances += 1

    print(f"labels_written={n_written} labels_with_instances={n_with_instances}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
