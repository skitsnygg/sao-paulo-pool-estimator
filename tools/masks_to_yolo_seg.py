#!/usr/bin/env python3
"""
Convert binary mask PNGs to YOLOv8 segmentation label format.

Output format per object:
class_id x1 y1 x2 y2 ... (normalized 0–1 polygon points)

Assumptions:
- masks are single-channel or readable as grayscale
- background = 0
- foreground > 0
- class_id = 0 (pool)

Improvements over the original version:
- filters tiny contour artifacts by minimum pixel area
- simplifies polygons with Douglas-Peucker
- handles recursive image discovery
- keeps deterministic output ordering
- guards against malformed contours
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CLASS_ID = 0
DEFAULT_MIN_AREA_PX = 20.0
DEFAULT_SIMPLIFY_EPSILON_RATIO = 0.002


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Directory containing source images.")
    ap.add_argument("--masks-dir", required=True, help="Directory containing binary mask PNGs.")
    ap.add_argument("--out-labels-dir", required=True, help="Directory to write YOLO txt labels.")
    ap.add_argument(
        "--class-id",
        type=int,
        default=DEFAULT_CLASS_ID,
        help=f"YOLO class id to write. Default: {DEFAULT_CLASS_ID}",
    )
    ap.add_argument(
        "--min-area-px",
        type=float,
        default=DEFAULT_MIN_AREA_PX,
        help=f"Minimum contour area in pixels to keep. Default: {DEFAULT_MIN_AREA_PX}",
    )
    ap.add_argument(
        "--simplify-epsilon-ratio",
        type=float,
        default=DEFAULT_SIMPLIFY_EPSILON_RATIO,
        help=(
            "Douglas-Peucker epsilon as a fraction of contour perimeter. "
            f"Default: {DEFAULT_SIMPLIFY_EPSILON_RATIO}"
        ),
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan images-dir for images.",
    )
    ap.add_argument(
        "--image-exts",
        default=".jpg,.jpeg,.png,.webp",
        help="Comma-separated image extensions to include. Default: .jpg,.jpeg,.png,.webp",
    )
    return ap.parse_args()


def clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def normalize_points(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = pts.astype(np.float32).copy()
    pts[:, 0] /= float(width)
    pts[:, 1] /= float(height)
    return clamp01(pts)


def contour_to_polygon(
    cnt: np.ndarray,
    width: int,
    height: int,
    min_area_px: float,
    simplify_epsilon_ratio: float,
) -> list[float]:
    if cnt is None or len(cnt) < 3:
        return []

    area = float(cv2.contourArea(cnt))
    if area < min_area_px:
        return []

    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter <= 0:
        return []

    epsilon = simplify_epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if approx is None or len(approx) < 3:
        approx = cnt

    pts = np.squeeze(approx, axis=1) if approx.ndim == 3 else np.squeeze(approx)

    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return []

    norm = normalize_points(pts, width=width, height=height)

    if norm.shape[0] < 3:
        return []

    return norm.flatten().tolist()


def mask_to_polygons(
    mask_path: Path,
    min_area_px: float = DEFAULT_MIN_AREA_PX,
    simplify_epsilon_ratio: float = DEFAULT_SIMPLIFY_EPSILON_RATIO,
) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    if mask.ndim != 2:
        return []

    height, width = mask.shape[:2]
    if height <= 0 or width <= 0:
        return []

    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return []

    # Deterministic ordering: sort largest-first, then by bounding box.
    def contour_sort_key(cnt: np.ndarray) -> tuple[float, int, int, int, int]:
        area = float(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        return (-area, y, x, h, w)

    contours = sorted(contours, key=contour_sort_key)

    polygons: list[list[float]] = []
    for cnt in contours:
        poly = contour_to_polygon(
            cnt=cnt,
            width=width,
            height=height,
            min_area_px=min_area_px,
            simplify_epsilon_ratio=simplify_epsilon_ratio,
        )
        if poly:
            polygons.append(poly)

    return polygons


def iter_images(images_dir: Path, recursive: bool, image_exts: set[str]) -> list[Path]:
    if recursive:
        paths = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
    else:
        paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    return sorted(paths)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    masks_dir = Path(args.masks_dir).expanduser().resolve()
    labels_dir = Path(args.out_labels_dir).expanduser().resolve()

    image_exts = {ext.strip().lower() for ext in args.image_exts.split(",") if ext.strip()}
    if not image_exts:
        image_exts = set(DEFAULT_EXTS)

    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(
        images_dir=images_dir,
        recursive=args.recursive,
        image_exts=image_exts,
    )

    count_images = 0
    count_with_masks = 0
    count_objects = 0
    count_missing_masks = 0
    count_empty_labels = 0

    for img_path in image_paths:
        rel = img_path.relative_to(images_dir)
        mask_rel = rel.with_suffix(".png")
        label_rel = rel.with_suffix(".txt")

        mask_path = masks_dir / mask_rel
        label_path = labels_dir / label_rel
        label_path.parent.mkdir(parents=True, exist_ok=True)

        polygons: list[list[float]] = []
        if mask_path.exists():
            count_with_masks += 1
            polygons = mask_to_polygons(
                mask_path=mask_path,
                min_area_px=args.min_area_px,
                simplify_epsilon_ratio=args.simplify_epsilon_ratio,
            )
        else:
            count_missing_masks += 1

        with label_path.open("w", encoding="utf-8") as f:
            for poly in polygons:
                f.write(
                    f"{args.class_id} " + " ".join(f"{x:.6f}" for x in poly) + "\n"
                )

        if not polygons:
            count_empty_labels += 1

        count_images += 1
        count_objects += len(polygons)

    print("Images processed:", count_images)
    print("Images with masks:", count_with_masks)
    print("Missing masks:", count_missing_masks)
    print("Empty label files:", count_empty_labels)
    print("Total objects:", count_objects)


if __name__ == "__main__":
    main()