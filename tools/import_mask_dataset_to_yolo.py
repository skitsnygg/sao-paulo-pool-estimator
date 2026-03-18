#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


@dataclass
class Stats:
    images_total: int = 0
    images_written: int = 0
    labels_written: int = 0
    labels_positive: int = 0
    labels_empty: int = 0
    objects_total: int = 0
    missing_masks: int = 0
    unreadable_masks: int = 0
    ambiguous_masks: int = 0
    skipped_existing: int = 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Convert a CVAT mask export (or a generic images+masks folder) into a YOLO "
            "segmentation dataset split. Always writes a label file per image (empty for negatives)."
        )
    )
    ap.add_argument("--src", required=True, help="Source export/workspace root.")
    ap.add_argument("--dataset", required=True, help="YOLO dataset root.")
    ap.add_argument("--split", default="train", help="Output split name (default: train).")
    ap.add_argument("--images-subdir", default="", help="Override image subdir (auto-detect by default).")
    ap.add_argument("--masks-subdir", default="", help="Override mask subdir (auto-detect by default).")
    ap.add_argument("--pool-label", default="pool", help="Class label name to read from labelmap.txt.")
    ap.add_argument("--pool-rgb", default="", help="Optional pool RGB override, e.g. 170,60,29.")
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id to emit.")
    ap.add_argument("--min-area-px", type=float, default=10.0, help="Minimum connected-component area in pixels.")
    ap.add_argument(
        "--simplify-epsilon-ratio",
        type=float,
        default=0.002,
        help="Douglas-Peucker epsilon ratio of contour perimeter.",
    )
    ap.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copying.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing image/label pairs.")
    ap.add_argument("--stats-json", default="", help="Optional explicit path for stats JSON.")
    ap.add_argument("--dry-run", action="store_true", help="Report actions without writing files.")
    return ap.parse_args()


def iter_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() in exts:
            yield p


def has_any_files(root: Path, exts: Tuple[str, ...]) -> bool:
    return any(True for _ in iter_files(root, exts))


def auto_find_images_dir(src: Path) -> Path:
    candidates = [
        src / "JPEGImages",
        src / "images",
        src / "Images",
        src,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir() and has_any_files(cand, IMG_EXTS):
            return cand
    raise SystemExit(
        "Could not auto-detect image directory in source. "
        "Pass --images-subdir explicitly."
    )


def auto_find_masks_dir(src: Path) -> Optional[Path]:
    candidates = [
        src / "SegmentationObject",
        src / "SegmentationClass",
        src / "masks",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir() and has_any_files(cand, MASK_EXTS):
            return cand
    return None


def parse_rgb(text: str) -> Optional[Tuple[int, int, int]]:
    raw = text.strip()
    if not raw:
        return None
    parts = [x.strip() for x in raw.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"Invalid --pool-rgb '{text}'. Expected format R,G,B.")
    try:
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise SystemExit(f"Invalid --pool-rgb '{text}': {exc}") from exc
    return r, g, b


def parse_labelmap(path: Path) -> Dict[str, Tuple[int, int, int]]:
    out: Dict[str, Tuple[int, int, int]] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        label, rgb_text = line.split(":", 1)
        label = label.strip()
        # CVAT exports may use "label:r,g,b:parts:actions".
        rgb_text = rgb_text.split(":", 1)[0].strip()
        parts = [p.strip() for p in rgb_text.split(",")]
        if len(parts) < 3:
            continue
        try:
            rgb = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        out[label] = rgb
    return out


def choose_pool_rgb(args_pool_rgb: str, pool_label: str, labelmap_path: Path) -> Optional[Tuple[int, int, int]]:
    override = parse_rgb(args_pool_rgb)
    if override is not None:
        return override
    lm = parse_labelmap(labelmap_path)
    if pool_label in lm:
        return lm[pool_label]
    return None


def find_mask_for_image(
    image_path: Path,
    images_dir: Path,
    masks_dir: Optional[Path],
    masks_by_stem: Dict[str, list[Path]],
) -> tuple[Optional[Path], bool]:
    if masks_dir is None:
        return None, False

    rel = image_path.relative_to(images_dir)
    stem = rel.with_suffix("").as_posix()
    # Prefer same relative path with a mask extension.
    for ext in MASK_EXTS:
        cand = masks_dir / f"{stem}{ext}"
        if cand.exists():
            return cand, False

    # Fallback: stem-only lookup.
    stem_only = image_path.stem
    matches = masks_by_stem.get(stem_only, [])
    if len(matches) == 1:
        return matches[0], False
    if len(matches) > 1:
        return None, True
    return None, False


def build_mask_stem_index(masks_dir: Optional[Path]) -> Dict[str, list[Path]]:
    out: Dict[str, list[Path]] = {}
    if masks_dir is None:
        return out
    for p in iter_files(masks_dir, MASK_EXTS):
        out.setdefault(p.stem, []).append(p)
    for k in out:
        out[k] = sorted(out[k], key=lambda x: x.as_posix())
    return out


def foreground_from_mask(mask: np.ndarray, pool_rgb: Optional[Tuple[int, int, int]]) -> np.ndarray:
    if mask.ndim == 2:
        return (mask > 0).astype(np.uint8)

    # OpenCV loads color as BGR.
    if mask.ndim == 3 and mask.shape[2] >= 3:
        if pool_rgb is not None:
            target_bgr = np.array([pool_rgb[2], pool_rgb[1], pool_rgb[0]], dtype=np.uint8).reshape(1, 1, 3)
            exact = np.all(mask[:, :, :3] == target_bgr, axis=2).astype(np.uint8)
            if int(exact.sum()) > 0:
                return exact
            # Some exports store instances in object-id colors rather than class RGB.
            # Fall back to generic foreground in that case.
            return np.any(mask[:, :, :3] > 0, axis=2).astype(np.uint8)
        return np.any(mask[:, :, :3] > 0, axis=2).astype(np.uint8)

    return np.zeros(mask.shape[:2], dtype=np.uint8)


def contour_to_line(
    contour: np.ndarray,
    width: int,
    height: int,
    class_id: int,
    simplify_epsilon_ratio: float,
) -> Optional[str]:
    if contour is None or len(contour) < 3:
        return None
    peri = float(cv2.arcLength(contour, True))
    if peri <= 0.0:
        return None
    eps = simplify_epsilon_ratio * peri
    approx = cv2.approxPolyDP(contour, eps, True)
    pts = approx.reshape(-1, 2) if approx is not None else contour.reshape(-1, 2)
    if pts.shape[0] < 3:
        return None

    coords: list[str] = []
    for x, y in pts:
        xn = min(max(float(x) / float(width), 0.0), 1.0)
        yn = min(max(float(y) / float(height), 0.0), 1.0)
        coords.append(f"{xn:.6f}")
        coords.append(f"{yn:.6f}")
    if len(coords) < 6:
        return None
    return f"{class_id} " + " ".join(coords)


def mask_to_yolo_lines(
    mask_path: Path,
    class_id: int,
    min_area_px: float,
    simplify_epsilon_ratio: float,
    pool_rgb: Optional[Tuple[int, int, int]],
) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask image: {mask_path}")

    h, w = mask.shape[:2]
    fg = foreground_from_mask(mask, pool_rgb)
    if fg is None or fg.shape[:2] != (h, w):
        return []

    num_labels, comp_ids, comp_stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return []

    lines: list[str] = []
    for component_id in range(1, num_labels):
        area = float(comp_stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        component_mask = (comp_ids == component_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            line = contour_to_line(
                contour=contour,
                width=w,
                height=h,
                class_id=class_id,
                simplify_epsilon_ratio=simplify_epsilon_ratio,
            )
            if line is not None:
                lines.append(line)
    return lines


def write_dataset_yaml(dataset_root: Path) -> None:
    yaml_path = dataset_root / "dataset.yaml"
    content = "\n".join(
        [
            f"path: {dataset_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names:",
            "  0: pool",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")


def write_split_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stem",
        "image_src",
        "image_dst",
        "label_dst",
        "mask_src",
        "objects",
        "empty_label",
        "ambiguous_mask",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()

    src = Path(args.src).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    if args.images_subdir.strip():
        images_dir = (src / args.images_subdir).resolve()
    else:
        images_dir = auto_find_images_dir(src)

    if args.masks_subdir.strip():
        masks_dir = (src / args.masks_subdir).resolve()
    else:
        masks_dir = auto_find_masks_dir(src)

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")

    split = str(args.split).strip()
    if not split:
        raise SystemExit("--split must be non-empty.")

    img_dst = dataset / "images" / split
    lbl_dst = dataset / "labels" / split
    stats_json = Path(args.stats_json).expanduser().resolve() if args.stats_json else dataset / f"import_{split}_stats.json"
    manifest_csv = dataset / f"import_{split}_manifest.csv"

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    pool_rgb = choose_pool_rgb(args.pool_rgb, args.pool_label, src / "labelmap.txt")
    masks_by_stem = build_mask_stem_index(masks_dir)

    stats = Stats()
    rows: list[dict] = []

    image_paths = list(iter_files(images_dir, IMG_EXTS))
    for img in image_paths:
        stats.images_total += 1
        stem = img.stem
        dst_img = img_dst / img.name
        dst_lbl = lbl_dst / f"{stem}.txt"

        if (dst_img.exists() or dst_lbl.exists()) and not args.overwrite:
            stats.skipped_existing += 1
            continue

        mask_path, ambiguous = find_mask_for_image(img, images_dir, masks_dir, masks_by_stem)
        if ambiguous:
            stats.ambiguous_masks += 1

        lines: list[str] = []
        if mask_path is None:
            stats.missing_masks += 1
        else:
            try:
                lines = mask_to_yolo_lines(
                    mask_path=mask_path,
                    class_id=int(args.class_id),
                    min_area_px=float(args.min_area_px),
                    simplify_epsilon_ratio=float(args.simplify_epsilon_ratio),
                    pool_rgb=pool_rgb,
                )
            except RuntimeError:
                stats.unreadable_masks += 1
                lines = []

        if lines:
            stats.labels_positive += 1
            stats.objects_total += len(lines)
        else:
            stats.labels_empty += 1

        if not args.dry_run:
            if args.symlink_images:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                dst_img.symlink_to(img)
            else:
                shutil.copy2(img, dst_img)

            # Always write a label file. Empty means "no pool".
            dst_lbl.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

        stats.images_written += 1
        stats.labels_written += 1
        rows.append(
            {
                "stem": stem,
                "image_src": str(img),
                "image_dst": str(dst_img),
                "label_dst": str(dst_lbl),
                "mask_src": "" if mask_path is None else str(mask_path),
                "objects": len(lines),
                "empty_label": int(len(lines) == 0),
                "ambiguous_mask": int(ambiguous),
            }
        )

    if not args.dry_run:
        write_dataset_yaml(dataset)
        write_split_manifest(manifest_csv, rows)
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats_json.write_text(
            json.dumps(
                {
                    "src": str(src),
                    "images_dir": str(images_dir),
                    "masks_dir": "" if masks_dir is None else str(masks_dir),
                    "dataset": str(dataset),
                    "split": split,
                    "pool_rgb": pool_rgb,
                    **stats.__dict__,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print("src:", src)
    print("images_dir:", images_dir)
    print("masks_dir:", "" if masks_dir is None else masks_dir)
    print("dataset:", dataset)
    print("split:", split)
    print("pool_rgb:", pool_rgb)
    print("images_total:", stats.images_total)
    print("images_written:", stats.images_written)
    print("labels_written:", stats.labels_written)
    print("labels_positive:", stats.labels_positive)
    print("labels_empty:", stats.labels_empty)
    print("objects_total:", stats.objects_total)
    print("missing_masks:", stats.missing_masks)
    print("unreadable_masks:", stats.unreadable_masks)
    print("ambiguous_masks:", stats.ambiguous_masks)
    print("skipped_existing:", stats.skipped_existing)
    print("dry_run:", bool(args.dry_run))
    if not args.dry_run:
        print("manifest_csv:", manifest_csv)
        print("stats_json:", stats_json)


if __name__ == "__main__":
    main()
