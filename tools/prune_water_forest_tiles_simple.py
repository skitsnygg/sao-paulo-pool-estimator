#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class BgSample:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    rel_image: str
    rel_label: str
    kind: str
    metrics: dict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prune obvious water/forest background tiles from an existing YOLO dataset using simple color rules."
    )
    ap.add_argument("--src-dataset", type=Path, required=True)
    ap.add_argument("--out-dataset", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")

    # copy/remove policy
    ap.add_argument("--drop-from-train", action="store_true", default=True)
    ap.add_argument("--drop-from-val", action="store_true", default=True)

    # speed
    ap.add_argument("--downsample-max-side", type=int, default=256)

    # water rules
    ap.add_argument("--water-blue-ratio", type=float, default=0.14)
    ap.add_argument("--water-edge-density-max", type=float, default=0.11)

    # forest rules
    ap.add_argument("--forest-green-ratio", type=float, default=0.26)
    ap.add_argument("--forest-edge-density-max", type=float, default=0.10)

    # generic natural scene rule
    ap.add_argument("--natural-color-ratio", type=float, default=0.45)
    ap.add_argument("--natural-edge-density-max", type=float, default=0.09)

    ap.add_argument("--clean-out", action="store_true", default=True)
    return ap.parse_args()


def find_image_for_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in sorted(IMG_EXTS):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def label_is_positive(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return bool(label_path.read_text(encoding="utf-8").strip())


def resize_for_scoring(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img_bgr
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def safe_mean(arr: np.ndarray) -> float:
    v = float(np.mean(arr))
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def score_tile(img_path: Path, downsample_max_side: int) -> dict:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    img = resize_for_scoring(img, downsample_max_side)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0].astype(np.float32) * 2.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Broad blue/cyan mask for lakes/rivers/water
    blue_mask = (
        (h >= 145) & (h <= 270) &
        (s >= 0.08) &
        (v >= 0.06)
    )

    # Broad green mask for vegetation/forest
    green_mask = (
        (h >= 40) & (h <= 165) &
        (s >= 0.14) &
        (v >= 0.10)
    )

    blue_ratio = safe_mean(blue_mask)
    green_ratio = safe_mean(green_mask)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = safe_mean(edges > 0)

    return {
        "blue_ratio": round(blue_ratio, 6),
        "green_ratio": round(green_ratio, 6),
        "natural_ratio": round(blue_ratio + green_ratio, 6),
        "edge_density": round(edge_density, 6),
    }


def classify_background(metrics: dict, args: argparse.Namespace) -> str:
    blue_ratio = float(metrics["blue_ratio"])
    green_ratio = float(metrics["green_ratio"])
    natural_ratio = float(metrics["natural_ratio"])
    edge_density = float(metrics["edge_density"])

    if blue_ratio >= args.water_blue_ratio and edge_density <= args.water_edge_density_max:
        return "drop_water"

    if green_ratio >= args.forest_green_ratio and edge_density <= args.forest_edge_density_max:
        return "drop_forest"

    if natural_ratio >= args.natural_color_ratio and edge_density <= args.natural_edge_density_max:
        return "drop_natural"

    return "keep"


def collect_backgrounds(dataset_root: Path, split: str, args: argparse.Namespace) -> list[BgSample]:
    img_dir = dataset_root / "images" / split
    lab_dir = dataset_root / "labels" / split
    if not img_dir.exists() or not lab_dir.exists():
        raise RuntimeError(f"Missing split dirs: {img_dir} / {lab_dir}")

    samples: list[BgSample] = []
    for label_path in sorted(lab_dir.glob("*.txt")):
        if label_is_positive(label_path):
            continue

        stem = label_path.stem
        image_path = find_image_for_stem(img_dir, stem)
        if image_path is None:
            raise RuntimeError(f"Missing image for label: {label_path}")

        metrics = score_tile(image_path, args.downsample_max_side)
        kind = classify_background(metrics, args)

        samples.append(
            BgSample(
                split=split,
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                rel_image=image_path.relative_to(dataset_root).as_posix(),
                rel_label=label_path.relative_to(dataset_root).as_posix(),
                kind=kind,
                metrics=metrics,
            )
        )
    return samples


def copy_dataset(src_root: Path, dst_root: Path, clean_out: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if clean_out and dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def remove_sample(dst_root: Path, sample: BgSample, dry_run: bool) -> None:
    if dry_run:
        return
    img = dst_root / sample.rel_image
    lab = dst_root / sample.rel_label
    if img.exists():
        img.unlink()
    if lab.exists():
        lab.unlink()


def main() -> int:
    args = parse_args()

    src_root = args.src_dataset.resolve()
    out_root = args.out_dataset.resolve()

    train_bgs = collect_backgrounds(src_root, "train", args)
    val_bgs = collect_backgrounds(src_root, "val", args)

    drop_train = [s for s in train_bgs if s.kind != "keep"] if args.drop_from_train else []
    drop_val = [s for s in val_bgs if s.kind != "keep"] if args.drop_from_val else []

    copy_dataset(src_root, out_root, args.clean_out, args.dry_run)

    for sample in drop_train + drop_val:
        remove_sample(out_root, sample, args.dry_run)

    summary = {
        "source_dataset": src_root.as_posix(),
        "output_dataset": out_root.as_posix(),
        "dry_run": args.dry_run,
        "params": {
            "water_blue_ratio": args.water_blue_ratio,
            "water_edge_density_max": args.water_edge_density_max,
            "forest_green_ratio": args.forest_green_ratio,
            "forest_edge_density_max": args.forest_edge_density_max,
            "natural_color_ratio": args.natural_color_ratio,
            "natural_edge_density_max": args.natural_edge_density_max,
        },
        "dropped": {
            "train_total": len(drop_train),
            "train_drop_water": sum(1 for s in drop_train if s.kind == "drop_water"),
            "train_drop_forest": sum(1 for s in drop_train if s.kind == "drop_forest"),
            "train_drop_natural": sum(1 for s in drop_train if s.kind == "drop_natural"),
            "val_total": len(drop_val),
            "val_drop_water": sum(1 for s in drop_val if s.kind == "drop_water"),
            "val_drop_forest": sum(1 for s in drop_val if s.kind == "drop_forest"),
            "val_drop_natural": sum(1 for s in drop_val if s.kind == "drop_natural"),
        },
        "preview": {
            "train_dropped": [
                {
                    "rel_image": s.rel_image,
                    "kind": s.kind,
                    **s.metrics,
                }
                for s in drop_train[:100]
            ],
            "val_dropped": [
                {
                    "rel_image": s.rel_image,
                    "kind": s.kind,
                    **s.metrics,
                }
                for s in drop_val[:100]
            ],
        },
    }

    if not args.dry_run:
        summary_path = out_root / "prune_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Source dataset: {src_root}")
    print(f"Output dataset: {out_root}")
    print(f"Dry run: {args.dry_run}")
    print()
    print(f"Train backgrounds scanned: {len(train_bgs)}")
    print(f"Val backgrounds scanned: {len(val_bgs)}")
    print()
    print(f"Drop from train: {len(drop_train)}")
    print(f"  water:   {sum(1 for s in drop_train if s.kind == 'drop_water')}")
    print(f"  forest:  {sum(1 for s in drop_train if s.kind == 'drop_forest')}")
    print(f"  natural: {sum(1 for s in drop_train if s.kind == 'drop_natural')}")
    print()
    print(f"Drop from val: {len(drop_val)}")
    print(f"  water:   {sum(1 for s in drop_val if s.kind == 'drop_water')}")
    print(f"  forest:  {sum(1 for s in drop_val if s.kind == 'drop_forest')}")
    print(f"  natural: {sum(1 for s in drop_val if s.kind == 'drop_natural')}")

    if not args.dry_run:
        print(f"\nSummary: {out_root / 'prune_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())