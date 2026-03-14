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
    score: dict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prune obvious water/forest background tiles from an existing YOLO segmentation dataset."
    )
    ap.add_argument("--src-dataset", type=Path, required=True)
    ap.add_argument("--out-dataset", type=Path, required=True)
    ap.add_argument("--downsample-max-side", type=int, default=256)

    # Drop policy
    ap.add_argument("--drop-from-train", action="store_true", default=True)
    ap.add_argument("--drop-from-val", action="store_true", default=True)

    # Very aggressive open-water rules
    ap.add_argument("--water-blue-ratio", type=float, default=0.22)
    ap.add_argument("--water-line-score-max", type=float, default=0.06)
    ap.add_argument("--water-edge-density-max", type=float, default=0.09)

    # Very aggressive vegetation rules
    ap.add_argument("--forest-green-ratio", type=float, default=0.32)
    ap.add_argument("--forest-line-score-max", type=float, default=0.05)
    ap.add_argument("--forest-edge-density-max", type=float, default=0.08)

    # Generic "obvious natural tile" rules
    ap.add_argument("--natural-score-min", type=float, default=0.62)
    ap.add_argument("--urban-score-max", type=float, default=0.22)

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-out", action="store_true", default=True)
    return ap.parse_args()


def find_image_for_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in sorted(IMG_EXTS):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_label_is_positive(label_path: Path) -> bool:
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


def safe_std(arr: np.ndarray) -> float:
    v = float(np.std(arr))
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_image(img_path: Path, downsample_max_side: int) -> dict:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    img = resize_for_scoring(img, downsample_max_side)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0].astype(np.float32) * 2.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    green_mask = ((h >= 45) & (h <= 165) & (s >= 0.16) & (v >= 0.10))
    blue_mask = ((h >= 150) & (h <= 265) & (s >= 0.10) & (v >= 0.08))

    green_ratio = safe_mean(green_mask)
    blue_ratio = safe_mean(blue_mask)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = safe_mean(edges > 0)

    sat_std = safe_std(s)
    val_std = safe_std(v)
    gray_std = safe_std(gray.astype(np.float32) / 255.0)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=max(10, int(min(gray.shape[:2]) * 0.08)),
        maxLineGap=6,
    )
    line_count = 0 if lines is None else len(lines)
    line_score = clamp01(line_count / 80.0)

    edge_score = clamp01(edge_density / 0.12)
    sat_var_score = clamp01(sat_std / 0.22)
    bright_var_score = clamp01(val_std / 0.22)
    texture_score = clamp01(gray_std / 0.20)

    natural_score = (
        green_ratio * 1.1 +
        blue_ratio * 1.5 +
        (1.0 - edge_score) * 1.0 +
        (1.0 - sat_var_score) * 0.5 +
        (1.0 - bright_var_score) * 0.4 +
        (1.0 - texture_score) * 0.4 +
        (1.0 - line_score) * 1.1
    ) / 6.0

    urban_score = (
        edge_score * 1.1 +
        sat_var_score * 0.5 +
        bright_var_score * 0.4 +
        texture_score * 0.7 +
        line_score * 1.3
    ) / 4.0

    return {
        "green_ratio": round(green_ratio, 6),
        "blue_ratio": round(blue_ratio, 6),
        "edge_density": round(edge_density, 6),
        "line_score": round(line_score, 6),
        "natural_score": round(natural_score, 6),
        "urban_score": round(urban_score, 6),
        "sat_std": round(sat_std, 6),
        "val_std": round(val_std, 6),
        "gray_std": round(gray_std, 6),
        "line_count": int(line_count),
    }


def classify_background(score: dict, args: argparse.Namespace) -> str:
    green_ratio = float(score["green_ratio"])
    blue_ratio = float(score["blue_ratio"])
    edge_density = float(score["edge_density"])
    line_score = float(score["line_score"])
    natural_score = float(score["natural_score"])
    urban_score = float(score["urban_score"])

    if (
        blue_ratio >= args.water_blue_ratio
        and line_score <= args.water_line_score_max
        and edge_density <= args.water_edge_density_max
    ):
        return "drop_water"

    if (
        green_ratio >= args.forest_green_ratio
        and line_score <= args.forest_line_score_max
        and edge_density <= args.forest_edge_density_max
    ):
        return "drop_forest"

    if natural_score >= args.natural_score_min and urban_score <= args.urban_score_max:
        return "drop_natural"

    return "keep"


def collect_backgrounds(dataset_root: Path, split: str, args: argparse.Namespace) -> list[BgSample]:
    img_dir = dataset_root / "images" / split
    lab_dir = dataset_root / "labels" / split
    if not img_dir.exists() or not lab_dir.exists():
        raise RuntimeError(f"Missing split dirs: {img_dir} / {lab_dir}")

    out: list[BgSample] = []
    for label_path in sorted(lab_dir.glob("*.txt")):
        if read_label_is_positive(label_path):
            continue
        stem = label_path.stem
        image_path = find_image_for_stem(img_dir, stem)
        if image_path is None:
            raise RuntimeError(f"Missing image for label: {label_path}")
        score = score_image(image_path, args.downsample_max_side)
        kind = classify_background(score, args)
        out.append(
            BgSample(
                split=split,
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                rel_image=image_path.relative_to(dataset_root).as_posix(),
                rel_label=label_path.relative_to(dataset_root).as_posix(),
                kind=kind,
                score=score,
            )
        )
    return out


def copy_dataset(src_root: Path, dst_root: Path, clean_out: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if clean_out and dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def remove_sample_files(dst_root: Path, sample: BgSample, dry_run: bool) -> None:
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

    to_drop_train = [
        s for s in train_bgs
        if s.kind != "keep" and args.drop_from_train
    ]
    to_drop_val = [
        s for s in val_bgs
        if s.kind != "keep" and args.drop_from_val
    ]

    copy_dataset(src_root, out_root, args.clean_out, args.dry_run)

    for s in to_drop_train + to_drop_val:
        remove_sample_files(out_root, s, args.dry_run)

    summary = {
        "source_dataset": src_root.as_posix(),
        "output_dataset": out_root.as_posix(),
        "dry_run": args.dry_run,
        "params": {
            "water_blue_ratio": args.water_blue_ratio,
            "water_line_score_max": args.water_line_score_max,
            "water_edge_density_max": args.water_edge_density_max,
            "forest_green_ratio": args.forest_green_ratio,
            "forest_line_score_max": args.forest_line_score_max,
            "forest_edge_density_max": args.forest_edge_density_max,
            "natural_score_min": args.natural_score_min,
            "urban_score_max": args.urban_score_max,
        },
        "dropped": {
            "train_total": len(to_drop_train),
            "train_drop_water": sum(1 for s in to_drop_train if s.kind == "drop_water"),
            "train_drop_forest": sum(1 for s in to_drop_train if s.kind == "drop_forest"),
            "train_drop_natural": sum(1 for s in to_drop_train if s.kind == "drop_natural"),
            "val_total": len(to_drop_val),
            "val_drop_water": sum(1 for s in to_drop_val if s.kind == "drop_water"),
            "val_drop_forest": sum(1 for s in to_drop_val if s.kind == "drop_forest"),
            "val_drop_natural": sum(1 for s in to_drop_val if s.kind == "drop_natural"),
        },
        "preview": {
            "train_dropped": [
                {
                    "rel_image": s.rel_image,
                    "kind": s.kind,
                    **s.score,
                }
                for s in to_drop_train[:100]
            ],
            "val_dropped": [
                {
                    "rel_image": s.rel_image,
                    "kind": s.kind,
                    **s.score,
                }
                for s in to_drop_val[:100]
            ],
        },
    }

    if not args.dry_run:
        summary_path = out_root / "prune_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        dataset_yaml = out_root / "dataset.yaml"
        if not dataset_yaml.exists():
            dataset_yaml.write_text(
                "\n".join(
                    [
                        f"path: {out_root.as_posix()}",
                        "train: images/train",
                        "val: images/val",
                        "nc: 1",
                        "names:",
                        "  0: pool",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    print(f"Source dataset: {src_root}")
    print(f"Output dataset: {out_root}")
    print(f"Dry run: {args.dry_run}")
    print()
    print(f"Train backgrounds scanned: {len(train_bgs)}")
    print(f"Val backgrounds scanned: {len(val_bgs)}")
    print()
    print(f"Drop from train: {len(to_drop_train)}")
    print(f"  water:   {sum(1 for s in to_drop_train if s.kind == 'drop_water')}")
    print(f"  forest:  {sum(1 for s in to_drop_train if s.kind == 'drop_forest')}")
    print(f"  natural: {sum(1 for s in to_drop_train if s.kind == 'drop_natural')}")
    print()
    print(f"Drop from val: {len(to_drop_val)}")
    print(f"  water:   {sum(1 for s in to_drop_val if s.kind == 'drop_water')}")
    print(f"  forest:  {sum(1 for s in to_drop_val if s.kind == 'drop_forest')}")
    print(f"  natural: {sum(1 for s in to_drop_val if s.kind == 'drop_natural')}")

    if not args.dry_run:
        print(f"\nSummary: {out_root / 'prune_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())