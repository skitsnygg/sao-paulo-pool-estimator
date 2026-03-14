#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class Sample:
    stem: str
    image_path: Path
    label_path: Path
    is_positive: bool
    split_source: str
    rel_image: str
    rel_label: str
    bg_kind: str  # positive | easy_natural | hard_urban
    score: dict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rebuild YOLO segmentation train/val split using stricter image-content filtering."
    )
    ap.add_argument("--src-dataset", type=Path, required=True)
    ap.add_argument("--out-dataset", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260314)

    ap.add_argument("--val-positives", type=int, default=116)
    ap.add_argument("--val-backgrounds", type=int, default=116)

    ap.add_argument(
        "--max-easy-backgrounds-in-val",
        type=int,
        default=10,
        help="Hard cap for easy natural backgrounds in validation.",
    )
    ap.add_argument(
        "--train-easy-ratio",
        type=float,
        default=0.35,
        help="Keep at most this many easy-natural train backgrounds relative to hard-urban train backgrounds.",
    )

    ap.add_argument(
        "--downsample-max-side",
        type=int,
        default=256,
        help="Resize long side to this for scoring speed.",
    )
    ap.add_argument(
        "--easy-threshold",
        type=float,
        default=0.62,
        help="Higher means fewer backgrounds become easy_natural.",
    )
    ap.add_argument(
        "--min-blue-ratio-for-water-bias",
        type=float,
        default=0.08,
        help="Boost easy-natural score for likely water-heavy tiles.",
    )
    ap.add_argument(
        "--min-green-ratio-for-vegetation-bias",
        type=float,
        default=0.16,
        help="Boost easy-natural score for likely vegetation-heavy tiles.",
    )
    ap.add_argument(
        "--edge-density-hard-threshold",
        type=float,
        default=0.055,
        help="Tiles above this edge density are more likely to be hard urban.",
    )
    ap.add_argument(
        "--line-score-hard-threshold",
        type=float,
        default=0.08,
        help="Tiles above this line score are more likely to be hard urban.",
    )

    ap.add_argument(
        "--clean-out",
        action="store_true",
        default=True,
        help="Delete output dataset if it exists.",
    )
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


def score_background_image(img_path: Path, downsample_max_side: int) -> dict:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    img = resize_for_scoring(img, downsample_max_side)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0].astype(np.float32) * 2.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    green_mask = (
        (h >= 45) & (h <= 165) &
        (s >= 0.18) &
        (v >= 0.12)
    )

    blue_mask = (
        (h >= 165) & (h <= 260) &
        (s >= 0.12) &
        (v >= 0.10)
    )

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

    natural_easy_score = (
        green_ratio * 1.2 +
        blue_ratio * 1.2 +
        (1.0 - edge_score) * 1.1 +
        (1.0 - sat_var_score) * 0.6 +
        (1.0 - bright_var_score) * 0.5 +
        (1.0 - texture_score) * 0.6 +
        (1.0 - line_score) * 0.9
    ) / 6.1

    urban_hard_score = (
        edge_score * 1.1 +
        sat_var_score * 0.5 +
        bright_var_score * 0.4 +
        texture_score * 0.8 +
        line_score * 1.2
    ) / 4.0

    return {
        "green_ratio": round(green_ratio, 6),
        "blue_ratio": round(blue_ratio, 6),
        "edge_density": round(edge_density, 6),
        "sat_std": round(sat_std, 6),
        "val_std": round(val_std, 6),
        "gray_std": round(gray_std, 6),
        "line_count": int(line_count),
        "line_score": round(line_score, 6),
        "edge_score": round(edge_score, 6),
        "sat_var_score": round(sat_var_score, 6),
        "bright_var_score": round(bright_var_score, 6),
        "texture_score": round(texture_score, 6),
        "natural_easy_score": round(natural_easy_score, 6),
        "urban_hard_score": round(urban_hard_score, 6),
    }


def classify_background(
    score: dict,
    easy_threshold: float,
    min_blue_ratio_for_water_bias: float,
    min_green_ratio_for_vegetation_bias: float,
    edge_density_hard_threshold: float,
    line_score_hard_threshold: float,
) -> str:
    green_ratio = float(score["green_ratio"])
    blue_ratio = float(score["blue_ratio"])
    natural_easy_score = float(score["natural_easy_score"])
    edge_density = float(score["edge_density"])
    line_score = float(score["line_score"])

    if edge_density >= edge_density_hard_threshold or line_score >= line_score_hard_threshold:
        return "hard_urban"

    if blue_ratio >= min_blue_ratio_for_water_bias and natural_easy_score >= (easy_threshold - 0.05):
        return "easy_natural"

    if green_ratio >= min_green_ratio_for_vegetation_bias and natural_easy_score >= (easy_threshold - 0.03):
        return "easy_natural"

    if natural_easy_score >= easy_threshold:
        return "easy_natural"

    return "hard_urban"


def load_split_samples(
    dataset_root: Path,
    split: str,
    args: argparse.Namespace,
) -> list[Sample]:
    img_dir = dataset_root / "images" / split
    lab_dir = dataset_root / "labels" / split
    if not img_dir.exists() or not lab_dir.exists():
        raise RuntimeError(f"Missing split dirs: {img_dir} / {lab_dir}")

    samples: list[Sample] = []

    for label_path in sorted(lab_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image_for_stem(img_dir, stem)
        if image_path is None:
            raise RuntimeError(f"Missing image for label: {label_path}")

        is_positive = read_label_is_positive(label_path)
        rel_image = image_path.relative_to(dataset_root).as_posix()
        rel_label = label_path.relative_to(dataset_root).as_posix()

        if is_positive:
            bg_kind = "positive"
            score = {}
        else:
            score = score_background_image(image_path, args.downsample_max_side)
            bg_kind = classify_background(
                score=score,
                easy_threshold=args.easy_threshold,
                min_blue_ratio_for_water_bias=args.min_blue_ratio_for_water_bias,
                min_green_ratio_for_vegetation_bias=args.min_green_ratio_for_vegetation_bias,
                edge_density_hard_threshold=args.edge_density_hard_threshold,
                line_score_hard_threshold=args.line_score_hard_threshold,
            )

        samples.append(
            Sample(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                is_positive=is_positive,
                split_source=split,
                rel_image=rel_image,
                rel_label=rel_label,
                bg_kind=bg_kind,
                score=score,
            )
        )

    return samples


def copy_sample(sample: Sample, dst_root: Path, split: str) -> None:
    img_dst = dst_root / "images" / split / sample.image_path.name
    lab_dst = dst_root / "labels" / split / sample.label_path.name
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lab_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, img_dst)
    shutil.copy2(sample.label_path, lab_dst)


def summarize(samples: list[Sample]) -> dict[str, int]:
    return {
        "total": len(samples),
        "positives": sum(1 for s in samples if s.is_positive),
        "backgrounds": sum(1 for s in samples if not s.is_positive),
        "easy_natural": sum(1 for s in samples if s.bg_kind == "easy_natural"),
        "hard_urban": sum(1 for s in samples if s.bg_kind == "hard_urban"),
    }


def main() -> int:
    args = parse_args()

    src_root = args.src_dataset.resolve()
    out_root = args.out_dataset.resolve()

    all_samples = (
        load_split_samples(src_root, "train", args)
        + load_split_samples(src_root, "val", args)
    )

    stems = [s.stem for s in all_samples]
    if len(stems) != len(set(stems)):
        dupes = sorted({x for x in stems if stems.count(x) > 1})[:20]
        raise RuntimeError(f"Duplicate stems found across source dataset: {dupes}")

    positives = [s for s in all_samples if s.is_positive]
    easy_bgs = [s for s in all_samples if s.bg_kind == "easy_natural"]
    hard_bgs = [s for s in all_samples if s.bg_kind == "hard_urban"]

    rng = random.Random(args.seed)
    rng.shuffle(positives)

    hard_bgs = sorted(
        hard_bgs,
        key=lambda s: (
            -float(s.score.get("urban_hard_score", 0.0)),
            float(s.score.get("natural_easy_score", 0.0)),
            s.stem,
        ),
    )
    easy_bgs = sorted(
        easy_bgs,
        key=lambda s: (
            -float(s.score.get("natural_easy_score", 0.0)),
            float(s.score.get("urban_hard_score", 0.0)),
            s.stem,
        ),
    )

    val_pos_n = min(args.val_positives, len(positives))
    val_bg_n = min(args.val_backgrounds, len(easy_bgs) + len(hard_bgs))
    val_easy_n = min(args.max_easy_backgrounds_in_val, len(easy_bgs), val_bg_n)
    val_hard_n = min(len(hard_bgs), val_bg_n - val_easy_n)

    remaining_val = val_bg_n - (val_easy_n + val_hard_n)
    if remaining_val > 0:
        extra_easy = min(remaining_val, len(easy_bgs) - val_easy_n)
        val_easy_n += extra_easy
        remaining_val -= extra_easy
    if remaining_val > 0:
        extra_hard = min(remaining_val, len(hard_bgs) - val_hard_n)
        val_hard_n += extra_hard
        remaining_val -= extra_hard

    val_samples = (
        positives[:val_pos_n]
        + hard_bgs[:val_hard_n]
        + easy_bgs[:val_easy_n]
    )
    val_stems = {s.stem for s in val_samples}

    remaining_samples = [s for s in all_samples if s.stem not in val_stems]
    train_pos = [s for s in remaining_samples if s.is_positive]
    train_easy = [s for s in remaining_samples if s.bg_kind == "easy_natural"]
    train_hard = [s for s in remaining_samples if s.bg_kind == "hard_urban"]

    max_train_easy = int(round(len(train_hard) * args.train_easy_ratio))
    max_train_easy = min(max_train_easy, len(train_easy))

    train_easy = sorted(
        train_easy,
        key=lambda s: (
            float(s.score.get("natural_easy_score", 0.0)),
            -float(s.score.get("urban_hard_score", 0.0)),
            s.stem,
        ),
    )
    kept_train_easy = train_easy[:max_train_easy]
    dropped_train_easy = train_easy[max_train_easy:]

    train_samples = train_pos + train_hard + kept_train_easy

    if args.clean_out and out_root.exists():
        shutil.rmtree(out_root)

    for s in sorted(train_samples, key=lambda x: x.stem):
        copy_sample(s, out_root, "train")
    for s in sorted(val_samples, key=lambda x: x.stem):
        copy_sample(s, out_root, "val")

    dataset_yaml = out_root / "dataset.yaml"
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

    src_summary = summarize(all_samples)
    train_summary = summarize(train_samples)
    val_summary = summarize(val_samples)

    summary = {
        "source_dataset": src_root.as_posix(),
        "output_dataset": out_root.as_posix(),
        "seed": args.seed,
        "params": {
            "val_positives": args.val_positives,
            "val_backgrounds": args.val_backgrounds,
            "max_easy_backgrounds_in_val": args.max_easy_backgrounds_in_val,
            "train_easy_ratio": args.train_easy_ratio,
            "downsample_max_side": args.downsample_max_side,
            "easy_threshold": args.easy_threshold,
            "min_blue_ratio_for_water_bias": args.min_blue_ratio_for_water_bias,
            "min_green_ratio_for_vegetation_bias": args.min_green_ratio_for_vegetation_bias,
            "edge_density_hard_threshold": args.edge_density_hard_threshold,
            "line_score_hard_threshold": args.line_score_hard_threshold,
        },
        "source_summary": src_summary,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "dropped": {
            "dropped_train_easy_natural": len(dropped_train_easy),
        },
        "preview": {
            "val_easy_natural": [
                {
                    "rel_image": s.rel_image,
                    "natural_easy_score": s.score.get("natural_easy_score"),
                    "urban_hard_score": s.score.get("urban_hard_score"),
                    "green_ratio": s.score.get("green_ratio"),
                    "blue_ratio": s.score.get("blue_ratio"),
                }
                for s in val_samples
                if s.bg_kind == "easy_natural"
            ][:30],
            "val_hard_urban": [
                {
                    "rel_image": s.rel_image,
                    "natural_easy_score": s.score.get("natural_easy_score"),
                    "urban_hard_score": s.score.get("urban_hard_score"),
                    "green_ratio": s.score.get("green_ratio"),
                    "blue_ratio": s.score.get("blue_ratio"),
                }
                for s in val_samples
                if s.bg_kind == "hard_urban"
            ][:30],
            "dropped_train_easy_natural": [
                {
                    "rel_image": s.rel_image,
                    "natural_easy_score": s.score.get("natural_easy_score"),
                    "urban_hard_score": s.score.get("urban_hard_score"),
                    "green_ratio": s.score.get("green_ratio"),
                    "blue_ratio": s.score.get("blue_ratio"),
                }
                for s in dropped_train_easy[:50]
            ],
        },
    }

    summary_path = out_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Source dataset: {src_root}")
    print(f"Output dataset: {out_root}")
    print(f"Source total: {src_summary['total']}")
    print(
        f"Source positives/backgrounds: "
        f"{src_summary['positives']}/{src_summary['backgrounds']}"
    )
    print(
        f"Source easy_natural/hard_urban: "
        f"{src_summary['easy_natural']}/{src_summary['hard_urban']}"
    )
    print()
    print(
        f"Train total: {train_summary['total']} | "
        f"positives={train_summary['positives']} | "
        f"backgrounds={train_summary['backgrounds']} | "
        f"easy_natural={train_summary['easy_natural']} | "
        f"hard_urban={train_summary['hard_urban']}"
    )
    print(
        f"Val total: {val_summary['total']} | "
        f"positives={val_summary['positives']} | "
        f"backgrounds={val_summary['backgrounds']} | "
        f"easy_natural={val_summary['easy_natural']} | "
        f"hard_urban={val_summary['hard_urban']}"
    )
    print(f"Dropped easy_natural from train: {len(dropped_train_easy)}")
    print(f"Summary: {summary_path}")
    print(f"Dataset YAML: {dataset_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())