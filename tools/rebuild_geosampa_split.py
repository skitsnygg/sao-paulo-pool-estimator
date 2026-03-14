#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class Sample:
    stem: str
    image_path: Path
    label_path: Path
    is_positive: bool
    rel_image: str
    rel_label: str
    split_source: str
    category: str  # positive | background_easy | background_hard


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rebuild train/val split for an existing YOLO segmentation dataset."
    )
    ap.add_argument(
        "--src-dataset",
        type=Path,
        required=True,
        help="Existing YOLO dataset root containing images/train, images/val, labels/train, labels/val",
    )
    ap.add_argument(
        "--out-dataset",
        type=Path,
        required=True,
        help="Output dataset root for rewritten split",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260314,
    )
    ap.add_argument(
        "--val-positives",
        type=int,
        default=116,
        help="Target number of positive tiles in val",
    )
    ap.add_argument(
        "--val-backgrounds",
        type=int,
        default=116,
        help="Target number of background tiles in val",
    )
    ap.add_argument(
        "--max-easy-backgrounds-in-val",
        type=int,
        default=25,
        help="Maximum number of easy backgrounds allowed in val",
    )
    ap.add_argument(
        "--easy-keywords",
        type=str,
        default="lake,lakes,water,river,forest,trees,woods,park,green,vegetation",
        help="Comma-separated keywords that mark a background as easy if found in its relative path",
    )
    ap.add_argument(
        "--hard-keywords",
        type=str,
        default="urban,residential,moema,pinheiros,jard,jardins,brooklin,itaim,vila,olimpia,roof,quadra,court,pool,condo,pred",
        help="Comma-separated keywords that mark a background as hard if found in its relative path",
    )
    ap.add_argument(
        "--clean-out",
        action="store_true",
        default=True,
        help="Delete output dataset first if it exists",
    )
    return ap.parse_args()


def stable_hash(text: str, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest(), 16)


def read_label_is_positive(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    txt = label_path.read_text(encoding="utf-8").strip()
    return bool(txt)


def classify_background(rel_path: str, easy_keywords: list[str], hard_keywords: list[str]) -> str:
    s = rel_path.lower()
    if any(k and k in s for k in easy_keywords):
        return "background_easy"
    if any(k and k in s for k in hard_keywords):
        return "background_hard"
    return "background_hard"


def find_image_for_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in sorted(IMG_EXTS):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_split_samples(
    dataset_root: Path,
    split: str,
    easy_keywords: list[str],
    hard_keywords: list[str],
) -> list[Sample]:
    lab_dir = dataset_root / "labels" / split
    img_dir = dataset_root / "images" / split

    if not lab_dir.exists() or not img_dir.exists():
        raise RuntimeError(f"Missing split dirs for {split}: {img_dir} / {lab_dir}")

    samples: list[Sample] = []

    for label_path in sorted(lab_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image_for_stem(img_dir, stem)
        if image_path is None:
            raise RuntimeError(f"Missing image for label stem={stem} split={split}")

        is_positive = read_label_is_positive(label_path)
        rel_image = image_path.relative_to(dataset_root).as_posix()
        rel_label = label_path.relative_to(dataset_root).as_posix()

        if is_positive:
            category = "positive"
        else:
            category = classify_background(rel_image, easy_keywords, hard_keywords)

        samples.append(
            Sample(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                is_positive=is_positive,
                rel_image=rel_image,
                rel_label=rel_label,
                split_source=split,
                category=category,
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
        "background_easy": sum(1 for s in samples if s.category == "background_easy"),
        "background_hard": sum(1 for s in samples if s.category == "background_hard"),
    }


def main() -> int:
    args = parse_args()

    src_root = args.src_dataset.resolve()
    out_root = args.out_dataset.resolve()

    easy_keywords = [x.strip().lower() for x in args.easy_keywords.split(",") if x.strip()]
    hard_keywords = [x.strip().lower() for x in args.hard_keywords.split(",") if x.strip()]

    all_samples = (
        load_split_samples(src_root, "train", easy_keywords, hard_keywords)
        + load_split_samples(src_root, "val", easy_keywords, hard_keywords)
    )

    stems = [s.stem for s in all_samples]
    if len(stems) != len(set(stems)):
        dupes = sorted({x for x in stems if stems.count(x) > 1})[:10]
        raise RuntimeError(f"Duplicate stems across source dataset: {dupes}")

    positives = [s for s in all_samples if s.is_positive]
    backgrounds_easy = [s for s in all_samples if s.category == "background_easy"]
    backgrounds_hard = [s for s in all_samples if s.category == "background_hard"]

    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(backgrounds_easy)
    rng.shuffle(backgrounds_hard)

    val_positives_target = min(args.val_positives, len(positives))
    val_backgrounds_target = min(args.val_backgrounds, len(backgrounds_easy) + len(backgrounds_hard))
    val_easy_target = min(args.max_easy_backgrounds_in_val, len(backgrounds_easy), val_backgrounds_target)
    val_hard_target = min(len(backgrounds_hard), val_backgrounds_target - val_easy_target)

    remaining_bg_needed = val_backgrounds_target - (val_easy_target + val_hard_target)
    if remaining_bg_needed > 0:
        extra_easy = min(remaining_bg_needed, len(backgrounds_easy) - val_easy_target)
        val_easy_target += extra_easy
        remaining_bg_needed -= extra_easy
    if remaining_bg_needed > 0:
        extra_hard = min(remaining_bg_needed, len(backgrounds_hard) - val_hard_target)
        val_hard_target += extra_hard
        remaining_bg_needed -= extra_hard

    val_samples = (
        positives[:val_positives_target]
        + backgrounds_hard[:val_hard_target]
        + backgrounds_easy[:val_easy_target]
    )

    val_stems = {s.stem for s in val_samples}
    train_samples = [s for s in all_samples if s.stem not in val_stems]

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

    train_summary = summarize(train_samples)
    val_summary = summarize(val_samples)
    src_summary = summarize(all_samples)

    summary = {
        "source_dataset": src_root.as_posix(),
        "output_dataset": out_root.as_posix(),
        "seed": args.seed,
        "val_targets": {
            "val_positives": args.val_positives,
            "val_backgrounds": args.val_backgrounds,
            "max_easy_backgrounds_in_val": args.max_easy_backgrounds_in_val,
        },
        "keywords": {
            "easy_keywords": easy_keywords,
            "hard_keywords": hard_keywords,
        },
        "source_summary": src_summary,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "selected_val_samples_preview": {
            "positives": [s.rel_image for s in val_samples if s.is_positive][:25],
            "background_hard": [s.rel_image for s in val_samples if s.category == "background_hard"][:25],
            "background_easy": [s.rel_image for s in val_samples if s.category == "background_easy"][:25],
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
        f"Source easy/hard backgrounds: "
        f"{src_summary['background_easy']}/{src_summary['background_hard']}"
    )
    print()
    print(
        f"Train total: {train_summary['total']} | "
        f"positives={train_summary['positives']} | "
        f"backgrounds={train_summary['backgrounds']} | "
        f"easy={train_summary['background_easy']} | "
        f"hard={train_summary['background_hard']}"
    )
    print(
        f"Val total: {val_summary['total']} | "
        f"positives={val_summary['positives']} | "
        f"backgrounds={val_summary['backgrounds']} | "
        f"easy={val_summary['background_easy']} | "
        f"hard={val_summary['background_hard']}"
    )
    print(f"Summary: {summary_path}")
    print(f"Dataset YAML: {dataset_yaml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())