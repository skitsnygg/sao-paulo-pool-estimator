#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys

# Constants
SRC_DATASET = Path("data/train_idesp_yolo11_v2")
DST_DATASET = Path("data/train_idesp_yolo11_v3")
SEED = 20250306
VAL_POS_FRACTION = 0.20
TRAIN_NEG_PER_POS = 1.5
VAL_NEG_PER_POS = 1.0

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    rel: Path

    @property
    def stem(self) -> str:
        return self.path.stem


def iter_images(base: Path) -> list[Path]:
    if not base.exists():
        return []
    images = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    images.sort(key=lambda p: p.as_posix())
    return images


def label_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.strip() != ""


def ensure_empty_destination(dst: Path) -> None:
    if not dst.exists():
        return
    if any(p.is_file() for p in dst.rglob("*")):
        raise SystemExit(
            f"Destination has files already: {dst}. Remove it or pick a new path."
        )


def copy_positive(rec: ImageRecord, dst_split: str, src_labels: Path, dst_images: Path, dst_labels: Path) -> None:
    dst_img = dst_images / dst_split / rec.rel
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rec.path, dst_img)

    src_label = src_labels / rec.split / rec.rel.with_suffix(".txt")
    dst_label = dst_labels / dst_split / rec.rel.with_suffix(".txt")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_label, dst_label)


def copy_negative(rec: ImageRecord, dst_split: str, dst_images: Path, dst_labels: Path) -> None:
    dst_img = dst_images / dst_split / rec.rel
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rec.path, dst_img)

    dst_label = dst_labels / dst_split / rec.rel.with_suffix(".txt")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("", encoding="utf-8")


def main() -> int:
    if not SRC_DATASET.exists():
        print(f"Source dataset not found: {SRC_DATASET}", file=sys.stderr)
        return 2

    src_images = SRC_DATASET / "images"
    src_labels = SRC_DATASET / "labels"
    if not src_images.exists() or not src_labels.exists():
        print("Source dataset must contain images/ and labels/", file=sys.stderr)
        return 2

    ensure_empty_destination(DST_DATASET)

    raw_images_by_split: dict[str, list[Path]] = {}
    for split in SPLITS:
        raw_images_by_split[split] = iter_images(src_images / split)

    train_stems = [p.stem for p in raw_images_by_split["train"]]
    val_stems = [p.stem for p in raw_images_by_split["val"]]
    dup_stems_cross = sorted(set(train_stems) & set(val_stems))
    dup_stems_train = [stem for stem, count in Counter(train_stems).items() if count > 1]
    dup_stems_val = [stem for stem, count in Counter(val_stems).items() if count > 1]

    records: list[ImageRecord] = []
    records_by_split: dict[str, list[ImageRecord]] = {"train": [], "val": []}
    seen_label_keys: dict[str, Path] = {}
    duplicate_label_keys: list[str] = []

    for split in SPLITS:
        base = src_images / split
        for img in raw_images_by_split[split]:
            rel = img.relative_to(base)
            label_key = rel.with_suffix("").as_posix()
            if label_key in seen_label_keys:
                duplicate_label_keys.append(label_key)
                continue
            rec = ImageRecord(path=img, split=split, rel=rel)
            records.append(rec)
            records_by_split[split].append(rec)
            seen_label_keys[label_key] = img

    image_keys_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        base = src_images / split
        keys = {
            p.relative_to(base).with_suffix("").as_posix()
            for p in raw_images_by_split[split]
        }
        image_keys_by_split[split] = keys

    label_files_by_split: dict[str, list[Path]] = {}
    labels_without_images: list[Path] = []
    labels_with_image_other_split: list[Path] = []

    for split in SPLITS:
        base = src_labels / split
        label_files = [p for p in base.rglob("*.txt") if p.is_file()]
        label_files.sort(key=lambda p: p.as_posix())
        label_files_by_split[split] = label_files

        for lf in label_files:
            rel = lf.relative_to(base)
            key = rel.with_suffix("").as_posix()
            if key not in image_keys_by_split.get(split, set()):
                labels_without_images.append(lf)
                other = "val" if split == "train" else "train"
                if key in image_keys_by_split.get(other, set()):
                    labels_with_image_other_split.append(lf)

    positives: list[ImageRecord] = []
    negatives: list[ImageRecord] = []
    missing_label_images: list[ImageRecord] = []
    empty_label_images: list[ImageRecord] = []
    label_in_other_split_images: list[ImageRecord] = []

    for rec in records:
        label_path = src_labels / rec.split / rec.rel.with_suffix(".txt")
        if label_path.exists():
            if label_has_content(label_path):
                positives.append(rec)
            else:
                negatives.append(rec)
                empty_label_images.append(rec)
        else:
            negatives.append(rec)
            missing_label_images.append(rec)
            other = "val" if rec.split == "train" else "train"
            other_label = src_labels / other / rec.rel.with_suffix(".txt")
            if other_label.exists():
                label_in_other_split_images.append(rec)

    positives.sort(key=lambda r: r.path.as_posix())
    negatives.sort(key=lambda r: r.path.as_posix())

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    val_pos_count = int(round(len(positives) * VAL_POS_FRACTION))
    val_pos_count = max(0, min(val_pos_count, len(positives)))
    train_pos_count = len(positives) - val_pos_count

    val_pos = positives[:val_pos_count]
    train_pos = positives[val_pos_count:]

    train_neg_target = int(round(train_pos_count * TRAIN_NEG_PER_POS))
    val_neg_target = int(round(val_pos_count * VAL_NEG_PER_POS))

    val_neg = negatives[:val_neg_target]
    train_neg = negatives[val_neg_target:val_neg_target + train_neg_target]

    if len(positives) == 0:
        print("Warning: no positive labels found.")

    if len(negatives) < (train_neg_target + val_neg_target):
        print(
            "Warning: not enough negatives for targets. "
            f"Available={len(negatives)}, Target={train_neg_target + val_neg_target}"
        )

    dst_images = DST_DATASET / "images"
    dst_labels = DST_DATASET / "labels"
    (dst_images / "train").mkdir(parents=True, exist_ok=True)
    (dst_images / "val").mkdir(parents=True, exist_ok=True)
    (dst_labels / "train").mkdir(parents=True, exist_ok=True)
    (dst_labels / "val").mkdir(parents=True, exist_ok=True)

    for rec in train_pos:
        copy_positive(rec, "train", src_labels, dst_images, dst_labels)
    for rec in train_neg:
        copy_negative(rec, "train", dst_images, dst_labels)
    for rec in val_pos:
        copy_positive(rec, "val", src_labels, dst_images, dst_labels)
    for rec in val_neg:
        copy_negative(rec, "val", dst_images, dst_labels)

    total_images_raw = len(raw_images_by_split["train"]) + len(raw_images_by_split["val"])
    total_labels_raw = len(label_files_by_split.get("train", [])) + len(label_files_by_split.get("val", []))

    print("Source scan:")
    print(f"  images total (raw): {total_images_raw}")
    print(f"  images total (deduped label keys): {len(records)}")
    print(f"  labels total (raw): {total_labels_raw}")

    print("Issues check:")
    print(f"  images missing label file: {len(missing_label_images)}")
    print(f"  images with empty label file: {len(empty_label_images)}")
    print(f"  labels without matching image: {len(labels_without_images)}")
    print(f"  labels with image only in other split: {len(labels_with_image_other_split)}")
    print(f"  images missing labels but label exists in other split: {len(label_in_other_split_images)}")
    print(f"  duplicate stems across old train/val: {len(dup_stems_cross)}")
    print(f"  duplicate stems within old train split: {len(dup_stems_train)}")
    print(f"  duplicate stems within old val split: {len(dup_stems_val)}")
    print(f"  duplicate label keys skipped: {len(duplicate_label_keys)}")

    print("Final split counts:")
    print(f"  train positives: {len(train_pos)}")
    print(f"  train negatives: {len(train_neg)}")
    print(f"  val positives: {len(val_pos)}")
    print(f"  val negatives: {len(val_neg)}")
    print(f"  total train images: {len(train_pos) + len(train_neg)}")
    print(f"  total val images: {len(val_pos) + len(val_neg)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
