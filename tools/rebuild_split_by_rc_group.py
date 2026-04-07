#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val")
RC_RE = re.compile(r"(r\d+_c\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PairedSample:
    source_split: str
    key: str
    stem: str
    rc_stem: str
    image_path: Path
    label_path: Path
    rel_image: Path
    rel_label: Path


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    reason: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild YOLO train/val split by grouping tiles on rc stem "
            "(e.g., r0012_c0008) so each rc group belongs to exactly one split."
        )
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Dataset root containing images/train, images/val, labels/train, labels/val "
            "(example: data/datasets/google_z21_v5)."
        ),
    )
    ap.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Target fraction of images in train after rc-group assignment (default: 0.8).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to deterministically shuffle rc groups (default: 42).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute and print the rebuild plan; do not write files.",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Replace dataset images/ and labels/ with rebuilt splits after writing clean "
            "temporary directories."
        ),
    )
    return ap.parse_args()


def iter_images(base: Path) -> list[Path]:
    if not base.exists():
        return []
    files = [
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    files.sort(key=lambda p: p.as_posix())
    return files


def iter_labels(base: Path) -> list[Path]:
    if not base.exists():
        return []
    files = [p for p in base.rglob("*.txt") if p.is_file()]
    files.sort(key=lambda p: p.as_posix())
    return files


def build_key(rel_path: Path) -> str:
    return rel_path.with_suffix("").as_posix()


def extract_rc_stem(stem: str) -> str | None:
    match = RC_RE.search(stem)
    if not match:
        return None
    return match.group(1).lower()


def collect_split_pairs(dataset_root: Path, split: str) -> tuple[list[PairedSample], list[SkippedFile]]:
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    if not image_root.exists() or not label_root.exists():
        raise SystemExit(f"Missing split directories: {image_root} / {label_root}")

    images_by_key: dict[str, list[Path]] = {}
    labels_by_key: dict[str, list[Path]] = {}
    skipped: list[SkippedFile] = []
    samples: list[PairedSample] = []

    for image_path in iter_images(image_root):
        rel = image_path.relative_to(image_root)
        images_by_key.setdefault(build_key(rel), []).append(image_path)

    for label_path in iter_labels(label_root):
        rel = label_path.relative_to(label_root)
        labels_by_key.setdefault(build_key(rel), []).append(label_path)

    all_keys = sorted(set(images_by_key.keys()) | set(labels_by_key.keys()))
    for key in all_keys:
        image_paths = images_by_key.get(key, [])
        label_paths = labels_by_key.get(key, [])

        if len(image_paths) == 0 and len(label_paths) == 1:
            skipped.append(SkippedFile(path=label_paths[0], reason="label_without_image"))
            continue
        if len(label_paths) == 0 and len(image_paths) == 1:
            skipped.append(SkippedFile(path=image_paths[0], reason="image_without_label"))
            continue
        if len(image_paths) > 1:
            for p in image_paths:
                skipped.append(SkippedFile(path=p, reason="duplicate_image_key"))
            continue
        if len(label_paths) > 1:
            for p in label_paths:
                skipped.append(SkippedFile(path=p, reason="duplicate_label_key"))
            continue
        if len(image_paths) != 1 or len(label_paths) != 1:
            continue

        image_path = image_paths[0]
        label_path = label_paths[0]
        stem = image_path.stem
        rc_stem = extract_rc_stem(stem)
        if rc_stem is None:
            skipped.append(SkippedFile(path=image_path, reason="unparseable_rc_stem"))
            skipped.append(SkippedFile(path=label_path, reason="unparseable_rc_stem"))
            continue

        samples.append(
            PairedSample(
                source_split=split,
                key=key,
                stem=stem,
                rc_stem=rc_stem,
                image_path=image_path,
                label_path=label_path,
                rel_image=image_path.relative_to(image_root),
                rel_label=label_path.relative_to(label_root),
            )
        )

    return samples, skipped


def assign_rc_groups(
    samples: list[PairedSample], train_ratio: float, seed: int
) -> tuple[list[PairedSample], list[PairedSample], set[str], set[str], int]:
    grouped: dict[str, list[PairedSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.rc_stem, []).append(sample)

    rc_keys = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(rc_keys)

    total_images = len(samples)
    target_train_images = int(round(total_images * train_ratio))
    train_rc_groups: set[str] = set()
    train_image_count = 0

    for rc_key in rc_keys:
        group_size = len(grouped[rc_key])
        if train_ratio <= 0.0:
            assign_train = False
        elif train_ratio >= 1.0:
            assign_train = True
        elif train_image_count >= target_train_images:
            assign_train = False
        else:
            take_gap = abs((train_image_count + group_size) - target_train_images)
            skip_gap = abs(train_image_count - target_train_images)
            assign_train = take_gap <= skip_gap

        if assign_train:
            train_rc_groups.add(rc_key)
            train_image_count += group_size

    train_samples: list[PairedSample] = []
    val_samples: list[PairedSample] = []
    for rc_key in rc_keys:
        rows = sorted(grouped[rc_key], key=lambda s: (s.stem, s.image_path.as_posix()))
        if rc_key in train_rc_groups:
            train_samples.extend(rows)
        else:
            val_samples.extend(rows)

    train_samples.sort(key=lambda s: (s.stem, s.image_path.as_posix()))
    val_samples.sort(key=lambda s: (s.stem, s.image_path.as_posix()))
    val_rc_groups = set(rc_keys) - train_rc_groups
    return train_samples, val_samples, train_rc_groups, val_rc_groups, target_train_images


def compute_overlaps(
    train_samples: list[PairedSample], val_samples: list[PairedSample]
) -> tuple[set[str], set[str]]:
    train_exact = {sample.stem for sample in train_samples}
    val_exact = {sample.stem for sample in val_samples}
    exact_overlap = train_exact & val_exact

    train_rc = {sample.rc_stem for sample in train_samples}
    val_rc = {sample.rc_stem for sample in val_samples}
    rc_overlap = train_rc & val_rc

    return exact_overlap, rc_overlap


def make_stage_dir(dataset_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dataset_root / f".tmp_rebuild_split_by_rc_group_{stamp}_{os.getpid()}"


def copy_split_samples(stage_root: Path, split: str, samples: list[PairedSample]) -> None:
    seen_img_dest: set[Path] = set()
    seen_lbl_dest: set[Path] = set()
    for sample in samples:
        dst_image = stage_root / "images" / split / sample.rel_image
        dst_label = stage_root / "labels" / split / sample.rel_label

        if dst_image in seen_img_dest:
            raise RuntimeError(f"Destination image collision: {dst_image}")
        if dst_label in seen_lbl_dest:
            raise RuntimeError(f"Destination label collision: {dst_label}")
        seen_img_dest.add(dst_image)
        seen_lbl_dest.add(dst_label)

        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, dst_image)
        shutil.copy2(sample.label_path, dst_label)


def write_stage(
    stage_root: Path, train_samples: list[PairedSample], val_samples: list[PairedSample]
) -> None:
    if stage_root.exists():
        raise RuntimeError(f"Temporary stage directory already exists: {stage_root}")
    (stage_root / "images" / "train").mkdir(parents=True, exist_ok=False)
    (stage_root / "images" / "val").mkdir(parents=True, exist_ok=False)
    (stage_root / "labels" / "train").mkdir(parents=True, exist_ok=False)
    (stage_root / "labels" / "val").mkdir(parents=True, exist_ok=False)
    copy_split_samples(stage_root, "train", train_samples)
    copy_split_samples(stage_root, "val", val_samples)


def swap_stage_in_place(dataset_root: Path, stage_root: Path) -> Path:
    new_images = stage_root / "images"
    new_labels = stage_root / "labels"
    if not new_images.exists() or not new_labels.exists():
        raise RuntimeError(f"Stage directory missing images/labels: {stage_root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dataset_root / f".backup_before_rebuild_split_by_rc_group_{stamp}_{os.getpid()}"
    backup_root.mkdir(parents=True, exist_ok=False)

    old_images = dataset_root / "images"
    old_labels = dataset_root / "labels"
    backup_images = backup_root / "images"
    backup_labels = backup_root / "labels"

    moved_old_images = False
    moved_old_labels = False
    placed_new_images = False
    placed_new_labels = False
    try:
        if old_images.exists():
            old_images.rename(backup_images)
            moved_old_images = True
        if old_labels.exists():
            old_labels.rename(backup_labels)
            moved_old_labels = True

        new_images.rename(old_images)
        placed_new_images = True
        new_labels.rename(old_labels)
        placed_new_labels = True
    except Exception:
        if placed_new_images and old_images.exists():
            shutil.rmtree(old_images, ignore_errors=True)
        if placed_new_labels and old_labels.exists():
            shutil.rmtree(old_labels, ignore_errors=True)
        if moved_old_images and backup_images.exists():
            backup_images.rename(old_images)
        if moved_old_labels and backup_labels.exists():
            backup_labels.rename(old_labels)
        raise
    finally:
        if stage_root.exists():
            # If rename succeeded, stage_root should be mostly empty.
            shutil.rmtree(stage_root, ignore_errors=True)
        if backup_root.exists() and not any(backup_root.iterdir()):
            backup_root.rmdir()

    return backup_root


def print_summary(
    total_images: int,
    total_rc_groups: int,
    train_samples: list[PairedSample],
    val_samples: list[PairedSample],
    train_rc_groups: set[str],
    val_rc_groups: set[str],
    skipped: list[SkippedFile],
    exact_overlap: set[str],
    rc_overlap: set[str],
    target_train_images: int,
) -> None:
    print(f"total images: {total_images}")
    print(f"total rc groups: {total_rc_groups}")
    print(f"target train images (ratio): {target_train_images}")
    print(f"train images: {len(train_samples)}")
    print(f"val images: {len(val_samples)}")
    print(f"train rc groups: {len(train_rc_groups)}")
    print(f"val rc groups: {len(val_rc_groups)}")
    print(f"skipped files: {len(skipped)}")
    if skipped:
        print("skipped file list:")
        for entry in skipped:
            print(f"  - {entry.reason}: {entry.path.as_posix()}")

    print("post-check:")
    print(f"  - exact overlap count: {len(exact_overlap)}")
    print(f"  - rc overlap count: {len(rc_overlap)}")
    if exact_overlap:
        print("  - exact overlap samples:", ", ".join(sorted(exact_overlap)[:20]))
    if rc_overlap:
        print("  - rc overlap samples:", ", ".join(sorted(rc_overlap)[:20]))


def validate_dataset_layout(dataset_root: Path) -> None:
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [p for p in required_dirs if not p.exists()]
    if missing:
        missing_text = ", ".join(p.as_posix() for p in missing)
        raise SystemExit(f"Dataset missing required directories: {missing_text}")


def main() -> int:
    args = parse_args()
    if args.train_ratio < 0.0 or args.train_ratio > 1.0:
        raise SystemExit("--train-ratio must be between 0.0 and 1.0")

    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset not found: {dataset_root}")
    validate_dataset_layout(dataset_root)

    all_samples: list[PairedSample] = []
    skipped: list[SkippedFile] = []
    for split in SPLITS:
        split_samples, split_skipped = collect_split_pairs(dataset_root, split)
        all_samples.extend(split_samples)
        skipped.extend(split_skipped)

    if not all_samples:
        raise SystemExit("No valid image/label pairs found after filtering.")

    train_samples, val_samples, train_rc_groups, val_rc_groups, target_train_images = assign_rc_groups(
        all_samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    exact_overlap, rc_overlap = compute_overlaps(train_samples, val_samples)
    skipped.sort(key=lambda s: (s.reason, s.path.as_posix()))

    print(f"dataset: {dataset_root.as_posix()}")
    print(f"seed: {args.seed}")
    print(f"train_ratio: {args.train_ratio}")
    print_summary(
        total_images=len(all_samples),
        total_rc_groups=len(train_rc_groups) + len(val_rc_groups),
        train_samples=train_samples,
        val_samples=val_samples,
        train_rc_groups=train_rc_groups,
        val_rc_groups=val_rc_groups,
        skipped=skipped,
        exact_overlap=exact_overlap,
        rc_overlap=rc_overlap,
        target_train_images=target_train_images,
    )

    if exact_overlap or rc_overlap:
        return 1

    if args.dry_run:
        print("dry_run: true")
        print(f"in_place requested: {args.in_place}")
        print("no files were written")
        return 0

    stage_root = make_stage_dir(dataset_root)
    write_stage(stage_root, train_samples, val_samples)
    print(f"staged rebuild written to: {stage_root.as_posix()}")

    if not args.in_place:
        print("in_place: false (original dataset unchanged)")
        return 0

    backup_root = swap_stage_in_place(dataset_root, stage_root)
    print(f"in-place swap complete")
    print(f"backup of previous split stored at: {backup_root.as_posix()}")

    # Re-scan after swap to guarantee post-conditions on current dataset.
    train_after, _ = collect_split_pairs(dataset_root, "train")
    val_after, _ = collect_split_pairs(dataset_root, "val")
    exact_after, rc_after = compute_overlaps(train_after, val_after)
    print("post-check after in-place swap:")
    print(f"  - exact overlap count: {len(exact_after)}")
    print(f"  - rc overlap count: {len(rc_after)}")
    if exact_after or rc_after:
        print("ERROR: overlap detected after in-place swap.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
