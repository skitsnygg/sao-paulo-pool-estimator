#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
IGNORE_DIR_PREFIXES = (".backup_", ".tmp_")
RC_RE = re.compile(r"(r\d+_c\d+)", flags=re.IGNORECASE)
DRY_RUN_PREVIEW_LIMIT = 20


@dataclass(frozen=True)
class RcSample:
    stem: str
    rc_stem: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "End-to-end cleanup for YOLO dataset splits: remove train/val stem overlap from val, "
            "merge remaining val into train, then run rebuild_split_by_rc_group."
        )
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/datasets/google_z21_v5"),
        help="Dataset root (default: data/datasets/google_z21_v5).",
    )
    ap.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Target train ratio for rebuild_split_by_rc_group (default: 0.8).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic rc-group split rebuild (default: 42).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and report actions without modifying files.",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Pass --in-place to rebuild_split_by_rc_group (ignored when --dry-run is set).",
    )
    return ap.parse_args()


def should_ignore_dir(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in IGNORE_DIR_PREFIXES)


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    files: list[Path] = []
    for current_dir, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(d for d in dir_names if not should_ignore_dir(d))
        for file_name in sorted(file_names):
            files.append(Path(current_dir) / file_name)
    files.sort(key=lambda p: p.as_posix())
    return files


def iter_images(root: Path) -> list[Path]:
    return [p for p in iter_files(root) if p.suffix.lower() in IMG_EXTS]


def iter_labels(root: Path) -> list[Path]:
    return [p for p in iter_files(root) if p.suffix.lower() == ".txt"]


def index_unique_by_stem(paths: list[Path], *, kind: str, split: str) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.stem, []).append(path)

    dup_stems = sorted([stem for stem, rows in grouped.items() if len(rows) > 1])
    if dup_stems:
        details: list[str] = []
        for stem in dup_stems[:20]:
            refs = ", ".join(p.as_posix() for p in sorted(grouped[stem], key=lambda x: x.as_posix()))
            details.append(f"{stem}: {refs}")
        joined = "\n".join(details)
        raise RuntimeError(
            f"Unexpected {kind} stem collisions inside split={split} (count={len(dup_stems)}).\n"
            f"{joined}"
        )

    return {stem: rows[0] for stem, rows in grouped.items()}


def validate_dataset_layout(dataset_root: Path) -> None:
    required = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_text = ", ".join(p.as_posix() for p in missing)
        raise SystemExit(f"Dataset missing required directories: {missing_text}")


def scan_split(dataset_root: Path, split: str) -> tuple[dict[str, Path], dict[str, Path]]:
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    images = index_unique_by_stem(iter_images(image_root), kind="image", split=split)
    labels = index_unique_by_stem(iter_labels(label_root), kind="label", split=split)
    return images, labels


def extract_rc_stem(stem: str) -> str | None:
    match = RC_RE.search(stem)
    if not match:
        return None
    return match.group(1).lower()


def preview_list(title: str, items: list[str], limit: int = 50) -> None:
    print(f"{title}: {len(items)}")
    if not items:
        return
    print(f"{title} (first {min(limit, len(items))}):")
    for stem in items[:limit]:
        print(f"  - {stem}")


def preflight_move_collisions(
    train_images_dir: Path,
    train_labels_dir: Path,
    train_images: dict[str, Path],
    train_labels: dict[str, Path],
    val_images: dict[str, Path],
    val_labels: dict[str, Path],
    duplicate_stems: set[str],
) -> None:
    errors: list[str] = []
    planned_image_dsts: set[Path] = set()
    planned_label_dsts: set[Path] = set()

    for stem in sorted(val_images.keys()):
        if stem in duplicate_stems:
            continue

        src_image = val_images[stem]
        dst_image = train_images_dir / src_image.name
        if stem in train_images:
            errors.append(f"image stem already exists in train: {stem}")
        if dst_image.exists():
            errors.append(f"destination image already exists: {dst_image.as_posix()}")
        if dst_image in planned_image_dsts:
            errors.append(f"planned image destination collision: {dst_image.as_posix()}")
        planned_image_dsts.add(dst_image)

        if stem in val_labels:
            dst_label = train_labels_dir / f"{stem}.txt"
            if stem in train_labels:
                errors.append(f"label stem already exists in train: {stem}")
            if dst_label.exists():
                errors.append(f"destination label already exists: {dst_label.as_posix()}")
            if dst_label in planned_label_dsts:
                errors.append(f"planned label destination collision: {dst_label.as_posix()}")
            planned_label_dsts.add(dst_label)

    if errors:
        details = "\n".join(f"  - {line}" for line in errors[:50])
        raise RuntimeError(
            "Unexpected collisions detected before val->train move. "
            "Aborting to avoid silent overwrite.\n"
            f"{details}"
        )


def remove_file(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.unlink()


def move_file(src: Path, dst: Path, dry_run: bool) -> None:
    if src == dst:
        raise RuntimeError(f"Source and destination are identical: {src.as_posix()}")
    if dst.exists():
        raise RuntimeError(f"Destination exists (refusing overwrite): {dst.as_posix()}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def build_rc_samples(
    train_images: dict[str, Path],
    val_images: dict[str, Path],
    train_labels: dict[str, Path],
    val_labels: dict[str, Path],
) -> tuple[list[RcSample], list[str], list[str]]:
    samples: list[RcSample] = []
    skipped_no_label: list[str] = []
    skipped_no_rc: list[str] = []

    for stem in sorted(train_images.keys()):
        if stem not in train_labels:
            skipped_no_label.append(stem)
            continue
        rc_stem = extract_rc_stem(stem)
        if rc_stem is None:
            skipped_no_rc.append(stem)
            continue
        samples.append(RcSample(stem=stem, rc_stem=rc_stem))

    for stem in sorted(val_images.keys()):
        if stem not in val_labels:
            skipped_no_label.append(stem)
            continue
        rc_stem = extract_rc_stem(stem)
        if rc_stem is None:
            skipped_no_rc.append(stem)
            continue
        samples.append(RcSample(stem=stem, rc_stem=rc_stem))

    return samples, skipped_no_label, skipped_no_rc


def assign_rc_groups(
    samples: list[RcSample], train_ratio: float, seed: int
) -> tuple[list[RcSample], list[RcSample], int]:
    grouped: dict[str, list[RcSample]] = {}
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

    train_samples: list[RcSample] = []
    val_samples: list[RcSample] = []
    for rc_key in rc_keys:
        rows = sorted(grouped[rc_key], key=lambda s: s.stem)
        if rc_key in train_rc_groups:
            train_samples.extend(rows)
        else:
            val_samples.extend(rows)

    train_samples.sort(key=lambda s: s.stem)
    val_samples.sort(key=lambda s: s.stem)
    return train_samples, val_samples, target_train_images


def compute_overlaps(train_samples: list[RcSample], val_samples: list[RcSample]) -> tuple[set[str], set[str]]:
    train_exact = {sample.stem for sample in train_samples}
    val_exact = {sample.stem for sample in val_samples}
    exact_overlap = train_exact & val_exact

    train_rc = {sample.rc_stem for sample in train_samples}
    val_rc = {sample.rc_stem for sample in val_samples}
    rc_overlap = train_rc & val_rc
    return exact_overlap, rc_overlap


def run_rebuild_split(args: argparse.Namespace, dataset_root: Path) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = repo_root / ".venv" / "bin" / "python"
    rebuild_script = repo_root / "tools" / "rebuild_split_by_rc_group.py"
    if python_exe.exists():
        py = python_exe.as_posix()
    else:
        py = sys.executable

    cmd = [
        py,
        rebuild_script.as_posix(),
        "--dataset",
        dataset_root.as_posix(),
        "--train-ratio",
        str(args.train_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    elif args.in_place:
        cmd.append("--in-place")

    print("STEP 5 — Run split rebuild")
    print("+", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=repo_root,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    return proc.returncode


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.train_ratio <= 1.0:
        raise SystemExit("--train-ratio must be between 0.0 and 1.0")

    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset not found: {dataset_root.as_posix()}")
    validate_dataset_layout(dataset_root)

    train_images_dir = dataset_root / "images" / "train"
    train_labels_dir = dataset_root / "labels" / "train"

    train_images, train_labels = scan_split(dataset_root, "train")
    val_images, val_labels = scan_split(dataset_root, "val")

    print(f"dataset: {dataset_root.as_posix()}")
    print(f"dry_run: {args.dry_run}")
    print(f"in_place: {args.in_place}")
    print(f"train_ratio: {args.train_ratio}")
    print(f"seed: {args.seed}")

    print("STEP 1 — Detect duplicates")
    duplicate_stems = sorted(set(train_images.keys()) & set(val_images.keys()))
    preview_list("duplicate canonical stems train∩val", duplicate_stems, limit=50)

    preflight_move_collisions(
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        duplicate_stems=set(duplicate_stems),
    )

    print("STEP 2 — Remove val duplicates (keep train untouched)")
    removed_dup_images = 0
    removed_dup_labels = 0
    dup_preview_count = 0
    for stem in duplicate_stems:
        dup_image = val_images.pop(stem)
        if args.dry_run and dup_preview_count < DRY_RUN_PREVIEW_LIMIT:
            print(f"DRY-RUN remove image: {dup_image.as_posix()}")
            dup_preview_count += 1
        remove_file(dup_image, dry_run=args.dry_run)
        removed_dup_images += 1
        dup_label = val_labels.pop(stem, None)
        if dup_label is not None:
            if args.dry_run and dup_preview_count < DRY_RUN_PREVIEW_LIMIT:
                print(f"DRY-RUN remove label: {dup_label.as_posix()}")
                dup_preview_count += 1
            remove_file(dup_label, dry_run=args.dry_run)
            removed_dup_labels += 1
    print(f"duplicate val images removed: {removed_dup_images}")
    print(f"duplicate val labels removed: {removed_dup_labels}")
    if args.dry_run and (removed_dup_images + removed_dup_labels) > DRY_RUN_PREVIEW_LIMIT:
        print("DRY-RUN duplicate-removal preview truncated.")

    print("STEP 3 — Move remaining val -> train")
    moved_images = 0
    moved_labels = 0
    missing_labels: list[str] = []
    move_preview_count = 0
    for stem in sorted(list(val_images.keys())):
        src_image = val_images.pop(stem)
        dst_image = train_images_dir / src_image.name
        if args.dry_run and move_preview_count < DRY_RUN_PREVIEW_LIMIT:
            print(f"DRY-RUN move image: {src_image.as_posix()} -> {dst_image.as_posix()}")
            move_preview_count += 1
        move_file(src_image, dst_image, dry_run=args.dry_run)
        train_images[stem] = dst_image
        moved_images += 1

        src_label = val_labels.pop(stem, None)
        if src_label is None:
            missing_labels.append(stem)
        else:
            dst_label = train_labels_dir / f"{stem}.txt"
            if args.dry_run and move_preview_count < DRY_RUN_PREVIEW_LIMIT:
                print(f"DRY-RUN move label: {src_label.as_posix()} -> {dst_label.as_posix()}")
                move_preview_count += 1
            move_file(src_label, dst_label, dry_run=args.dry_run)
            train_labels[stem] = dst_label
            moved_labels += 1

    print(f"val images moved to train: {moved_images}")
    print(f"val labels moved to train: {moved_labels}")
    if args.dry_run and (moved_images + moved_labels) > DRY_RUN_PREVIEW_LIMIT:
        print("DRY-RUN move preview truncated.")
    if missing_labels:
        preview_list("moved images missing label", sorted(missing_labels), limit=50)
    else:
        print("moved images missing label: 0")

    print("STEP 4 — Verify post-merge")
    if args.dry_run:
        check_train_images = train_images
        check_train_labels = train_labels
        check_val_images = val_images
        check_val_labels = val_labels
    else:
        check_train_images, check_train_labels = scan_split(dataset_root, "train")
        check_val_images, check_val_labels = scan_split(dataset_root, "val")

    exact_overlap_now = sorted(set(check_train_images.keys()) & set(check_val_images.keys()))
    train_rc_now = {extract_rc_stem(s) for s in check_train_images.keys() if extract_rc_stem(s)}
    val_rc_now = {extract_rc_stem(s) for s in check_val_images.keys() if extract_rc_stem(s)}
    rc_overlap_now = sorted(train_rc_now & val_rc_now)

    print(f"post-merge train images: {len(check_train_images)}")
    print(f"post-merge val images: {len(check_val_images)}")
    print(f"post-merge val labels remaining: {len(check_val_labels)}")
    print(f"post-merge exact overlap count: {len(exact_overlap_now)}")
    print(f"post-merge rc overlap count: {len(rc_overlap_now)}")

    if check_val_images:
        raise RuntimeError(
            "Validation failed: images/val is not empty after duplicate removal + merge."
        )
    if exact_overlap_now:
        raise RuntimeError(
            f"Validation failed: exact overlap remains after merge ({len(exact_overlap_now)} stems)."
        )
    if rc_overlap_now:
        raise RuntimeError(
            f"Validation failed: rc overlap remains after merge ({len(rc_overlap_now)} rc groups)."
        )

    rebuild_rc = run_rebuild_split(args, dataset_root)
    if rebuild_rc != 0:
        raise SystemExit(rebuild_rc)

    print("STEP 6 — Final report")
    final_samples, skipped_no_label, skipped_no_rc = build_rc_samples(
        train_images=check_train_images,
        val_images=check_val_images,
        train_labels=check_train_labels,
        val_labels=check_val_labels,
    )
    if not final_samples:
        print("rebuildable image/label pairs: 0")
        print("WARNING: no valid paired samples available for rc split forecasting.")
        return 0

    final_train_samples, final_val_samples, target_train_images = assign_rc_groups(
        final_samples, train_ratio=args.train_ratio, seed=args.seed
    )
    final_exact_overlap, final_rc_overlap = compute_overlaps(final_train_samples, final_val_samples)

    print(f"duplicates removed (val images): {removed_dup_images}")
    print(f"val items moved (images): {moved_images}")
    print(f"forecast target train images: {target_train_images}")
    print(f"forecast final train images: {len(final_train_samples)}")
    print(f"forecast final val images: {len(final_val_samples)}")
    print(f"forecast skipped image_without_label: {len(skipped_no_label)}")
    print(f"forecast skipped unparseable_rc_stem: {len(skipped_no_rc)}")
    print(f"confirm exact overlap = {len(final_exact_overlap)}")
    print(f"confirm rc overlap = {len(final_rc_overlap)}")

    if final_exact_overlap or final_rc_overlap:
        raise RuntimeError("Final overlap check failed after rebuild forecasting.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
