#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys
from typing import Iterable
import re

# Constants
SRC_DATASET = Path("data/train_idesp_yolo11_v2")
DST_DATASET = Path("data/train_idesp_yolo11_v4")
SEED = 20250306
VAL_POS_FRACTION = 0.20
TRAIN_NEG_PER_POS = 1.5
VAL_NEG_PER_POS = 1.0

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")
REPORT_PATH = Path("runs/debug/rebuild_v4_grouped_by_cell_report.txt")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    source_split: str
    rel: Path
    stem: str
    cell_key: str
    label_path: Path
    label_exists: bool
    label_nonempty: bool

    @property
    def is_positive(self) -> bool:
        return self.label_nonempty


def iter_images(base: Path) -> list[Path]:
    if not base.exists():
        return []
    images = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    images.sort(key=lambda p: p.as_posix())
    return images


def iter_labels(base: Path) -> list[Path]:
    if not base.exists():
        return []
    labels = [p for p in base.rglob("*.txt") if p.is_file()]
    labels.sort(key=lambda p: p.as_posix())
    return labels


def parse_cell_key(stem: str) -> str | None:
    # Example: cell_0024_0008__r0002_c0007
    m = re.match(r"^(cell_\d{4}_\d{4})__r\d{4}_c\d{4}$", stem)
    if not m:
        return None
    return m.group(1)


def label_has_content(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return text.strip() != ""


def ensure_empty_destination(dst: Path) -> None:
    if not dst.exists():
        return
    if any(p.is_file() for p in dst.rglob("*")):
        raise SystemExit(
            f"Destination has files already: {dst}. Remove it or pick a new path."
        )


def build_image_key(rel: Path) -> str:
    return rel.with_suffix("").as_posix()


def collect_records(src_images: Path, src_labels: Path) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    duplicate_keys: list[str] = []
    seen_keys: set[str] = set()

    for split in SPLITS:
        base = src_images / split
        for img in iter_images(base):
            rel = img.relative_to(base)
            stem = img.stem
            key = build_image_key(rel)
            if key in seen_keys:
                duplicate_keys.append(key)
                continue
            seen_keys.add(key)

            cell_key = parse_cell_key(stem)
            if cell_key is None:
                cell_key = f"__unparsed__/{stem}"

            label_path = src_labels / split / rel.with_suffix(".txt")
            label_exists = label_path.exists()
            label_nonempty = label_has_content(label_path)

            records.append(
                ImageRecord(
                    path=img,
                    source_split=split,
                    rel=rel,
                    stem=stem,
                    cell_key=cell_key,
                    label_path=label_path,
                    label_exists=label_exists,
                    label_nonempty=label_nonempty,
                )
            )

    return records, duplicate_keys


def group_by_cell(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.cell_key, []).append(rec)
    return grouped


def choose_val_cells(
    cell_keys: list[str],
    cell_pos_counts: dict[str, int],
    total_pos: int,
    target_val_pos: int,
    rng: random.Random,
) -> set[str]:
    val_cells: set[str] = set()
    processed_pos = 0
    val_pos = 0

    rng.shuffle(cell_keys)
    for cell in cell_keys:
        cell_pos = cell_pos_counts[cell]
        remaining_pos = total_pos - processed_pos - cell_pos

        if target_val_pos <= 0:
            # Put everything in train if no val positives are desired
            processed_pos += cell_pos
            continue

        if val_pos < target_val_pos:
            # If we skip this cell and can't reach target, we must take it
            if val_pos + remaining_pos < target_val_pos:
                val_cells.add(cell)
                val_pos += cell_pos
            else:
                # Take the cell if it improves closeness to target
                if abs((val_pos + cell_pos) - target_val_pos) <= abs(val_pos - target_val_pos):
                    val_cells.add(cell)
                    val_pos += cell_pos
        processed_pos += cell_pos

    return val_cells


def assign_negative_only_cells(
    negative_only_cells: list[str],
    cell_neg_counts: dict[str, int],
    train_cells: set[str],
    val_cells: set[str],
    train_neg_target: int,
    val_neg_target: int,
    rng: random.Random,
) -> None:
    rng.shuffle(negative_only_cells)

    train_neg_available = sum(cell_neg_counts[c] for c in train_cells)
    val_neg_available = sum(cell_neg_counts[c] for c in val_cells)

    for cell in negative_only_cells:
        train_def = max(0, train_neg_target - train_neg_available)
        val_def = max(0, val_neg_target - val_neg_available)

        if train_def == 0 and val_def == 0:
            # No deficit: assign deterministically using RNG
            if rng.random() < 0.5:
                train_cells.add(cell)
                train_neg_available += cell_neg_counts[cell]
            else:
                val_cells.add(cell)
                val_neg_available += cell_neg_counts[cell]
        else:
            if train_def >= val_def:
                train_cells.add(cell)
                train_neg_available += cell_neg_counts[cell]
            else:
                val_cells.add(cell)
                val_neg_available += cell_neg_counts[cell]


def sample_negatives(records: list[ImageRecord], target: int, rng: random.Random) -> list[ImageRecord]:
    if len(records) <= target:
        return records
    shuffled = records[:]
    rng.shuffle(shuffled)
    return shuffled[:target]


def copy_positive(rec: ImageRecord, dst_split: str, dst_images: Path, dst_labels: Path) -> None:
    dst_img = dst_images / dst_split / rec.rel
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rec.path, dst_img)

    dst_label = dst_labels / dst_split / rec.rel.with_suffix(".txt")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rec.label_path, dst_label)


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

    records, duplicate_keys = collect_records(src_images, src_labels)

    # Source integrity checks
    image_keys_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        base = src_images / split
        image_keys_by_split[split] = {build_image_key(p.relative_to(base)) for p in iter_images(base)}

    labels_without_images: list[Path] = []
    for split in SPLITS:
        base = src_labels / split
        for label in iter_labels(base):
            rel = label.relative_to(base)
            key = build_image_key(rel)
            if key not in image_keys_by_split.get(split, set()):
                labels_without_images.append(label)

    images_without_labels = [r for r in records if not r.label_exists]

    grouped = group_by_cell(records)
    cell_keys = list(grouped.keys())
    cell_pos_counts = {cell: sum(r.is_positive for r in grouped[cell]) for cell in cell_keys}
    cell_neg_counts = {cell: sum(not r.is_positive for r in grouped[cell]) for cell in cell_keys}

    total_images = len(records)
    total_pos = sum(cell_pos_counts.values())
    total_neg = total_images - total_pos

    target_val_pos = int(round(total_pos * VAL_POS_FRACTION))

    rng_cells = random.Random(SEED)
    positive_cells = [cell for cell in cell_keys if cell_pos_counts[cell] > 0]
    negative_only_cells = [cell for cell in cell_keys if cell_pos_counts[cell] == 0]

    val_cells = choose_val_cells(
        positive_cells,
        cell_pos_counts,
        total_pos,
        target_val_pos,
        rng_cells,
    )
    train_cells = set(positive_cells) - val_cells

    val_pos = sum(cell_pos_counts[c] for c in val_cells)
    train_pos = total_pos - val_pos

    train_neg_target = int(round(train_pos * TRAIN_NEG_PER_POS))
    val_neg_target = int(round(val_pos * VAL_NEG_PER_POS))

    assign_negative_only_cells(
        negative_only_cells,
        cell_neg_counts,
        train_cells,
        val_cells,
        train_neg_target,
        val_neg_target,
        rng_cells,
    )

    train_cells_count = len(train_cells)
    val_cells_count = len(val_cells)

    train_records = [r for r in records if r.cell_key in train_cells]
    val_records = [r for r in records if r.cell_key in val_cells]

    train_pos_records = [r for r in train_records if r.is_positive]
    val_pos_records = [r for r in val_records if r.is_positive]

    train_neg_records = [r for r in train_records if not r.is_positive]
    val_neg_records = [r for r in val_records if not r.is_positive]

    rng_neg = random.Random(SEED + 1)
    train_neg_selected = sample_negatives(train_neg_records, train_neg_target, rng_neg)
    val_neg_selected = sample_negatives(val_neg_records, val_neg_target, rng_neg)

    dst_images = DST_DATASET / "images"
    dst_labels = DST_DATASET / "labels"
    (dst_images / "train").mkdir(parents=True, exist_ok=True)
    (dst_images / "val").mkdir(parents=True, exist_ok=True)
    (dst_labels / "train").mkdir(parents=True, exist_ok=True)
    (dst_labels / "val").mkdir(parents=True, exist_ok=True)

    for rec in train_pos_records:
        copy_positive(rec, "train", dst_images, dst_labels)
    for rec in val_pos_records:
        copy_positive(rec, "val", dst_images, dst_labels)
    for rec in train_neg_selected:
        copy_negative(rec, "train", dst_images, dst_labels)
    for rec in val_neg_selected:
        copy_negative(rec, "val", dst_images, dst_labels)

    train_neg = len(train_neg_selected)
    val_neg = len(val_neg_selected)

    total_train = len(train_pos_records) + train_neg
    total_val = len(val_pos_records) + val_neg

    # Leakage checks
    leaked_cells = sorted(train_cells & val_cells)
    train_stems = {r.stem for r in train_pos_records + train_neg_selected}
    val_stems = {r.stem for r in val_pos_records + val_neg_selected}
    duplicate_stems_cross = sorted(train_stems & val_stems)

    suspicious: list[str] = []
    if duplicate_keys:
        suspicious.append(f"duplicate image keys across source splits: {len(duplicate_keys)}")
    if labels_without_images:
        suspicious.append(f"labels without images: {len(labels_without_images)}")
    if images_without_labels:
        suspicious.append(f"images without labels: {len(images_without_labels)}")
    if leaked_cells:
        suspicious.append(f"cell leakage across splits: {len(leaked_cells)}")
    if duplicate_stems_cross:
        suspicious.append(f"duplicate stems across splits: {len(duplicate_stems_cross)}")
    if val_pos != target_val_pos:
        suspicious.append(f"val positives target {target_val_pos} achieved {val_pos}")
    if len(train_neg_records) < train_neg_target:
        suspicious.append("not enough train negatives to meet target")
    if len(val_neg_records) < val_neg_target:
        suspicious.append("not enough val negatives to meet target")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Rebuild v4 grouped-by-cell report\n")
        f.write(f"Source dataset: {SRC_DATASET}\n")
        f.write(f"Destination dataset: {DST_DATASET}\n")
        f.write("\nSource summary\n")
        f.write(f"  total images: {total_images}\n")
        f.write(f"  total positives: {total_pos}\n")
        f.write(f"  total negatives: {total_neg}\n")
        f.write(f"  unique cells: {len(cell_keys)}\n")

        f.write("\nSplit summary\n")
        f.write(f"  train cells: {train_cells_count}\n")
        f.write(f"  val cells: {val_cells_count}\n")
        f.write(f"  train positives: {len(train_pos_records)}\n")
        f.write(f"  train negatives: {train_neg}\n")
        f.write(f"  val positives: {len(val_pos_records)}\n")
        f.write(f"  val negatives: {val_neg}\n")
        f.write(f"  total train images: {total_train}\n")
        f.write(f"  total val images: {total_val}\n")

        f.write("\nLeakage checks\n")
        f.write(f"  leaked cells across splits: {len(leaked_cells)}\n")
        f.write(f"  duplicate stems across splits: {len(duplicate_stems_cross)}\n")

        f.write("\nSource integrity\n")
        f.write(f"  labels without images: {len(labels_without_images)}\n")
        f.write(f"  images without labels: {len(images_without_labels)}\n")
        f.write(f"  duplicate image keys skipped: {len(duplicate_keys)}\n")

        f.write("\nSuspicious issues\n")
        if suspicious:
            for item in suspicious:
                f.write(f"  - {item}\n")
        else:
            f.write("  (none)\n")

    # Console output
    print("Source summary:")
    print(f"  total images: {total_images}")
    print(f"  total positives: {total_pos}")
    print(f"  total negatives: {total_neg}")
    print(f"  unique cells: {len(cell_keys)}")

    print("Split summary:")
    print(f"  train cells: {train_cells_count}")
    print(f"  val cells: {val_cells_count}")
    print(f"  train positives: {len(train_pos_records)}")
    print(f"  train negatives: {train_neg}")
    print(f"  val positives: {len(val_pos_records)}")
    print(f"  val negatives: {val_neg}")
    print(f"  total train images: {total_train}")
    print(f"  total val images: {total_val}")

    print("Leakage checks:")
    print(f"  leaked cells across splits: {len(leaked_cells)}")
    print(f"  duplicate stems across splits: {len(duplicate_stems_cross)}")

    print("Source integrity:")
    print(f"  labels without images: {len(labels_without_images)}")
    print(f"  images without labels: {len(images_without_labels)}")

    if suspicious:
        print("Suspicious issues:")
        for item in suspicious:
            print(f"  - {item}")

    print(f"Report written to: {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
