#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import sys
from typing import Iterable

# Constants
DATASET_V3 = Path("data/train_idesp_yolo11_v3")
DATASET_V2 = Path("data/train_idesp_yolo11_v2")
REPORT_PATH = Path("runs/debug/tile_overlap_report.txt")
CONTACT_SHEET_PATH = Path("runs/debug/tile_overlap_suspects.jpg")

SEED = 20250306
AHASH_THRESHOLD = 6
MAX_SUSPECTS = 20
NEIGHBOR_DELTA = 1

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    split: str
    rel: Path
    stem: str
    cell_id: str | None
    r: int | None
    c: int | None
    size: tuple[int, int] | None
    sha256: str
    ahash: int | None
    label_present: bool
    label_nonempty: bool


def iter_images(base: Path) -> list[Path]:
    if not base.exists():
        return []
    images = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    images.sort(key=lambda p: p.as_posix())
    return images


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_stem(stem: str) -> tuple[str | None, int | None, int | None]:
    # Example: cell_0024_0008__r0002_c0007
    m = re.match(r"^cell_(\d{4})_(\d{4})__r(\d{4})_c(\d{4})$", stem)
    if not m:
        return None, None, None
    cell_id = f"cell_{m.group(1)}_{m.group(2)}"
    r = int(m.group(3))
    c = int(m.group(4))
    return cell_id, r, c


def compute_ahash(img: Image.Image, size: int = 8) -> int:
    gray = ImageOps.grayscale(img)
    small = gray.resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(small.get_flattened_data())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def collect_images(
    base_dir: Path,
    split: str,
    labels_root: Path | None = None,
    compute_hash: bool = True,
    compute_ah: bool = True,
) -> tuple[list[ImageInfo], list[Path]]:
    errors: list[Path] = []
    infos: list[ImageInfo] = []
    for path in iter_images(base_dir):
        rel = path.relative_to(base_dir)
        stem = path.stem
        cell_id, r, c = parse_stem(stem)
        sha = sha256_file(path) if compute_hash else ""
        label_present = False
        label_nonempty = False
        if labels_root is not None:
            label_path = labels_root / split / rel.with_suffix(".txt")
            if label_path.exists():
                label_present = True
                try:
                    label_nonempty = label_path.read_text(encoding="utf-8", errors="ignore").strip() != ""
                except Exception:
                    label_nonempty = False
        size: tuple[int, int] | None = None
        ahash: int | None = None
        if compute_ah and PIL_AVAILABLE:
            try:
                with Image.open(path) as img:
                    size = img.size
                    ahash = compute_ahash(img)
            except Exception:
                errors.append(path)
        elif PIL_AVAILABLE:
            try:
                with Image.open(path) as img:
                    size = img.size
            except Exception:
                errors.append(path)

        info = ImageInfo(
            path=path,
            split=split,
            rel=rel,
            stem=stem,
            cell_id=cell_id,
            r=r,
            c=c,
            size=size,
            sha256=sha,
            ahash=ahash,
            label_present=label_present,
            label_nonempty=label_nonempty,
        )
        infos.append(info)

    return infos, errors


def group_by_hash(infos: Iterable[ImageInfo]) -> dict[str, list[ImageInfo]]:
    groups: dict[str, list[ImageInfo]] = {}
    for info in infos:
        groups.setdefault(info.sha256, []).append(info)
    return groups


def group_by_stem(infos: Iterable[ImageInfo]) -> dict[str, list[ImageInfo]]:
    groups: dict[str, list[ImageInfo]] = {}
    for info in infos:
        groups.setdefault(info.stem, []).append(info)
    return groups


def group_by_cell(infos: Iterable[ImageInfo]) -> dict[str, list[ImageInfo]]:
    groups: dict[str, list[ImageInfo]] = {}
    for info in infos:
        if info.cell_id is None:
            continue
        groups.setdefault(info.cell_id, []).append(info)
    return groups


def neighbor_pairs(infos: list[ImageInfo]) -> int:
    by_key: dict[tuple[str, int, int], ImageInfo] = {}
    count = 0
    for info in infos:
        if info.cell_id is None or info.r is None or info.c is None:
            continue
        by_key[(info.cell_id, info.r, info.c)] = info

    for info in infos:
        if info.cell_id is None or info.r is None or info.c is None:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor_key = (info.cell_id, info.r + dr, info.c + dc)
                if neighbor_key in by_key:
                    count += 1
    # Each neighbor is counted twice
    return count // 2


def neighbor_pairs_cross(train_infos: list[ImageInfo], val_infos: list[ImageInfo]) -> list[tuple[ImageInfo, ImageInfo]]:
    train_lookup: dict[tuple[str, int, int], ImageInfo] = {}
    for info in train_infos:
        if info.cell_id is None or info.r is None or info.c is None:
            continue
        train_lookup[(info.cell_id, info.r, info.c)] = info

    pairs: list[tuple[ImageInfo, ImageInfo]] = []
    for val in val_infos:
        if val.cell_id is None or val.r is None or val.c is None:
            continue
        for dr in range(-NEIGHBOR_DELTA, NEIGHBOR_DELTA + 1):
            for dc in range(-NEIGHBOR_DELTA, NEIGHBOR_DELTA + 1):
                if dr == 0 and dc == 0:
                    continue
                key = (val.cell_id, val.r + dr, val.c + dc)
                train = train_lookup.get(key)
                if train:
                    pairs.append((train, val))
    return pairs


def cross_split_ahash_suspects(train_infos: list[ImageInfo], val_infos: list[ImageInfo]) -> list[tuple[int, ImageInfo, ImageInfo]]:
    if not PIL_AVAILABLE:
        return []
    suspects: list[tuple[int, ImageInfo, ImageInfo]] = []
    for t in train_infos:
        if t.ahash is None:
            continue
        for v in val_infos:
            if v.ahash is None:
                continue
            dist = hamming(t.ahash, v.ahash)
            if dist <= AHASH_THRESHOLD:
                suspects.append((dist, t, v))
    suspects.sort(key=lambda x: (x[0], x[1].stem, x[2].stem))
    return suspects


def create_contact_sheet(pairs: list[tuple[int, ImageInfo, ImageInfo]]) -> None:
    CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not PIL_AVAILABLE:
        return

    if not pairs:
        img = Image.new("RGB", (640, 80), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "No suspicious pairs found", fill=(0, 0, 0))
        img.save(CONTACT_SHEET_PATH)
        return

    pairs = pairs[:MAX_SUSPECTS]
    thumb_w, thumb_h = 256, 256
    gap = 12
    margin = 12
    text_h = 36

    sheet_w = margin * 2 + thumb_w * 2 + gap
    sheet_h = margin * 2 + (thumb_h + text_h) * len(pairs)

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (dist, train, val) in enumerate(pairs):
        y = margin + idx * (thumb_h + text_h)
        with Image.open(train.path) as img_train:
            img_train = ImageOps.fit(img_train.convert("RGB"), (thumb_w, thumb_h), Image.Resampling.LANCZOS)
        with Image.open(val.path) as img_val:
            img_val = ImageOps.fit(img_val.convert("RGB"), (thumb_w, thumb_h), Image.Resampling.LANCZOS)

        sheet.paste(img_train, (margin, y))
        sheet.paste(img_val, (margin + thumb_w + gap, y))

        text_y = y + thumb_h + 4
        text = f"d={dist}  train={train.stem}  val={val.stem}"
        if len(text) > 120:
            text = text[:117] + "..."
        draw.text((margin, text_y), text, fill=(0, 0, 0), font=font)

    sheet.save(CONTACT_SHEET_PATH)


def main() -> int:
    if not DATASET_V3.exists():
        print(f"Dataset not found: {DATASET_V3}", file=sys.stderr)
        return 2

    train_dir = DATASET_V3 / "images" / "train"
    val_dir = DATASET_V3 / "images" / "val"

    labels_dir = DATASET_V3 / "labels"
    train_infos, train_errors = collect_images(
        train_dir,
        "train",
        labels_root=labels_dir,
        compute_hash=True,
        compute_ah=True,
    )
    val_infos, val_errors = collect_images(
        val_dir,
        "val",
        labels_root=labels_dir,
        compute_hash=True,
        compute_ah=True,
    )

    all_infos = train_infos + val_infos

    train_stems = {i.stem for i in train_infos}
    val_stems = {i.stem for i in val_infos}
    dup_stems_cross = sorted(train_stems & val_stems)

    dup_stems_train = [stem for stem, items in group_by_stem(train_infos).items() if len(items) > 1]
    dup_stems_val = [stem for stem, items in group_by_stem(val_infos).items() if len(items) > 1]

    hash_groups = group_by_hash(all_infos)
    exact_dups = [items for items in hash_groups.values() if len(items) > 1]
    exact_dups_cross = [items for items in exact_dups if {i.split for i in items} == {"train", "val"} or len({i.split for i in items}) > 1]

    sizes = {}
    for info in all_infos:
        if info.size is None:
            continue
        sizes.setdefault(info.size, 0)
        sizes[info.size] += 1

    parsed = [i for i in all_infos if i.cell_id is not None]
    unparsed = [i for i in all_infos if i.cell_id is None]

    neighbor_train = neighbor_pairs(train_infos)
    neighbor_val = neighbor_pairs(val_infos)
    neighbor_cross = neighbor_pairs_cross(train_infos, val_infos)

    suspects = cross_split_ahash_suspects(train_infos, val_infos)

    train_label_present = sum(1 for i in train_infos if i.label_present)
    train_label_nonempty = sum(1 for i in train_infos if i.label_nonempty)
    val_label_present = sum(1 for i in val_infos if i.label_present)
    val_label_nonempty = sum(1 for i in val_infos if i.label_nonempty)
    train_label_missing = len(train_infos) - train_label_present
    val_label_missing = len(val_infos) - val_label_present

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Tile overlap / leakage report\n")
        f.write(f"Dataset v3: {DATASET_V3}\n")
        f.write(f"PIL available: {PIL_AVAILABLE}\n")
        f.write("\nCounts\n")
        f.write(f"  train images: {len(train_infos)}\n")
        f.write(f"  val images: {len(val_infos)}\n")
        f.write(f"  total images: {len(all_infos)}\n")

        f.write("\nLabel coverage (v3)\n")
        f.write(
            f"  train labels present: {train_label_present}  missing: {train_label_missing}  non-empty: {train_label_nonempty}\n"
        )
        f.write(
            f"  val labels present: {val_label_present}  missing: {val_label_missing}  non-empty: {val_label_nonempty}\n"
        )

        f.write("\nFilename parsing\n")
        f.write(f"  parsed: {len(parsed)}\n")
        f.write(f"  unparsed: {len(unparsed)}\n")

        f.write("\nImage sizes\n")
        if sizes:
            for size, count in sorted(sizes.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"  {size[0]}x{size[1]}: {count}\n")
        else:
            f.write("  (size info unavailable)\n")

        f.write("\nDuplicate stems\n")
        f.write(f"  cross split: {len(dup_stems_cross)}\n")
        f.write(f"  within train: {len(dup_stems_train)}\n")
        f.write(f"  within val: {len(dup_stems_val)}\n")

        f.write("\nExact duplicate files (sha256)\n")
        f.write(f"  total duplicate groups: {len(exact_dups)}\n")
        f.write(f"  duplicate groups spanning splits: {len(exact_dups_cross)}\n")
        if exact_dups_cross:
            for group in exact_dups_cross[:20]:
                f.write("  group:\n")
                for item in group:
                    f.write(f"    - {item.split}: {item.path}\n")

        f.write("\nNeighboring tiles by filename (possible overlap indicator)\n")
        f.write(f"  neighbor pairs within train (delta<=1): {neighbor_train}\n")
        f.write(f"  neighbor pairs within val (delta<=1): {neighbor_val}\n")
        f.write(f"  neighbor pairs across train/val (delta<=1): {len(neighbor_cross)}\n")
        if neighbor_cross:
            for train, val in neighbor_cross[:20]:
                f.write(f"    - train={train.stem} val={val.stem}\n")

        f.write("\nNear-duplicate pairs across train/val (aHash)\n")
        if PIL_AVAILABLE:
            f.write(f"  threshold: {AHASH_THRESHOLD}\n")
            f.write(f"  suspects: {len(suspects)}\n")
            for dist, t, v in suspects[:20]:
                f.write(f"    - d={dist} train={t.stem} val={v.stem}\n")
        else:
            f.write("  skipped (PIL not available)\n")

        if train_errors or val_errors:
            f.write("\nImage read errors\n")
            for path in (train_errors + val_errors):
                f.write(f"  - {path}\n")

        if DATASET_V2.exists():
            f.write("\nSource dataset v2 quick scan\n")
            v2_train = iter_images(DATASET_V2 / "images" / "train")
            v2_val = iter_images(DATASET_V2 / "images" / "val")
            f.write(f"  v2 train images: {len(v2_train)}\n")
            f.write(f"  v2 val images: {len(v2_val)}\n")
            v2_train_stems = {p.stem for p in v2_train}
            v2_val_stems = {p.stem for p in v2_val}
            v2_dup_cross = sorted(v2_train_stems & v2_val_stems)
            f.write(f"  v2 duplicate stems across train/val: {len(v2_dup_cross)}\n")

    create_contact_sheet(suspects)

    print(f"Report written to: {REPORT_PATH}")
    print(f"Contact sheet written to: {CONTACT_SHEET_PATH}")
    print(f"Train images: {len(train_infos)}  Val images: {len(val_infos)}")
    print(f"Cross-split duplicate stems: {len(dup_stems_cross)}")
    print(f"Cross-split exact-duplicate groups: {len(exact_dups_cross)}")
    print(f"Cross-split aHash suspects: {len(suspects)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
