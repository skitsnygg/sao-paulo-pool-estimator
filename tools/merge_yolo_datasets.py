#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge multiple YOLO segmentation datasets")
    ap.add_argument("--input-dirs", required=True, nargs="+", help="Input dataset directories")
    ap.add_argument("--out-dir", required=True, help="Output directory for the merged dataset")
    ap.add_argument("--symlink", action="store_true", help="Create symbolic links instead of copying files")
    ap.add_argument("--overwrite", action="store_true", help="Delete out-dir before writing")
    return ap.parse_args()


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    token = token.strip("_")
    return token or "dataset"


def ensure_output_dirs(out_dir: Path, overwrite: bool) -> Dict[str, Path]:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {out_dir} (use --overwrite)")

    paths = {
        "images_train": out_dir / "images" / "train",
        "images_val": out_dir / "images" / "val",
        "labels_train": out_dir / "labels" / "train",
        "labels_val": out_dir / "labels" / "val",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def allocate_name(base_name: str, used_stems: set[str], dataset_tag: str) -> tuple[str, bool]:
    stem = Path(base_name).stem
    ext = Path(base_name).suffix.lower()
    if stem not in used_stems:
        used_stems.add(stem)
        return f"{stem}{ext}", False

    candidate_stem = f"{dataset_tag}__{stem}"
    idx = 1
    while candidate_stem in used_stems:
        candidate_stem = f"{dataset_tag}__{stem}__{idx:04d}"
        idx += 1
    used_stems.add(candidate_stem)
    return f"{candidate_stem}{ext}", True


def validate_label_file(label_path: Path, class_id: int = 0) -> list[str]:
    try:
        raw = label_path.read_bytes()
    except OSError as exc:
        return [f"read_error:{exc}"]

    if not raw:
        return []
    if not raw.strip():
        return ["whitespace_only_empty_label"]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"utf8_decode_error:{exc}"]

    errors: list[str] = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            errors.append(f"line_{line_no}:too_few_tokens:{len(parts)}")
            continue

        try:
            cls = float(parts[0])
        except ValueError:
            errors.append(f"line_{line_no}:non_numeric_class")
            continue

        if not math.isfinite(cls):
            errors.append(f"line_{line_no}:non_finite_class")
            continue
        if not cls.is_integer():
            errors.append(f"line_{line_no}:non_integer_class")
        elif int(cls) != class_id:
            errors.append(f"line_{line_no}:unexpected_class:{int(cls)}")

        coords: list[float] = []
        bad_coord = False
        for token in parts[1:]:
            try:
                value = float(token)
            except ValueError:
                errors.append(f"line_{line_no}:non_numeric_coord")
                bad_coord = True
                break
            if not math.isfinite(value):
                errors.append(f"line_{line_no}:non_finite_coord")
                bad_coord = True
                break
            coords.append(value)
        if bad_coord:
            continue

        if len(coords) % 2 != 0:
            errors.append(f"line_{line_no}:odd_coord_count:{len(coords)}")
            continue
        n_points = len(coords) // 2
        if n_points < 3:
            errors.append(f"line_{line_no}:too_few_points:{n_points}")
        if any(value < 0.0 or value > 1.0 for value in coords):
            errors.append(f"line_{line_no}:coord_out_of_range")

    return errors


def link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel_target = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel_target)
    else:
        shutil.copy2(src, dst)


def write_dataset_yaml(out_dir: Path) -> Path:
    dataset_yaml = out_dir / "dataset.yaml"
    content = "\n".join(
        [
            f"path: {out_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names:",
            "  0: pool",
            "",
        ]
    )
    dataset_yaml.write_text(content, encoding="utf-8")
    return dataset_yaml


def write_manifest(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "source_dataset",
        "split",
        "src_image",
        "src_label",
        "dst_image",
        "dst_label",
        "renamed",
        "source_stem_collision",
        "label_created_empty",
        "label_positive",
        "label_valid",
        "label_error_count",
        "label_errors",
        "skipped",
        "skip_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()

    input_dirs = [Path(d).expanduser().resolve() for d in args.input_dirs]
    out_dir = Path(args.out_dir).expanduser().resolve()

    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"Input dataset not found: {d}")

    out_paths = ensure_output_dirs(out_dir, overwrite=bool(args.overwrite))
    used_stems = {split: set() for split in SPLITS}
    manifest_rows: List[dict] = []

    stats = {
        "datasets_seen": len(input_dirs),
        "images_written": 0,
        "labels_written": 0,
        "labels_positive": 0,
        "labels_empty": 0,
        "labels_created_empty": 0,
        "renamed_due_collision": 0,
        "source_duplicate_stem_groups": 0,
        "source_duplicate_stem_images": 0,
        "samples_skipped_invalid_label": 0,
        "invalid_label_files": 0,
        "invalid_label_rows": 0,
    }

    for dataset_idx, input_dir in enumerate(input_dirs):
        dataset_tag = f"d{dataset_idx:02d}_{sanitize_token(input_dir.name)}"
        print(f"Processing: {input_dir}")

        for split in SPLITS:
            src_img_dir = input_dir / "images" / split
            src_lbl_dir = input_dir / "labels" / split
            if not src_img_dir.exists():
                continue

            dst_img_dir = out_paths[f"images_{split}"]
            dst_lbl_dir = out_paths[f"labels_{split}"]
            src_images = list(iter_images(src_img_dir))
            src_stem_counts = Counter([p.stem for p in src_images])
            duplicate_groups = sum(1 for count in src_stem_counts.values() if count > 1)
            duplicate_images = sum(count for count in src_stem_counts.values() if count > 1)
            if duplicate_groups:
                stats["source_duplicate_stem_groups"] += duplicate_groups
                stats["source_duplicate_stem_images"] += duplicate_images
                print(
                    f"  [warn] split={split} has {duplicate_groups} duplicate stem groups "
                    f"({duplicate_images} images) before merge."
                )

            for src_img in src_images:
                out_name, renamed = allocate_name(src_img.name, used_stems[split], dataset_tag)
                out_stem = Path(out_name).stem
                dst_img = dst_img_dir / out_name
                dst_lbl = dst_lbl_dir / f"{out_stem}.txt"

                src_lbl = src_lbl_dir / f"{src_img.stem}.txt"
                label_exists = src_lbl.exists()
                source_stem_collision = src_stem_counts[src_img.stem] > 1

                label_errors: list[str] = []
                if label_exists:
                    label_errors = validate_label_file(src_lbl, class_id=0)
                    if label_errors:
                        stats["samples_skipped_invalid_label"] += 1
                        stats["invalid_label_files"] += 1
                        stats["invalid_label_rows"] += len(label_errors)
                        manifest_rows.append(
                            {
                                "source_dataset": str(input_dir),
                                "split": split,
                                "src_image": str(src_img),
                                "src_label": str(src_lbl),
                                "dst_image": "",
                                "dst_label": "",
                                "renamed": int(renamed),
                                "source_stem_collision": int(source_stem_collision),
                                "label_created_empty": 0,
                                "label_positive": 0,
                                "label_valid": 0,
                                "label_error_count": len(label_errors),
                                "label_errors": ";".join(label_errors),
                                "skipped": 1,
                                "skip_reason": "invalid_label",
                            }
                        )
                        continue

                link_or_copy(src_img, dst_img, symlink=bool(args.symlink))
                if label_exists:
                    link_or_copy(src_lbl, dst_lbl, symlink=bool(args.symlink))
                    try:
                        positive = bool(src_lbl.read_text(encoding="utf-8").strip())
                    except OSError:
                        positive = False
                else:
                    dst_lbl.write_text("", encoding="utf-8")
                    stats["labels_created_empty"] += 1
                    positive = False

                stats["images_written"] += 1
                stats["labels_written"] += 1
                if positive:
                    stats["labels_positive"] += 1
                else:
                    stats["labels_empty"] += 1
                if renamed:
                    stats["renamed_due_collision"] += 1

                manifest_rows.append(
                    {
                        "source_dataset": str(input_dir),
                        "split": split,
                        "src_image": str(src_img),
                        "src_label": str(src_lbl) if label_exists else "",
                        "dst_image": str(dst_img),
                        "dst_label": str(dst_lbl),
                        "renamed": int(renamed),
                        "source_stem_collision": int(source_stem_collision),
                        "label_created_empty": int(not label_exists),
                        "label_positive": int(positive),
                        "label_valid": 1,
                        "label_error_count": 0,
                        "label_errors": "",
                        "skipped": 0,
                        "skip_reason": "",
                    }
                )

    dataset_yaml = write_dataset_yaml(out_dir)
    manifest_csv = out_dir / "merge_manifest.csv"
    stats_json = out_dir / "merge_stats.json"
    write_manifest(manifest_csv, manifest_rows)
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Merged dataset created in {out_dir}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Stats JSON: {stats_json}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
