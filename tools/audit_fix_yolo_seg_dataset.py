#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
DEFAULT_SPLITS = ("train", "val")
CROSS_SPLIT_CODES = {
    "duplicate_image_files_same_stem_cross_split",
    "conflicting_image_files_same_stem_cross_split",
    "duplicate_label_files_same_stem_cross_split",
    "conflicting_label_files_same_stem_cross_split",
}


@dataclass
class Issue:
    split: str
    stem: str
    code: str
    detail: str
    image_paths: list[str]
    label_paths: list[str]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Audit and optionally quarantine problematic samples in a YOLO segmentation dataset "
            "(image/label mismatches, malformed label rows, stem collisions)."
        )
    )
    ap.add_argument("--dataset", type=Path, required=True, help="YOLO dataset root containing images/ and labels/.")
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Dataset splits to audit.")
    ap.add_argument("--class-id", type=int, default=0, help="Expected class id in label rows (default: 0).")
    ap.add_argument("--quarantine", action="store_true", help="Move bad samples to a quarantine folder.")
    ap.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Explicit quarantine folder. Default: <dataset>/quarantine_<timestamp>.",
    )
    ap.add_argument(
        "--quarantine-cross-split",
        action="store_true",
        help="Also quarantine cross-split duplicate/conflict stems (default: report-only).",
    )
    ap.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Path for JSON report. Default: <dataset>/audit_yolo_seg_report.json",
    )
    ap.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Path for issue CSV report. Default: <dataset>/audit_yolo_seg_issues.csv",
    )
    ap.add_argument(
        "--quarantine-manifest-csv",
        type=Path,
        default=None,
        help="Path for quarantine manifest CSV. Default: <quarantine_dir>/quarantine_manifest.csv",
    )
    ap.add_argument("--print-limit", type=int, default=200, help="Max bad samples to print to console.")
    ap.add_argument("--dry-run", action="store_true", help="Report quarantine actions without moving files.")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero when any issue is detected.")
    ap.add_argument(
        "--ultralytics-smoke-check",
        action="store_true",
        help="Run an Ultralytics train-loss smoke check to catch TAL shape-mismatch failures.",
    )
    ap.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Dataset YAML for Ultralytics smoke check. Default: <dataset>/dataset.yaml",
    )
    ap.add_argument("--smoke-model", default="yolov8n-seg.pt", help="Model checkpoint for smoke check.")
    ap.add_argument("--smoke-batch", type=int, default=8, help="Batch size for smoke check.")
    ap.add_argument("--smoke-imgsz", type=int, default=1024, help="Image size for smoke check.")
    ap.add_argument("--smoke-device", default="cpu", help="Device for smoke check.")
    ap.add_argument("--smoke-workers", type=int, default=0, help="Workers for smoke check.")
    ap.add_argument("--smoke-seed", type=int, default=123, help="Seed for smoke check.")
    ap.add_argument("--smoke-epochs", type=int, default=3, help="Epoch count for smoke check.")
    ap.add_argument("--smoke-max-batches", type=int, default=0, help="Max batches per epoch (0 = full epoch).")
    return ap.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("._"):
            yield p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def add_issue(
    issues: list[Issue],
    issue_counts: Counter[str],
    split: str,
    stem: str,
    code: str,
    detail: str,
    image_paths: list[Path] | None = None,
    label_paths: list[Path] | None = None,
) -> None:
    issue_counts[code] += 1
    issues.append(
        Issue(
            split=split,
            stem=stem,
            code=code,
            detail=detail,
            image_paths=[p.as_posix() for p in (image_paths or [])],
            label_paths=[p.as_posix() for p in (label_paths or [])],
        )
    )


def validate_label_file(
    label_path: Path,
    split: str,
    stem: str,
    class_id: int,
    issues: list[Issue],
    issue_counts: Counter[str],
) -> int:
    try:
        raw = label_path.read_bytes()
    except OSError as exc:
        add_issue(
            issues,
            issue_counts,
            split,
            stem,
            "label_read_error",
            f"Could not read label file: {exc}",
            label_paths=[label_path],
        )
        return 0

    if raw and not raw.strip():
        add_issue(
            issues,
            issue_counts,
            split,
            stem,
            "malformed_empty_label_whitespace",
            "Label file is whitespace-only. Truly empty labels should be zero-byte files.",
            label_paths=[label_path],
        )
        return 0

    if not raw:
        return 0

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        add_issue(
            issues,
            issue_counts,
            split,
            stem,
            "label_decode_error",
            f"Label is not valid UTF-8: {exc}",
            label_paths=[label_path],
        )
        return 0

    if "\x00" in text:
        add_issue(
            issues,
            issue_counts,
            split,
            stem,
            "label_contains_nul",
            "Label file contains NUL bytes.",
            label_paths=[label_path],
        )

    rows_checked = 0
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        rows_checked += 1
        parts = line.split()
        if len(parts) < 7:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "malformed_label_row_too_few_tokens",
                f"line {line_no}: expected at least 7 tokens, got {len(parts)}",
                label_paths=[label_path],
            )
            continue

        class_token = parts[0]
        try:
            class_value = float(class_token)
        except ValueError:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "non_numeric_class_value",
                f"line {line_no}: class token is not numeric ({class_token!r})",
                label_paths=[label_path],
            )
            continue

        if not math.isfinite(class_value):
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "non_finite_class_value",
                f"line {line_no}: class token is not finite ({class_token!r})",
                label_paths=[label_path],
            )
            continue

        if not class_value.is_integer():
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "non_integer_class_value",
                f"line {line_no}: class token is not an integer ({class_token!r})",
                label_paths=[label_path],
            )
        elif int(class_value) != class_id:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "unexpected_class_id",
                f"line {line_no}: expected class {class_id}, got {int(class_value)}",
                label_paths=[label_path],
            )

        coords: list[float] = []
        has_non_numeric = False
        has_non_finite = False
        for token in parts[1:]:
            try:
                value = float(token)
            except ValueError:
                has_non_numeric = True
                break
            if not math.isfinite(value):
                has_non_finite = True
                break
            coords.append(value)

        if has_non_numeric:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "non_numeric_coordinate_value",
                f"line {line_no}: coordinate token is not numeric",
                label_paths=[label_path],
            )
            continue

        if has_non_finite:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "non_finite_coordinate_value",
                f"line {line_no}: coordinate token is not finite",
                label_paths=[label_path],
            )
            continue

        if len(coords) % 2 != 0:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "odd_number_of_segmentation_coordinates",
                f"line {line_no}: coordinate count is odd ({len(coords)})",
                label_paths=[label_path],
            )
            continue

        n_points = len(coords) // 2
        if n_points < 3:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "too_few_polygon_points",
                f"line {line_no}: polygon has {n_points} points (minimum is 3)",
                label_paths=[label_path],
            )

        out_of_range = [v for v in coords if v < 0.0 or v > 1.0]
        if out_of_range:
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "coordinate_out_of_range_0_1",
                (
                    f"line {line_no}: found {len(out_of_range)} coordinates outside [0,1], "
                    f"min={min(coords):.6f}, max={max(coords):.6f}"
                ),
                label_paths=[label_path],
            )

    return rows_checked


def audit_dataset(dataset_root: Path, splits: list[str], class_id: int) -> tuple[dict, list[Issue]]:
    issues: list[Issue] = []
    issue_counts: Counter[str] = Counter()
    summary: Counter[str] = Counter()

    image_refs_by_stem: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    label_refs_by_stem: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for split in splits:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        summary["splits_checked"] += 1

        if not images_dir.exists():
            add_issue(
                issues,
                issue_counts,
                split,
                "<split>",
                "missing_images_split_dir",
                f"Missing directory: {images_dir}",
            )
            continue
        if not labels_dir.exists():
            add_issue(
                issues,
                issue_counts,
                split,
                "<split>",
                "missing_labels_split_dir",
                f"Missing directory: {labels_dir}",
            )
            continue

        images = list(iter_images(images_dir))
        labels = sorted([p for p in labels_dir.glob("*.txt") if p.is_file() and not p.name.startswith("._")])
        summary["images_total"] += len(images)
        summary["labels_total"] += len(labels)

        images_by_stem: dict[str, list[Path]] = defaultdict(list)
        for image_path in images:
            images_by_stem[image_path.stem].append(image_path)
            image_refs_by_stem[image_path.stem].append((split, image_path))

        labels_by_stem: dict[str, Path] = {}
        for label_path in labels:
            labels_by_stem[label_path.stem] = label_path
            label_refs_by_stem[label_path.stem].append((split, label_path))

        for stem, image_paths in sorted(images_by_stem.items()):
            if len(image_paths) > 1:
                hashes = {file_sha256(p) for p in image_paths}
                if len(hashes) > 1:
                    code = "conflicting_image_files_same_stem"
                    detail = f"Split has {len(image_paths)} images for stem {stem} with different file contents."
                else:
                    code = "duplicate_image_files_same_stem"
                    detail = f"Split has {len(image_paths)} duplicate images for stem {stem}."
                add_issue(issues, issue_counts, split, stem, code, detail, image_paths=image_paths)

        image_stems = set(images_by_stem.keys())
        label_stems = set(labels_by_stem.keys())

        for stem in sorted(image_stems - label_stems):
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "image_without_label_stem_mismatch",
                "Image exists but label file is missing.",
                image_paths=images_by_stem.get(stem, []),
            )

        for stem in sorted(label_stems - image_stems):
            add_issue(
                issues,
                issue_counts,
                split,
                stem,
                "label_without_image_stem_mismatch",
                "Label exists but image file is missing.",
                label_paths=[labels_by_stem[stem]],
            )

        for stem, label_path in sorted(labels_by_stem.items()):
            summary["label_rows_checked"] += validate_label_file(
                label_path=label_path,
                split=split,
                stem=stem,
                class_id=class_id,
                issues=issues,
                issue_counts=issue_counts,
            )

    for stem, refs in sorted(image_refs_by_stem.items()):
        split_set = {split for split, _ in refs}
        if len(split_set) <= 1:
            continue
        refs_by_split: dict[str, list[Path]] = defaultdict(list)
        for split, image_path in refs:
            refs_by_split[split].append(image_path)
        all_hashes = {file_sha256(path) for _, path in refs}
        if len(all_hashes) > 1:
            code = "conflicting_image_files_same_stem_cross_split"
            detail = f"Stem appears across splits with different image content: {sorted(split_set)}"
        else:
            code = "duplicate_image_files_same_stem_cross_split"
            detail = f"Stem appears across splits with identical image content: {sorted(split_set)}"
        for split, split_paths in sorted(refs_by_split.items()):
            add_issue(issues, issue_counts, split, stem, code, detail, image_paths=split_paths)

    for stem, refs in sorted(label_refs_by_stem.items()):
        split_set = {split for split, _ in refs}
        if len(split_set) <= 1:
            continue
        refs_by_split: dict[str, list[Path]] = defaultdict(list)
        for split, label_path in refs:
            refs_by_split[split].append(label_path)
        all_hashes = {file_sha256(path) for _, path in refs}
        if len(all_hashes) > 1:
            code = "conflicting_label_files_same_stem_cross_split"
            detail = f"Stem appears across splits with different label content: {sorted(split_set)}"
        else:
            code = "duplicate_label_files_same_stem_cross_split"
            detail = f"Stem appears across splits with identical label content: {sorted(split_set)}"
        for split, split_paths in sorted(refs_by_split.items()):
            add_issue(issues, issue_counts, split, stem, code, detail, label_paths=split_paths)

    summary_dict = {
        "dataset": dataset_root.as_posix(),
        "splits": list(splits),
        "images_total": int(summary["images_total"]),
        "labels_total": int(summary["labels_total"]),
        "label_rows_checked": int(summary["label_rows_checked"]),
        "issues_total": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
    }
    return summary_dict, issues


def issues_by_sample(issues: list[Issue]) -> dict[tuple[str, str], list[Issue]]:
    sample_map: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in issues:
        sample_map[(issue.split, issue.stem)].append(issue)
    return sample_map


def should_quarantine_issue(issue_code: str, quarantine_cross_split: bool) -> bool:
    if issue_code in CROSS_SPLIT_CODES and not quarantine_cross_split:
        return False
    return True


def choose_unique_destination(base_dst: Path, used: set[Path]) -> Path:
    candidate = base_dst
    idx = 1
    while candidate in used or candidate.exists():
        candidate = candidate.with_name(f"{base_dst.stem}__dup{idx:04d}{base_dst.suffix}")
        idx += 1
    used.add(candidate)
    return candidate


def quarantine_bad_samples(
    dataset_root: Path,
    sample_map: dict[tuple[str, str], list[Issue]],
    quarantine_dir: Path,
    quarantine_cross_split: bool,
    dry_run: bool,
) -> tuple[dict, list[dict[str, str]]]:
    manifest_rows: list[dict[str, str]] = []
    moved_files = 0
    moved_samples = 0
    used_destinations: set[Path] = set()

    for split, stem in sorted(sample_map.keys()):
        issues = sample_map[(split, stem)]
        codes = sorted({issue.code for issue in issues if should_quarantine_issue(issue.code, quarantine_cross_split)})
        if not codes:
            continue

        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        source_files: list[Path] = []
        for ext in IMG_EXTS:
            img_path = image_dir / f"{stem}{ext}"
            if img_path.exists():
                source_files.append(img_path)
        label_path = label_dir / f"{stem}.txt"
        if label_path.exists():
            source_files.append(label_path)

        if not source_files:
            continue

        moved_this_sample = False
        for src in sorted(source_files, key=lambda p: p.as_posix()):
            rel = src.relative_to(dataset_root)
            dst_base = quarantine_dir / rel
            dst = choose_unique_destination(dst_base, used_destinations)

            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src.as_posix(), dst.as_posix())

            manifest_rows.append(
                {
                    "split": split,
                    "stem": stem,
                    "reasons": ";".join(codes),
                    "src_path": src.as_posix(),
                    "dst_path": dst.as_posix(),
                    "file_type": "label" if src.suffix.lower() == ".txt" else "image",
                    "dry_run": str(int(dry_run)),
                }
            )
            moved_files += 1
            moved_this_sample = True

        if moved_this_sample:
            moved_samples += 1

    return {
        "quarantine_dir": quarantine_dir.as_posix(),
        "moved_samples": moved_samples,
        "moved_files": moved_files,
        "dry_run": bool(dry_run),
    }, manifest_rows


def write_issue_csv(path: Path, issues: list[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "stem", "code", "detail", "image_paths", "label_paths"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(
                {
                    "split": issue.split,
                    "stem": issue.stem,
                    "code": issue.code,
                    "detail": issue.detail,
                    "image_paths": ";".join(issue.image_paths),
                    "label_paths": ";".join(issue.label_paths),
                }
            )


def write_quarantine_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "stem", "reasons", "src_path", "dst_path", "file_type", "dry_run"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_smoke_batch_samples(batch: dict, labels_train_dir: Path) -> list[dict]:
    samples: list[dict] = []
    im_files = list(batch.get("im_file", []))
    batch_idx = batch.get("batch_idx", None)
    for i, image_str in enumerate(im_files):
        image_path = Path(image_str).resolve()
        label_path = labels_train_dir / f"{image_path.stem}.txt"
        objects = None
        if hasattr(batch_idx, "shape"):
            try:
                objects = int((batch_idx == i).sum().item())
            except Exception:
                objects = None
        samples.append(
            {
                "batch_pos": i,
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "label_exists": bool(label_path.exists()),
                "objects_in_batch_for_image": objects,
            }
        )
    return samples


def run_ultralytics_smoke_check(
    dataset_yaml: Path,
    labels_train_dir: Path,
    model: str,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    seed: int,
    epochs: int,
    max_batches_per_epoch: int,
    project_dir: Path,
    run_name: str,
) -> dict:
    import random
    import traceback

    import numpy as np
    import torch
    from ultralytics.models.yolo.segment.train import SegmentationTrainer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    overrides = dict(
        model=model,
        data=dataset_yaml.as_posix(),
        epochs=1,
        batch=int(batch),
        imgsz=int(imgsz),
        workers=int(workers),
        device=str(device),
        project=project_dir.resolve().as_posix(),
        name=run_name,
        exist_ok=True,
        deterministic=True,
        seed=int(seed),
        cache=False,
        amp=False,
        save=False,
        plots=False,
        val=False,
        verbose=False,
    )

    trainer = SegmentationTrainer(overrides=overrides)
    trainer._setup_train()
    for epoch in range(int(epochs)):
        if hasattr(trainer.train_loader, "sampler") and hasattr(trainer.train_loader.sampler, "set_epoch"):
            trainer.train_loader.sampler.set_epoch(epoch)
        for batch_idx, raw_batch in enumerate(trainer.train_loader):
            if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                break
            batch_data = trainer.preprocess_batch(raw_batch)
            try:
                with torch.no_grad():
                    trainer.model(batch_data)
            except Exception as exc:
                msg = str(exc)
                tb = traceback.format_exc()
                is_tal = "ultralytics/utils/tal.py" in tb
                is_shape = "shape mismatch" in msg and "cannot be broadcast" in msg
                is_indexing = ("out of bounds" in msg and "index" in msg) or "cannot be broadcast" in msg
                if is_tal and (is_shape or is_indexing):
                    kind = "tal_shape_mismatch" if is_shape else "tal_indexing_error"
                    return {
                        "crashed": True,
                        "failure_kind": kind,
                        "runtime_error": msg,
                        "traceback": tb,
                        "epoch": epoch,
                        "batch_index": batch_idx,
                        "batch_samples": extract_smoke_batch_samples(batch_data, labels_train_dir),
                    }
                raise
    return {
        "crashed": False,
        "failure_kind": "",
        "runtime_error": "",
        "traceback": "",
        "epoch": -1,
        "batch_index": -1,
        "batch_samples": [],
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset not found: {dataset_root}")

    splits = [split.strip() for split in args.splits if split.strip()]
    if not splits:
        raise SystemExit("No valid splits provided.")

    report_json = (
        args.report_json.expanduser().resolve()
        if args.report_json
        else (dataset_root / "audit_yolo_seg_report.json").resolve()
    )
    report_csv = (
        args.report_csv.expanduser().resolve()
        if args.report_csv
        else (dataset_root / "audit_yolo_seg_issues.csv").resolve()
    )

    summary, issues = audit_dataset(dataset_root=dataset_root, splits=splits, class_id=int(args.class_id))
    smoke_result = None
    if args.ultralytics_smoke_check:
        if "train" not in splits:
            smoke_result = {
                "crashed": False,
                "runtime_error": "",
                "traceback": "",
                "epoch": -1,
                "batch_index": -1,
                "batch_samples": [],
                "note": "Skipped smoke check because train split was not requested.",
            }
        else:
            smoke_dataset_yaml = (
                args.dataset_yaml.expanduser().resolve()
                if args.dataset_yaml
                else (dataset_root / "dataset.yaml").resolve()
            )
            if not smoke_dataset_yaml.exists():
                smoke_result = {
                    "crashed": False,
                    "runtime_error": "",
                    "traceback": "",
                    "epoch": -1,
                    "batch_index": -1,
                    "batch_samples": [],
                    "note": f"Skipped smoke check because dataset YAML was not found: {smoke_dataset_yaml}",
                }
            else:
                smoke_result = run_ultralytics_smoke_check(
                    dataset_yaml=smoke_dataset_yaml,
                    labels_train_dir=(dataset_root / "labels" / "train").resolve(),
                    model=str(args.smoke_model),
                    batch=max(1, int(args.smoke_batch)),
                    imgsz=int(args.smoke_imgsz),
                    device=str(args.smoke_device),
                    workers=int(args.smoke_workers),
                    seed=int(args.smoke_seed),
                    epochs=max(1, int(args.smoke_epochs)),
                    max_batches_per_epoch=int(args.smoke_max_batches),
                    project_dir=Path("runs/diagnostics").resolve(),
                    run_name="audit_smoke_check",
                )
                if smoke_result.get("crashed"):
                    detail = (
                        f"Ultralytics smoke check reproduced TAL failure ({smoke_result.get('failure_kind', 'tal_unknown')}) "
                        f"(epoch={smoke_result.get('epoch')}, batch={smoke_result.get('batch_index')})."
                    )
                    for sample in smoke_result.get("batch_samples", []):
                        image_path = Path(sample["image_path"]).resolve()
                        label_path = Path(sample["label_path"]).resolve()
                        issues.append(
                            Issue(
                                split="train",
                                stem=image_path.stem,
                                code="ultralytics_tal_shape_mismatch_smoke",
                                detail=detail,
                                image_paths=[image_path.as_posix()],
                                label_paths=[label_path.as_posix()],
                            )
                        )

    issue_counts = Counter([issue.code for issue in issues])
    summary["issues_total"] = len(issues)
    summary["issue_counts"] = dict(sorted(issue_counts.items()))

    sample_map = issues_by_sample(issues)

    bad_sample_map: dict[str, dict[str, object]] = {}
    for (split, stem), sample_issues in sorted(sample_map.items()):
        bad_sample_map[f"{split}:{stem}"] = {
            "split": split,
            "stem": stem,
            "codes": sorted({issue.code for issue in sample_issues}),
            "details": [issue.detail for issue in sample_issues],
        }

    quarantine_summary = None
    quarantine_manifest: list[dict[str, str]] = []
    quarantine_manifest_csv = None
    if args.quarantine:
        quarantine_dir = (
            args.quarantine_dir.expanduser().resolve()
            if args.quarantine_dir
            else (dataset_root / f"quarantine_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
        )
        quarantine_summary, quarantine_manifest = quarantine_bad_samples(
            dataset_root=dataset_root,
            sample_map=sample_map,
            quarantine_dir=quarantine_dir,
            quarantine_cross_split=bool(args.quarantine_cross_split),
            dry_run=bool(args.dry_run),
        )
        quarantine_manifest_csv = (
            args.quarantine_manifest_csv.expanduser().resolve()
            if args.quarantine_manifest_csv
            else (quarantine_dir / "quarantine_manifest.csv").resolve()
        )
        write_quarantine_manifest_csv(quarantine_manifest_csv, quarantine_manifest)

    report = {
        "summary": summary,
        "bad_samples_count": len(bad_sample_map),
        "bad_samples": bad_sample_map,
        "issues": [asdict(issue) for issue in issues],
        "quarantine": quarantine_summary,
        "ultralytics_smoke_check": smoke_result,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_issue_csv(report_csv, issues)

    print("dataset:", dataset_root)
    print("splits:", ",".join(splits))
    print("images_total:", summary["images_total"])
    print("labels_total:", summary["labels_total"])
    print("label_rows_checked:", summary["label_rows_checked"])
    print("issues_total:", summary["issues_total"])
    print("bad_samples_count:", len(bad_sample_map))
    for code, count in sorted(summary["issue_counts"].items()):
        print(f"{code}: {count}")

    if bad_sample_map:
        print("bad_samples_list:")
        for idx, key in enumerate(sorted(bad_sample_map.keys())):
            if idx >= int(args.print_limit):
                print(f"... truncated after {args.print_limit} samples")
                break
            entry = bad_sample_map[key]
            codes = ",".join(entry["codes"])
            print(f"{entry['split']}:{entry['stem']} -> {codes}")

    print("report_json:", report_json)
    print("report_csv:", report_csv)
    if smoke_result is not None:
        print("ultralytics_smoke_check_crashed:", bool(smoke_result.get("crashed")))
        if smoke_result.get("crashed"):
            print("ultralytics_smoke_check_epoch:", smoke_result.get("epoch"))
            print("ultralytics_smoke_check_batch:", smoke_result.get("batch_index"))
    if quarantine_summary is not None:
        print("quarantine_dir:", quarantine_summary["quarantine_dir"])
        print("quarantine_moved_samples:", quarantine_summary["moved_samples"])
        print("quarantine_moved_files:", quarantine_summary["moved_files"])
        print("quarantine_manifest_csv:", quarantine_manifest_csv)

    if args.strict and summary["issues_total"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
