#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
import shlex
import shutil
import tempfile
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import YAML

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


@dataclass
class PairingIssue:
    split: str
    stem: str
    code: str
    detail: str
    image_paths: list[str]
    label_paths: list[str]


@dataclass
class SmokeResult:
    crashed: bool
    crash_kind: str
    runtime_error: str
    traceback_text: str
    epoch: int
    batch_index: int
    batch_samples: list[dict]
    dataset_yaml: str
    run_name: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Isolate offending YOLO segmentation samples that trigger Ultralytics TAL shape mismatch by "
            "reproducing the loss path and bisecting the failing batch."
        )
    )
    ap.add_argument("--dataset-yaml", type=Path, required=True, help="Dataset YAML path.")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="Segmentation model checkpoint for smoke runs.")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for smoke checks.")
    ap.add_argument("--imgsz", type=int, default=1024, help="Image size for smoke checks.")
    ap.add_argument("--device", default="cpu", help="Training device for smoke checks (cpu or cuda:0).")
    ap.add_argument("--workers", type=int, default=0, help="Dataloader workers (default 0 for determinism).")
    ap.add_argument("--seed", type=int, default=123, help="Deterministic seed.")
    ap.add_argument("--max-epochs-search", type=int, default=20, help="Max epochs to run while reproducing crash.")
    ap.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=0,
        help="Optional max batches per epoch during smoke run (0 means full epoch).",
    )
    ap.add_argument(
        "--subset-epochs",
        type=int,
        default=5,
        help="Epochs for subset smoke checks during bisection.",
    )
    ap.add_argument(
        "--subset-max-batches",
        type=int,
        default=0,
        help="Optional max batches per epoch during subset smoke checks (0 means full epoch).",
    )
    ap.add_argument("--max-pair-tests", type=int, default=64, help="Max pairwise subset tests for interaction fallback.")
    ap.add_argument("--project", type=Path, default=Path("runs/diagnostics"), help="Diagnostics output directory.")
    ap.add_argument("--name", default="tal_isolation", help="Run name prefix.")
    ap.add_argument("--keep-temp", action="store_true", help="Keep temporary subset datasets.")
    ap.add_argument("--smoke-only", action="store_true", help="Only run full-dataset smoke check, skip bisection.")
    ap.add_argument("--verbose", action="store_true", help="Print extra progress details.")
    ap.add_argument("--progress-every", type=int, default=0, help="Print progress every N batches (0 disables).")
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def jsonable(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    return value


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in sorted(images_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("._"):
            yield p


def resolve_dataset(dataset_yaml: Path) -> dict:
    data = YAML.load(dataset_yaml.as_posix())
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid dataset yaml: {dataset_yaml}")

    dataset_root = Path(data.get("path", dataset_yaml.parent)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    train_rel = str(data.get("train", "images/train"))
    val_rel = str(data.get("val", "images/val"))

    train_images_dir = (dataset_root / train_rel).resolve()
    if train_rel.endswith("train"):
        labels_train_rel = train_rel.replace("images", "labels", 1)
    else:
        labels_train_rel = "labels/train"
    labels_train_dir = (dataset_root / labels_train_rel).resolve()

    val_images_dir = (dataset_root / val_rel).resolve()
    if val_rel.endswith("val"):
        labels_val_rel = val_rel.replace("images", "labels", 1)
    else:
        labels_val_rel = "labels/val"
    labels_val_dir = (dataset_root / labels_val_rel).resolve()

    return {
        "yaml_path": dataset_yaml.resolve().as_posix(),
        "yaml_data": data,
        "dataset_root": dataset_root,
        "train_images_dir": train_images_dir,
        "train_labels_dir": labels_train_dir,
        "val_images_dir": val_images_dir,
        "val_labels_dir": labels_val_dir,
        "train_rel": train_rel,
        "val_rel": val_rel,
    }


def validate_train_pairing(train_images_dir: Path, train_labels_dir: Path) -> tuple[dict, list[PairingIssue], dict[str, Path]]:
    issues: list[PairingIssue] = []
    images = list(iter_images(train_images_dir))
    labels = sorted([p for p in train_labels_dir.glob("*.txt") if p.is_file() and not p.name.startswith("._")])
    image_by_stem: dict[str, list[Path]] = {}
    label_by_stem: dict[str, Path] = {}

    for img in images:
        image_by_stem.setdefault(img.stem, []).append(img)
    for lbl in labels:
        label_by_stem[lbl.stem] = lbl

    for stem, img_paths in sorted(image_by_stem.items()):
        if len(img_paths) > 1:
            issues.append(
                PairingIssue(
                    split="train",
                    stem=stem,
                    code="duplicate_image_stem_train",
                    detail=f"Found {len(img_paths)} images for stem {stem} in train split.",
                    image_paths=[p.as_posix() for p in img_paths],
                    label_paths=[label_by_stem[stem].as_posix()] if stem in label_by_stem else [],
                )
            )

    image_stems = set(image_by_stem.keys())
    label_stems = set(label_by_stem.keys())
    for stem in sorted(image_stems - label_stems):
        issues.append(
            PairingIssue(
                split="train",
                stem=stem,
                code="image_without_label",
                detail="Train image exists but label is missing (allowed for negatives, but tracked).",
                image_paths=[p.as_posix() for p in image_by_stem.get(stem, [])],
                label_paths=[],
            )
        )
    for stem in sorted(label_stems - image_stems):
        issues.append(
            PairingIssue(
                split="train",
                stem=stem,
                code="label_without_image",
                detail="Train label exists but image is missing.",
                image_paths=[],
                label_paths=[label_by_stem[stem].as_posix()],
            )
        )

    summary = {
        "images_train_total": len(images),
        "labels_train_total": len(labels),
        "pairing_issues_total": len(issues),
        "pairing_issue_counts": {},
    }
    for issue in issues:
        summary["pairing_issue_counts"][issue.code] = summary["pairing_issue_counts"].get(issue.code, 0) + 1

    stem_to_image: dict[str, Path] = {}
    for stem, img_paths in image_by_stem.items():
        stem_to_image[stem] = sorted(img_paths, key=lambda p: p.as_posix())[0]
    return summary, issues, stem_to_image


def extract_batch_samples(batch: dict, labels_train_dir: Path) -> list[dict]:
    im_files_raw = list(batch.get("im_file", []))
    batch_idx = batch.get("batch_idx", None)
    batch_samples: list[dict] = []

    for i, im_file in enumerate(im_files_raw):
        image_path = Path(im_file).resolve()
        stem = image_path.stem
        label_path = labels_train_dir / f"{stem}.txt"
        obj_count = None
        if isinstance(batch_idx, torch.Tensor):
            obj_count = int((batch_idx == i).sum().item())
        batch_samples.append(
            {
                "batch_pos": i,
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "label_exists": bool(label_path.exists()),
                "objects_in_batch_for_image": obj_count,
            }
        )
    return batch_samples


def run_loss_smoke(
    dataset_yaml: Path,
    model: str,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    seed: int,
    max_epochs: int,
    max_batches_per_epoch: int,
    progress_every: int,
    verbose: bool,
    project: Path,
    run_name: str,
    labels_train_dir: Path,
) -> SmokeResult:
    seed_everything(seed)

    overrides = dict(
        model=model,
        data=dataset_yaml.as_posix(),
        epochs=1,
        batch=int(batch),
        imgsz=int(imgsz),
        workers=int(workers),
        device=str(device),
        project=project.resolve().as_posix(),
        name=run_name,
        exist_ok=True,
        deterministic=True,
        seed=int(seed),
        cache=False,
        amp=False,
        plots=False,
        save=False,
        val=False,
        verbose=False,
    )

    trainer = SegmentationTrainer(overrides=overrides)
    trainer._setup_train()
    for epoch in range(int(max_epochs)):
        if hasattr(trainer.train_loader, "sampler") and hasattr(trainer.train_loader.sampler, "set_epoch"):
            trainer.train_loader.sampler.set_epoch(epoch)
        for batch_idx, batch_data in enumerate(trainer.train_loader):
            if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                break
            batch_pre = trainer.preprocess_batch(batch_data)
            try:
                with torch.no_grad():
                    trainer.model(batch_pre)
                if verbose and progress_every > 0 and (batch_idx + 1) % progress_every == 0:
                    print(f"[progress] epoch={epoch} batch={batch_idx + 1} run={run_name}", flush=True)
            except Exception as exc:
                msg = str(exc)
                tb_text = traceback.format_exc()
                is_tal = "ultralytics/utils/tal.py" in tb_text
                is_shape = "shape mismatch" in msg and "cannot be broadcast" in msg
                is_indexing = ("out of bounds" in msg and "index" in msg) or "cannot be broadcast" in msg
                if is_tal and (is_shape or is_indexing):
                    crash_kind = "tal_shape_mismatch" if is_shape else "tal_indexing_error"
                    return SmokeResult(
                        crashed=True,
                        crash_kind=crash_kind,
                        runtime_error=msg,
                        traceback_text=tb_text,
                        epoch=epoch,
                        batch_index=batch_idx,
                        batch_samples=extract_batch_samples(batch_pre, labels_train_dir),
                        dataset_yaml=dataset_yaml.as_posix(),
                        run_name=run_name,
                    )
                raise

    return SmokeResult(
        crashed=False,
        crash_kind="",
        runtime_error="",
        traceback_text="",
        epoch=-1,
        batch_index=-1,
        batch_samples=[],
        dataset_yaml=dataset_yaml.as_posix(),
        run_name=run_name,
    )


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_subset_dataset(
    image_paths: list[Path],
    labels_train_dir: Path,
    tmp_root: Path,
) -> Path:
    images_train = tmp_root / "images" / "train"
    labels_train = tmp_root / "labels" / "train"
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    for img in sorted(image_paths, key=lambda p: p.as_posix()):
        dst_img = images_train / img.name
        link_or_copy(img, dst_img)
        label_src = labels_train_dir / f"{img.stem}.txt"
        if label_src.exists():
            dst_lbl = labels_train / f"{img.stem}.txt"
            link_or_copy(label_src, dst_lbl)

    dataset_yaml = tmp_root / "dataset.yaml"
    YAML.save(
        dataset_yaml,
        {
            "path": tmp_root.resolve().as_posix(),
            "train": "images/train",
            "val": "images/train",
            "nc": 1,
            "names": {0: "pool"},
        },
    )
    return dataset_yaml


def run_subset_smoke(
    image_paths: list[Path],
    labels_train_dir: Path,
    args: argparse.Namespace,
    run_tag: str,
) -> tuple[SmokeResult, str]:
    tmp_root = Path(tempfile.mkdtemp(prefix=f"tal_isolate_{run_tag}_"))
    subset_yaml = build_subset_dataset(image_paths=image_paths, labels_train_dir=labels_train_dir, tmp_root=tmp_root)
    subset_batch = max(1, min(int(args.batch), len(image_paths)))
    result = run_loss_smoke(
        dataset_yaml=subset_yaml,
        model=str(args.model),
        batch=subset_batch,
        imgsz=int(args.imgsz),
        device=str(args.device),
        workers=int(args.workers),
        seed=int(args.seed),
        max_epochs=int(args.subset_epochs),
        max_batches_per_epoch=int(args.subset_max_batches),
        progress_every=int(args.progress_every),
        verbose=bool(args.verbose),
        project=Path(args.project),
        run_name=f"{args.name}_{run_tag}",
        labels_train_dir=(tmp_root / "labels" / "train"),
    )
    if not args.keep_temp:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return result, tmp_root.as_posix()


def inspect_label_file(label_path: Path) -> dict:
    info = {
        "label_path": label_path.as_posix(),
        "exists": bool(label_path.exists()),
        "size_bytes": 0,
        "lines_nonempty": 0,
        "objects": 0,
        "max_polygon_points": 0,
        "issues": [],
    }
    if not label_path.exists():
        info["issues"].append("missing_label_file")
        return info

    raw = label_path.read_bytes()
    info["size_bytes"] = len(raw)
    if not raw:
        return info
    if not raw.strip():
        info["issues"].append("whitespace_only_label")
        return info

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        info["issues"].append(f"decode_error:{exc}")
        return info

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        info["lines_nonempty"] += 1
        parts = line.split()
        if len(parts) < 7:
            info["issues"].append(f"line_{line_no}:too_few_tokens:{len(parts)}")
            continue
        try:
            class_value = float(parts[0])
        except ValueError:
            info["issues"].append(f"line_{line_no}:non_numeric_class")
            continue
        if not class_value.is_integer():
            info["issues"].append(f"line_{line_no}:non_integer_class")
        elif int(class_value) != 0:
            info["issues"].append(f"line_{line_no}:unexpected_class:{int(class_value)}")

        coords: list[float] = []
        bad_coords = False
        for token in parts[1:]:
            try:
                coords.append(float(token))
            except ValueError:
                info["issues"].append(f"line_{line_no}:non_numeric_coord")
                bad_coords = True
                break
        if bad_coords:
            continue
        if len(coords) % 2 != 0:
            info["issues"].append(f"line_{line_no}:odd_coord_count:{len(coords)}")
            continue
        points = len(coords) // 2
        info["objects"] += 1
        info["max_polygon_points"] = max(info["max_polygon_points"], points)
        if points < 3:
            info["issues"].append(f"line_{line_no}:too_few_points:{points}")
        if any((not np.isfinite(v)) for v in coords):
            info["issues"].append(f"line_{line_no}:non_finite_coord")
        if any(v < 0.0 or v > 1.0 for v in coords):
            info["issues"].append(f"line_{line_no}:coord_out_of_range")

    return info


def make_quarantine_commands(dataset_root: Path, offenders: list[dict]) -> list[str]:
    q_images = dataset_root / "quarantine_tal_isolation" / "images" / "train"
    q_labels = dataset_root / "quarantine_tal_isolation" / "labels" / "train"
    cmds = [
        f"mkdir -p {shlex.quote(q_images.as_posix())}",
        f"mkdir -p {shlex.quote(q_labels.as_posix())}",
    ]
    seen = set()
    for item in offenders:
        image_path = Path(item["image_path"])
        label_path = Path(item["label_path"])

        image_cmd = (
            f"mv {shlex.quote(image_path.as_posix())} "
            f"{shlex.quote((q_images / image_path.name).as_posix())}"
        )
        if image_cmd not in seen:
            cmds.append(image_cmd)
            seen.add(image_cmd)

        if label_path.exists():
            label_cmd = (
                f"mv {shlex.quote(label_path.as_posix())} "
                f"{shlex.quote((q_labels / label_path.name).as_posix())}"
            )
            if label_cmd not in seen:
                cmds.append(label_cmd)
                seen.add(label_cmd)
    return cmds


def bisect_candidates(
    candidates: list[Path],
    labels_train_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[Path], list[dict], bool]:
    active = sorted(candidates, key=lambda p: p.as_posix())
    logs: list[dict] = []
    interaction = False
    round_idx = 0

    while len(active) > 1:
        round_idx += 1
        mid = len(active) // 2
        left = active[:mid]
        right = active[mid:]

        left_res, left_tmp = run_subset_smoke(left, labels_train_dir, args, run_tag=f"bisect_{round_idx}_L")
        logs.append(
            {
                "stage": "bisect",
                "round": round_idx,
                "subset": "left",
                "size": len(left),
                "crashed": left_res.crashed,
                "tmp_dataset": left_tmp,
                "images": [p.as_posix() for p in left],
            }
        )
        if left_res.crashed:
            active = left
            continue

        right_res, right_tmp = run_subset_smoke(right, labels_train_dir, args, run_tag=f"bisect_{round_idx}_R")
        logs.append(
            {
                "stage": "bisect",
                "round": round_idx,
                "subset": "right",
                "size": len(right),
                "crashed": right_res.crashed,
                "tmp_dataset": right_tmp,
                "images": [p.as_posix() for p in right],
            }
        )
        if right_res.crashed:
            active = right
            continue

        interaction = True
        break

    return active, logs, interaction


def isolate_offenders(
    candidates: list[Path],
    labels_train_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[Path], list[dict], str]:
    logs: list[dict] = []
    bisected, bisect_logs, interaction = bisect_candidates(candidates, labels_train_dir, args)
    logs.extend(bisect_logs)

    if len(bisected) == 1:
        verify_res, verify_tmp = run_subset_smoke(
            bisected,
            labels_train_dir,
            args,
            run_tag="verify_single",
        )
        logs.append(
            {
                "stage": "verify_single",
                "size": 1,
                "crashed": verify_res.crashed,
                "tmp_dataset": verify_tmp,
                "images": [bisected[0].as_posix()],
            }
        )
        if verify_res.crashed:
            return bisected, logs, "single_sample"
        interaction = True

    single_hits: list[Path] = []
    for idx, image_path in enumerate(sorted(set(candidates), key=lambda p: p.as_posix()), 1):
        res, tmp_path = run_subset_smoke([image_path], labels_train_dir, args, run_tag=f"single_{idx:03d}")
        logs.append(
            {
                "stage": "single_test",
                "size": 1,
                "crashed": res.crashed,
                "tmp_dataset": tmp_path,
                "images": [image_path.as_posix()],
            }
        )
        if res.crashed:
            single_hits.append(image_path)
    if single_hits:
        return single_hits, logs, "single_sample"

    pair_tests = 0
    for a, b in itertools.combinations(sorted(set(candidates), key=lambda p: p.as_posix()), 2):
        pair_tests += 1
        if pair_tests > int(args.max_pair_tests):
            break
        res, tmp_path = run_subset_smoke([a, b], labels_train_dir, args, run_tag=f"pair_{pair_tests:03d}")
        logs.append(
            {
                "stage": "pair_test",
                "size": 2,
                "crashed": res.crashed,
                "tmp_dataset": tmp_path,
                "images": [a.as_posix(), b.as_posix()],
            }
        )
        if res.crashed:
            return [a, b], logs, "pair_interaction"

    if interaction:
        return bisected, logs, "interaction_unresolved"
    return candidates, logs, "batch_subset_unresolved"


def main() -> int:
    args = parse_args()
    dataset_yaml = args.dataset_yaml.expanduser().resolve()
    if not dataset_yaml.exists():
        raise SystemExit(f"Dataset YAML not found: {dataset_yaml}")

    resolved = resolve_dataset(dataset_yaml)
    dataset_root = Path(resolved["dataset_root"])
    train_images_dir = Path(resolved["train_images_dir"])
    train_labels_dir = Path(resolved["train_labels_dir"])
    if not train_images_dir.exists():
        raise SystemExit(f"Train images directory not found: {train_images_dir}")
    if not train_labels_dir.exists():
        raise SystemExit(f"Train labels directory not found: {train_labels_dir}")

    pairing_summary, pairing_issues, stem_to_image = validate_train_pairing(train_images_dir, train_labels_dir)
    run_dir = Path(args.project).resolve() / f"{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "isolation_report.json"

    smoke_result = run_loss_smoke(
        dataset_yaml=dataset_yaml,
        model=str(args.model),
        batch=max(1, int(args.batch)),
        imgsz=int(args.imgsz),
        device=str(args.device),
        workers=int(args.workers),
        seed=int(args.seed),
        max_epochs=int(args.max_epochs_search),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        progress_every=int(args.progress_every),
        verbose=bool(args.verbose),
        project=Path(args.project),
        run_name=f"{args.name}_full_smoke",
        labels_train_dir=train_labels_dir,
    )

    report: dict = {
        "timestamp": datetime.now().isoformat(),
        "dataset_yaml": dataset_yaml.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "train_images_dir": train_images_dir.as_posix(),
        "train_labels_dir": train_labels_dir.as_posix(),
        "args": jsonable(vars(args)),
        "pairing_summary": pairing_summary,
        "pairing_issues": [asdict(issue) for issue in pairing_issues],
        "full_smoke_result": asdict(smoke_result),
        "bisection_logs": [],
        "offenders": [],
        "quarantine_commands": [],
        "isolation_mode": "",
    }

    if not smoke_result.crashed:
        report["status"] = "not_reproduced"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Smoke check did not reproduce TAL shape mismatch within configured search window.")
        print(f"Report: {report_path}")
        return 0

    failing_images = [
        Path(sample["image_path"]).resolve()
        for sample in smoke_result.batch_samples
        if sample.get("image_path")
    ]
    unique_failing_images = sorted(set(failing_images), key=lambda p: p.as_posix())

    if args.smoke_only:
        offenders = unique_failing_images
        isolation_mode = "smoke_only_batch"
        logs = []
    else:
        offenders, logs, isolation_mode = isolate_offenders(unique_failing_images, train_labels_dir, args)

    offender_entries: list[dict] = []
    for image_path in offenders:
        label_path = train_labels_dir / f"{image_path.stem}.txt"
        label_info = inspect_label_file(label_path)
        offender_entries.append(
            {
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "label_info": label_info,
            }
        )

    quarantine_commands = make_quarantine_commands(dataset_root=dataset_root, offenders=offender_entries)

    report["status"] = "reproduced"
    report["bisection_logs"] = logs
    report["offenders"] = offender_entries
    report["quarantine_commands"] = quarantine_commands
    report["isolation_mode"] = isolation_mode
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("TAL shape-mismatch reproduced.")
    print("Failure kind:", smoke_result.crash_kind)
    print("Failing batch sample count:", len(unique_failing_images))
    print("Isolation mode:", isolation_mode)
    print("Offending candidate image(s):")
    for entry in offender_entries:
        print("image:", entry["image_path"])
        print("label:", entry["label_path"])
        issues = entry["label_info"]["issues"]
        if issues:
            print("label_issues:", ",".join(issues))
        else:
            print("label_issues: none_detected")
    print("Suggested quarantine commands:")
    for cmd in quarantine_commands:
        print(cmd)
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
