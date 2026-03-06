#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Defaults tuned for your project layout.
DEFAULT_MODEL = Path("checkpoints/pools_idesp_v26_best.pt")
DEFAULT_SOURCE_ROOT = Path(
    "data/raw/idesp_ortho/FEHIDRO_ORTOMOSAICO_IGC_RMSP_2023_2024_3857_jpg"
)
DEFAULT_PROJECT = Path("runs/segment")
DEFAULT_NAME = "idesp_full_v26_predict_unique"
DEFAULT_STAGING_ROOT = Path("runs/debug")
DEFAULT_IMGSZ = 1024
DEFAULT_BATCH = 4
DEFAULT_DEVICE = "mps"
DEFAULT_CONF = 0.15

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run YOLO segmentation on nested tile folders without filename collisions "
            "by creating unique symlink names that include the cell directory."
        )
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    p.add_argument("--name", type=str, default=DEFAULT_NAME)
    p.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--limit", type=int, default=0, help="Optional: process only first N images for testing")
    p.add_argument("--clean", action="store_true", help="Delete previous staging/output dirs before running")
    return p.parse_args()


def iter_images(root: Path) -> list[Path]:
    images = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    images.sort(key=lambda p: p.as_posix())
    return images


def make_unique_name(rel: Path) -> str:
    stem_parts = list(rel.with_suffix("").parts)
    # Example: cell_0000_0003/r0000_c0000.jpg -> cell_0000_0003__r0000_c0000.jpg
    base = "__".join(stem_parts)
    return f"{base}{rel.suffix.lower()}"


def ensure_empty_dir(path: Path, clean: bool) -> None:
    if path.exists() and clean:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_mapping(mapping_path: Path, rows: list[tuple[str, Path]]) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        f.write("unique_name\trelative_source\n")
        for unique_name, rel in rows:
            f.write(f"{unique_name}\t{rel.as_posix()}\n")


def build_symlink_stage(
    source_root: Path,
    stage_dir: Path,
    mapping_path: Path,
    limit: int,
    clean: bool,
) -> tuple[list[tuple[str, Path]], int]:
    # Stage directory should always be rebuilt to avoid stale symlinks affecting source size.
    ensure_empty_dir(stage_dir, clean=True)

    images = iter_images(source_root)
    if limit > 0:
        images = images[:limit]

    rows: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    collision_count = 0

    for src in images:
        rel = src.relative_to(source_root)
        unique_name = make_unique_name(rel)
        if unique_name in seen_names:
            collision_count += 1
            continue
        seen_names.add(unique_name)

        dst = stage_dir / unique_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # Use relative symlink for portability inside project directory.
        rel_target = os.path.relpath(src, start=stage_dir)
        os.symlink(rel_target, dst)
        rows.append((unique_name, rel))

    write_mapping(mapping_path, rows)
    return rows, collision_count


def run_yolo_predict(
    model: Path,
    stage_dir: Path,
    project: Path,
    name: str,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
) -> int:
    cmd = [
        ".venv/bin/yolo",
        "segment",
        "predict",
        f"model={model.as_posix()}",
        f"source={stage_dir.as_posix()}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={device}",
        f"conf={conf}",
        "save=False",
        "save_txt=True",
        "save_conf=True",
        "retina_masks=True",
        f"project={project.as_posix()}",
        f"name={name}",
        "exist_ok=True",
    ]

    print("Running:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def rebuild_labels_by_cell(
    output_dir: Path,
    mapping_rows: list[tuple[str, Path]],
    clean: bool,
) -> tuple[int, int]:
    labels_flat = output_dir / "labels"
    labels_by_cell = output_dir / "labels_by_cell"
    ensure_empty_dir(labels_by_cell, clean=clean)

    copied = 0
    missing = 0

    for unique_name, rel in mapping_rows:
        flat_label = labels_flat / Path(unique_name).with_suffix(".txt")
        if not flat_label.exists():
            missing += 1
            continue

        out_label = labels_by_cell / rel.with_suffix(".txt")
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(flat_label, out_label)
        copied += 1

    return copied, missing


def main() -> int:
    args = parse_args()

    model = args.model.resolve()
    source_root = args.source_root.resolve()
    project = args.project.resolve()
    name = args.name
    output_dir = project / name

    staging_root = args.staging_root.resolve()
    stage_dir = staging_root / f"{name}_stage"
    mapping_path = staging_root / f"{name}_mapping.tsv"

    if not model.exists():
        print(f"Model not found: {model}", file=sys.stderr)
        return 2
    if not source_root.exists():
        print(f"Source root not found: {source_root}", file=sys.stderr)
        return 2

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    mapping_rows, name_collisions = build_symlink_stage(
        source_root=source_root,
        stage_dir=stage_dir,
        mapping_path=mapping_path,
        limit=args.limit,
        clean=args.clean,
    )

    if not mapping_rows:
        print("No images found after filtering.", file=sys.stderr)
        return 2

    print(f"Staged images: {len(mapping_rows)}")
    print(f"Name collisions skipped: {name_collisions}")
    print(f"Stage dir: {stage_dir}")
    print(f"Mapping file: {mapping_path}")

    rc = run_yolo_predict(
        model=model,
        stage_dir=stage_dir,
        project=project,
        name=name,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
    )
    if rc != 0:
        print(f"YOLO predict failed with code {rc}", file=sys.stderr)
        return rc

    copied, missing = rebuild_labels_by_cell(
        output_dir=output_dir,
        mapping_rows=mapping_rows,
        clean=True,
    )

    print("Done.")
    print(f"Output dir: {output_dir}")
    print(f"Flat labels dir: {output_dir / 'labels'}")
    print(f"Cell-preserved labels dir: {output_dir / 'labels_by_cell'}")
    print(f"Labels copied to cell tree: {copied}")
    print(f"Images with no detections (missing label file): {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
