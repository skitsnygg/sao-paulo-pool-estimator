#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class SampleRef:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    rel_image: str
    rel_label: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Copy a YOLO dataset and remove samples by stem blacklist."
    )
    ap.add_argument("--src-dataset", type=Path, required=True)
    ap.add_argument("--out-dataset", type=Path, required=True)
    ap.add_argument("--blacklist", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-out", action="store_true", default=True)
    return ap.parse_args()


def read_blacklist(path: Path) -> set[str]:
    stems: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        stems.add(s)
    return stems


def find_image_for_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in sorted(IMG_EXTS):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_samples(dataset_root: Path, split: str) -> list[SampleRef]:
    img_dir = dataset_root / "images" / split
    lab_dir = dataset_root / "labels" / split
    if not img_dir.exists() or not lab_dir.exists():
        raise RuntimeError(f"Missing split dirs: {img_dir} / {lab_dir}")

    out: list[SampleRef] = []
    for label_path in sorted(lab_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image_for_stem(img_dir, stem)
        if image_path is None:
            raise RuntimeError(f"Missing image for label: {label_path}")
        out.append(
            SampleRef(
                split=split,
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                rel_image=image_path.relative_to(dataset_root).as_posix(),
                rel_label=label_path.relative_to(dataset_root).as_posix(),
            )
        )
    return out


def copy_dataset(src_root: Path, dst_root: Path, clean_out: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if clean_out and dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def remove_sample(dst_root: Path, sample: SampleRef, dry_run: bool) -> None:
    if dry_run:
        return
    img = dst_root / sample.rel_image
    lab = dst_root / sample.rel_label
    if img.exists():
        img.unlink()
    if lab.exists():
        lab.unlink()


def write_dataset_yaml(out_root: Path, dry_run: bool) -> None:
    if dry_run:
        return
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


def count_files(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file())


def main() -> int:
    args = parse_args()

    src_root = args.src_dataset.resolve()
    out_root = args.out_dataset.resolve()
    blacklist_path = args.blacklist.resolve()

    if not src_root.exists():
        raise SystemExit(f"Source dataset not found: {src_root}")
    if not blacklist_path.exists():
        raise SystemExit(f"Blacklist file not found: {blacklist_path}")

    blacklist = read_blacklist(blacklist_path)

    train_samples = collect_samples(src_root, "train")
    val_samples = collect_samples(src_root, "val")
    all_samples = train_samples + val_samples

    matches = [s for s in all_samples if s.stem in blacklist]
    missing_blacklist_stems = sorted(blacklist - {s.stem for s in matches})

    copy_dataset(src_root, out_root, args.clean_out, args.dry_run)

    for sample in matches:
        remove_sample(out_root, sample, args.dry_run)

    write_dataset_yaml(out_root, args.dry_run)

    summary = {
        "source_dataset": src_root.as_posix(),
        "output_dataset": out_root.as_posix(),
        "blacklist": blacklist_path.as_posix(),
        "dry_run": args.dry_run,
        "blacklist_count": len(blacklist),
        "matched_count": len(matches),
        "missing_blacklist_stems_count": len(missing_blacklist_stems),
        "removed": {
            "train": sum(1 for s in matches if s.split == "train"),
            "val": sum(1 for s in matches if s.split == "val"),
        },
        "preview_removed": [
            {
                "split": s.split,
                "stem": s.stem,
                "rel_image": s.rel_image,
                "rel_label": s.rel_label,
            }
            for s in matches[:200]
        ],
        "missing_blacklist_stems_preview": missing_blacklist_stems[:200],
    }

    if not args.dry_run:
        summary["remaining_counts"] = {
            "train_images": count_files(out_root / "images" / "train"),
            "train_labels": count_files(out_root / "labels" / "train"),
            "val_images": count_files(out_root / "images" / "val"),
            "val_labels": count_files(out_root / "labels" / "val"),
        }
        summary_path = out_root / "blacklist_prune_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Source dataset: {src_root}")
    print(f"Output dataset: {out_root}")
    print(f"Blacklist: {blacklist_path}")
    print(f"Dry run: {args.dry_run}")
    print()
    print(f"Blacklist stems: {len(blacklist)}")
    print(f"Matched stems: {len(matches)}")
    print(f"Missing blacklist stems: {len(missing_blacklist_stems)}")
    print(f"Removed from train: {sum(1 for s in matches if s.split == 'train')}")
    print(f"Removed from val: {sum(1 for s in matches if s.split == 'val')}")

    if not args.dry_run:
        print(f"Summary: {out_root / 'blacklist_prune_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())