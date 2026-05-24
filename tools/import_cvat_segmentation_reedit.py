#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Reuse polygon extraction behavior from the existing converter.
try:
    from masks_to_yolo_seg import mask_to_polygons
except ImportError:  # pragma: no cover - fallback when imported as module
    from tools.masks_to_yolo_seg import mask_to_polygons


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
MAX_EXAMPLES = 25


@dataclass
class DatasetLayout:
    layout_name: str
    train_images_dir: Path
    val_images_dir: Path
    train_labels_dir: Path
    val_labels_dir: Path


@dataclass
class StemManifest:
    stem: str
    listed_in_default: bool = False
    jpeg_image_candidates: list[Path] = field(default_factory=list)
    segobj_mask_candidates: list[Path] = field(default_factory=list)
    segclass_mask_candidates: list[Path] = field(default_factory=list)

    def any_image_candidate(self) -> bool:
        return bool(self.jpeg_image_candidates)

    def any_mask_candidate(self) -> bool:
        return bool(self.segobj_mask_candidates or self.segclass_mask_candidates)

    def choose_image_source(self) -> Path | None:
        if self.jpeg_image_candidates:
            return self.jpeg_image_candidates[0]
        return None

    def choose_mask_source(self) -> Path | None:
        for group in (self.segobj_mask_candidates, self.segclass_mask_candidates):
            if group:
                return group[0]
        return None


@dataclass
class Summary:
    total_export_stems_discovered: int = 0
    stems_listed_in_default_txt: int = 0
    stems_actually_found_in_image_folders: int = 0
    stems_with_masks_found: int = 0
    matches_in_train: int = 0
    matches_in_val: int = 0
    collisions_in_both: int = 0
    missing_from_dataset_entirely: int = 0
    items_would_overwrite_train: int = 0
    items_would_overwrite_val: int = 0
    items_skipped_missing_mask: int = 0
    items_skipped_invalid_empty_polygons: int = 0
    items_skipped_split_internal_duplicates: int = 0


@dataclass
class RoleRoots:
    source_image_root: Path | None
    object_mask_root: Path | None
    class_mask_root: Path | None
    source_image_file_count: int
    object_mask_file_count: int
    class_mask_file_count: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Import a CVAT segmentation-mask export into an existing YOLOv8 "
            "segmentation dataset, with split-safe matching and dry-run support."
        )
    )
    ap.add_argument("--src-root", required=True, help="CVAT export root.")
    ap.add_argument("--dataset-root", required=True, help="Target dataset root.")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class id. Default: 0")
    ap.add_argument(
        "--min-area-px",
        type=float,
        default=20.0,
        help="Minimum contour area in px for polygon keep. Default: 20.0",
    )
    ap.add_argument(
        "--simplify-epsilon-ratio",
        type=float,
        default=0.002,
        help="Douglas-Peucker epsilon ratio. Default: 0.002",
    )
    ap.add_argument(
        "--prefer-default-txt",
        action="store_true",
        help="Restrict processing to stems listed in ImageSets/Segmentation/default.txt.",
    )
    ap.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    return ap.parse_args()


def parse_default_stems(default_txt: Path) -> set[str]:
    stems: set[str] = set()
    if not default_txt.exists():
        return stems

    for raw_line in default_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # CVAT lines can be "images/foo", "foo", "foo.png", etc.
        name_part = Path(line).name
        stem = Path(name_part).stem
        if stem:
            stems.add(stem)
    return stems


def find_role_dirs(src_root: Path, role_name: str) -> list[Path]:
    """Find candidate role directories while supporting slight layout variation."""
    role_lower = role_name.lower()
    candidates: list[Path] = []

    direct_images = src_root / role_name / "images"
    direct_role = src_root / role_name

    if direct_images.is_dir():
        candidates.append(direct_images)
    elif direct_role.is_dir():
        candidates.append(direct_role)

    for p in src_root.rglob("*"):
        if not p.is_dir() or p.name.lower() != role_lower:
            continue
        maybe_images = p / "images"
        if maybe_images.is_dir():
            candidates.append(maybe_images)
        else:
            candidates.append(p)

    seen: set[Path] = set()
    unique: list[Path] = []
    for cand in sorted(candidates, key=lambda x: x.as_posix()):
        resolved = cand.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(cand)
    return unique


def iter_image_like_files(dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in dirs:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith("._"):
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(p)
    return files


def add_unique_path(paths: list[Path], path: Path) -> None:
    if path not in paths:
        paths.append(path)


def choose_role_root(src_root: Path, role_name: str) -> tuple[Path | None, list[Path]]:
    """
    Choose a single root for each role.
    We intentionally bind behavior to export folder role, not '/images/' path text.
    """
    for root in find_role_dirs(src_root, role_name):
        files = iter_image_like_files([root])
        if files:
            return root, files
    return None, []


def build_manifest(
    src_root: Path,
    default_stems: set[str],
    verbose: bool,
) -> tuple[dict[str, StemManifest], set[str], set[str], set[str], RoleRoots]:
    manifest: dict[str, StemManifest] = {}

    def get_entry(stem: str) -> StemManifest:
        if stem not in manifest:
            manifest[stem] = StemManifest(stem=stem)
        return manifest[stem]

    for stem in sorted(default_stems):
        get_entry(stem).listed_in_default = True

    jpeg_root, jpeg_files = choose_role_root(src_root, "JPEGImages")
    segobj_root, segobj_files = choose_role_root(src_root, "SegmentationObject")
    segclass_root, segclass_files = choose_role_root(src_root, "SegmentationClass")
    role_roots = RoleRoots(
        source_image_root=jpeg_root,
        object_mask_root=segobj_root,
        class_mask_root=segclass_root,
        source_image_file_count=len(jpeg_files),
        object_mask_file_count=len(segobj_files),
        class_mask_file_count=len(segclass_files),
    )

    if verbose:
        print(
            f"[ROLE_ROOTS] source_image_root={jpeg_root} object_mask_root={segobj_root} "
            f"class_mask_root={segclass_root}"
        )

    for p in jpeg_files:
        add_unique_path(get_entry(p.stem).jpeg_image_candidates, p)

    # Folder role controls mask interpretation here.
    for p in segobj_files:
        add_unique_path(get_entry(p.stem).segobj_mask_candidates, p)

    # Class masks are fallback when object masks are unavailable for a stem.
    for p in segclass_files:
        add_unique_path(get_entry(p.stem).segclass_mask_candidates, p)

    for entry in manifest.values():
        entry.jpeg_image_candidates.sort(key=lambda x: x.as_posix())
        entry.segobj_mask_candidates.sort(key=lambda x: x.as_posix())
        entry.segclass_mask_candidates.sort(key=lambda x: x.as_posix())

    discovered_stems = set(manifest.keys())
    stems_with_any_images = {stem for stem, entry in manifest.items() if entry.any_image_candidate()}
    stems_with_any_masks = {stem for stem, entry in manifest.items() if entry.any_mask_candidate()}
    return manifest, discovered_stems, stems_with_any_images, stems_with_any_masks, role_roots


def resolve_dataset_layout(dataset_root: Path) -> DatasetLayout:
    # Standard YOLO layout: images/train + images/val
    yolo_train_images = dataset_root / "images" / "train"
    yolo_val_images = dataset_root / "images" / "val"
    if yolo_train_images.is_dir() and yolo_val_images.is_dir():
        return DatasetLayout(
            layout_name="yolo_images_labels",
            train_images_dir=yolo_train_images,
            val_images_dir=yolo_val_images,
            train_labels_dir=dataset_root / "labels" / "train",
            val_labels_dir=dataset_root / "labels" / "val",
        )

    # Alternative layout requested in prompt: train/images + val/images
    alt_train_images = dataset_root / "train" / "images"
    alt_val_images = dataset_root / "val" / "images"
    if alt_train_images.is_dir() and alt_val_images.is_dir():
        train_labels_candidates = [dataset_root / "train" / "labels", dataset_root / "labels" / "train"]
        val_labels_candidates = [dataset_root / "val" / "labels", dataset_root / "labels" / "val"]
        train_labels = next((p for p in train_labels_candidates if p.exists()), train_labels_candidates[0])
        val_labels = next((p for p in val_labels_candidates if p.exists()), val_labels_candidates[0])
        return DatasetLayout(
            layout_name="split_scoped_train_val",
            train_images_dir=alt_train_images,
            val_images_dir=alt_val_images,
            train_labels_dir=train_labels,
            val_labels_dir=val_labels,
        )

    raise SystemExit(
        "Could not resolve dataset layout. Expected either:\n"
        " - <dataset>/images/train + <dataset>/images/val\n"
        " - <dataset>/train/images + <dataset>/val/images"
    )


def build_split_index(images_dir: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(images_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        idx[p.stem].append(p)
    return dict(idx)


def format_yolo_lines(polygons: list[list[float]], class_id: int) -> list[str]:
    lines: list[str] = []
    for poly in polygons:
        if len(poly) < 6:
            continue
        lines.append(f"{class_id} " + " ".join(f"{val:.6f}" for val in poly))
    return lines


def print_examples(title: str, stems: list[str]) -> None:
    print(f"{title} (showing up to {MAX_EXAMPLES}, total={len(stems)}):")
    if not stems:
        print("  - none")
        return
    for stem in stems[:MAX_EXAMPLES]:
        print(f"  - {stem}")


def main() -> None:
    args = parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not src_root.exists():
        raise SystemExit(f"Source root not found: {src_root}")
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    default_txt = src_root / "ImageSets" / "Segmentation" / "default.txt"
    default_stems = parse_default_stems(default_txt)
    manifest, discovered_stems, stems_with_images, stems_with_masks, role_roots = build_manifest(
        src_root=src_root,
        default_stems=default_stems,
        verbose=bool(args.verbose),
    )

    if args.prefer_default_txt and default_stems:
        process_stems = sorted(default_stems)
    else:
        process_stems = sorted(discovered_stems.union(default_stems))

    layout = resolve_dataset_layout(dataset_root)
    train_index = build_split_index(layout.train_images_dir)
    val_index = build_split_index(layout.val_images_dir)

    if not args.dry_run:
        layout.train_labels_dir.mkdir(parents=True, exist_ok=True)
        layout.val_labels_dir.mkdir(parents=True, exist_ok=True)

    summary = Summary(
        total_export_stems_discovered=len(discovered_stems),
        stems_listed_in_default_txt=len(default_stems),
        stems_actually_found_in_image_folders=len(stems_with_images),
        stems_with_masks_found=len(stems_with_masks),
    )

    train_matches_examples: list[str] = []
    val_matches_examples: list[str] = []
    collision_examples: list[str] = []
    missing_dataset_examples: list[str] = []
    missing_mask_examples: list[str] = []

    print("=== Import Configuration ===")
    print(f"src_root: {src_root}")
    print(f"dataset_root: {dataset_root}")
    print(f"dataset_layout: {layout.layout_name}")
    print(f"train_images_dir: {layout.train_images_dir}")
    print(f"val_images_dir: {layout.val_images_dir}")
    print(f"train_labels_dir: {layout.train_labels_dir}")
    print(f"val_labels_dir: {layout.val_labels_dir}")
    print(f"default_txt: {default_txt} (exists={default_txt.exists()})")
    print(f"prefer_default_txt: {bool(args.prefer_default_txt)}")
    print(f"dry_run: {bool(args.dry_run)}")
    print(f"class_id: {int(args.class_id)}")
    print(f"min_area_px: {float(args.min_area_px)}")
    print(f"simplify_epsilon_ratio: {float(args.simplify_epsilon_ratio)}")
    print(f"process_stems_count: {len(process_stems)}")
    print("=== Export Role Roots ===")
    print(
        f"chosen_source_image_root: {role_roots.source_image_root} "
        f"(files={role_roots.source_image_file_count})"
    )
    print(
        f"chosen_object_mask_root: {role_roots.object_mask_root} "
        f"(files={role_roots.object_mask_file_count})"
    )
    print(
        f"chosen_class_mask_root: {role_roots.class_mask_root} "
        f"(files={role_roots.class_mask_file_count})"
    )
    print("=== Decisions ===")

    for stem in process_stems:
        entry = manifest.get(stem, StemManifest(stem=stem))
        train_paths = train_index.get(stem, [])
        val_paths = val_index.get(stem, [])

        in_train = bool(train_paths)
        in_val = bool(val_paths)

        if in_train and not in_val:
            summary.matches_in_train += 1
            train_matches_examples.append(stem)
        elif in_val and not in_train:
            summary.matches_in_val += 1
            val_matches_examples.append(stem)
        elif in_train and in_val:
            summary.collisions_in_both += 1
            collision_examples.append(stem)
        else:
            summary.missing_from_dataset_entirely += 1
            missing_dataset_examples.append(stem)

        if in_train and in_val:
            print(
                f"[SKIP_COLLISION_BOTH_SPLITS] stem={stem} "
                f"train_count={len(train_paths)} val_count={len(val_paths)}"
            )
            continue

        if (in_train and len(train_paths) > 1) or (in_val and len(val_paths) > 1):
            summary.items_skipped_split_internal_duplicates += 1
            print(
                f"[SKIP_SPLIT_INTERNAL_DUPLICATES] stem={stem} "
                f"train_count={len(train_paths)} val_count={len(val_paths)}"
            )
            continue

        if not in_train and not in_val:
            print(f"[SKIP_NEW_TILE_NOT_CURRENTLY_IN_DATASET] stem={stem}")
            continue

        split = "train" if in_train else "val"
        target_image_path = train_paths[0] if in_train else val_paths[0]
        target_label_path = (
            layout.train_labels_dir / f"{stem}.txt"
            if in_train
            else layout.val_labels_dir / f"{stem}.txt"
        )

        image_src = entry.choose_image_source()
        mask_src = entry.choose_mask_source()

        if mask_src is None:
            summary.items_skipped_missing_mask += 1
            missing_mask_examples.append(stem)
            print(
                f"[SKIP_MISSING_MASK] stem={stem} split={split} "
                f"target_image={target_image_path}"
            )
            continue

        polygons = mask_to_polygons(
            mask_path=mask_src,
            min_area_px=float(args.min_area_px),
            simplify_epsilon_ratio=float(args.simplify_epsilon_ratio),
        )
        lines = format_yolo_lines(polygons=polygons, class_id=int(args.class_id))

        if not lines:
            summary.items_skipped_invalid_empty_polygons += 1
            print(
                f"[SKIP_INVALID_OR_EMPTY_POLYGONS] stem={stem} split={split} "
                f"mask={mask_src}"
            )
            continue

        if in_train:
            summary.items_would_overwrite_train += 1
        else:
            summary.items_would_overwrite_val += 1

        action_prefix = "DRYRUN_WOULD_OVERWRITE" if args.dry_run else "APPLY_OVERWRITE"
        print(
            f"[{action_prefix}] stem={stem} split={split} "
            f"overwrite_label={target_label_path} "
            f"overwrite_image={'yes' if image_src else 'no'} "
            f"source_image={image_src if image_src else 'NONE'} "
            f"target_image={target_image_path} "
            f"mask={mask_src} polygons={len(lines)}"
        )

        if args.dry_run:
            continue

        if image_src is not None:
            shutil.copy2(image_src, target_image_path)
        target_label_path.parent.mkdir(parents=True, exist_ok=True)
        target_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Summary ===")
    print(f"total_export_stems_discovered: {summary.total_export_stems_discovered}")
    print(f"stems_listed_in_default.txt: {summary.stems_listed_in_default_txt}")
    print(f"stems_actually_found_in_image_folders: {summary.stems_actually_found_in_image_folders}")
    print(f"stems_with_masks_found: {summary.stems_with_masks_found}")
    print(f"matches_in_train: {summary.matches_in_train}")
    print(f"matches_in_val: {summary.matches_in_val}")
    print(f"collisions_in_both: {summary.collisions_in_both}")
    print(f"missing_from_dataset_entirely: {summary.missing_from_dataset_entirely}")
    print(f"items_would_overwrite_train: {summary.items_would_overwrite_train}")
    print(f"items_would_overwrite_val: {summary.items_would_overwrite_val}")
    print(f"items_skipped_due_to_missing_mask: {summary.items_skipped_missing_mask}")
    print(
        "items_skipped_due_to_invalid_or_empty_polygons: "
        f"{summary.items_skipped_invalid_empty_polygons}"
    )
    print(
        "items_skipped_due_to_split_internal_duplicates: "
        f"{summary.items_skipped_split_internal_duplicates}"
    )

    print_examples("train matches", sorted(set(train_matches_examples)))
    print_examples("val matches", sorted(set(val_matches_examples)))
    print_examples("both-split collisions", sorted(set(collision_examples)))
    print_examples("missing from dataset", sorted(set(missing_dataset_examples)))
    print_examples("missing mask", sorted(set(missing_mask_examples)))


if __name__ == "__main__":
    main()
