#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

CELL_RE = re.compile(r"(cell_\d+_\d+)", re.IGNORECASE)
TILE_RE = re.compile(r"(r\d+_c\d+)", re.IGNORECASE)
CANON_RE = re.compile(r"^(cell_\d+_\d+__r\d+_c\d+)$", re.IGNORECASE)


@dataclass
class Row:
    kind: str
    src: str
    dst: str
    canonical_stem: str
    status: str
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract a CVAT zip export, flatten images/masks to canonical names "
            "(cell_XXXX_XXXX__rXXXX_cXXXX), and write to a flat output folder."
        )
    )
    p.add_argument(
        "--zip",
        required=True,
        help="Path to CVAT zip export, e.g. /Users/admin/Downloads/v1.zip",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output flat directory, e.g. /tmp/z21_v1_flat",
    )
    p.add_argument(
        "--copy-masks",
        action="store_true",
        help="Copy SegmentationObject masks into output/SegmentationObject",
    )
    p.add_argument(
        "--copy-segmentation-class",
        action="store_true",
        help="Copy SegmentationClass masks into output/SegmentationClass",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing canonical files in output if encountered again",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing files",
    )
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def canonical_from_path(p: Path) -> str | None:
    stem = p.stem.lower()

    if CANON_RE.match(stem):
        return stem

    tile_match = TILE_RE.search(stem)
    cell_match = CELL_RE.search(str(p).lower())

    if tile_match and cell_match:
        return f"{cell_match.group(1).lower()}__{tile_match.group(1).lower()}"

    return None


def collect_under_anchor(root: Path, anchor_name: str, exts: set[str]) -> list[Path]:
    out: list[Path] = []
    anchor_name = anchor_name.lower()

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        parts_lower = [x.lower() for x in p.parts]
        if anchor_name in parts_lower:
            out.append(p)

    return sorted(out)


def build_mask_lookup(paths: list[Path]) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        canon = canonical_from_path(p)
        if canon:
            lookup[canon].append(p)
    return lookup


def copy_one(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "exists_skipped"
    if not dry_run:
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
    return "written_overwrite" if dst.exists() and overwrite else "written"


def main() -> int:
    args = parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not zip_path.exists():
        print(f"ERROR: zip does not exist: {zip_path}", file=sys.stderr)
        return 2
    if not zipfile.is_zipfile(zip_path):
        print(f"ERROR: not a valid zip file: {zip_path}", file=sys.stderr)
        return 2

    img_out = out / "JPEGImages"
    obj_out = out / "SegmentationObject"
    cls_out = out / "SegmentationClass"
    report_dir = out / "_report"

    ensure_dir(out)
    ensure_dir(img_out)
    ensure_dir(report_dir)
    if args.copy_masks:
        ensure_dir(obj_out)
    if args.copy_segmentation_class:
        ensure_dir(cls_out)

    rows: list[Row] = []
    unresolved_images: list[str] = []
    collisions: dict[str, list[str]] = defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="flatten_cvat_zip_") as tmpdir:
        tmp_root = Path(tmpdir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_root)

        images = collect_under_anchor(tmp_root, "JPEGImages", IMAGE_EXTS)
        segobj_paths = (
            collect_under_anchor(tmp_root, "SegmentationObject", MASK_EXTS)
            if args.copy_masks
            else []
        )
        segcls_paths = (
            collect_under_anchor(tmp_root, "SegmentationClass", MASK_EXTS)
            if args.copy_segmentation_class
            else []
        )

        segobj_lookup = build_mask_lookup(segobj_paths) if args.copy_masks else {}
        segcls_lookup = build_mask_lookup(segcls_paths) if args.copy_segmentation_class else {}

        images_found = 0
        images_written = 0
        segobj_written = 0
        segcls_written = 0

        for img in images:
            images_found += 1
            canon = canonical_from_path(img)

            if not canon:
                unresolved_images.append(str(img))
                rows.append(
                    Row(
                        kind="image",
                        src=str(img),
                        dst="",
                        canonical_stem="",
                        status="unresolved",
                        note="Could not derive canonical name from path",
                    )
                )
                continue

            dst_img = img_out / f"{canon}{img.suffix.lower()}"
            existed_before = dst_img.exists()

            status = copy_one(img, dst_img, args.overwrite, args.dry_run)
            if status == "exists_skipped":
                rows.append(
                    Row(
                        kind="image",
                        src=str(img),
                        dst=str(dst_img),
                        canonical_stem=canon,
                        status=status,
                        note="Destination exists and overwrite is disabled",
                    )
                )
            else:
                images_written += 1
                rows.append(
                    Row(
                        kind="image",
                        src=str(img),
                        dst=str(dst_img),
                        canonical_stem=canon,
                        status="written_overwrite" if existed_before and args.overwrite else "written",
                        note="",
                    )
                )

            if args.copy_masks:
                hits = segobj_lookup.get(canon, [])
                if len(hits) == 1:
                    src_mask = hits[0]
                    dst_mask = obj_out / f"{canon}{src_mask.suffix.lower()}"
                    existed_before = dst_mask.exists()
                    status = copy_one(src_mask, dst_mask, args.overwrite, args.dry_run)
                    if status != "exists_skipped":
                        segobj_written += 1
                    rows.append(
                        Row(
                            kind="segmentation_object",
                            src=str(src_mask),
                            dst=str(dst_mask),
                            canonical_stem=canon,
                            status="written_overwrite" if existed_before and args.overwrite else status,
                            note="" if status != "exists_skipped" else "Destination exists and overwrite is disabled",
                        )
                    )
                elif len(hits) == 0:
                    rows.append(
                        Row(
                            kind="segmentation_object",
                            src="",
                            dst="",
                            canonical_stem=canon,
                            status="missing",
                            note="No SegmentationObject mask found",
                        )
                    )
                else:
                    collisions[canon].extend(str(x) for x in hits)
                    rows.append(
                        Row(
                            kind="segmentation_object",
                            src=" | ".join(str(x) for x in hits[:10]),
                            dst="",
                            canonical_stem=canon,
                            status="ambiguous",
                            note=f"Multiple SegmentationObject matches: {len(hits)}",
                        )
                    )

            if args.copy_segmentation_class:
                hits = segcls_lookup.get(canon, [])
                if len(hits) == 1:
                    src_mask = hits[0]
                    dst_mask = cls_out / f"{canon}{src_mask.suffix.lower()}"
                    existed_before = dst_mask.exists()
                    status = copy_one(src_mask, dst_mask, args.overwrite, args.dry_run)
                    if status != "exists_skipped":
                        segcls_written += 1
                    rows.append(
                        Row(
                            kind="segmentation_class",
                            src=str(src_mask),
                            dst=str(dst_mask),
                            canonical_stem=canon,
                            status="written_overwrite" if existed_before and args.overwrite else status,
                            note="" if status != "exists_skipped" else "Destination exists and overwrite is disabled",
                        )
                    )
                elif len(hits) == 0:
                    rows.append(
                        Row(
                            kind="segmentation_class",
                            src="",
                            dst="",
                            canonical_stem=canon,
                            status="missing",
                            note="No SegmentationClass mask found",
                        )
                    )
                else:
                    collisions[canon].extend(str(x) for x in hits)
                    rows.append(
                        Row(
                            kind="segmentation_class",
                            src=" | ".join(str(x) for x in hits[:10]),
                            dst="",
                            canonical_stem=canon,
                            status="ambiguous",
                            note=f"Multiple SegmentationClass matches: {len(hits)}",
                        )
                    )

    csv_path = report_dir / f"{zip_path.stem}_flatten_manifest.csv"
    json_path = report_dir / f"{zip_path.stem}_flatten_stats.json"
    unresolved_path = report_dir / f"{zip_path.stem}_unresolved_images.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["kind", "src", "dst", "canonical_stem", "status", "note"],
        )
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))

    unresolved_path.write_text(
        "\n".join(unresolved_images) + ("\n" if unresolved_images else ""),
        encoding="utf-8",
    )

    stats = {
        "zip": str(zip_path),
        "out": str(out),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "copy_masks": args.copy_masks,
        "copy_segmentation_class": args.copy_segmentation_class,
        "rows_total": len(rows),
        "unresolved_images": len(unresolved_images),
        "collision_keys": sorted(collisions.keys()),
        "manifest_csv": str(csv_path),
        "stats_json": str(json_path),
        "unresolved_images_txt": str(unresolved_path),
    }
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("zip:", zip_path)
    print("out:", out)
    print("overwrite:", args.overwrite)
    print("dry_run:", args.dry_run)
    print("manifest_csv:", csv_path)
    print("stats_json:", json_path)
    print("unresolved_images_txt:", unresolved_path)
    print("unresolved_images:", len(unresolved_images))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())