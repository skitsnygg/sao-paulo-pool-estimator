#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
CELL_RE = re.compile(r"cell_\d{4}_\d{4}$")
RC_RE = re.compile(r"r\d{4}_c\d{4}$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Import positive CVAT segmentation zip(s) into an existing YOLO segmentation dataset."
    )
    ap.add_argument("--archive", type=Path, required=True, help="Path to outer zip, e.g. ~/Downloads/3_MISS.zip")
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="YOLO dataset root, e.g. data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned",
    )
    ap.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Root imagery folder, e.g. data/raw/geosampa_ortho/sp_city_2020_rebuild_official",
    )
    ap.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which split to add imported samples to.",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional extraction directory. Defaults to a temporary directory.",
    )
    ap.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep temporary extraction directory after completion.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing files.",
    )
    return ap.parse_args()


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def find_nested_zips(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.zip"):
        if "__MACOSX" in p.parts:
            continue
        if p.name.startswith("._"):
            continue
        out.append(p)
    return sorted(out)


def find_first(root: Path, rel_pattern: str) -> Path | None:
    matches = list(root.rglob(rel_pattern))
    return matches[0] if matches else None


def read_default_entries(default_txt: Path) -> list[str]:
    entries = []
    for ln in default_txt.read_text(encoding="utf-8").splitlines():
        s = ln.strip().replace("\\", "/")
        if not s:
            continue
        s = s.strip("/")
        s = re.sub(r"\.(png|jpg|jpeg|tif|tiff|webp)$", "", s, flags=re.IGNORECASE)
        entries.append(s)
    return entries


def canonical_from_identity(identity: str) -> str:
    """
    Accepts things like:
      cell_0026_0007/r0000_c0008
      some/path/cell_0026_0007/r0000_c0008
      cell_0026_0007__r0000_c0008
      r0000_c0008
    Returns best canonical form:
      cell_0026_0007__r0000_c0008
      or fallback r0000_c0008
    """
    p = Path(identity)
    stem = p.name

    if "__" in stem:
        left, right = stem.split("__", 1)
        if CELL_RE.fullmatch(left) and RC_RE.fullmatch(right):
            return stem

    parts = [x for x in p.parts if x not in ("", ".", "..")]
    cell = None
    rc = None
    for token in parts:
        if CELL_RE.fullmatch(token):
            cell = token
        if RC_RE.fullmatch(token):
            rc = token

    if cell and rc:
        return f"{cell}__{rc}"
    if RC_RE.fullmatch(stem):
        return stem
    raise ValueError(f"Could not parse canonical id from identity: {identity}")


def resolve_source_image(image_root: Path, canonical_id: str) -> Path | None:
    if "__" in canonical_id:
        cell, rc = canonical_id.split("__", 1)
        for ext in IMG_EXTS:
            cand = image_root / cell / f"{rc}{ext}"
            if cand.exists():
                return cand
    else:
        rc = canonical_id
        matches = []
        for ext in IMG_EXTS:
            matches.extend(image_root.rglob(f"{rc}{ext}"))
        if len(matches) == 1:
            return matches[0]
    return None


def find_mask_path(mask_root: Path, canonical_id: str) -> Path | None:
    candidates: list[Path] = []

    if "__" in canonical_id:
        cell, rc = canonical_id.split("__", 1)
        candidates.extend(
            [
                mask_root / f"{canonical_id}.png",
                mask_root / cell / f"{rc}.png",
                mask_root / f"{rc}.png",
            ]
        )
    else:
        rc = canonical_id
        candidates.append(mask_root / f"{rc}.png")

    for cand in candidates:
        if cand.exists():
            return cand

    # fallback search
    needle = canonical_id.split("__", 1)[-1] + ".png"
    matches = sorted(mask_root.rglob(needle))
    return matches[0] if matches else None


def mask_to_yolo_segments(mask_path: Path) -> list[str]:
    """
    Convert nonzero mask regions to YOLO segmentation lines.
    Class id is always 0.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines: list[str] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue

        coords: list[str] = []
        for x, y in pts:
            xn = min(max(float(x) / w, 0.0), 1.0)
            yn = min(max(float(y) / h, 0.0), 1.0)
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")

        lines.append("0 " + " ".join(coords))

    return lines


def main() -> int:
    args = parse_args()

    archive = args.archive.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()

    img_dst = dataset / "images" / args.split
    lbl_dst = dataset / "labels" / args.split

    if not archive.exists():
        raise SystemExit(f"Archive not found: {archive}")
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")
    if not image_root.exists():
        raise SystemExit(f"Image root not found: {image_root}")

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="cvat_positive_import_"))
        cleanup = not args.keep_work_dir

    outer_dir = work_dir / "outer"
    nested_dir = work_dir / "nested"
    outer_dir.mkdir(parents=True, exist_ok=True)
    nested_dir.mkdir(parents=True, exist_ok=True)

    extract_zip(archive, outer_dir)
    nested_zips = find_nested_zips(outer_dir)

    if not nested_zips:
        # maybe the archive itself is the CVAT export
        nested_zips = [archive]

    imported = 0
    skipped_existing = 0
    missing_image = 0
    missing_mask = 0

    for idx, zp in enumerate(nested_zips):
        export_dir = nested_dir / f"export_{idx:03d}"
        if zp == archive:
            # already extracted outer archive; use outer_dir directly
            export_dir = outer_dir
        else:
            extract_zip(zp, export_dir)

        default_txt = find_first(export_dir, "ImageSets/Segmentation/default.txt")
        if default_txt is None:
            continue

        segobj = find_first(export_dir, "SegmentationObject")
        if segobj is None:
            segobj = find_first(export_dir, "SegmentationClass")
        if segobj is None:
            continue

        entries = read_default_entries(default_txt)

        for entry in entries:
            canonical_id = canonical_from_identity(entry)

            src_img = resolve_source_image(image_root, canonical_id)
            if src_img is None:
                missing_image += 1
                print(f"Missing source image for: {canonical_id}")
                continue

            mask_path = find_mask_path(segobj, canonical_id)
            if mask_path is None:
                missing_mask += 1
                print(f"Missing mask for: {canonical_id}")
                continue

            out_img = img_dst / f"{canonical_id}{src_img.suffix.lower()}"
            out_lbl = lbl_dst / f"{canonical_id}.txt"

            if out_img.exists() or out_lbl.exists():
                skipped_existing += 1
                print(f"Skipping existing: {canonical_id}")
                continue

            yolo_lines = mask_to_yolo_segments(mask_path)
            if not yolo_lines:
                print(f"Mask had no usable contours: {canonical_id}")
                continue

            if not args.dry_run:
                shutil.copy2(src_img, out_img)
                out_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

            imported += 1
            print(f"Imported: {canonical_id}")

    print()
    print(f"Imported: {imported}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Missing image: {missing_image}")
    print(f"Missing mask: {missing_mask}")
    print(f"Dry run: {args.dry_run}")

    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"Kept work dir: {work_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())