#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
CELL_RE = re.compile(r"^cell_\d{4}_\d{4}$")
RC_RE = re.compile(r"^r\d{4}_c\d{4}$")
TASK_NUM_RE = re.compile(r"task_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ExportSpec:
    zip_path: Path
    extracted_dir: Path
    name: str


def slugify(text: str) -> str:
    s = text.lower().replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = s.strip("_")
    return s or "sample"


def natural_sort_key(name: str) -> tuple[int, str]:
    m = TASK_NUM_RE.search(name)
    if m:
        return int(m.group(1)), name.lower()
    if "jard" in name.lower():
        return 0, name.lower()
    return -1, name.lower()


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def find_nested_zips(extract_root: Path) -> list[Path]:
    zips: list[Path] = []
    for p in extract_root.rglob("*.zip"):
        parts = set(p.parts)
        if "__MACOSX" in parts:
            continue
        if p.name.startswith("._"):
            continue
        zips.append(p)
    return sorted(zips, key=lambda x: natural_sort_key(x.name))


def find_default_txt(export_dir: Path) -> Path | None:
    matches = sorted(export_dir.rglob("ImageSets/Segmentation/default.txt"))
    if not matches:
        return None
    return matches[0]


def find_seg_dir(export_base: Path, name: str) -> Path | None:
    direct = export_base / name
    if direct.exists() and direct.is_dir():
        return direct
    matches = sorted([p for p in export_base.rglob(name) if p.is_dir()])
    return matches[0] if matches else None


def read_default_entries(default_txt: Path) -> list[str]:
    entries = [
        ln.strip().replace("\\", "/")
        for ln in default_txt.read_text(encoding="utf-8").splitlines()
    ]
    return [
        e.strip("/").removesuffix(".png").removesuffix(".jpg").removesuffix(".jpeg")
        for e in entries
        if e.strip()
    ]


def parse_cell_rc(identity: str) -> tuple[str | None, str | None]:
    p = Path(identity)
    parts = [x for x in p.with_suffix("").parts if x not in ("", ".", "..")]

    stem = p.stem if p.suffix else p.name
    if "__" in stem:
        left, right = stem.split("__", 1)
        if CELL_RE.match(left) and RC_RE.match(right):
            return left, right

    cell = None
    rc = None
    for token in parts:
        if CELL_RE.match(token):
            cell = token
        if RC_RE.match(token):
            rc = token

    return cell, rc


def parse_rc_only(identity: str) -> str | None:
    stem = Path(identity).stem if Path(identity).suffix else Path(identity).name
    return stem if RC_RE.match(stem) else None


def candidate_relative_image_paths(identity: str) -> list[str]:
    norm = Path(identity).with_suffix("").as_posix().strip("/")
    candidates: list[str] = []

    def add(v: str) -> None:
        if v and v not in candidates:
            candidates.append(v)

    add(norm)
    add(Path(norm).name)

    cell, rc = parse_cell_rc(norm)
    if cell and rc:
        add(f"{cell}/{rc}")
        add(f"{cell}__{rc}")
        prefix = "/".join(Path(norm).parts[:-2])
        if prefix:
            add(f"{prefix}/{cell}/{rc}")

    rc_only = parse_rc_only(norm)
    if rc_only and not (cell and rc):
        add(rc_only)

    return candidates


def resolve_image_path(image_root: Path, identity: str) -> Path | None:
    for rel in candidate_relative_image_paths(identity):
        for ext in IMG_EXTS:
            cand = image_root / f"{rel}{ext}"
            if cand.exists():
                return cand
    return None


def detect_format_variant(default_entries: list[str]) -> str:
    joined = " ".join(default_entries[:200]).lower()
    if any(
        k in joined
        for k in (
            "annotate_round2/",
            "large_masks/",
            "low_conf/",
            "hard_empty/",
            "many_preds/",
            "random/",
        )
    ):
        return "nested_round2"
    if any("cell_" in e for e in default_entries):
        return "cell_aware"
    return "flat"


def image_root_hint_score(export_name: str, format_variant: str, image_root: Path) -> int:
    name = export_name.lower()
    root = image_root.as_posix().lower()
    score = 0

    if "jard" in name and "/jardins_2020" in root and "infer" not in root:
        score += 100
    if "task_5" in name and "/moema_2020" in root:
        score += 100
    if "task_3" in name and "/pinheiros_2020" in root:
        score += 100

    if format_variant == "nested_round2":
        if "/runs/annotate_round2" in root:
            score += 80
        if "/sp_city_2020_rebuild_official" in root:
            score += 60
    elif format_variant == "cell_aware":
        if "/sp_city_2020_rebuild_official" in root:
            score += 80
        if "/runs/annotate_round2" in root:
            score += 40

    return score


def choose_image_root(
    export_name: str,
    format_variant: str,
    default_entries: list[str],
    image_roots: list[Path],
) -> tuple[Path | None, dict[str, int]]:
    coverage: dict[str, int] = {}
    ranked: list[tuple[int, int, int, str, Path]] = []

    for idx, root in enumerate(image_roots):
        if not root.exists():
            continue
        found = 0
        for entry in default_entries:
            if resolve_image_path(root, entry) is not None:
                found += 1
        coverage[root.as_posix()] = found
        hint = image_root_hint_score(export_name, format_variant, root)
        ranked.append((found, hint, -idx, root.as_posix(), root))

    if not ranked:
        return None, coverage

    ranked.sort(reverse=True)
    best = ranked[0][-1]
    if ranked[0][0] == 0:
        return None, coverage
    return best, coverage


def build_mask_lookup(mask_root: Path, recursive: bool = True) -> set[str]:
    rels: set[str] = set()
    paths = mask_root.rglob("*.png") if recursive else mask_root.glob("*.png")
    for p in paths:
        rels.add(p.relative_to(mask_root).with_suffix("").as_posix())
    return rels


def resolve_mask_path(mask_root: Path, identity: str) -> Path | None:
    norm = Path(identity).with_suffix("").as_posix().strip("/")
    candidates: list[str] = []

    def add(v: str) -> None:
        if v and v not in candidates:
            candidates.append(v)

    add(norm)
    add(Path(norm).name)

    cell, rc = parse_cell_rc(norm)
    if cell and rc:
        add(f"{cell}__{rc}")
        add(f"{cell}/{rc}")

    rc_only = parse_rc_only(norm)
    if rc_only and not (cell and rc):
        add(rc_only)

    for rel in candidates:
        cand = mask_root / f"{rel}.png"
        if cand.exists():
            return cand

    matches = sorted(mask_root.rglob(f"{Path(norm).name}.png"))
    return matches[0] if matches else None


def default_image_roots(repo_root: Path) -> list[Path]:
    candidates = [
        repo_root / "runs" / "annotate_round2",
        repo_root / "data" / "raw" / "geosampa_ortho" / "sp_city_2020_rebuild_official",
        repo_root / "data" / "raw" / "geosampa_ortho" / "moema_2020",
        repo_root / "data" / "raw" / "geosampa_ortho" / "pinheiros_2020",
        repo_root / "data" / "raw" / "geosampa_ortho" / "jardins_2020",
        repo_root / "data" / "raw" / "geosampa_ortho" / "jardins_2020_infer",
        repo_root / "data" / "raw" / "geosampa_ortho" / "brooklin_2020",
        repo_root / "data" / "raw" / "geosampa_ortho" / "vila_olimpia_2020",
        repo_root / "data" / "raw" / "geosampa_ortho" / "itaim_bibi_2020",
    ]
    return [p for p in candidates if p.exists()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create explicit black masks for reviewed-empty frames inside nested GeoSampa CVAT export zips."
    )
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional persistent extraction directory. Defaults to a temporary directory.",
    )
    ap.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep temporary extraction directory after completion.",
    )
    ap.add_argument(
        "--exclude-export-substring",
        action="append",
        default=[],
        help="Case-insensitive substring filter. Matching exports are skipped.",
    )
    ap.add_argument(
        "--include-export-substring",
        action="append",
        default=[],
        help="If provided, only exports matching at least one case-insensitive substring are processed.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be created without writing masks.",
    )
    ap.add_argument(
        "--overwrite-existing-black",
        action="store_true",
        help="Overwrite existing all-zero masks if they already exist.",
    )
    return ap.parse_args()


def is_existing_mask_all_zero(mask_path: Path) -> bool:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return False
    return int(np.count_nonzero(mask)) == 0


def write_black_mask(mask_path: Path, shape_hw: tuple[int, int], dry_run: bool) -> None:
    h, w = shape_hw
    if not dry_run:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        blank = np.zeros((h, w), dtype=np.uint8)
        ok = cv2.imwrite(str(mask_path), blank)
        if not ok:
            raise RuntimeError(f"Failed to write mask: {mask_path}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    archive = args.archive.expanduser().resolve()
    image_roots = default_image_roots(repo_root)

    if not archive.exists():
        raise SystemExit(f"Archive not found: {archive}")
    if not image_roots:
        raise SystemExit("No candidate image roots found in repository.")

    if args.work_dir is not None:
        work_dir = args.work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="geosampa_black_masks_"))
        cleanup_work = not args.keep_work_dir

    extracted_archive_dir = work_dir / "archive_contents"
    nested_extract_dir = work_dir / "nested_exports"
    extracted_archive_dir.mkdir(parents=True, exist_ok=True)
    nested_extract_dir.mkdir(parents=True, exist_ok=True)

    extract_zip(archive, extracted_archive_dir)
    nested_zips = find_nested_zips(extracted_archive_dir)
    if not nested_zips:
        raise SystemExit(f"No nested CVAT zip exports found in: {archive}")

    include_substrings = [s.lower() for s in args.include_export_substring if s.strip()]
    exclude_substrings = [s.lower() for s in args.exclude_export_substring if s.strip()]

    filtered_zips: list[Path] = []
    excluded_exports: list[tuple[str, str]] = []
    for zp in nested_zips:
        rel = zp.relative_to(extracted_archive_dir).as_posix()
        haystack = f"{rel} {zp.name}".lower()
        excluded = False

        if include_substrings and not any(sub in haystack for sub in include_substrings):
            excluded_exports.append((zp.name, "did_not_match_include_filter"))
            excluded = True

        if any(sub in haystack for sub in exclude_substrings):
            excluded_exports.append((zp.name, "matched_exclude_filter"))
            excluded = True

        if not excluded:
            filtered_zips.append(zp)

    specs: list[ExportSpec] = []
    for idx, zp in enumerate(filtered_zips):
        name = zp.stem
        out = nested_extract_dir / f"{idx:03d}_{slugify(name)}"
        extract_zip(zp, out)
        specs.append(ExportSpec(zip_path=zp, extracted_dir=out, name=name))

    total_exports = 0
    total_default_entries = 0
    total_images_resolved = 0
    total_masks_present = 0
    total_masks_created = 0
    total_missing_images = 0
    total_skipped_no_default = 0
    total_skipped_no_mask_root = 0
    per_export_stats: list[dict[str, object]] = []

    for spec in specs:
        total_exports += 1
        default_txt = find_default_txt(spec.extracted_dir)
        if default_txt is None:
            total_skipped_no_default += 1
            per_export_stats.append(
                {
                    "export": spec.name,
                    "status": "skipped_missing_default_txt",
                    "zip_path": spec.zip_path.as_posix(),
                }
            )
            continue

        export_base = default_txt.parents[2]
        segobj_dir = find_seg_dir(export_base, "SegmentationObject")
        segclass_dir = find_seg_dir(export_base, "SegmentationClass")
        mask_root = segobj_dir if segobj_dir is not None else segclass_dir

        if mask_root is None:
            total_skipped_no_mask_root += 1
            per_export_stats.append(
                {
                    "export": spec.name,
                    "status": "skipped_missing_mask_root",
                    "zip_path": spec.zip_path.as_posix(),
                }
            )
            continue

        default_entries = read_default_entries(default_txt)
        format_variant = detect_format_variant(default_entries)
        image_root, coverage = choose_image_root(
            spec.name,
            format_variant,
            default_entries,
            image_roots,
        )

        if image_root is None:
            per_export_stats.append(
                {
                    "export": spec.name,
                    "status": "skipped_no_image_root",
                    "zip_path": spec.zip_path.as_posix(),
                    "default_items": len(default_entries),
                    "coverage": coverage,
                }
            )
            continue

        export_default_entries = 0
        export_images_resolved = 0
        export_masks_present = 0
        export_masks_created = 0
        export_missing_images = 0
        export_existing_black = 0

        for entry in default_entries:
            export_default_entries += 1
            total_default_entries += 1

            img_path = resolve_image_path(image_root, entry)
            if img_path is None:
                export_missing_images += 1
                total_missing_images += 1
                continue

            export_images_resolved += 1
            total_images_resolved += 1

            mask_path = resolve_mask_path(mask_root, entry)
            if mask_path is not None and mask_path.exists():
                if args.overwrite_existing_black and is_existing_mask_all_zero(mask_path):
                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        export_missing_images += 1
                        total_missing_images += 1
                        continue
                    h, w = img.shape[:2]
                    write_black_mask(mask_path, (h, w), args.dry_run)
                    export_existing_black += 1
                else:
                    export_masks_present += 1
                    total_masks_present += 1
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                export_missing_images += 1
                total_missing_images += 1
                continue

            h, w = img.shape[:2]

            rel_candidates = candidate_relative_image_paths(entry)
            rel_stem = Path(rel_candidates[0])
            if format_variant == "nested_round2":
                rel_stem = Path(entry).with_suffix("")
            target_mask_path = mask_root / rel_stem.with_suffix(".png")

            write_black_mask(target_mask_path, (h, w), args.dry_run)
            export_masks_created += 1
            total_masks_created += 1

        per_export_stats.append(
            {
                "export": spec.name,
                "status": "processed",
                "zip_path": spec.zip_path.as_posix(),
                "image_root": image_root.as_posix(),
                "mask_root": mask_root.as_posix(),
                "default_items": export_default_entries,
                "images_resolved": export_images_resolved,
                "masks_already_present": export_masks_present,
                "black_masks_created": export_masks_created,
                "existing_black_overwritten": export_existing_black,
                "missing_images": export_missing_images,
                "format_variant": format_variant,
            }
        )

    print(f"Archive: {archive}")
    print(f"Exports discovered: {len(nested_zips)}")
    print(f"Exports processed: {len(specs)}")
    print(f"Exports excluded: {len(excluded_exports)}")
    print(f"Default entries scanned: {total_default_entries}")
    print(f"Images resolved: {total_images_resolved}")
    print(f"Masks already present: {total_masks_present}")
    print(f"Black masks created: {total_masks_created}")
    print(f"Missing images: {total_missing_images}")
    print(f"Skipped missing default.txt: {total_skipped_no_default}")
    print(f"Skipped missing mask root: {total_skipped_no_mask_root}")
    print(f"Dry run: {args.dry_run}")

    print("\nPer-export stats:")
    for row in per_export_stats:
        print(row)

    if cleanup_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"\nKept work dir: {work_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())