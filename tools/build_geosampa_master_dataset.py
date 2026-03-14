#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    rank: int


@dataclass
class ExportProcessStats:
    export: str
    zip_path: str
    format_variant: str
    default_items: int
    image_root: str
    images_resolved: int
    images_missing: int
    masks_found: int
    positives_generated: int
    empties_generated: int
    used_after_dedup: int = 0


@dataclass(frozen=True)
class SampleCandidate:
    sample_id: str
    normalized_name: str
    export_name: str
    export_rank: int
    default_entry: str
    image_path: Path
    image_ext: str
    source_image_root: str
    mask_path: Path | None
    label_lines: tuple[str, ...]

    @property
    def is_positive(self) -> bool:
        return len(self.label_lines) > 0


def natural_sort_key(name: str) -> tuple[int, str]:
    m = TASK_NUM_RE.search(name)
    if m:
        return int(m.group(1)), name.lower()
    if "jard" in name.lower():
        return 0, name.lower()
    return -1, name.lower()


def rank_from_name(name: str) -> int:
    m = TASK_NUM_RE.search(name)
    if m:
        return int(m.group(1))
    return 0


def slugify(text: str) -> str:
    s = text.lower().replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = s.strip("_")
    return s or "sample"


def md5_bucket(text: str, seed: int) -> int:
    key = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % 10000


def load_mask_to_polygons(repo_root: Path) -> Callable[[Path], list[list[float]]]:
    mod_path = repo_root / "tools" / "masks_to_yolo_seg.py"
    spec = importlib.util.spec_from_file_location("masks_to_yolo_seg", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load converter module: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "mask_to_polygons", None)
    if fn is None:
        raise RuntimeError(f"mask_to_polygons not found in: {mod_path}")
    return fn


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


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


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
    entries = [ln.strip().replace("\\", "/") for ln in default_txt.read_text(encoding="utf-8").splitlines()]
    return [e.strip("/").removesuffix(".png").removesuffix(".jpg").removesuffix(".jpeg") for e in entries if e.strip()]


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


def entry_aliases(identity: str, *, allow_rc_only_fallback: bool = False) -> list[str]:
    norm = Path(identity).with_suffix("").as_posix().strip("/")
    aliases: list[str] = []

    def add(v: str) -> None:
        if v and v not in aliases:
            aliases.append(v)

    add(norm)
    add(Path(norm).name)

    cell, rc = parse_cell_rc(norm)
    if cell and rc:
        add(f"{cell}__{rc}")
        add(f"{cell}/{rc}")

    rc_only = parse_rc_only(norm)
    if allow_rc_only_fallback and rc_only and not (cell and rc):
        add(rc_only)

    return aliases


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
    if any(k in joined for k in ("annotate_round2/", "large_masks/", "low_conf/", "hard_empty/", "many_preds/")):
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


def build_mask_lookup(mask_root: Path, *, allow_rc_only_fallback: bool = False) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    if not mask_root.exists():
        return lookup
    for p in sorted(mask_root.rglob("*.png"), key=lambda x: x.as_posix()):
        rel = p.relative_to(mask_root).with_suffix("").as_posix()
        for alias in entry_aliases(rel, allow_rc_only_fallback=allow_rc_only_fallback):
            lookup[alias].append(p)
    for k in list(lookup.keys()):
        lookup[k] = sorted(lookup[k], key=lambda x: x.as_posix())
    return lookup


def resolve_mask(
    mask_lookup: dict[str, list[Path]],
    identity: str,
    *,
    allow_rc_only_fallback: bool = False,
) -> Path | None:
    for alias in entry_aliases(identity, allow_rc_only_fallback=allow_rc_only_fallback):
        candidates = mask_lookup.get(alias, [])
        if candidates:
            return candidates[0]
    return None


def polygons_to_yolo_lines(mask_to_polygons_fn: Callable[[Path], list[list[float]]], mask_path: Path) -> tuple[str, ...]:
    polys = mask_to_polygons_fn(mask_path)
    lines: list[str] = []
    for poly in polys:
        if len(poly) < 6:
            continue
        coords = " ".join(f"{float(x):.6f}" for x in poly)
        lines.append(f"0 {coords}")
    lines.sort()
    return tuple(lines)


def classify_empty_mask(mask_path: Path) -> tuple[str, int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return "decode_fail", -1
    nonzero = int(np.count_nonzero(mask))
    if nonzero == 0:
        return "explicit_background", 0
    return "nonzero_empty_polygon", nonzero


def build_sample_id(identity: str, export_name: str, image_root: Path) -> str:
    cell, rc = parse_cell_rc(identity)
    if cell and rc:
        return f"{cell}__{rc}"

    rc_only = parse_rc_only(identity)
    source_key = slugify(image_root.name or export_name)
    if rc_only:
        return f"{source_key}__{rc_only}"

    full_norm = slugify(Path(identity).with_suffix("").as_posix().replace("/", "__"))
    return f"{source_key}__{full_norm}"


def choose_best_candidate(candidates: list[SampleCandidate]) -> SampleCandidate:
    def score(c: SampleCandidate) -> tuple[int, int, int, str, str]:
        return (
            1 if c.is_positive else 0,
            c.export_rank,
            len(c.label_lines),
            c.export_name,
            c.default_entry,
        )

    return max(candidates, key=score)


def verify_final_dataset(out_dir: Path) -> dict[str, int]:
    summary: dict[str, int] = {}
    for split in ("train", "val"):
        img_dir = out_dir / "images" / split
        lab_dir = out_dir / "labels" / split

        image_stems = sorted([p.stem for p in img_dir.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
        label_stems = sorted([p.stem for p in lab_dir.glob("*.txt") if p.is_file()])

        img_set = set(image_stems)
        lab_set = set(label_stems)
        if img_set != lab_set:
            only_images = sorted(list(img_set - lab_set))[:10]
            only_labels = sorted(list(lab_set - img_set))[:10]
            raise RuntimeError(
                f"Integrity mismatch for split={split}: "
                f"images_only={only_images}, labels_only={only_labels}"
            )

        if len(image_stems) != len(img_set):
            raise RuntimeError(f"Duplicate image stems found in split={split}")
        if len(label_stems) != len(lab_set):
            raise RuntimeError(f"Duplicate label stems found in split={split}")

        summary[f"{split}_image_count"] = len(image_stems)
        summary[f"{split}_label_count"] = len(label_stems)

    return summary


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
    ap = argparse.ArgumentParser(description="Build clean GeoSampa YOLO-seg master dataset from raw CVAT export zips.")
    ap.add_argument("--archive", type=Path, default=Path("~/Downloads/Archive.zip").expanduser())
    ap.add_argument("--out-dir", type=Path, default=Path("data/datasets/geosampa_master_v1"))
    ap.add_argument("--work-dir", type=Path, default=None, help="Optional persistent work dir. Defaults to temporary dir.")
    ap.add_argument("--keep-work-dir", action="store_true", help="Keep temporary work dir after completion.")
    ap.add_argument("--seed", type=int, default=20260314)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--train-empty-pos-ratio", type=float, default=1.0)

    ap.add_argument(
        "--allow-implicit-empties",
        action="store_true",
        help=(
            "Unsafe mode. Treat default.txt entries without matching masks, decode-fail masks, "
            "and nonzero masks with zero generated polygons as empty negatives."
        ),
    )
    ap.add_argument(
        "--allow-rc-only-mask-fallback",
        action="store_true",
        help=(
            "Allow bare rXXXX_cXXXX alias fallback during mask lookup. Disabled by default to avoid cross-cell collisions."
        ),
    )

    ap.add_argument(
        "--fail-on-suspicious",
        action="store_true",
        help="Fail build if any suspicious mask cases are found.",
    )
    ap.add_argument(
        "--fail-on-missing-mask",
        action="store_true",
        help="Fail build if any default.txt entries resolve to an image but not to a matching mask.",
    )
    ap.add_argument(
        "--fail-on-nonzero-empty",
        action="store_true",
        help="Fail build if any nonzero masks produce zero YOLO polygons.",
    )
    ap.add_argument(
        "--fail-on-decode-fail",
        action="store_true",
        help="Fail build if any mask exists but cannot be decoded.",
    )

    ap.add_argument(
        "--exclude-export-substring",
        action="append",
        default=[],
        help=(
            "Case-insensitive substring filter applied to discovered raw export zip path/name. "
            "Matching exports are excluded. Can be repeated."
        ),
    )
    ap.add_argument(
        "--include-export-substring",
        action="append",
        default=[],
        help=(
            "Case-insensitive substring filter applied to discovered raw export zip path/name. "
            "If provided, only exports matching at least one include substring are kept. Can be repeated."
        ),
    )
    ap.add_argument(
        "--suspicious-log-limit",
        type=int,
        default=200,
        help="Maximum number of suspicious sample examples stored in build_summary.json.",
    )
    ap.add_argument("--clean-out", action="store_true", default=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    archive = args.archive.expanduser().resolve()
    out_dir = args.out_dir.resolve()
    image_roots = default_image_roots(repo_root)

    if not archive.exists():
        raise SystemExit(f"Archive not found: {archive}")
    if not image_roots:
        raise SystemExit("No candidate image roots found in repository.")

    mask_to_polygons_fn = load_mask_to_polygons(repo_root)

    if args.work_dir is not None:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_work = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="geosampa_master_build_"))
        cleanup_work = not args.keep_work_dir

    extracted_archive_dir = work_dir / "archive_contents"
    nested_extract_dir = work_dir / "nested_exports"
    extracted_archive_dir.mkdir(parents=True, exist_ok=True)
    nested_extract_dir.mkdir(parents=True, exist_ok=True)

    extract_zip(archive, extracted_archive_dir)
    nested_zips = find_nested_zips(extracted_archive_dir)
    if not nested_zips:
        raise SystemExit(f"No nested CVAT zip exports found in: {archive}")

    discovered_exports: list[dict[str, str]] = []
    for zp in nested_zips:
        rel = zp.relative_to(extracted_archive_dir).as_posix()
        discovered_exports.append(
            {
                "name": zp.name,
                "archive_rel_path": rel,
                "zip_path": zp.as_posix(),
            }
        )

    include_substrings = [s.lower() for s in args.include_export_substring if s.strip()]
    exclude_substrings = [s.lower() for s in args.exclude_export_substring if s.strip()]

    filtered_zips: list[Path] = []
    excluded_exports: list[dict[str, str]] = []
    for zp in nested_zips:
        rel = zp.relative_to(extracted_archive_dir).as_posix()
        haystack = f"{rel} {zp.name}".lower()
        reasons: list[str] = []

        if include_substrings and not any(sub in haystack for sub in include_substrings):
            reasons.append(
                "did_not_match_include_substring:"
                + ",".join(args.include_export_substring)
            )

        matched_excludes = [sub for sub in exclude_substrings if sub in haystack]
        if matched_excludes:
            reasons.append(
                "matched_exclude_substring:" + ",".join(matched_excludes)
            )

        if reasons:
            excluded_exports.append(
                {
                    "name": zp.name,
                    "archive_rel_path": rel,
                    "zip_path": zp.as_posix(),
                    "reason": "; ".join(reasons),
                }
            )
            continue

        filtered_zips.append(zp)

    if not filtered_zips:
        raise SystemExit(
            "No exports left after include/exclude filtering. "
            f"found={len(nested_zips)} excluded={len(excluded_exports)}"
        )

    specs: list[ExportSpec] = []
    for idx, zp in enumerate(filtered_zips):
        name = zp.stem
        out = nested_extract_dir / f"{idx:03d}_{slugify(name)}"
        extract_zip(zp, out)
        specs.append(
            ExportSpec(
                zip_path=zp,
                extracted_dir=out,
                name=name,
                rank=rank_from_name(name),
            )
        )

    all_candidates: list[SampleCandidate] = []
    export_stats: list[ExportProcessStats] = []
    skipped_broken_exports: list[dict[str, str]] = []
    suspicious_missing_mask_by_export: Counter[str] = Counter()
    suspicious_decode_fail_by_export: Counter[str] = Counter()
    suspicious_nonzero_empty_by_export: Counter[str] = Counter()
    suspicious_examples: list[dict[str, str | int]] = []

    def log_suspicious(kind: str, spec_name: str, entry: str, image_root: Path, mask_path: Path | None, detail: str) -> None:
        if kind == "missing_mask":
            suspicious_missing_mask_by_export[spec_name] += 1
        elif kind == "decode_fail":
            suspicious_decode_fail_by_export[spec_name] += 1
        elif kind == "nonzero_empty_polygon":
            suspicious_nonzero_empty_by_export[spec_name] += 1

        if len(suspicious_examples) < args.suspicious_log_limit:
            suspicious_examples.append(
                {
                    "kind": kind,
                    "export": spec_name,
                    "entry": entry,
                    "image_root": image_root.as_posix(),
                    "mask_path": mask_path.as_posix() if mask_path is not None else "",
                    "detail": detail,
                }
            )

    strict_empty_mode = not args.allow_implicit_empties

    for spec in specs:
        default_txt = find_default_txt(spec.extracted_dir)
        if default_txt is None:
            skipped_broken_exports.append(
                {"export": spec.name, "zip_path": spec.zip_path.as_posix(), "reason": "missing_default_txt"}
            )
            continue

        export_base = default_txt.parents[2]
        segobj_dir = find_seg_dir(export_base, "SegmentationObject")
        segclass_dir = find_seg_dir(export_base, "SegmentationClass")

        if segobj_dir is None and segclass_dir is None:
            skipped_broken_exports.append(
                {"export": spec.name, "zip_path": spec.zip_path.as_posix(), "reason": "missing_segmentation_dirs"}
            )
            continue

        default_entries = read_default_entries(default_txt)
        if not default_entries:
            skipped_broken_exports.append(
                {"export": spec.name, "zip_path": spec.zip_path.as_posix(), "reason": "empty_default_txt"}
            )
            continue

        format_variant = detect_format_variant(default_entries)
        image_root, _coverage = choose_image_root(spec.name, format_variant, default_entries, image_roots)
        if image_root is None:
            skipped_broken_exports.append(
                {"export": spec.name, "zip_path": spec.zip_path.as_posix(), "reason": "no_image_root_resolved"}
            )
            continue

        mask_root = segobj_dir if segobj_dir is not None else segclass_dir
        if mask_root is None:
            skipped_broken_exports.append(
                {"export": spec.name, "zip_path": spec.zip_path.as_posix(), "reason": "no_mask_root"}
            )
            continue

        mask_lookup = build_mask_lookup(
            mask_root,
            allow_rc_only_fallback=args.allow_rc_only_mask_fallback,
        )

        images_resolved = 0
        images_missing = 0
        masks_found = 0
        positives = 0
        empties = 0

        for entry in default_entries:
            img_path = resolve_image_path(image_root, entry)
            if img_path is None:
                images_missing += 1
                continue

            mask_path = resolve_mask(
                mask_lookup,
                entry,
                allow_rc_only_fallback=args.allow_rc_only_mask_fallback,
            )

            label_lines: tuple[str, ...] = tuple()

            if mask_path is None:
                log_suspicious(
                    kind="missing_mask",
                    spec_name=spec.name,
                    entry=entry,
                    image_root=image_root,
                    mask_path=None,
                    detail="No matching mask for default.txt entry.",
                )
                if strict_empty_mode:
                    continue
            else:
                masks_found += 1
                label_lines = polygons_to_yolo_lines(mask_to_polygons_fn, mask_path)
                if not label_lines:
                    empty_kind, nonzero = classify_empty_mask(mask_path)
                    if empty_kind == "decode_fail":
                        log_suspicious(
                            kind="decode_fail",
                            spec_name=spec.name,
                            entry=entry,
                            image_root=image_root,
                            mask_path=mask_path,
                            detail="Mask exists but failed to decode.",
                        )
                        if strict_empty_mode:
                            continue
                    elif empty_kind == "nonzero_empty_polygon":
                        log_suspicious(
                            kind="nonzero_empty_polygon",
                            spec_name=spec.name,
                            entry=entry,
                            image_root=image_root,
                            mask_path=mask_path,
                            detail=f"Mask has nonzero pixels ({nonzero}) but produced no YOLO polygons.",
                        )
                        if strict_empty_mode:
                            continue

            sample_id = build_sample_id(entry, spec.name, image_root)
            normalized_name = slugify(sample_id)
            cand = SampleCandidate(
                sample_id=sample_id,
                normalized_name=normalized_name,
                export_name=spec.name,
                export_rank=spec.rank,
                default_entry=entry,
                image_path=img_path,
                image_ext=img_path.suffix.lower(),
                source_image_root=image_root.as_posix(),
                mask_path=mask_path,
                label_lines=label_lines,
            )
            all_candidates.append(cand)
            images_resolved += 1
            if cand.is_positive:
                positives += 1
            else:
                empties += 1

        export_stats.append(
            ExportProcessStats(
                export=spec.name,
                zip_path=spec.zip_path.as_posix(),
                format_variant=format_variant,
                default_items=len(default_entries),
                image_root=image_root.as_posix(),
                images_resolved=images_resolved,
                images_missing=images_missing,
                masks_found=masks_found,
                positives_generated=positives,
                empties_generated=empties,
            )
        )

    if not all_candidates:
        raise SystemExit("No samples were generated from the discovered exports.")

    missing_mask_total = int(sum(suspicious_missing_mask_by_export.values()))
    decode_fail_total = int(sum(suspicious_decode_fail_by_export.values()))
    nonzero_empty_total = int(sum(suspicious_nonzero_empty_by_export.values()))
    suspicious_total = missing_mask_total + decode_fail_total + nonzero_empty_total

    fail_reasons: list[str] = []
    if args.fail_on_suspicious and suspicious_total > 0:
        fail_reasons.append(f"suspicious_total={suspicious_total}")
    if args.fail_on_missing_mask and missing_mask_total > 0:
        fail_reasons.append(f"missing_mask_total={missing_mask_total}")
    if args.fail_on_decode_fail and decode_fail_total > 0:
        fail_reasons.append(f"decode_fail_total={decode_fail_total}")
    if args.fail_on_nonzero_empty and nonzero_empty_total > 0:
        fail_reasons.append(f"nonzero_empty_total={nonzero_empty_total}")

    if fail_reasons:
        summary = {
            "seed": args.seed,
            "source_archive": archive.as_posix(),
            "work_dir": work_dir.as_posix(),
            "failure": {
                "reason": "suspicious_mask_audit_failed",
                "details": fail_reasons,
            },
            "suspicious_mask_audit": {
                "strict_empty_mode_enabled": strict_empty_mode,
                "allow_implicit_empties": args.allow_implicit_empties,
                "allow_rc_only_mask_fallback": args.allow_rc_only_mask_fallback,
                "missing_mask_entries_total": missing_mask_total,
                "decode_fail_entries_total": decode_fail_total,
                "nonzero_empty_polygon_entries_total": nonzero_empty_total,
                "missing_mask_entries_by_export": dict(sorted(suspicious_missing_mask_by_export.items())),
                "decode_fail_entries_by_export": dict(sorted(suspicious_decode_fail_by_export.items())),
                "nonzero_empty_polygon_entries_by_export": dict(sorted(suspicious_nonzero_empty_by_export.items())),
                "examples": suspicious_examples,
            },
            "export_stats": [st.__dict__ for st in export_stats],
            "skipped_broken_exports": skipped_broken_exports,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "build_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        raise SystemExit(
            "Build failed due to suspicious mask audit thresholds: "
            + ", ".join(fail_reasons)
            + f". Summary: {summary_path}"
        )

    grouped: dict[str, list[SampleCandidate]] = defaultdict(list)
    for c in all_candidates:
        grouped[c.sample_id].append(c)

    deduped_samples: list[SampleCandidate] = []
    duplicates_removed = 0
    for sample_id in sorted(grouped.keys()):
        group = grouped[sample_id]
        best = choose_best_candidate(group)
        deduped_samples.append(best)
        duplicates_removed += max(0, len(group) - 1)

    used_counts: dict[str, int] = defaultdict(int)
    for s in deduped_samples:
        used_counts[s.export_name] += 1
    for st in export_stats:
        st.used_after_dedup = used_counts.get(st.export, 0)

    positives = sorted([s for s in deduped_samples if s.is_positive], key=lambda x: x.normalized_name)
    empties = sorted([s for s in deduped_samples if not s.is_positive], key=lambda x: x.normalized_name)

    def split_name(sample: SampleCandidate) -> str:
        return "val" if md5_bucket(sample.sample_id, args.seed) < int(args.val_fraction * 10000) else "train"

    train_pos = [s for s in positives if split_name(s) == "train"]
    val_pos = [s for s in positives if split_name(s) == "val"]
    if len(positives) > 1:
        if not val_pos:
            move = min(train_pos, key=lambda s: md5_bucket(s.sample_id, args.seed))
            train_pos.remove(move)
            val_pos.append(move)
        if not train_pos:
            move = min(val_pos, key=lambda s: md5_bucket(s.sample_id, args.seed))
            val_pos.remove(move)
            train_pos.append(move)

    train_empty_all = [s for s in empties if split_name(s) == "train"]
    val_empty = [s for s in empties if split_name(s) == "val"]

    if len(positives) == 0:
        train_empty_target = len(train_empty_all)
    else:
        train_empty_target = int(round(len(train_pos) * args.train_empty_pos_ratio))
    train_empty_target = max(0, min(train_empty_target, len(train_empty_all)))

    train_empty_sorted = sorted(train_empty_all, key=lambda s: (md5_bucket(s.sample_id, args.seed + 17), s.sample_id))
    train_empty = train_empty_sorted[:train_empty_target]
    dropped_train_empties = len(train_empty_all) - len(train_empty)

    split_map: dict[str, str] = {}
    for s in train_pos + train_empty:
        split_map[s.sample_id] = "train"
    for s in val_pos + val_empty:
        split_map[s.sample_id] = "val"

    selected = sorted([s for s in deduped_samples if s.sample_id in split_map], key=lambda x: x.normalized_name)

    if args.clean_out and out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    for s in selected:
        if s.sample_id in seen_ids:
            raise RuntimeError(f"Duplicate normalized sample id in final selection: {s.sample_id}")
        seen_ids.add(s.sample_id)

        split = split_map[s.sample_id]
        img_dst = out_dir / "images" / split / f"{s.normalized_name}{s.image_ext}"
        lab_dst = out_dir / "labels" / split / f"{s.normalized_name}.txt"
        shutil.copy2(s.image_path, img_dst)
        if s.label_lines:
            lab_dst.write_text("\n".join(s.label_lines) + "\n", encoding="utf-8")
        else:
            lab_dst.write_text("", encoding="utf-8")

    count_summary = verify_final_dataset(out_dir)

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.as_posix()}",
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

    source_exports_found = discovered_exports
    source_exports_used = [
        {
            "name": st.export,
            "zip_path": st.zip_path,
            "used_after_dedup": st.used_after_dedup,
        }
        for st in export_stats
        if st.used_after_dedup > 0
    ]

    summary = {
        "seed": args.seed,
        "source_archive": archive.as_posix(),
        "work_dir": work_dir.as_posix(),
        "source_exports_found": source_exports_found,
        "source_exports_used": source_exports_used,
        "source_exports_excluded": excluded_exports,
        "positives_kept": len(train_pos) + len(val_pos),
        "empties_kept": len(train_empty) + len(val_empty),
        "empties_dropped": dropped_train_empties,
        "duplicates_removed": duplicates_removed,
        "train_image_count": count_summary["train_image_count"],
        "train_label_count": count_summary["train_label_count"],
        "val_image_count": count_summary["val_image_count"],
        "val_label_count": count_summary["val_label_count"],
        "skipped_broken_exports": skipped_broken_exports,
        "split_policy": {
            "preserve_export_split": False,
            "reason": "No train/val files were present in raw CVAT exports (only default.txt).",
            "deterministic_val_fraction": args.val_fraction,
            "train_empty_positive_ratio_target": args.train_empty_pos_ratio,
        },
        "suspicious_mask_audit": {
            "strict_empty_mode_enabled": strict_empty_mode,
            "allow_implicit_empties": args.allow_implicit_empties,
            "allow_rc_only_mask_fallback": args.allow_rc_only_mask_fallback,
            "missing_mask_entries_total": missing_mask_total,
            "decode_fail_entries_total": decode_fail_total,
            "nonzero_empty_polygon_entries_total": nonzero_empty_total,
            "missing_mask_entries_by_export": dict(sorted(suspicious_missing_mask_by_export.items())),
            "decode_fail_entries_by_export": dict(sorted(suspicious_decode_fail_by_export.items())),
            "nonzero_empty_polygon_entries_by_export": dict(sorted(suspicious_nonzero_empty_by_export.items())),
            "examples": suspicious_examples,
        },
        "export_stats": [st.__dict__ for st in export_stats],
    }

    summary_path = out_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Built dataset at: {out_dir}")
    print(f"Nested export zips found: {len(source_exports_found)}")
    print(f"Exports excluded by filter: {len(excluded_exports)}")
    print(f"Exports used: {len(source_exports_used)}")
    print(f"Strict empty mode enabled: {strict_empty_mode}")
    print(f"Allow RC-only mask fallback: {args.allow_rc_only_mask_fallback}")
    print(f"Suspicious missing-mask entries: {missing_mask_total}")
    print(f"Suspicious decode-fail entries: {decode_fail_total}")
    print(f"Suspicious nonzero-empty entries: {nonzero_empty_total}")
    print(f"Positives kept: {summary['positives_kept']}")
    print(f"Empties kept: {summary['empties_kept']}")
    print(f"Empties dropped: {summary['empties_dropped']}")
    print(f"Duplicates removed: {summary['duplicates_removed']}")
    print(
        f"Train images/labels: {summary['train_image_count']}/{summary['train_label_count']} | "
        f"Val images/labels: {summary['val_image_count']}/{summary['val_label_count']}"
    )
    print(f"Summary: {summary_path}")
    print(f"Dataset YAML: {dataset_yaml}")

    if cleanup_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"Kept work dir: {work_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())