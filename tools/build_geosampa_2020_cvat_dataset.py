#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


@dataclass
class ExportStats:
    export_name: str
    image_source: str
    default_items: int = 0
    copied_images: int = 0
    missing_images: int = 0
    segclass_masks_present: int = 0
    segclass_masks_valid: int = 0
    segclass_masks_invalid: int = 0
    xml_images_present: int = 0
    labels_positive: int = 0
    labels_empty: int = 0


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_labelmap(path: Path) -> Dict[str, Tuple[int, int, int]]:
    mapping: Dict[str, Tuple[int, int, int]] = {}
    if not path.exists():
        return mapping
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        label = parts[0].strip()
        rgb_txt = parts[1].strip()
        if not label or not rgb_txt:
            continue
        rgb_parts = [p.strip() for p in rgb_txt.split(",")]
        if len(rgb_parts) != 3:
            continue
        try:
            rgb = (int(rgb_parts[0]), int(rgb_parts[1]), int(rgb_parts[2]))
        except ValueError:
            continue
        mapping[label] = rgb
    return mapping


def resolve_image_path(images_dir: Path, stem: str) -> Optional[Path]:
    # Citywide batch format: cell_XXXX_YYYY__r####_c####
    if stem.startswith("cell_") and "__" in stem:
        cell, tile_stem = stem.split("__", 1)
        for ext in IMG_EXTS:
            cand = images_dir / cell / f"{tile_stem}{ext}"
            if cand.exists():
                return cand
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def deterministic_split(stem: str, val_frac: float, split_seed: str) -> str:
    key = f"{split_seed}:{stem}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    bucket = h % 10000
    return "val" if bucket < int(val_frac * 10000) else "train"


def parse_cvat_xml_polygons(xml_path: Path, pool_label: str) -> Dict[str, List[List[Tuple[float, float]]]]:
    out: Dict[str, List[List[Tuple[float, float]]]] = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image_node in root.findall(".//image"):
        name = image_node.attrib.get("name", "").strip()
        if not name:
            continue
        stem = Path(name).stem
        polys: List[List[Tuple[float, float]]] = []

        for poly in image_node.findall("polygon"):
            if poly.attrib.get("label", "") != pool_label:
                continue
            points_txt = poly.attrib.get("points", "").strip()
            if not points_txt:
                continue
            pts: List[Tuple[float, float]] = []
            for pair in points_txt.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                xy = pair.split(",")
                if len(xy) != 2:
                    continue
                try:
                    x = float(xy[0])
                    y = float(xy[1])
                except ValueError:
                    continue
                pts.append((x, y))
            if len(pts) >= 3:
                polys.append(pts)

        if polys:
            out[stem] = polys
    return out


def polygon_to_yolo_line(poly: List[Tuple[float, float]], w: int, h: int) -> Optional[str]:
    if len(poly) < 3:
        return None
    coords: List[str] = []
    for x, y in poly:
        xn = min(1.0, max(0.0, x / float(w)))
        yn = min(1.0, max(0.0, y / float(h)))
        coords.append(f"{xn:.6f}")
        coords.append(f"{yn:.6f}")
    return "0 " + " ".join(coords)


def mask_rgb_to_yolo_lines(mask_rgb: np.ndarray, pool_rgb: Tuple[int, int, int], w: int, h: int, min_area_px: int) -> List[str]:
    pool_arr = np.array(pool_rgb, dtype=np.uint8).reshape(1, 1, 3)
    binary = np.all(mask_rgb == pool_arr, axis=-1).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[str] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        poly = contour.reshape(-1, 2)
        area = cv2.contourArea(poly.astype(np.int32))
        if area < min_area_px:
            continue
        coords: List[str] = []
        for x, y in poly:
            xn = min(1.0, max(0.0, float(x) / float(w)))
            yn = min(1.0, max(0.0, float(y) / float(h)))
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")
        out.append("0 " + " ".join(coords))
    return out


def ensure_clean_dir(path: Path, clean: bool) -> None:
    if path.exists() and clean:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_base_dataset(
    base_dataset: Path,
    out_img_train: Path,
    out_img_val: Path,
    out_lab_train: Path,
    out_lab_val: Path,
    used_out_stems: Set[str],
) -> Dict[str, int]:
    stats = {
        "images": 0,
        "labels": 0,
        "positive_labels": 0,
        "empty_labels": 0,
    }
    for split in ("train", "val"):
        src_img_dir = base_dataset / "images" / split
        src_lab_dir = base_dataset / "labels" / split
        dst_img_dir = out_img_val if split == "val" else out_img_train
        dst_lab_dir = out_lab_val if split == "val" else out_lab_train
        if not src_img_dir.exists():
            continue
        for src_img in sorted(src_img_dir.glob("*")):
            if not src_img.is_file() or src_img.suffix.lower() not in IMG_EXTS:
                continue
            stem = src_img.stem
            used_out_stems.add(stem)
            dst_img = dst_img_dir / src_img.name
            shutil.copy2(src_img, dst_img)

            src_lab = src_lab_dir / f"{stem}.txt"
            dst_lab = dst_lab_dir / f"{stem}.txt"
            if src_lab.exists():
                shutil.copy2(src_lab, dst_lab)
            else:
                dst_lab.write_text("", encoding="utf-8")

            stats["images"] += 1
            stats["labels"] += 1
            if dst_lab.stat().st_size > 0:
                stats["positive_labels"] += 1
            else:
                stats["empty_labels"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build clean YOLO-seg dataset from GeoSampa 2020 CVAT exports.")
    ap.add_argument("--exports-root", type=Path, default=Path("data/cvat_exports_2020_fresh"))
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/geosampa_ortho"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/datasets/geosampa_2020_cvat_all"))
    ap.add_argument("--base-dataset", type=Path, default=None, help="Optional existing dataset to copy first.")
    ap.add_argument("--include-exports", type=str, default="", help="Optional comma-separated export folder names to include.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=str, default="20260310")
    ap.add_argument("--pool-label", type=str, default="pool")
    ap.add_argument("--min-area-px", type=int, default=100)
    ap.add_argument("--clean", action="store_true", default=True)
    ap.add_argument("--strict", action="store_true", default=False)
    args = ap.parse_args()

    image_sources = {
        normalize_key("Moema"): args.raw_root / "moema_2020",
        normalize_key("Pinheiros"): args.raw_root / "pinheiros_2020",
        normalize_key("Jardins"): args.raw_root / "jardins_2020",
        normalize_key("Brooklin"): args.raw_root / "brooklin_2020",
        normalize_key("Vila_Olimpia"): args.raw_root / "vila_olimpia_2020",
        normalize_key("batch_2000_600"): args.raw_root / "sp_city_2020_rebuild_official",
    }

    if args.out_dir.exists() and args.clean:
        shutil.rmtree(args.out_dir)

    out_img_train = args.out_dir / "images" / "train"
    out_img_val = args.out_dir / "images" / "val"
    out_lab_train = args.out_dir / "labels" / "train"
    out_lab_val = args.out_dir / "labels" / "val"

    ensure_clean_dir(out_img_train, False)
    ensure_clean_dir(out_img_val, False)
    ensure_clean_dir(out_lab_train, False)
    ensure_clean_dir(out_lab_val, False)

    exports = sorted([p for p in args.exports_root.glob("*") if p.is_dir()])
    if args.include_exports.strip():
        include_keys = {normalize_key(x.strip()) for x in args.include_exports.split(",") if x.strip()}
        exports = [p for p in exports if normalize_key(p.name) in include_keys]
    if not exports:
        raise SystemExit(f"No export folders found under {args.exports_root}")

    invalid_masks: List[Dict[str, str]] = []
    report: Dict[str, object] = {"exports": {}, "totals": {}}
    used_out_stems: set[str] = set()

    total_images = 0
    total_labels = 0
    total_positive = 0
    total_empty = 0
    total_missing_images = 0

    base_stats = None
    if args.base_dataset is not None:
        if not args.base_dataset.exists():
            raise SystemExit(f"Base dataset not found: {args.base_dataset}")
        base_stats = copy_base_dataset(
            base_dataset=args.base_dataset,
            out_img_train=out_img_train,
            out_img_val=out_img_val,
            out_lab_train=out_lab_train,
            out_lab_val=out_lab_val,
            used_out_stems=used_out_stems,
        )
        total_images += base_stats["images"]
        total_labels += base_stats["labels"]
        total_positive += base_stats["positive_labels"]
        total_empty += base_stats["empty_labels"]
        report["base_dataset"] = {
            "path": str(args.base_dataset),
            **base_stats,
        }

    for export_dir in exports:
        export_key = normalize_key(export_dir.name)

        default_txt = export_dir / "ImageSets" / "Segmentation" / "default.txt"
        segclass_dir = export_dir / "SegmentationClass"
        labelmap_path = export_dir / "labelmap.txt"
        xml_paths = sorted(export_dir.glob("*.xml"))

        if not default_txt.exists():
            msg = f"Missing default.txt in {export_dir}"
            print(msg)
            if args.strict:
                raise SystemExit(msg)
            continue

        image_dir = image_sources.get(export_key)
        if image_dir is None:
            msg = f"Skipping export with unknown image source mapping: {export_dir.name}"
            print(msg)
            if args.strict:
                raise SystemExit(msg)
            continue
        if not image_dir.exists():
            msg = f"Image source not found for {export_dir.name}: {image_dir}"
            print(msg)
            if args.strict:
                raise SystemExit(msg)
            continue

        labelmap = parse_labelmap(labelmap_path)
        if args.pool_label in labelmap:
            pool_rgb = labelmap[args.pool_label]
        else:
            pool_rgb = None
            for label, rgb in labelmap.items():
                if normalize_key(label) != "background":
                    pool_rgb = rgb
                    break
            if pool_rgb is None:
                pool_rgb = (250, 50, 83)

        xml_polys: Dict[str, List[List[Tuple[float, float]]]] = {}
        for xml_path in xml_paths:
            try:
                parsed = parse_cvat_xml_polygons(xml_path, args.pool_label)
                for stem, polys in parsed.items():
                    xml_polys.setdefault(stem, []).extend(polys)
            except Exception as exc:
                print(f"[warn] Failed to parse XML {xml_path}: {exc}")

        stems = [ln.strip() for ln in default_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        stats = ExportStats(export_name=export_dir.name, image_source=str(image_dir), default_items=len(stems))
        stats.xml_images_present = len(xml_polys)

        for stem in stems:
            src_img = resolve_image_path(image_dir, stem)
            if src_img is None:
                stats.missing_images += 1
                total_missing_images += 1
                continue

            out_stem = f"{export_key}__{stem}"
            if out_stem in used_out_stems:
                msg = f"Duplicate output stem detected: {out_stem}"
                if args.strict:
                    raise SystemExit(msg)
                print(f"[warn] {msg}, skipping")
                continue
            used_out_stems.add(out_stem)

            split = deterministic_split(out_stem, args.val_frac, args.split_seed)
            dst_img = (out_img_val if split == "val" else out_img_train) / f"{out_stem}{src_img.suffix.lower()}"
            dst_lab = (out_lab_val if split == "val" else out_lab_train) / f"{out_stem}.txt"
            shutil.copy2(src_img, dst_img)
            stats.copied_images += 1
            total_images += 1

            with Image.open(src_img) as im:
                w, h = im.size

            lines: List[str] = []
            mask_path = segclass_dir / f"{stem}.png"
            if mask_path.exists():
                stats.segclass_masks_present += 1
                if mask_path.stat().st_size == 0:
                    stats.segclass_masks_invalid += 1
                    invalid_masks.append(
                        {
                            "export": export_dir.name,
                            "stem": stem,
                            "path": str(mask_path),
                            "reason": "zero_bytes",
                        }
                    )
                else:
                    try:
                        with Image.open(mask_path) as m:
                            mask_rgb = np.array(m.convert("RGB"))
                        lines = mask_rgb_to_yolo_lines(mask_rgb, pool_rgb, w, h, args.min_area_px)
                        stats.segclass_masks_valid += 1
                    except Exception as exc:
                        stats.segclass_masks_invalid += 1
                        invalid_masks.append(
                            {
                                "export": export_dir.name,
                                "stem": stem,
                                "path": str(mask_path),
                                "reason": f"decode_error:{exc.__class__.__name__}",
                            }
                        )
                        lines = []
            elif stem in xml_polys:
                for poly in xml_polys[stem]:
                    line = polygon_to_yolo_line(poly, w, h)
                    if line:
                        lines.append(line)

            dst_lab.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
            total_labels += 1
            if lines:
                stats.labels_positive += 1
                total_positive += 1
            else:
                stats.labels_empty += 1
                total_empty += 1

        report["exports"][export_dir.name] = stats.__dict__

    dataset_yaml = args.out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {args.out_dir.resolve().as_posix()}",
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

    report["totals"] = {
        "images": total_images,
        "labels": total_labels,
        "positive_labels": total_positive,
        "empty_labels": total_empty,
        "missing_images": total_missing_images,
        "invalid_masks": len(invalid_masks),
        "train_images": len(list(out_img_train.glob("*"))),
        "val_images": len(list(out_img_val.glob("*"))),
    }

    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    invalid_csv = args.out_dir / "invalid_masks.csv"
    with invalid_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["export", "stem", "path", "reason"])
        writer.writeheader()
        writer.writerows(invalid_masks)

    print(f"Wrote dataset: {args.out_dir}")
    print(f"Wrote dataset yaml: {dataset_yaml}")
    print(f"Wrote build report: {report_path}")
    print(f"Wrote invalid mask list: {invalid_csv}")
    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
