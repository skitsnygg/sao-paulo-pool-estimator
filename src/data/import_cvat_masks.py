from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def _parse_labelmap(text: str) -> dict[str, tuple[int, int, int]]:
    mapping: dict[str, tuple[int, int, int]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        label = parts[0].strip()
        rgb = parts[1].strip()
        if not label or not rgb:
            continue
        try:
            r_str, g_str, b_str = rgb.split(",")
            mapping[label] = (int(r_str), int(g_str), int(b_str))
        except ValueError:
            continue
    return mapping


def _load_segmentation_mask(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as handle:
        img = Image.open(handle).convert("RGB")
        return np.array(img)


def import_masks(
    export_zip: Path,
    workspace: Path,
    pool_label: str,
    foreground_value: int,
    fill_empty: bool,
) -> tuple[int, int]:
    masks_dir = workspace / "masks"
    images_dir = workspace / "images"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not export_zip.exists():
        raise FileNotFoundError(f"Missing CVAT export zip: {export_zip}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    with zipfile.ZipFile(export_zip, "r") as zf:
        try:
            labelmap_text = zf.read("labelmap.txt").decode("utf-8")
        except KeyError as exc:
            raise FileNotFoundError("labelmap.txt not found in CVAT export.") from exc

        labelmap = _parse_labelmap(labelmap_text)
        if pool_label not in labelmap:
            raise ValueError(f"Pool label '{pool_label}' not found in labelmap.")
        pool_color = labelmap[pool_label]

        annotated = set()
        for name in zf.namelist():
            if not name.startswith("SegmentationClass/") or not name.endswith(".png"):
                continue
            mask_rgb = _load_segmentation_mask(zf, name)
            pool_mask = np.all(mask_rgb == pool_color, axis=-1)
            out = (pool_mask.astype(np.uint8) * foreground_value)
            out_path = masks_dir / Path(name).name
            Image.fromarray(out, mode="L").save(out_path)
            annotated.add(out_path.name)

    empty_created = 0
    if fill_empty:
        for image_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / image_path.name
            if mask_path.exists():
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            Image.new("L", (width, height), color=0).save(mask_path)
            empty_created += 1

    return len(annotated), empty_created


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CVAT segmentation masks into the workspace.")
    parser.add_argument(
        "--export-zip",
        default="data/annotations/jardins_seg/cvat_segmentation_mask.zip",
    )
    parser.add_argument("--workspace", default="data/annotations/jardins_seg")
    parser.add_argument("--pool-label", default="pool")
    parser.add_argument("--foreground-value", type=int, default=255)
    parser.add_argument("--fill-empty", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    annotated_count, empty_count = import_masks(
        export_zip=Path(args.export_zip),
        workspace=Path(args.workspace),
        pool_label=args.pool_label,
        foreground_value=args.foreground_value,
        fill_empty=args.fill_empty,
    )
    print(f"Imported {annotated_count} mask(s).")
    if args.fill_empty:
        print(f"Created {empty_count} empty mask(s).")


if __name__ == "__main__":
    main()
