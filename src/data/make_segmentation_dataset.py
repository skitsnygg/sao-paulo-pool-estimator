from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
from pathlib import Path
import shutil
from typing import Iterable
from xml.etree import ElementTree as ET
import zipfile

from src.utils.config import load_config, project_root


@dataclass(frozen=True)
class ImageEntry:
    name: str
    width: int
    height: int
    polygons: list[list[tuple[float, float]]]


def _parse_points(points: str) -> list[tuple[float, float]]:
    coords = []
    for pair in points.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        try:
            x_str, y_str = pair.split(",")
            coords.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return coords


def _clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _load_annotations(xml_path: Path, label: str) -> list[ImageEntry]:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    images: list[ImageEntry] = []
    for image_el in root.findall(".//image"):
        name = image_el.get("name")
        if not name:
            continue
        try:
            width = int(float(image_el.get("width", "0")))
            height = int(float(image_el.get("height", "0")))
        except ValueError:
            continue
        polygons: list[list[tuple[float, float]]] = []
        for poly_el in image_el.findall("polygon"):
            if poly_el.get("label") != label:
                continue
            points = poly_el.get("points", "")
            coords = _parse_points(points)
            if len(coords) < 3:
                continue
            polygons.append(coords)
        images.append(ImageEntry(name=name, width=width, height=height, polygons=polygons))
    return images


def _read_annotations_zip(zip_path: Path, label: str) -> list[ImageEntry]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("annotations.xml") as handle:
            xml_bytes = handle.read()
    xml_path = zip_path.with_suffix(".xml")
    xml_path.write_bytes(xml_bytes)
    try:
        return _load_annotations(xml_path, label)
    finally:
        xml_path.unlink(missing_ok=True)


def _assign_splits(items: list[ImageEntry], ratios: dict[str, float], seed: int) -> dict[str, list[ImageEntry]]:
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    total = len(items)

    train_count = int(total * ratios["train"])
    val_count = int(total * ratios["val"])
    test_count = total - train_count - val_count

    return {
        "train": items[:train_count],
        "val": items[train_count:train_count + val_count],
        "test": items[train_count + val_count:train_count + val_count + test_count],
    }


def _normalize_polygon(
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> Iterable[float]:
    for x, y in polygon:
        x_norm = _clamp(x / width, 0.0, 1.0)
        y_norm = _clamp(y / height, 0.0, 1.0)
        yield x_norm
        yield y_norm


def _write_dataset_yaml(dataset_root: Path) -> None:
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: pool",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_dataset(
    annotations_zip: Path,
    images_dir: Path,
    output_dir: Path,
    seed: int,
    ratios: dict[str, float],
    label: str,
    use_symlinks: bool,
) -> Path:
    entries = _read_annotations_zip(annotations_zip, label)
    images_by_name = {entry.name: entry for entry in entries}

    # Process all images in the images directory
    all_entries = []
    for image_path in sorted(images_dir.glob("*.png")):
        entry = images_by_name.get(image_path.name)
        if entry is not None:
            # Image has annotations
            all_entries.append(entry)
        else:
            # Image has no annotations - create an entry with empty polygons
            # We need to get the image dimensions to create a proper entry
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                # If we can't open the image, skip it
                continue
            all_entries.append(ImageEntry(name=image_path.name, width=width, height=height, polygons=[]))

    if not all_entries:
        raise ValueError("No images found in the images directory.")

    splits = _assign_splits(all_entries, ratios, seed)

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, split_entries in splits.items():
        for entry in split_entries:
            src = images_dir / entry.name
            dst = output_dir / "images" / split / entry.name
            if not dst.exists():
                if use_symlinks:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)

            label_path = output_dir / "labels" / split / f"{Path(entry.name).stem}.txt"
            lines = []
            for polygon in entry.polygons:
                coords = list(_normalize_polygon(polygon, entry.width, entry.height))
                if len(coords) < 6:
                    continue
                line = "0 " + " ".join(f"{value:.6f}" for value in coords)
                lines.append(line)
            # Write label file if there are valid polygons, or don't create it for empty masks
            if lines:
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                # For images with no annotations, we don't create a label file
                # This approach is consistent with how YOLO handles empty masks
                # The absence of a label file indicates no object in the image
                pass

    _write_dataset_yaml(output_dir)
    return output_dir / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO segmentation dataset from CVAT polygons.")
    parser.add_argument("--annotations-zip", default="data/annotations/jardins_seg/cvat_annotations.zip")
    parser.add_argument("--images-dir", default="data/annotations/jardins_seg/images")
    parser.add_argument("--output-dir", default="data/processed/yolo_seg")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--label", default="pool")
    parser.add_argument("--symlink", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ratios = {
        "train": config["dataset"]["train_ratio"],
        "val": config["dataset"]["val_ratio"],
        "test": config["dataset"]["test_ratio"],
    }
    seed = config["dataset"].get("random_seed", 42)

    dataset_yaml = build_dataset(
        annotations_zip=project_root() / args.annotations_zip,
        images_dir=project_root() / args.images_dir,
        output_dir=project_root() / args.output_dir,
        seed=seed,
        ratios=ratios,
        label=args.label,
        use_symlinks=args.symlink,
    )
    print(f"Saved segmentation dataset to {dataset_yaml}")


if __name__ == "__main__":
    main()
