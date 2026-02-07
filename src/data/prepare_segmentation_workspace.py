from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import pandas as pd


README_TEXT = """Jardins pool segmentation workspace

This folder is for manual mask annotation.

Expected mask format:
- One PNG mask per image in `masks/`
- Same base filename as the image (e.g., image `r0000_c0000.png` -> mask `r0000_c0000.png`)
- Single channel, 8-bit PNG
- Pixel values: 0 = background, 255 = pool
"""


def prepare_workspace(chips_csv: Path, out_dir: Path, use_symlinks: bool, init_masks: bool) -> Path:
    df = pd.read_csv(chips_csv)
    if "path" not in df.columns:
        raise ValueError("chips_csv must include a path column.")

    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = out_dir / "annotations.csv"
    classes_path = out_dir / "classes.txt"
    readme_path = out_dir / "README.txt"

    if not classes_path.exists():
        classes_path.write_text("pool\n", encoding="utf-8")
    if not readme_path.exists():
        readme_path.write_text(README_TEXT, encoding="utf-8")

    rows = []
    for row in df.itertuples():
        src = Path(row.path)
        if not src.exists():
            continue
        dst = images_dir / src.name
        if not dst.exists():
            if use_symlinks:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

        mask = masks_dir / src.name
        if init_masks and not mask.exists():
            # Placeholder mask can be created by the annotator; leave blank by default.
            mask.touch()

        rows.append(
            {
                "image": str(dst),
                "mask": str(mask),
                "chip_id": src.stem,
                "xmin": getattr(row, "xmin", ""),
                "ymin": getattr(row, "ymin", ""),
                "xmax": getattr(row, "xmax", ""),
                "ymax": getattr(row, "ymax", ""),
                "crs": getattr(row, "crs", ""),
                "width_px": getattr(row, "width_px", ""),
                "height_px": getattr(row, "height_px", ""),
            }
        )

    with annotations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "mask",
                "chip_id",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "crs",
                "width_px",
                "height_px",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return annotations_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a segmentation workspace for Jardins chips.")
    parser.add_argument("--chips-csv", default="data/raw/geosampa_ortho/jardins_2020/chips.csv")
    parser.add_argument("--out-dir", default="data/annotations/jardins_seg")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--init-masks", action="store_true")
    args = parser.parse_args()

    chips_csv = Path(args.chips_csv)
    out_dir = Path(args.out_dir)
    annotations_path = prepare_workspace(chips_csv, out_dir, args.symlink, args.init_masks)
    print(f"Prepared workspace: {annotations_path}")


if __name__ == "__main__":
    main()
