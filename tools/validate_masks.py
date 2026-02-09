#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import cv2


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    # Update these if your folders differ:
    images_dir = Path("data/annotations/jardins_seg/images")
    masks_dir = Path("data/annotations/jardins_seg/masks")  # change if yours is different
    chips_csv = Path("data/raw/geosampa_ortho/jardins_2020/chips.csv")

    if not images_dir.exists():
        fail(f"Missing images dir: {images_dir}")
    if not masks_dir.exists():
        fail(f"Missing masks dir: {masks_dir}")
    if not chips_csv.exists():
        fail(f"Missing chips.csv: {chips_csv}")

    # Load expected image filenames from chips.csv if it has a filename column,
    # otherwise just validate against images_dir.
    expected = []
    with chips_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        name_col = None
        for c in ["filename", "image", "chip", "path"]:
            if c in cols:
                name_col = c
                break
        if name_col:
            for row in r:
                expected.append(Path(row[name_col]).name)
        else:
            expected = [p.name for p in sorted(images_dir.glob("*.png"))]

    if not expected:
        fail("No images found to validate.")

    missing_masks = []
    bad_masks = []
    ok = 0

    for name in expected:
        img_path = images_dir / name
        mask_path = masks_dir / name

        if not img_path.exists():
            continue  # allow chips.csv to include non-workspace images

        if not mask_path.exists():
            missing_masks.append(name)
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            bad_masks.append((name, "could_not_read"))
            continue

        # Accept either HxW single channel or HxWx3, but convert/check properly.
        if mask.ndim == 3:
            # If someone saved RGB masks, require all channels equal.
            if not (np.all(mask[..., 0] == mask[..., 1]) and np.all(mask[..., 1] == mask[..., 2])):
                bad_masks.append((name, f"rgb_channels_not_equal shape={mask.shape}"))
                continue
            mask = mask[..., 0]

        if mask.dtype != np.uint8:
            bad_masks.append((name, f"dtype_not_uint8 dtype={mask.dtype}"))
            continue

        uniques = np.unique(mask)
        # Allow {0,255} strictly; if you used {0,1} we can fix later but flag now.
        if not set(uniques.tolist()).issubset({0, 255}):
            bad_masks.append((name, f"unexpected_values uniques={uniques[:20]}"))
            continue

        ok += 1

    print(f"OK masks: {ok}")
    print(f"Missing masks: {len(missing_masks)}")
    if missing_masks[:10]:
        print("  examples:", ", ".join(missing_masks[:10]))
    print(f"Bad masks: {len(bad_masks)}")
    if bad_masks[:10]:
        print("  examples:", bad_masks[:10])

    # Nonzero exit if something is wrong
    if missing_masks or bad_masks:
        sys.exit(1)


if __name__ == "__main__":
    main()
