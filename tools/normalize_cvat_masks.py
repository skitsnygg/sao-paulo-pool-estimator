#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


def main() -> None:
    masks_dir = Path("data/annotations/jardins_seg/masks")
    if not masks_dir.exists():
        print(f"Missing masks dir: {masks_dir}", file=sys.stderr)
        sys.exit(2)

    changed = 0
    total = 0

    for p in sorted(masks_dir.glob("*.png")):
        total += 1
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            print(f"Could not read: {p}", file=sys.stderr)
            continue

        # If already single-channel, normalize to {0,255} and continue
        if m.ndim == 2:
            out = (m > 0).astype(np.uint8) * 255
            # Only rewrite if needed
            if not (out.dtype == np.uint8 and out.shape == m.shape and np.array_equal(out, m)):
                cv2.imwrite(str(p), out)
                changed += 1
            continue

        # If RGB/RGBA: treat any non-zero pixel in any channel as pool
        if m.ndim == 3:
            # m could be HxWx3 or HxWx4
            rgb = m[..., :3]
            binary = (np.any(rgb != 0, axis=2)).astype(np.uint8) * 255
            cv2.imwrite(str(p), binary)
            changed += 1
            continue

        print(f"Unexpected mask shape {m.shape} for {p}", file=sys.stderr)

    print(f"Processed {total} masks. Normalized/rewrote: {changed}")


if __name__ == "__main__":
    main()
