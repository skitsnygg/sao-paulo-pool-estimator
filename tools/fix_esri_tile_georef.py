#!/usr/bin/env python3
"""
Fix broken EPSG:3857 affine transforms on Esri XYZ tiles by recomputing transform
from tile x/y indices and zoom.

Assumes:
- Files named: <x>_<y>.tif (or .tiff)
- Each tile is 256x256 RGB uint8 (but works for other band counts too)
- Zoom is fixed for the directory (pass --zoom)

Writes fixed tiles to --out-dir (does not modify originals by default).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple, Optional

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

TILE_SIZE = 256
ORIGIN_SHIFT = 20037508.342789244  # EPSG:3857 half-world width in meters

FNAME_RE = re.compile(r"(?P<x>\d+)_(?P<y>\d+)\.(tif|tiff)$", re.IGNORECASE)


def parse_xy(p: Path) -> Optional[Tuple[int, int]]:
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    return int(m.group("x")), int(m.group("y"))


def tile_transform_3857(x: int, y: int, z: int) -> Affine:
    # exact web mercator tile bounds formula
    res = (2 * ORIGIN_SHIFT) / (TILE_SIZE * (2 ** z))
    minx = -ORIGIN_SHIFT + x * TILE_SIZE * res
    maxy = ORIGIN_SHIFT - y * TILE_SIZE * res
    # Affine(a,b,c,d,e,f) where:
    # x = a*col + b*row + c
    # y = d*col + e*row + f
    # For north-up rasters: b=d=0, a=+res, e=-res, c=minx, f=maxy
    return Affine(res, 0.0, minx, 0.0, -res, maxy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True, help="Directory containing tiles named <x>_<y>.tif")
    ap.add_argument("--zoom", type=int, required=True, help="Zoom level (e.g., 18)")
    ap.add_argument("--out-dir", required=True, help="Output directory for fixed tiles")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files in out-dir if they exist")
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_paths = sorted([p for p in tiles_dir.iterdir() if p.is_file() and parse_xy(p)])
    if not src_paths:
        raise SystemExit(f"No tiles found in {tiles_dir} matching <x>_<y>.tif/.tiff")

    fixed = 0
    skipped = 0

    for p in src_paths:
        xy = parse_xy(p)
        assert xy is not None
        x, y = xy

        out_path = out_dir / p.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with rasterio.open(p) as ds:
            arr = ds.read()  # (bands, H, W)
            profile = ds.profile.copy()
            # force correct georef
            profile["crs"] = CRS.from_epsg(3857)
            profile["transform"] = tile_transform_3857(x, y, args.zoom)
            # ensure tiling/compat
            profile["driver"] = "GTiff"
            profile.pop("compress", None)  # keep simple; optional
            profile.pop("predictor", None)
            # preserve dtype/count/size
            profile["height"] = ds.height
            profile["width"] = ds.width
            profile["count"] = ds.count
            profile["dtype"] = ds.dtypes[0]

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)

        fixed += 1

    print(f"Fixed tiles written: {fixed} (skipped existing: {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())