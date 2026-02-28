#!/usr/bin/env python3
"""
Stitch Esri XYZ 256x256 EPSG:3857 GeoTIFF tiles into 1024x1024 chips (4x4 blocks).

Input tiles:
  - 256x256
  - RGB uint8 (3 bands)
  - EPSG:3857
  - Filenames like: <x>_<y>.tif   (tile x,y at fixed zoom)

Output chips:
  - 1024x1024
  - RGB uint8
  - EPSG:3857
  - Correct Affine transform derived from the top-left tile of each 4x4 block:
      Affine(t.a, 0, t.c, 0, t.e, t.f)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import rasterio
from rasterio.transform import Affine

FNAME_RE = re.compile(r"(?P<x>\d+)_(?P<y>\d+)\.tif$", re.IGNORECASE)


def parse_xy(path: Path) -> Tuple[int, int]:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Tile filename does not match <x>_<y>.tif: {path.name}")
    return int(m.group("x")), int(m.group("y"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True, help="Directory containing 256x256 tiles named <x>_<y>.tif")
    ap.add_argument("--neighborhood", required=True, help="Neighborhood name (output subdir)")
    ap.add_argument("--out-root", default="data/esri_annotate_1024/z18", help="Root output directory")
    ap.add_argument("--block", type=int, default=4, help="Tiles per side (4 -> 1024)")
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.is_dir():
        raise SystemExit(f"tiles-dir not found: {tiles_dir}")

    out_dir = Path(args.out_root) / args.neighborhood
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index tiles by (x, y)
    tiles: Dict[Tuple[int, int], Path] = {}
    for p in sorted(tiles_dir.glob("*.tif")):
        try:
            xy = parse_xy(p)
        except ValueError:
            continue
        tiles[xy] = p

    if not tiles:
        raise SystemExit(f"No tiles found in {tiles_dir} matching <x>_<y>.tif")

    xs = sorted({x for x, _ in tiles.keys()})
    ys = sorted({y for _, y in tiles.keys()})

    block = args.block
    created = 0

    # Deterministic: iterate top-left candidates in sorted x0,y0
    for x0 in xs:
        if x0 % block != 0:
            continue
        for y0 in ys:
            if y0 % block != 0:
                continue

            # Need a full block of 4x4 tiles: (x0..x0+3, y0..y0+3)
            coords: List[Tuple[int, int]] = [
                (x0 + dx, y0 + dy) for dy in range(block) for dx in range(block)
            ]
            if any(c not in tiles for c in coords):
                continue

            # Use the top-left tile (x0,y0) for metadata
            tl_path = tiles[(x0, y0)]
            with rasterio.open(tl_path) as tl:
                if tl.crs is None:
                    raise SystemExit(f"Missing CRS in {tl_path}")
                if str(tl.crs).upper() not in ("EPSG:3857", "WGS 84 / PSEUDO-MERCATOR"):
                    # rasterio prints "EPSG:3857" typically; accept that
                    pass
                if tl.count != 3:
                    raise SystemExit(f"Expected 3-band RGB tile, got {tl.count} bands: {tl_path}")
                if tl.width != 256 or tl.height != 256:
                    raise SystemExit(f"Expected 256x256 tile, got {tl.width}x{tl.height}: {tl_path}")

                t = tl.transform
                # The ONLY correct way: preserve pixel size and top-left origin
                out_transform = Affine(t.a, 0.0, t.c, 0.0, t.e, t.f)
                out_crs = tl.crs

                # Create mosaic array (bands, H, W)
                mosaic = np.zeros((3, 256 * block, 256 * block), dtype=np.uint8)

            # Fill mosaic row-major by dy,dx where y increases downward in XYZ.
            for dy in range(block):
                for dx in range(block):
                    p = tiles[(x0 + dx, y0 + dy)]
                    with rasterio.open(p) as ds:
                        arr = ds.read(out_dtype=np.uint8)  # (3,256,256)
                        if arr.shape != (3, 256, 256):
                            raise SystemExit(f"Unexpected tile array shape {arr.shape} in {p}")
                    y_off = dy * 256
                    x_off = dx * 256
                    mosaic[:, y_off : y_off + 256, x_off : x_off + 256] = arr

            out_path = out_dir / f"{x0}_{y0}.tif"
            profile = {
                "driver": "GTiff",
                "width": 256 * block,
                "height": 256 * block,
                "count": 3,
                "dtype": "uint8",
                "crs": out_crs,
                "transform": out_transform,
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic)

            created += 1

    print(f"Created 1024x1024 chips: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())