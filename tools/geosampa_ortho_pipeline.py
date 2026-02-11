#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WMS_BASE = "https://raster.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms"


@dataclass(frozen=True)
class BBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def tile_bbox(self, ix: int, iy: int, nx: int, ny: int) -> "BBox":
        dx = (self.maxx - self.minx) / nx
        dy = (self.maxy - self.miny) / ny
        x1 = self.minx + ix * dx
        y1 = self.miny + iy * dy
        x2 = x1 + dx
        y2 = y1 + dy
        return BBox(x1, y1, x2, y2)


def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    # Uncomment to debug:
    # print(p.stdout)


def ensure_gdal_tools() -> None:
    # minimal sanity: gdalinfo in PATH
    try:
        run(["gdalinfo", "--version"])
    except Exception as e:
        raise SystemExit(
            "GDAL tools not found (need gdalinfo/gdalbuildvrt/gdal_translate/gdal_retile.py). "
            "On macOS: brew install gdal"
        ) from e


def write_tfw(tfw_path: Path, bbox: BBox, width: int, height: int) -> None:
    # worldfile for north-up images
    A = (bbox.maxx - bbox.minx) / width
    E = - (bbox.maxy - bbox.miny) / height
    C = bbox.minx + A / 2.0
    F = bbox.maxy + E / 2.0
    tfw_path.write_text(f"{A}\n0.0\n0.0\n{E}\n{C}\n{F}\n")


def http_download(url: str, out_path: Path, timeout_s: int, retries: int, retry_delay_s: float) -> None:
    # Skip if file already exists and is non-trivial
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "geosampa-ortho-pipeline/1.0"})
            with urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            # Basic “blank tile” protection: keep very small files to inspect, but flag them.
            if out_path.stat().st_size < 5_000:
                print(f"Warning: tiny tile ({out_path.stat().st_size} bytes): {out_path}", file=sys.stderr)
            return
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_delay_s * (2 ** attempt))
                continue
            break
    raise RuntimeError(f"Failed to download after retries: {out_path}\n{last_err}") from last_err


def build_getmap_url(
    layer: str,
    bbox: BBox,
    srs: str,
    width: int,
    height: int,
    fmt: str,
) -> str:
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": fmt,
        "SRS": srs,
        "BBOX": f"{bbox.minx},{bbox.miny},{bbox.maxx},{bbox.maxy}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
    }
    return f"{WMS_BASE}?{urlencode(params)}"


def process_neighborhood(
    name: str,
    bbox: BBox,
    layer: str,
    srs: str,
    nx: int,
    ny: int,
    width: int,
    height: int,
    fmt: str,
    out_root: Path,
    max_workers: int,
    timeout_s: int,
    retries: int,
    retry_delay_s: float,
    chip_px: int,
) -> None:
    safe_name = name.replace(" ", "_")
    outdir = out_root / safe_name
    tiles_dir = outdir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== {name} ==")
    print(f"Layer: {layer}")
    print(f"BBox ({srs}): {bbox.minx},{bbox.miny},{bbox.maxx},{bbox.maxy}")
    print(f"Grid: {nx}x{ny}, tile px: {width}x{height}")
    print(f"Out: {outdir}")

    def one_tile(ix: int, iy: int) -> Tuple[int, int]:
        tb = bbox.tile_bbox(ix, iy, nx, ny)
        tif = tiles_dir / f"tile_{ix}_{iy}.tif"
        tfw = tiles_dir / f"tile_{ix}_{iy}.tfw"
        url = build_getmap_url(layer=layer, bbox=tb, srs=srs, width=width, height=height, fmt=fmt)
        http_download(url, tif, timeout_s=timeout_s, retries=retries, retry_delay_s=retry_delay_s)
        write_tfw(tfw, tb, width=width, height=height)
        return ix, iy

    # Download tiles (bounded parallelism to be polite to GeoServer)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(one_tile, ix, iy) for ix in range(nx) for iy in range(ny)]
        for f in cf.as_completed(futures):
            ix, iy = f.result()
            # light progress signal
            print(f"  got tile {ix}_{iy}")

    # Build mosaic
    mosaic_vrt = outdir / "mosaic.vrt"
    mosaic_tif = outdir / "mosaic.tif"
    crop_tif = outdir / "mosaic_aoi.tif"

    run(["gdalbuildvrt", str(mosaic_vrt), str(tiles_dir / "tile_*.tif")])

    run([
        "gdal_translate", "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        str(mosaic_vrt),
        str(mosaic_tif),
    ])

    # Crop exactly to bbox (same SRS)
    run([
        "gdalwarp",
        "-te", str(bbox.minx), str(bbox.miny), str(bbox.maxx), str(bbox.maxy),
        "-te_srs", srs,
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        str(mosaic_tif),
        str(crop_tif),
    ])

    # Chip as GeoTIFF then convert to PNG
    chips_tif_dir = outdir / f"chips_{chip_px}_tif"
    chips_png_dir = outdir / f"chips_{chip_px}_png"
    chips_tif_dir.mkdir(exist_ok=True)
    chips_png_dir.mkdir(exist_ok=True)

    run([
        "gdal_retile.py",
        "-ps", str(chip_px), str(chip_px),
        "-targetDir", str(chips_tif_dir),
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        str(crop_tif),
    ])

    # Convert tif chips to png
    tifs = sorted(chips_tif_dir.glob("*.tif"))
    for tif in tifs:
        png = chips_png_dir / (tif.stem + ".png")
        if png.exists() and png.stat().st_size > 1000:
            continue
        run(["gdal_translate", "-of", "PNG", str(tif), str(png)])

    # Manifest for downstream steps
    manifest = {
        "name": name,
        "layer": layer,
        "srs": srs,
        "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
        "nx": nx,
        "ny": ny,
        "tile_size_px": [width, height],
        "chip_px": chip_px,
        "chips_png": [str(p) for p in sorted(chips_png_dir.glob("*.png"))],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Done: {name}")
    print(f"  {crop_tif}")
    print(f"  chips: {chips_png_dir} ({len(manifest['chips_png'])} PNGs)")
    print(f"  manifest: {outdir / 'manifest.json'}")


def load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if "neighborhoods" not in data or not isinstance(data["neighborhoods"], list):
        raise ValueError("Config must contain a 'neighborhoods' list.")
    return data


def main() -> None:
    ensure_gdal_tools()

    ap = argparse.ArgumentParser(description="GeoSampa ortho pipeline: tiles -> mosaic -> crop -> chips (multi-neighborhood).")
    ap.add_argument("--config", required=True, help="Path to JSON config file.")
    ap.add_argument("--out-root", default="data/geosampa_ortho", help="Output root directory.")
    ap.add_argument("--layer", default="Orto_PMD_RGB_2017", help="WMS layer name (e.g., Orto_PMD_RGB_2017, ORTO_RGB_2020, MOSAICO_ORTO_RGB_10CM_20CM).")
    ap.add_argument("--srs", default="EPSG:3857", help="SRS for bbox and requests (default EPSG:3857).")
    ap.add_argument("--nx", type=int, default=4, help="Tiles in X direction per neighborhood.")
    ap.add_argument("--ny", type=int, default=4, help="Tiles in Y direction per neighborhood.")
    ap.add_argument("--tile-w", type=int, default=512, help="Tile width in pixels.")
    ap.add_argument("--tile-h", type=int, default=512, help="Tile height in pixels.")
    ap.add_argument("--format", default="image/geotiff", help="WMS output format (recommend image/geotiff).")
    ap.add_argument("--chip-px", type=int, default=1024, help="Chip size in pixels.")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent tile downloads per neighborhood.")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds.")
    ap.add_argument("--retries", type=int, default=10, help="HTTP retries.")
    ap.add_argument("--retry-delay", type=float, default=2.0, help="Base retry delay seconds (exponential backoff).")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for n in cfg["neighborhoods"]:
        name = n["name"]
        bb = n["bbox"]
        bbox = BBox(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))

        process_neighborhood(
            name=name,
            bbox=bbox,
            layer=args.layer,
            srs=args.srs,
            nx=args.nx,
            ny=args.ny,
            width=args.tile_w,
            height=args.tile_h,
            fmt=args.format,
            out_root=out_root,
            max_workers=args.workers,
            timeout_s=args.timeout,
            retries=args.retries,
            retry_delay_s=args.retry_delay,
            chip_px=args.chip_px,
        )


if __name__ == "__main__":
    main()
