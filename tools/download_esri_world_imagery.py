#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import numpy as np
from PIL import Image
from shapely.geometry import box, shape
from shapely.ops import unary_union
from pyproj import Transformer
import mercantile


ESRI_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ORIGIN_SHIFT = 20037508.342789244
TILE_SIZE = 256


@dataclass
class TileJob:
    z: int
    x: int
    y: int


def read_aoi_geometry(geojson_path: Path) -> Tuple[object, Optional[str]]:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    crs_name: Optional[str] = None
    crs = data.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties", {}) if isinstance(crs.get("properties"), dict) else {}
        name = props.get("name")
        if isinstance(name, str) and name:
            crs_name = name

    geoms = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if geom:
            geoms.append(shape(geom))

    if not geoms:
        raise SystemExit(f"No geometries found in {geojson_path}")

    return unary_union(geoms), crs_name


def reproject_geom(geom, src_crs: str, dst_crs: str):
    if src_crs == dst_crs:
        return geom
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform_geom(geom, transformer)


def shapely_transform_geom(geom, transformer: Transformer):
    # Avoid importing shapely.ops.transform at import time (keeps deps light)
    try:
        from shapely.ops import transform as shapely_transform
    except Exception as exc:
        raise SystemExit("shapely.ops.transform is required for reprojection") from exc

    return shapely_transform(transformer.transform, geom)


def tiles_covering_geom(geom, zoom: int) -> List[TileJob]:
    minx, miny, maxx, maxy = geom.bounds
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zooms=zoom))

    # Filter by polygon intersection for tighter coverage
    filtered = []
    for t in tiles:
        tb = mercantile.bounds(t)
        tile_poly = box(tb.west, tb.south, tb.east, tb.north)
        if geom.intersects(tile_poly):
            filtered.append(TileJob(t.z, t.x, t.y))

    filtered.sort(key=lambda t: (t.x, t.y))
    return filtered


def tile_bounds_mercator(x: int, y: int, z: int) -> Tuple[float, float, float, float, float]:
    res = (2 * ORIGIN_SHIFT) / (TILE_SIZE * (2 ** z))
    minx = -ORIGIN_SHIFT + x * TILE_SIZE * res
    maxx = -ORIGIN_SHIFT + (x + 1) * TILE_SIZE * res
    maxy = ORIGIN_SHIFT - y * TILE_SIZE * res
    miny = ORIGIN_SHIFT - (y + 1) * TILE_SIZE * res
    return minx, miny, maxx, maxy, res


def write_rgb_geotiff(path: Path, arr: np.ndarray, transform, crs: str = "EPSG:3857") -> None:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB array with shape (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)

    try:
        import rasterio
        from rasterio.transform import Affine
    except Exception:
        rasterio = None

    if rasterio is not None:
        if not isinstance(transform, tuple):
            transform = tuple(transform)
        affine = Affine(*transform)
        height, width = arr.shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs=crs,
            transform=affine,
        ) as dst:
            dst.write(arr[:, :, 0], 1)
            dst.write(arr[:, :, 1], 2)
            dst.write(arr[:, :, 2], 3)
        return

    try:
        from osgeo import gdal, osr
    except Exception as exc:
        raise SystemExit("rasterio or GDAL Python bindings are required to write GeoTIFFs") from exc

    height, width = arr.shape[:2]
    gdal_dtype = gdal.GDT_Byte

    path.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), width, height, 3, gdal_dtype)
    ds.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(crs.split(":")[1]))
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr[:, :, 0])
    ds.GetRasterBand(2).WriteArray(arr[:, :, 1])
    ds.GetRasterBand(3).WriteArray(arr[:, :, 2])
    ds.FlushCache()
    ds = None


def fetch_tile(session: requests.Session, url: str, timeout: float) -> bytes:
    resp = session.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    return resp.content


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Esri World Imagery XYZ tiles and save as EPSG:3857 GeoTIFFs.")
    ap.add_argument("--aoi-geojson", required=True, type=Path, help="AOI GeoJSON path")
    ap.add_argument("--zoom", type=int, default=19, help="XYZ zoom level (default: 19)")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    ap.add_argument("--max-tiles", type=int, default=0, help="If >0, limit tile count (debug)")
    ap.add_argument("--aoi-crs", type=str, default=None, help="Override AOI CRS (e.g. EPSG:4326)")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests (seconds)")
    ap.add_argument("--user-agent", type=str, default="esri-world-imagery-downloader/1.0")
    args = ap.parse_args()

    geom, crs_from_geojson = read_aoi_geometry(args.aoi_geojson)
    src_crs = args.aoi_crs or crs_from_geojson or "EPSG:4326"
    geom_ll = reproject_geom(geom, src_crs, "EPSG:4326")

    tiles = tiles_covering_geom(geom_ll, args.zoom)
    if args.max_tiles and args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]

    if not tiles:
        raise SystemExit("No tiles to download (check AOI/zoom)")

    print(f"AOI CRS: {src_crs} -> EPSG:4326")
    print(f"Tiles to download: {len(tiles)} (z={args.zoom})")

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    ok = 0
    skipped = 0
    failed = 0

    for idx, t in enumerate(tiles, 1):
        url = ESRI_URL.format(z=t.z, x=t.x, y=t.y)
        out_path = args.out_dir / f"z{t.z}" / f"{t.x}_{t.y}.tif"
        tmp_png = out_path.with_name(out_path.stem + ".tmp.png")

        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        try:
            data = fetch_tile(session, url, timeout=args.timeout)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_png.write_bytes(data)
            with Image.open(tmp_png) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise RuntimeError(f"Unexpected tile shape after RGB conversion: {arr.shape}")

            minx, miny, maxx, maxy, res = tile_bounds_mercator(t.x, t.y, t.z)
            transform = (minx, res, 0.0, maxy, 0.0, -res)
            write_rgb_geotiff(out_path, arr, transform, crs="EPSG:3857")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[fail] z={t.z} x={t.x} y={t.y}: {exc}")
        finally:
            if tmp_png.exists():
                tmp_png.unlink()

        if args.sleep > 0:
            time.sleep(args.sleep)

        if idx % 25 == 0:
            print(f"progress {idx}/{len(tiles)} ok={ok} skipped={skipped} failed={failed}")

    print(f"done. ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
