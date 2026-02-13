#!/usr/bin/env python3
"""
Build 1024x1024 chips by stitching 4x4 Esri World Imagery XYZ tiles (256x256).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

TILE_URL_TEMPLATE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE = 256
MAX_LAT = 85.05112878


class RateLimiter:
    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.interval = 1.0 / rate if rate and rate > 0 else 0.0
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.monotonic()
            self.next_time = now + self.interval


def latlon_to_tile(lat: float, lon: float, z: int) -> Tuple[int, int]:
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    xtile = int(math.floor(x))
    ytile = int(math.floor(y))
    xtile = max(0, min(xtile, n - 1))
    ytile = max(0, min(ytile, n - 1))
    return xtile, ytile


def tile_to_bounds(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    n = 2 ** z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return lon_left, lat_bottom, lon_right, lat_top


def parse_bbox(s: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'min_lon,min_lat,max_lon,max_lat'")
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if min_lon > max_lon:
        raise ValueError("bbox crosses antimeridian (min_lon > max_lon), not supported")
    if min_lat > max_lat:
        raise ValueError("bbox min_lat > max_lat")
    return min_lon, min_lat, max_lon, max_lat


def parse_center(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("center must be 'lon,lat'")
    lon, lat = map(float, parts)
    return lon, lat


def bbox_to_tile_range(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, z: int
) -> Tuple[int, int, int, int]:
    x_min, y_min = latlon_to_tile(max_lat, min_lon, z)
    x_max, y_max = latlon_to_tile(min_lat, max_lon, z)
    if x_min > x_max:
        raise ValueError("computed tile range invalid (x_min > x_max)")
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    return x_min, x_max, y_min, y_max


def center_to_tile_range(lon: float, lat: float, radius_tiles: int, z: int) -> Tuple[int, int, int, int]:
    x_c, y_c = latlon_to_tile(lat, lon, z)
    n = 2 ** z
    x_min = max(0, x_c - radius_tiles)
    x_max = min(n - 1, x_c + radius_tiles)
    y_min = max(0, y_c - radius_tiles)
    y_max = min(n - 1, y_c + radius_tiles)
    return x_min, x_max, y_min, y_max


def cache_paths(cache_dir: Path, z: int, x: int, y: int) -> List[Path]:
    base = cache_dir / str(z) / str(x)
    return [base / f"{y}.jpg", base / f"{y}.jpeg", base / f"{y}.png"]


def find_cached_tile(cache_dir: Path, z: int, x: int, y: int) -> Optional[Path]:
    for p in cache_paths(cache_dir, z, x, y):
        if p.exists():
            return p
    return None


def validate_tile_file(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
            return img.size == (TILE_SIZE, TILE_SIZE)
    except Exception:
        return False


def download_tile(
    z: int,
    x: int,
    y: int,
    cache_dir: Path,
    rate_limiter: RateLimiter,
    jpeg_quality: int,
    max_retries: int = 3,
    timeout_s: int = 30,
) -> Optional[Path]:
    url = TILE_URL_TEMPLATE.format(z=z, y=y, x=x)
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            rate_limiter.wait()
            resp = requests.get(url, timeout=timeout_s)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            data = resp.content
            if not data:
                raise RuntimeError("empty response")
            img = Image.open(io.BytesIO(data))
            img.load()
            if img.size != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError(f"unexpected tile size {img.size}")
            content_type = (resp.headers.get("Content-Type") or "").lower()
            ext = "png" if ("png" in content_type or (img.format or "").upper() == "PNG") else "jpg"
            out_dir = cache_dir / str(z) / str(x)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{y}.{ext}"
            out_path.write_bytes(data)
            return out_path
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                sleep_s = (0.5 * (2 ** attempt)) + (0.1 * attempt)
                time.sleep(sleep_s)
            else:
                break
    print(f"Failed to download z={z} x={x} y={y}: {last_err}", file=sys.stderr)
    return None


def build_chip(
    z: int,
    x0: int,
    y0: int,
    block: int,
    tile_paths: Dict[Tuple[int, int], Path],
    out_path: Path,
    jpeg_quality: int,
) -> bool:
    chip_size = block * TILE_SIZE
    chip = Image.new("RGB", (chip_size, chip_size))
    try:
        for dy in range(block):
            for dx in range(block):
                key = (x0 + dx, y0 + dy)
                path = tile_paths.get(key)
                if path is None:
                    return False
                with Image.open(path) as img:
                    img.load()
                    if img.size != (TILE_SIZE, TILE_SIZE):
                        return False
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    chip.paste(img, (dx * TILE_SIZE, dy * TILE_SIZE))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chip.save(out_path, "JPEG", quality=jpeg_quality)
        return True
    except Exception as e:
        print(f"Failed to build chip z={z} x={x0} y={y0}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 1024x1024 chips by stitching 4x4 Esri XYZ tiles."
    )
    parser.add_argument("--z", type=int, required=True, help="Zoom level (e.g., 18)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bbox", type=str, help="min_lon,min_lat,max_lon,max_lat (EPSG:4326)")
    group.add_argument("--center", type=str, help="lon,lat (EPSG:4326)")
    parser.add_argument("--radius-tiles", type=int, default=None, help="Radius in tiles around center")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for stitched chips")
    parser.add_argument("--cache-dir", type=str, required=True, help="Cache directory for 256x256 tiles")
    parser.add_argument("--chip-size", type=int, default=1024, help="Chip size in pixels (default 1024)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for output chips")
    parser.add_argument("--max-workers", type=int, default=8, help="Max concurrent download workers")
    parser.add_argument("--rate-limit", type=float, default=8.0, help="Max requests per second (0 to disable)")
    parser.add_argument("--manifest-csv", type=str, default=None, help="Optional CSV manifest path")

    args = parser.parse_args()

    if args.chip_size % TILE_SIZE != 0:
        raise SystemExit(f"--chip-size must be a multiple of {TILE_SIZE}")
    block = args.chip_size // TILE_SIZE
    if block <= 0:
        raise SystemExit("Invalid chip size")

    if args.center:
        if args.radius_tiles is None:
            raise SystemExit("--radius-tiles is required with --center")
        lon, lat = parse_center(args.center)
        x_min, x_max, y_min, y_max = center_to_tile_range(lon, lat, args.radius_tiles, args.z)
    else:
        min_lon, min_lat, max_lon, max_lat = parse_bbox(args.bbox)
        x_min, x_max, y_min, y_max = bbox_to_tile_range(min_lon, min_lat, max_lon, max_lat, args.z)

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    coords: List[Tuple[int, int]] = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            coords.append((x, y))

    rate_limiter = RateLimiter(args.rate_limit)
    tile_paths: Dict[Tuple[int, int], Path] = {}
    cache_hits = 0
    downloaded = 0
    failed = 0

    missing: List[Tuple[int, int]] = []
    for x, y in coords:
        cached = find_cached_tile(cache_dir, args.z, x, y)
        if cached and validate_tile_file(cached):
            tile_paths[(x, y)] = cached
            cache_hits += 1
        else:
            if cached:
                try:
                    cached.unlink()
                except Exception:
                    pass
            missing.append((x, y))

    if missing:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            future_map = {
                ex.submit(download_tile, args.z, x, y, cache_dir, rate_limiter, args.jpeg_quality): (x, y)
                for x, y in missing
            }
            for fut in as_completed(future_map):
                x, y = future_map[fut]
                path = fut.result()
                if path and validate_tile_file(path):
                    tile_paths[(x, y)] = path
                    downloaded += 1
                else:
                    failed += 1

    anchors_skipped = 0
    chips_written = 0

    manifest_writer = None
    manifest_file = None
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_path.open("w", newline="", encoding="utf-8")
        manifest_writer = csv.writer(manifest_file)
        manifest_writer.writerow(["filename", "min_lon", "min_lat", "max_lon", "max_lat", "z", "x0", "y0"])

    anchor_x_max = x_max - block + 1
    anchor_y_max = y_max - block + 1
    if anchor_x_max >= x_min and anchor_y_max >= y_min:
        for y0 in range(y_min, anchor_y_max + 1):
            for x0 in range(x_min, anchor_x_max + 1):
                complete = True
                for dy in range(block):
                    for dx in range(block):
                        if (x0 + dx, y0 + dy) not in tile_paths:
                            complete = False
                            break
                    if not complete:
                        break
                if not complete:
                    anchors_skipped += 1
                    continue

                chip_name = f"chip_z{args.z}_x{x0}_y{y0}.jpg"
                chip_path = out_dir / chip_name
                ok = build_chip(args.z, x0, y0, block, tile_paths, chip_path, args.jpeg_quality)
                if ok:
                    chips_written += 1
                    if manifest_writer:
                        tl = tile_to_bounds(x0, y0, args.z)
                        br = tile_to_bounds(x0 + block - 1, y0 + block - 1, args.z)
                        min_lon = tl[0]
                        max_lon = br[2]
                        max_lat = tl[3]
                        min_lat = br[1]
                        manifest_writer.writerow([chip_name, min_lon, min_lat, max_lon, max_lat, args.z, x0, y0])
                else:
                    anchors_skipped += 1
    else:
        anchors_skipped = len(coords)

    if manifest_file:
        manifest_file.close()

    print("Summary")
    print(f"tiles_downloaded: {downloaded}")
    print(f"cache_hits: {cache_hits}")
    print(f"tiles_failed: {failed}")
    print(f"chips_written: {chips_written}")
    print(f"anchors_skipped: {anchors_skipped}")


if __name__ == "__main__":
    main()
