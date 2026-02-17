#!/usr/bin/env python3
"""
Build 1024x1024 chips by stitching Esri World Imagery XYZ tiles (256x256) into NxN mosaics.

Key features:
- Downloads a contiguous XYZ tile range (bbox or center+radius) at a given zoom.
- Caches raw tiles on disk and reuses them.
- Stitches complete NxN blocks into chips (default 4x4 => 1024x1024).
- Skips incomplete anchors (does not crash).
- Optional stride to avoid sliding-window explosion (default: non-overlapping blocks).
- Optional filtering of low-information/placeholder tiles via file size + image variance.
- Writes optional manifest CSV with WGS84 bounds for each chip.

Note:
- Stitching increases chip pixel dimensions, NOT spatial resolution. Use higher zoom (z) for sharper detail.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image, ImageStat

TILE_URL_TEMPLATE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE = 256
MAX_LAT = 85.05112878


# -------------------------
# Rate limiting
# -------------------------
class RateLimiter:
    """Simple global rate limiter across threads (max requests/sec)."""

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


# -------------------------
# Slippy tile math
# -------------------------
def latlon_to_tile(lat: float, lon: float, z: int) -> Tuple[int, int]:
    """Convert WGS84 lat/lon to XYZ tile coordinates at zoom z."""
    lat = max(min(lat, MAX_LAT), -MAX_LAT)
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    xtile = int(math.floor(x))
    ytile = int(math.floor(y))
    xtile = max(0, min(xtile, n - 1))
    ytile = max(0, min(ytile, n - 1))
    return xtile, ytile


def tile_to_bounds(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    """
    Return WGS84 bounds for a single tile:
    (min_lon, min_lat, max_lon, max_lat)
    """
    n = 2**z
    min_lon = x / n * 360.0 - 180.0
    max_lon = (x + 1) / n * 360.0 - 180.0
    max_lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    min_lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return min_lon, min_lat, max_lon, max_lat


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
    # top-left from (max_lat, min_lon), bottom-right from (min_lat, max_lon)
    x_min, y_min = latlon_to_tile(max_lat, min_lon, z)
    x_max, y_max = latlon_to_tile(min_lat, max_lon, z)
    if x_min > x_max:
        raise ValueError("computed tile range invalid (x_min > x_max)")
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    return x_min, x_max, y_min, y_max


def center_to_tile_range(lon: float, lat: float, radius_tiles: int, z: int) -> Tuple[int, int, int, int]:
    x_c, y_c = latlon_to_tile(lat, lon, z)
    n = 2**z
    x_min = max(0, x_c - radius_tiles)
    x_max = min(n - 1, x_c + radius_tiles)
    y_min = max(0, y_c - radius_tiles)
    y_max = min(n - 1, y_c + radius_tiles)
    return x_min, x_max, y_min, y_max


# -------------------------
# Tile caching + validation
# -------------------------
def tile_cache_path(cache_dir: Path, z: int, x: int, y: int, ext: str) -> Path:
    return cache_dir / str(z) / str(x) / f"{y}.{ext}"


def find_cached_tile(cache_dir: Path, z: int, x: int, y: int) -> Optional[Path]:
    base = cache_dir / str(z) / str(x)
    for ext in ("jpg", "jpeg", "png"):
        p = base / f"{y}.{ext}"
        if p.exists():
            return p
    return None


def validate_tile_file(
    path: Path,
    min_bytes: int = 0,
    min_variance: float = 0.0,
) -> bool:
    """
    Validate tile:
    - readable image
    - correct size (256x256)
    - optional file size >= min_bytes
    - optional grayscale variance >= min_variance (filters near-blank tiles)
    """
    try:
        if min_bytes and path.stat().st_size < min_bytes:
            return False
        with Image.open(path) as img:
            img.load()
            if img.size != (TILE_SIZE, TILE_SIZE):
                return False
            if min_variance and min_variance > 0:
                g = img.convert("L")
                stat = ImageStat.Stat(g)
                # variance is stat.var[0]
                if (stat.var[0] or 0.0) < min_variance:
                    return False
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class DownloadResult:
    x: int
    y: int
    path: Optional[Path]
    from_cache: bool
    ok: bool
    err: Optional[str] = None


def download_tile(
    session: requests.Session,
    z: int,
    x: int,
    y: int,
    cache_dir: Path,
    rate_limiter: RateLimiter,
    min_tile_bytes: int,
    min_tile_variance: float,
    max_retries: int = 3,
    timeout_s: int = 30,
) -> DownloadResult:
    cached = find_cached_tile(cache_dir, z, x, y)
    if cached and validate_tile_file(cached, min_bytes=min_tile_bytes, min_variance=min_tile_variance):
        return DownloadResult(x=x, y=y, path=cached, from_cache=True, ok=True)

    if cached:
        try:
            cached.unlink()
        except Exception:
            pass

    url = TILE_URL_TEMPLATE.format(z=z, y=y, x=x)
    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            rate_limiter.wait()
            resp = session.get(url, timeout=timeout_s)
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
            img_fmt = (img.format or "").upper()
            ext = "png" if ("png" in content_type or img_fmt == "PNG") else "jpg"

            out_dir = cache_dir / str(z) / str(x)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{y}.{ext}"
            out_path.write_bytes(data)

            if not validate_tile_file(out_path, min_bytes=min_tile_bytes, min_variance=min_tile_variance):
                try:
                    out_path.unlink()
                except Exception:
                    pass
                raise RuntimeError("tile failed validation (size/bytes/variance)")

            return DownloadResult(x=x, y=y, path=out_path, from_cache=False, ok=True)

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                # exponential-ish backoff
                time.sleep(0.5 * (2**attempt) + 0.1 * attempt)
            else:
                break

    return DownloadResult(x=x, y=y, path=None, from_cache=False, ok=False, err=last_err)


# -------------------------
# Stitching
# -------------------------
def build_chip(
    z: int,
    x0: int,
    y0: int,
    block: int,
    tile_paths: Dict[Tuple[int, int], Path],
    out_path: Path,
    jpeg_quality: int,
) -> bool:
    """Stitch a block x block set of tiles into one chip."""
    chip_size = block * TILE_SIZE
    chip = Image.new("RGB", (chip_size, chip_size))

    try:
        for dy in range(block):
            for dx in range(block):
                p = tile_paths.get((x0 + dx, y0 + dy))
                if p is None:
                    return False
                with Image.open(p) as img:
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


def chip_bounds_from_anchor(z: int, x0: int, y0: int, block: int) -> Tuple[float, float, float, float]:
    """
    Compute WGS84 bounds for the full chip from top-left anchor tile (x0,y0) and block size.
    """
    # top-left tile bounds
    tl_min_lon, tl_min_lat, tl_max_lon, tl_max_lat = tile_to_bounds(x0, y0, z)
    # bottom-right tile bounds
    br_min_lon, br_min_lat, br_max_lon, br_max_lat = tile_to_bounds(x0 + block - 1, y0 + block - 1, z)

    # chip min_lon from top-left min_lon; chip max_lon from bottom-right max_lon
    min_lon = tl_min_lon
    max_lon = br_max_lon
    # chip max_lat from top-left max_lat; chip min_lat from bottom-right min_lat
    max_lat = tl_max_lat
    min_lat = br_min_lat
    return min_lon, min_lat, max_lon, max_lat


# -------------------------
# Main
# -------------------------
def iter_tile_coords(x_min: int, x_max: int, y_min: int, y_max: int) -> Iterable[Tuple[int, int]]:
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            yield x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch Esri XYZ tiles into larger chips.")
    parser.add_argument("--z", type=int, required=True, help="Zoom level (e.g., 18, 19, 20)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bbox", type=str, help="min_lon,min_lat,max_lon,max_lat (EPSG:4326)")
    group.add_argument("--center", type=str, help="lon,lat (EPSG:4326)")

    parser.add_argument("--radius-tiles", type=int, default=None, help="Radius in tiles around center (required with --center)")

    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for stitched chips")
    parser.add_argument("--cache-dir", type=str, required=True, help="Cache directory for 256x256 tiles")
    parser.add_argument("--chip-size", type=int, default=1024, help="Chip size in pixels (default 1024)")
    parser.add_argument("--stride-tiles", type=int, default=None, help="Anchor stride in tiles (default: block size, i.e. non-overlapping)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for output chips")
    parser.add_argument("--max-workers", type=int, default=16, help="Max concurrent download workers")
    parser.add_argument("--rate-limit", type=float, default=8.0, help="Max requests per second (0 to disable)")

    parser.add_argument("--min-tile-bytes", type=int, default=0, help="Reject tiles smaller than this size in bytes (0 disables)")
    parser.add_argument("--min-tile-variance", type=float, default=0.0, help="Reject near-blank tiles by grayscale variance threshold (0 disables)")

    parser.add_argument("--manifest-csv", type=str, default=None, help="Optional CSV manifest path")

    args = parser.parse_args()

    if args.z < 0 or args.z > 23:
        raise SystemExit("--z must be within a reasonable range (0-23)")

    if args.center and args.radius_tiles is None:
        raise SystemExit("--radius-tiles is required with --center")
    if args.radius_tiles is not None and args.radius_tiles < 0:
        raise SystemExit("--radius-tiles must be >= 0")

    if args.chip_size % TILE_SIZE != 0:
        raise SystemExit(f"--chip-size must be a multiple of {TILE_SIZE}")
    block = args.chip_size // TILE_SIZE
    if block <= 0:
        raise SystemExit("Invalid chip size (block <= 0)")

    stride = args.stride_tiles if args.stride_tiles is not None else block
    if stride <= 0:
        raise SystemExit("--stride-tiles must be >= 1")

    if args.center:
        lon, lat = parse_center(args.center)
        x_min, x_max, y_min, y_max = center_to_tile_range(lon, lat, args.radius_tiles, args.z)
    else:
        min_lon, min_lat, max_lon, max_lat = parse_bbox(args.bbox)
        x_min, x_max, y_min, y_max = bbox_to_tile_range(min_lon, min_lat, max_lon, max_lat, args.z)

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    coords = list(iter_tile_coords(x_min, x_max, y_min, y_max))
    total_tiles = len(coords)

    rate_limiter = RateLimiter(args.rate_limit)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "esri-tile-stitcher/1.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
    )

    # Download tiles (or load from cache)
    tile_paths: Dict[Tuple[int, int], Path] = {}
    cache_hits = 0
    downloaded = 0
    failed = 0

    # First, attempt cache fast-path (single-thread)
    to_fetch: List[Tuple[int, int]] = []
    for x, y in coords:
        cached = find_cached_tile(cache_dir, args.z, x, y)
        if cached and validate_tile_file(cached, min_bytes=args.min_tile_bytes, min_variance=args.min_tile_variance):
            tile_paths[(x, y)] = cached
            cache_hits += 1
        else:
            if cached:
                try:
                    cached.unlink()
                except Exception:
                    pass
            to_fetch.append((x, y))

    # Fetch missing concurrently
    if to_fetch:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [
                ex.submit(
                    download_tile,
                    session,
                    args.z,
                    x,
                    y,
                    cache_dir,
                    rate_limiter,
                    args.min_tile_bytes,
                    args.min_tile_variance,
                )
                for x, y in to_fetch
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if res.ok and res.path:
                    tile_paths[(res.x, res.y)] = res.path
                    downloaded += 1
                else:
                    failed += 1
                    if res.err:
                        print(f"Failed tile z={args.z} x={res.x} y={res.y}: {res.err}", file=sys.stderr)

    # Stitch chips
    anchors_skipped = 0
    chips_written = 0

    manifest_writer = None
    manifest_file = None
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_path.open("w", newline="", encoding="utf-8")
        manifest_writer = csv.writer(manifest_file)
        manifest_writer.writerow(["filename", "min_lon", "min_lat", "max_lon", "max_lat", "z", "x0", "y0", "block", "stride"])

    anchor_x_max = x_max - block + 1
    anchor_y_max = y_max - block + 1

    if anchor_x_max >= x_min and anchor_y_max >= y_min:
        for y0 in range(y_min, anchor_y_max + 1, stride):
            for x0 in range(x_min, anchor_x_max + 1, stride):
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

                chip_name = f"chip_z{args.z}_x{x0}_y{y0}_b{block}.jpg"
                chip_path = out_dir / chip_name
                ok = build_chip(args.z, x0, y0, block, tile_paths, chip_path, args.jpeg_quality)
                if ok:
                    chips_written += 1
                    if manifest_writer:
                        min_lon, min_lat, max_lon, max_lat = chip_bounds_from_anchor(args.z, x0, y0, block)
                        manifest_writer.writerow([chip_name, min_lon, min_lat, max_lon, max_lat, args.z, x0, y0, block, stride])
                else:
                    anchors_skipped += 1
    else:
        anchors_skipped = max(0, (len(range(y_min, y_max + 1, stride)) * len(range(x_min, x_max + 1, stride))))

    if manifest_file:
        manifest_file.close()

    print("Summary")
    print(f"zoom: {args.z}")
    print(f"tile_range: x={x_min}..{x_max} y={y_min}..{y_max} (tiles={total_tiles})")
    print(f"chip_size: {args.chip_size} (block={block} tiles) stride_tiles: {stride}")
    print(f"tiles_downloaded: {downloaded}")
    print(f"cache_hits: {cache_hits}")
    print(f"tiles_failed: {failed}")
    print(f"chips_written: {chips_written}")
    print(f"anchors_skipped: {anchors_skipped}")


if __name__ == "__main__":
    main()
