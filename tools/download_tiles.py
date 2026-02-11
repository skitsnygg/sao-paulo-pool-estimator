#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def deg2num(lon_deg: float, lat_deg: float, zoom: int):
    """Convert lon/lat to XYZ tile indices."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(x: int, y: int, zoom: int):
    """Convert XYZ tile indices to lon/lat of NW corner."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lon_deg, lat_deg

def tile_bounds_lonlat(x: int, y: int, zoom: int):
    """Return (min_lon, min_lat, max_lon, max_lat) for a tile."""
    west, north = num2deg(x, y, zoom)
    east, south = num2deg(x + 1, y + 1, zoom)
    return (west, south, east, north)

def load_polygon(geojson_path: str):
    fc = json.load(open(geojson_path, "r", encoding="utf-8"))
    geom = fc["features"][0]["geometry"]
    gtype = geom["type"]

    # Grab the outer ring of the first polygon.
    if gtype == "Polygon":
        ring = geom["coordinates"][0]
    elif gtype == "MultiPolygon":
        ring = geom["coordinates"][0][0]
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")

    # ring is [ [lon,lat], ... ]
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    bbox = (min(lons), min(lats), max(lons), max(lats))  # (minlon, minlat, maxlon, maxlat)
    return ring, bbox

def try_make_shapely_polygon(ring):
    try:
        from shapely.geometry import Polygon, box
        poly = Polygon(ring)
        return poly, box
    except Exception:
        return None, None

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download(url: str, out_path: str, user_agent: str, timeout: float, retries: int, sleep_s: float):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skip"

    ensure_dir(os.path.dirname(out_path))

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": user_agent})
            with urlopen(req, timeout=timeout) as r:
                data = r.read()
            with open(out_path, "wb") as f:
                f.write(data)
            return "ok"
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s * (1.0 + random.random()))
            else:
                return f"fail: {type(last_err).__name__}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True, help="Polygon GeoJSON in EPSG:4326 (lon/lat)")
    ap.add_argument("--out-dir", required=True, help="Where to save z/x/y.ext")
    ap.add_argument("--url-template", required=True, help="e.g. https://server/tiles/{z}/{x}/{y}.png")
    ap.add_argument("--ext", default="png", help="File extension (png/jpg/webp)")
    ap.add_argument("--zmin", type=int, required=True)
    ap.add_argument("--zmax", type=int, required=True)
    ap.add_argument("--sleep", type=float, default=0.05, help="Seconds between requests")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--user-agent", default="tile-downloader/1.0")
    ap.add_argument("--no-poly-filter", action="store_true", help="Download bbox tiles only (no polygon intersection)")
    args = ap.parse_args()

    ring, bbox = load_polygon(args.geojson)
    minlon, minlat, maxlon, maxlat = bbox
    print("bbox lon/lat:", bbox)

    poly, shapely_box = (None, None)
    if not args.no_poly_filter:
        poly, shapely_box = try_make_shapely_polygon(ring)
        if poly is None:
            print("Note: shapely not available; falling back to bbox-only. (pip install shapely)")
    use_poly = (poly is not None) and (not args.no_poly_filter)

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for z in range(args.zmin, args.zmax + 1):
        x0, y1 = deg2num(minlon, minlat, z)  # minlon/minlat is SW
        x1, y0 = deg2num(maxlon, maxlat, z)  # maxlon/maxlat is NE

        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])

        print(f"z={z} tiles x[{xmin}..{xmax}] y[{ymin}..{ymax}]")
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                if use_poly:
                    west, south, east, north = tile_bounds_lonlat(x, y, z)
                    tb = shapely_box(west, south, east, north)
                    if not poly.intersects(tb):
                        continue

                url = args.url_template.format(z=z, x=x, y=y)
                out_path = os.path.join(args.out_dir, str(z), str(x), f"{y}.{args.ext}")

                total += 1
                status = download(
                    url=url,
                    out_path=out_path,
                    user_agent=args.user_agent,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep_s=max(args.sleep, 0.0),
                )
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1

                if args.sleep > 0:
                    time.sleep(args.sleep)

    print(f"done. total={total} ok={ok} skipped={skipped} failed={failed}")
    if failed > 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
