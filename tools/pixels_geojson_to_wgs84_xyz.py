#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mercator_lat_from_t(t: float) -> float:
    # t in [0,1] from top (0) to bottom (1)
    # Web Mercator inverse: lat = atan(sinh(pi*(1-2t)))
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * t))))


def lonlat_from_xyz_pixel(z: int, x: int, y: int, px: float, py: float, w: int, h: int) -> tuple[float, float]:
    n = 2 ** z
    # fractional tile coordinate in global tile space
    fx = x + (px / w)
    fy = y + (py / h)

    lon = (fx / n) * 360.0 - 180.0
    lat = mercator_lat_from_t(fy / n)
    return lon, lat


def parse_xy_from_name(name: str) -> tuple[int, int]:
    # expects "x_y.jpg" or "x_y.png"
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Filename does not look like x_y.*: {name}")
    return int(parts[0]), int(parts[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-geojson", required=True, type=Path, help="pixel-coordinate GeoJSON from merge_pred_polygons.py")
    ap.add_argument("--tiles-dir", required=True, type=Path, help="directory containing the tiles (e.g. data/raw/tiles/18)")
    ap.add_argument("--z", required=True, type=int, help="zoom level (e.g. 18)")
    ap.add_argument("--out-geojson", required=True, type=Path, help="output WGS84 GeoJSON")
    args = ap.parse_args()

    # lazy import to avoid forcing dependency unless used
    from PIL import Image

    fc = json.loads(args.in_geojson.read_text())
    out_features = []

    # cache tile sizes by image filename (avoid reopening repeatedly)
    size_cache: dict[str, tuple[int, int]] = {}

    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        img_name = props.get("image")
        if not img_name:
            continue

        try:
            x, y = parse_xy_from_name(img_name)
        except Exception:
            # skip anything not matching x_y.*
            continue

        img_path = args.tiles_dir / img_name
        if not img_path.exists():
            # if images were copied/renamed, we can't geo-reference
            continue

        if img_name not in size_cache:
            w, h = Image.open(img_path).size
            size_cache[img_name] = (w, h)
        else:
            w, h = size_cache[img_name]

        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype != "Polygon" or not coords:
            continue

        # coords: [ [ [px,py], [px,py], ... ] ]  (outer ring; we ignore holes for now)
        rings_ll = []
        for ring in coords:
            ring_ll = []
            for pt in ring:
                px, py = float(pt[0]), float(pt[1])
                lon, lat = lonlat_from_xyz_pixel(args.z, x, y, px, py, w, h)
                ring_ll.append([lon, lat])
            rings_ll.append(ring_ll)

        out_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Polygon",
                "coordinates": rings_ll
            }
        })

    out_fc = {"type": "FeatureCollection", "features": out_features}
    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(json.dumps(out_fc))
    print("Wrote", args.out_geojson)
    print("Features:", len(out_features))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
