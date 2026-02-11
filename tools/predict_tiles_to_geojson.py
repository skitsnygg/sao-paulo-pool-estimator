from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ultralytics import YOLO


def mercator_lat_from_t(t: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * t))))


def lonlat_from_xyz_pixel(z: int, x: int, y: int, px: float, py: float, w: int, h: int):
    n = 2 ** z
    fx = x + (px / w)
    fy = y + (py / h)
    lon = (fx / n) * 360.0 - 180.0
    lat = mercator_lat_from_t(fy / n)
    return lon, lat


def parse_xy_from_path(p: Path) -> tuple[int, int]:
    stem = p.stem
    a, b = stem.split("_")
    return int(a), int(b)


def poly_area_px(poly_xy):
    # shoelace formula in pixel space
    if not poly_xy or len(poly_xy) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly_xy)):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % len(poly_xy)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tiles-dir", required=True, type=Path)
    ap.add_argument("--z", required=True, type=int)
    ap.add_argument("--out-geojson", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-area-px", type=float, default=200.0)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all tiles, else limit for testing")
    args = ap.parse_args()

    from PIL import Image

    tiles = sorted([p for p in args.tiles_dir.glob("*.jpg")])
    if args.max_tiles and args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]

    model = YOLO(args.model)

    features = []
    for p in tiles:
        x, y = parse_xy_from_path(p)
        w, h = Image.open(p).size

        results = model.predict(
            source=str(p),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        r = results[0]
        if r.masks is None or r.masks.xy is None:
            continue

        # r.masks.xy: list of (N,2) float arrays in pixel coords
        for poly in r.masks.xy:
            poly = [(float(px), float(py)) for px, py in poly]
            if poly_area_px(poly) < args.min_area_px:
                continue

            ring_ll = []
            for px, py in poly:
                lon, lat = lonlat_from_xyz_pixel(args.z, x, y, px, py, w, h)
                ring_ll.append([lon, lat])
            # close ring
            if ring_ll and ring_ll[0] != ring_ll[-1]:
                ring_ll.append(ring_ll[0])

            features.append({
                "type": "Feature",
                "properties": {
                    "tile": p.name,
                    "z": args.z,
                    "x": x,
                    "y": y,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring_ll],
                }
            })

    out = {"type": "FeatureCollection", "features": features}
    args.out_geojson.parent.mkdir(parents=True, exist_ok=True)
    args.out_geojson.write_text(json.dumps(out))
    print("Wrote:", args.out_geojson)
    print("Features:", len(features))


if __name__ == "__main__":
    main()
