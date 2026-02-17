#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


def main() -> int:
    ap = argparse.ArgumentParser(description="Build grid AOI GeoJSON cells covering a boundary (expects EPSG:31983).")
    ap.add_argument("--boundary", required=True, help="Boundary GeoJSON (EPSG:31983)")
    ap.add_argument("--out-dir", required=True, help="Output dir for AOI cell GeoJSONs")
    ap.add_argument("--cell-m", type=float, default=2000.0, help="Cell size in meters (default 2000)")
    ap.add_argument("--min-area-m2", type=float, default=1.0, help="Skip tiny slivers below this area")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.boundary)
    if gdf.crs is None:
        raise SystemExit("Boundary has no CRS; expected EPSG:31983.")
    gdf = gdf.to_crs("EPSG:31983")
    geom = gdf.geometry.unary_union

    minx, miny, maxx, maxy = geom.bounds
    cell = float(args.cell_m)

    written = 0
    ix = 0
    y = miny
    while y < maxy:
        x = minx
        jx = 0
        while x < maxx:
            cell_poly = box(x, y, x + cell, y + cell)
            inter = cell_poly.intersection(geom)
            if (not inter.is_empty) and (inter.area >= float(args.min_area_m2)):
                out = out_dir / f"cell_{ix:04d}_{jx:04d}.geojson"
                gpd.GeoDataFrame({"cell_id": [out.stem]}, geometry=[inter], crs="EPSG:31983").to_file(out, driver="GeoJSON")
                written += 1
            x += cell
            jx += 1
        y += cell
        ix += 1

    print("wrote cells:", written)
    print("out_dir:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
