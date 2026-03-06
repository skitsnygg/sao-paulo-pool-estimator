#!/usr/bin/env python3
"""
Folium viewer for GeoSampa 2020 ortho tiles + predicted polygons.

Reproducible approach:
- Mosaic a folder of PNG tiles (with .pgw worldfiles) into a VRT
- Warp to EPSG:3857 (web mercator)
- Render to a PNG for fast browser overlay
- Compute bounds in EPSG:4326 for Leaflet
- Load predicted polygons (GeoJSON), reproject to EPSG:4326, and render on top

Outputs:
- runs/folium/<name>/overlay_3857.tif
- runs/folium/<name>/overlay_3857.png
- runs/folium/<name>/viewer.html
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Tuple

import folium
import geopandas as gpd
import rasterio
from folium.plugins import MousePosition
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_overlay_from_tiles(
    tiles_dir: Path,
    out_dir: Path,
    name: str,
    resampling: str = "bilinear",
) -> Tuple[Path, Path, Tuple[float, float, float, float]]:
    """
    Returns:
      (tif_3857_path, png_3857_path, bounds_4326=(west,south,east,north))
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vrt_path = out_dir / f"{name}.vrt"
    tif_path = out_dir / f"{name}_3857.tif"
    png_path = out_dir / f"{name}_3857.png"

    pngs = sorted(tiles_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNGs found in: {tiles_dir}")

    # 1) Build a VRT from tiles (absolute paths)
    run(["gdalbuildvrt", str(vrt_path), *[str(p) for p in pngs]])

    # 2) Warp VRT -> EPSG:3857 GeoTIFF (web mercator)
    # Use deterministic options (no random, no temp naming leakage beyond out_dir)
    run(
        [
            "gdalwarp",
            "-t_srs",
            "EPSG:3857",
            "-r",
            resampling,
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            str(vrt_path),
            str(tif_path),
        ]
    )

    # 3) Render to PNG for fast Leaflet overlay
    run(["gdal_translate", "-of", "PNG", str(tif_path), str(png_path)])

    # 4) Compute bounds in EPSG:4326 for Leaflet
    with rasterio.open(tif_path) as src:
        b = src.bounds
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
        )

    return tif_path, png_path, (west, south, east, north)


def add_geojson_layer(
    m: folium.Map,
    geojson_path: Path,
    layer_name: str = "Predictions",
) -> None:
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        print(f"WARN: GeoJSON has no features: {geojson_path}")
        return

    # If missing CRS, assume your pipeline output is EPSG:31983 or EPSG:4326?
    # Best practice: ensure your pipeline writes CRS explicitly.
    # We'll handle common cases:
    if gdf.crs is None:
        # If your prediction geojson is already EPSG:4326, set it here.
        # If it's EPSG:31983, set it accordingly. Change if needed.
        print("WARN: GeoJSON has no CRS. Assuming EPSG:4326. (Change if needed.)")
        gdf = gdf.set_crs("EPSG:4326")

    # Reproject to EPSG:4326 for Leaflet
    gdf4326 = gdf.to_crs("EPSG:4326")

    # Style: simple, readable
    def style_fn(_feature):
        return {
            "fillOpacity": 0.2,
            "weight": 2,
        }

    folium.GeoJson(
        data=json.loads(gdf4326.to_json()),
        name=layer_name,
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(fields=list(gdf4326.columns)[:1]),
    ).add_to(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True, help="Folder containing GeoSampa PNG tiles + PGW")
    ap.add_argument("--name", required=True, help="Name for output folder/layers")
    ap.add_argument("--out-dir", default="runs/folium", help="Base output directory")
    ap.add_argument("--pred-geojson", default="", help="Optional: path to prediction GeoJSON to overlay")
    ap.add_argument("--center-lat", type=float, default=None, help="Optional override map center lat")
    ap.add_argument("--center-lon", type=float, default=None, help="Optional override map center lon")
    ap.add_argument("--zoom", type=int, default=18, help="Initial zoom")
    ap.add_argument("--resampling", default="bilinear", help="gdalwarp resampling: nearest|bilinear|cubic...")
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    base_out = Path(args.out_dir) / args.name

    tif3857, png3857, (west, south, east, north) = build_overlay_from_tiles(
        tiles_dir=tiles_dir,
        out_dir=base_out,
        name=args.name,
        resampling=args.resampling,
    )

    # Choose center
    if args.center_lat is None or args.center_lon is None:
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2
    else:
        center_lat = args.center_lat
        center_lon = args.center_lon

    m = folium.Map(location=[center_lat, center_lon], zoom_start=args.zoom, tiles="OpenStreetMap")
    MousePosition(
        position="topright",
        separator=" | ",
        prefix="Lat/Lon:",
    ).add_to(m)

    # Add imagery overlay
    ImageOverlay(
        name=f"GeoSampa 2020 ({args.name})",
        image=str(png3857),
        bounds=[[south, west], [north, east]],
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Add predictions if provided
    if args.pred_geojson:
        add_geojson_layer(m, Path(args.pred_geojson), layer_name="Predicted pools")

    folium.LayerControl(collapsed=False).add_to(m)

    out_html = base_out / "viewer.html"
    m.save(str(out_html))

    meta = {
        "tiles_dir": str(tiles_dir),
        "tif_3857": str(tif3857),
        "png_3857": str(png3857),
        "bounds_4326": {"west": west, "south": south, "east": east, "north": north},
        "pred_geojson": args.pred_geojson,
    }
    (base_out / "viewer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(" ", out_html)
    print(" ", base_out / "viewer_meta.json")


if __name__ == "__main__":
    main()
