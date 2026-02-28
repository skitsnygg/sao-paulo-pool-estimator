#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional, Tuple

from pyproj import Transformer


ESRI_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def pick_geotiff(geotiff: Optional[Path], tiles_dir: Optional[Path]) -> Path:
    if geotiff:
        if not geotiff.exists():
            raise SystemExit(f"GeoTIFF not found: {geotiff}")
        return geotiff

    if not tiles_dir:
        raise SystemExit("Provide --geotiff or --tiles-dir")

    candidates = sorted([p for p in tiles_dir.rglob("*.tif") if p.is_file()])
    if not candidates:
        raise SystemExit(f"No .tif files found under {tiles_dir}")
    return candidates[0]


def transform_bounds(bounds: Tuple[float, float, float, float], src_crs: str, dst_crs: str) -> Tuple[float, float, float, float]:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny, maxx, maxy = bounds
    corners = [
        transformer.transform(minx, miny),
        transformer.transform(minx, maxy),
        transformer.transform(maxx, miny),
        transformer.transform(maxx, maxy),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def build_qgs(
    *,
    geotiff_path: Path,
    qgs_path: Path,
    xyz_url: str,
    project_crs: str,
) -> None:
    try:
        from xml.sax.saxutils import escape
    except Exception as exc:
        raise SystemExit("xml.sax.saxutils.escape is required to write the QGS") from exc

    raster_id = "raster_" + hashlib.md5(str(geotiff_path).encode("utf-8")).hexdigest()[:12]
    xyz_id = "xyz_" + hashlib.md5(xyz_url.encode("utf-8")).hexdigest()[:12]

    raster_name = geotiff_path.stem
    datasource_raster = escape(str(geotiff_path.resolve()))
    datasource_xyz = escape(f"type=xyz&url={xyz_url}&zmin=0&zmax=23")

    qgs = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<qgis projectname=\"esri_georef_check\" version=\"3.28.0\">
  <title>esri_georef_check</title>
  <projectCrs>
    <spatialrefsys>
      <authid>{project_crs}</authid>
      <description>WGS 84 / Pseudo-Mercator</description>
      <projectionacronym>merc</projectionacronym>
      <ellipsoidacronym>WGS84</ellipsoidacronym>
      <geographicflag>false</geographicflag>
    </spatialrefsys>
  </projectCrs>
  <layer-tree-group name=\"Layers\" expanded=\"1\">
    <layer-tree-layer id=\"{raster_id}\" name=\"{raster_name}\" checked=\"Qt::Checked\"/>
    <layer-tree-layer id=\"{xyz_id}\" name=\"Esri World Imagery\" checked=\"Qt::Checked\"/>
  </layer-tree-group>
  <maplayers>
    <maplayer type=\"raster\" name=\"{raster_name}\" id=\"{raster_id}\">
      <datasource>{datasource_raster}</datasource>
      <layername>{raster_name}</layername>
      <provider>gdal</provider>
    </maplayer>
    <maplayer type=\"raster\" name=\"Esri World Imagery\" id=\"{xyz_id}\">
      <datasource>{datasource_xyz}</datasource>
      <layername>Esri World Imagery</layername>
      <provider>wms</provider>
    </maplayer>
  </maplayers>
  <layerorder>
    <layer id=\"{xyz_id}\"/>
    <layer id=\"{raster_id}\"/>
  </layerorder>
</qgis>
"""

    qgs_path.parent.mkdir(parents=True, exist_ok=True)
    qgs_path.write_text(qgs, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick georef sanity check for Esri GeoTIFF tiles.")
    ap.add_argument("--geotiff", type=Path, default=None, help="Path to a GeoTIFF tile")
    ap.add_argument("--tiles-dir", type=Path, default=None, help="Directory containing GeoTIFF tiles")
    ap.add_argument("--qgs-out", type=Path, default=None, help="Optional output QGIS project (.qgs)")
    ap.add_argument("--xyz-url", type=str, default=ESRI_URL, help="XYZ URL template for basemap")
    ap.add_argument("--project-crs", type=str, default="EPSG:3857", help="QGIS project CRS (default EPSG:3857)")
    args = ap.parse_args()

    geotiff = pick_geotiff(args.geotiff, args.tiles_dir)

    try:
        import rasterio
    except Exception as exc:
        raise SystemExit("rasterio is required to read GeoTIFF bounds") from exc

    with rasterio.open(geotiff) as src:
        if src.crs is None:
            raise SystemExit(f"GeoTIFF has no CRS: {geotiff}")
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        src_crs = src.crs.to_string()

    bounds_3857 = bounds if src_crs == "EPSG:3857" else transform_bounds(bounds, src_crs, "EPSG:3857")
    bounds_4326 = transform_bounds(bounds_3857, "EPSG:3857", "EPSG:4326")

    print(f"GeoTIFF: {geotiff}")
    print(f"CRS: {src_crs}")
    print(f"Bounds EPSG:3857: {bounds_3857}")
    print(f"Bounds EPSG:4326: {bounds_4326}")

    if args.qgs_out:
        build_qgs(
            geotiff_path=geotiff,
            qgs_path=args.qgs_out,
            xyz_url=args.xyz_url,
            project_crs=args.project_crs,
        )
        print(f"Wrote QGIS project: {args.qgs_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
