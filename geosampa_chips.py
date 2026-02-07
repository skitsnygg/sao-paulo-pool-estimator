#!/usr/bin/env python3
"""
GeoSampa WMS chip downloader (for training a pool detector).

Usage examples:

1) List layers:
   python geosampa_chips.py --wms https://wms.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms --list-layers

2) Download chips over a bbox (EPSG:31983 meters):
   python geosampa_chips.py --wms https://wms.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms \
     --layer <LAYER_NAME> \
     --download \
     --bbox 330000 7395000 334000 7399000 \
     --chip-size 1024 \
     --meters-per-pixel 0.10 \
     --out-dir ./data/jardins_chips
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Iterable, List, Tuple
from xml.etree import ElementTree as ET

import requests
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer


DEFAULT_WMS = "https://wms.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms"
DEFAULT_TIMEOUT = 60


def wms_getcapabilities(wms_url: str, timeout: int) -> bytes:
    params = {"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"}
    response = requests.get(wms_url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_wms_layers(xml_bytes: bytes) -> List[Tuple[str, str]]:
    root = ET.fromstring(xml_bytes)
    layers: List[Tuple[str, str]] = []
    seen = set()
    for layer in root.findall(".//{*}Layer"):
        name_el = layer.find("{*}Name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        if name in seen:
            continue
        title_el = layer.find("{*}Title")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        layers.append((name, title))
        seen.add(name)
    return layers


def _iter_geoms(geojson_payload: dict) -> Iterable:
    gtype = geojson_payload.get("type")
    if gtype == "FeatureCollection":
        for feature in geojson_payload.get("features", []):
            geom = feature.get("geometry")
            if geom:
                yield shape(geom)
    elif gtype == "Feature":
        geom = geojson_payload.get("geometry")
        if geom:
            yield shape(geom)
    else:
        yield shape(geojson_payload)


def bbox_from_geojson(path: Path, source_crs: str, target_crs: str) -> Tuple[float, float, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    geoms = [geom for geom in _iter_geoms(payload) if not geom.is_empty]
    if not geoms:
        raise ValueError("No geometries found in GeoJSON.")

    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geoms = [transform(transformer.transform, geom) for geom in geoms]

    minx = min(geom.bounds[0] for geom in geoms)
    miny = min(geom.bounds[1] for geom in geoms)
    maxx = max(geom.bounds[2] for geom in geoms)
    maxy = max(geom.bounds[3] for geom in geoms)
    return minx, miny, maxx, maxy


def _write_world_file(path: Path, minx: float, miny: float, maxx: float, maxy: float, width: int, height: int) -> None:
    pixel_size_x = (maxx - minx) / float(width)
    pixel_size_y = (maxy - miny) / float(height)
    x_center = minx + pixel_size_x / 2.0
    y_center = maxy - pixel_size_y / 2.0
    world_path = path.with_suffix(".pgw")
    world_path.write_text(
        "\n".join(
            [
                f"{pixel_size_x:.10f}",
                "0.0",
                "0.0",
                f"{-pixel_size_y:.10f}",
                f"{x_center:.10f}",
                f"{y_center:.10f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def download_chips(
    wms_url: str,
    layer: str,
    bbox: Tuple[float, float, float, float],
    out_dir: Path,
    crs: str,
    chip_size: int,
    meters_per_pixel: float,
    fmt: str,
    timeout: int,
    sleep_s: float,
    max_chips: int | None,
    overwrite: bool,
    write_worldfile: bool,
) -> Path:
    minx, miny, maxx, maxy = bbox
    chip_m = chip_size * meters_per_pixel
    cols = math.ceil((maxx - minx) / chip_m)
    rows = math.ceil((maxy - miny) / chip_m)

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "chips.csv"

    total = rows * cols
    written = 0
    skipped = 0
    errors = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chip_id",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "crs",
                "width_px",
                "height_px",
                "path",
                "status",
                "http_status",
                "content_type",
            ],
        )
        writer.writeheader()

        index = 0
        for row in range(rows):
            for col in range(cols):
                if max_chips is not None and index >= max_chips:
                    break
                x0 = minx + col * chip_m
                y0 = miny + row * chip_m
                x1 = x0 + chip_m
                y1 = y0 + chip_m

                chip_id = f"r{row:04d}_c{col:04d}"
                filename = out_dir / f"{chip_id}.png"

                if filename.exists() and not overwrite:
                    status = "cached"
                    skipped += 1
                    writer.writerow(
                        {
                            "chip_id": chip_id,
                            "xmin": f"{x0:.3f}",
                            "ymin": f"{y0:.3f}",
                            "xmax": f"{x1:.3f}",
                            "ymax": f"{y1:.3f}",
                            "crs": crs,
                            "width_px": chip_size,
                            "height_px": chip_size,
                            "path": str(filename),
                            "status": status,
                            "http_status": "",
                            "content_type": "",
                        }
                    )
                    index += 1
                    continue

                params = {
                    "service": "WMS",
                    "request": "GetMap",
                    "version": "1.3.0",
                    "layers": layer,
                    "styles": "",
                    "crs": crs,
                    "bbox": f"{x0},{y0},{x1},{y1}",
                    "width": chip_size,
                    "height": chip_size,
                    "format": fmt,
                    "transparent": "true",
                }

                status = "downloaded"
                http_status = ""
                content_type = ""
                try:
                    response = requests.get(wms_url, params=params, timeout=timeout)
                    http_status = str(response.status_code)
                    content_type = response.headers.get("Content-Type", "")
                    if response.status_code != 200 or not content_type.startswith("image"):
                        status = "error"
                        errors += 1
                    else:
                        filename.write_bytes(response.content)
                        if write_worldfile:
                            _write_world_file(filename, x0, y0, x1, y1, chip_size, chip_size)
                        written += 1
                except requests.RequestException:
                    status = "error"
                    errors += 1

                writer.writerow(
                    {
                        "chip_id": chip_id,
                        "xmin": f"{x0:.3f}",
                        "ymin": f"{y0:.3f}",
                        "xmax": f"{x1:.3f}",
                        "ymax": f"{y1:.3f}",
                        "crs": crs,
                        "width_px": chip_size,
                        "height_px": chip_size,
                        "path": str(filename),
                        "status": status,
                        "http_status": http_status,
                        "content_type": content_type,
                    }
                )

                index += 1
                if sleep_s > 0:
                    time.sleep(sleep_s)
            if max_chips is not None and index >= max_chips:
                break

    print(f"Total chips: {total} | downloaded: {written} | cached: {skipped} | errors: {errors}")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoSampa WMS chip downloader.")
    parser.add_argument("--wms", default=DEFAULT_WMS)
    parser.add_argument("--list-layers", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    parser.add_argument("--aoi-geojson", default=None)
    parser.add_argument("--aoi-crs", default="EPSG:4326")
    parser.add_argument("--crs", default="EPSG:31983")
    parser.add_argument("--chip-size", type=int, default=1024)
    parser.add_argument("--meters-per-pixel", type=float, default=0.10)
    parser.add_argument("--format", default="image/png")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-chips", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-worldfile", action="store_true")
    args = parser.parse_args()

    if args.list_layers:
        xml_bytes = wms_getcapabilities(args.wms, args.timeout)
        layers = parse_wms_layers(xml_bytes)
        for name, title in layers:
            print(f"{name}\t{title}")
        return

    if not args.download:
        raise SystemExit("Use --list-layers or --download.")
    if not args.layer:
        raise SystemExit("--layer is required for downloads.")
    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.aoi_geojson:
        bbox = bbox_from_geojson(Path(args.aoi_geojson), args.aoi_crs, args.crs)
    else:
        raise SystemExit("Provide --bbox or --aoi-geojson.")
    if not args.out_dir:
        raise SystemExit("--out-dir is required for downloads.")

    out_dir = Path(args.out_dir)
    download_chips(
        wms_url=args.wms,
        layer=args.layer,
        bbox=bbox,
        out_dir=out_dir,
        crs=args.crs,
        chip_size=args.chip_size,
        meters_per_pixel=args.meters_per_pixel,
        fmt=args.format,
        timeout=args.timeout,
        sleep_s=args.sleep,
        max_chips=args.max_chips,
        overwrite=args.overwrite,
        write_worldfile=not args.no_worldfile,
    )


if __name__ == "__main__":
    main()
