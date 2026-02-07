from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Iterable
from xml.etree import ElementTree as ET

import requests
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

from src.data.geosampa_config import ORTHO_LAYER, SUPPORTED_CRS, WMS_BASE_URL


DEFAULT_TIMEOUT = 60
DEFAULT_CANDIDATE_WMS = [
    "https://raster.geosampa.prefeitura.sp.gov.br/geoserver/wms",
    "https://raster.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms",
    "https://wms.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms",
    "https://wms.geosampa.prefeitura.sp.gov.br/geoserver/wms",
]


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


def bbox_from_geojson(path: Path, source_crs: str, target_crs: str) -> tuple[float, float, float, float]:
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


def get_capabilities(base_url: str, timeout: int = DEFAULT_TIMEOUT) -> ET.Element:
    params = {"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"}
    response = requests.get(base_url, params=params, timeout=timeout)
    response.raise_for_status()
    return ET.fromstring(response.content)


def discover_ortho_layer(base_urls: list[str]) -> tuple[str, str, list[str]]:
    keywords = ["orto", "ortho", "ortofoto", "mosaico", "mosaic", "rgb", "2020", "2017", "10cm", "20cm"]
    best = None
    for base in base_urls:
        try:
            root = get_capabilities(base)
        except requests.RequestException:
            continue

        for layer in root.findall(".//{*}Layer"):
            name_el = layer.find("{*}Name")
            if name_el is None or not name_el.text:
                continue
            title_el = layer.find("{*}Title")
            name = name_el.text.strip()
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            hay = f"{name} {title}".lower()
            score = sum(1 for key in keywords if key in hay)
            if score == 0:
                continue
            crs_list = [crs.text.strip() for crs in layer.findall("{*}CRS") if crs.text]
            crs_list = sorted(set(crs_list))
            if "EPSG:31983" in crs_list:
                score += 3
            if "2020" in hay:
                score += 2
            if best is None or score > best[0]:
                best = (score, base, name, crs_list)

    if best is None:
        raise RuntimeError("Unable to discover an orthophoto layer from the candidate WMS endpoints.")
    _, base, name, crs_list = best
    return base, name, crs_list


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


def _parse_chip_id(name: str) -> tuple[int, int] | None:
    if not name.startswith("r") or "_c" not in name:
        return None
    parts = name.replace(".png", "").split("_")
    if len(parts) != 2:
        return None
    try:
        row = int(parts[0][1:])
        col = int(parts[1][1:])
    except ValueError:
        return None
    return row, col


def rebuild_metadata(
    out_dir: Path,
    bbox: tuple[float, float, float, float],
    crs: str,
    chip_size: int,
    meters_per_pixel: float,
) -> Path:
    minx, miny, _, _ = bbox
    chip_m = chip_size * meters_per_pixel
    metadata_path = out_dir / "chips.csv"
    rows = []
    for path in sorted(out_dir.glob("*.png")):
        parsed = _parse_chip_id(path.name)
        if parsed is None:
            continue
        row, col = parsed
        x0 = minx + col * chip_m
        y0 = miny + row * chip_m
        x1 = x0 + chip_m
        y1 = y0 + chip_m
        rows.append(
            {
                "chip_id": path.stem,
                "xmin": f"{x0:.3f}",
                "ymin": f"{y0:.3f}",
                "xmax": f"{x1:.3f}",
                "ymax": f"{y1:.3f}",
                "crs": crs,
                "width_px": chip_size,
                "height_px": chip_size,
                "path": str(path),
                "status": "cached",
                "http_status": "",
                "content_type": "",
            }
        )

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
        writer.writerows(rows)
    return metadata_path


def download_chips(
    wms_url: str,
    layer: str,
    bbox: tuple[float, float, float, float],
    out_dir: Path,
    crs: str,
    chip_size: int,
    meters_per_pixel: float,
    timeout: int,
    sleep_s: float,
    max_chips: int | None,
    overwrite: bool,
) -> Path:
    minx, miny, maxx, maxy = bbox
    chip_m = chip_size * meters_per_pixel
    cols = math.ceil((maxx - minx) / chip_m)
    rows = math.ceil((maxy - miny) / chip_m)

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "chips.csv"

    total = rows * cols
    downloaded = 0
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
                    "format": "image/png",
                    "transparent": "false",
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
                        _write_world_file(filename, x0, y0, x1, y1, chip_size, chip_size)
                        downloaded += 1
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

    print(f"Total chips: {total} | downloaded: {downloaded} | cached: {skipped} | errors: {errors}")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GeoSampa orthophoto chips for the Jardins AOI.")
    parser.add_argument("--aoi-geojson", default="data/external/jardins_union.geojson")
    parser.add_argument("--aoi-crs", default="EPSG:4326")
    parser.add_argument("--crs", default="EPSG:31983")
    parser.add_argument("--chip-size", type=int, default=1024)
    parser.add_argument("--meters-per-pixel", type=float, default=0.10)
    parser.add_argument("--out-dir", default="data/raw/geosampa_ortho/jardins_2020")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-chips", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--wms", default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--rebuild-metadata", action="store_true")
    args = parser.parse_args()

    if args.discover:
        base_url, layer, crs_list = discover_ortho_layer(DEFAULT_CANDIDATE_WMS)
    else:
        base_url = args.wms or WMS_BASE_URL
        layer = args.layer or ORTHO_LAYER
        crs_list = SUPPORTED_CRS

    if args.crs not in crs_list:
        raise ValueError(f"CRS {args.crs} is not supported by layer {layer}. Supported: {crs_list}")

    bbox = bbox_from_geojson(Path(args.aoi_geojson), args.aoi_crs, args.crs)
    out_dir = Path(args.out_dir)
    if args.rebuild_metadata:
        metadata_path = rebuild_metadata(out_dir, bbox, args.crs, args.chip_size, args.meters_per_pixel)
        print(f"Rebuilt metadata at {metadata_path}")
    else:
        download_chips(
            wms_url=base_url,
            layer=layer,
            bbox=bbox,
            out_dir=out_dir,
            crs=args.crs,
            chip_size=args.chip_size,
            meters_per_pixel=args.meters_per_pixel,
            timeout=args.timeout,
            sleep_s=args.sleep,
            max_chips=args.max_chips,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
