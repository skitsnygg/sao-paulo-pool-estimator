from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from src.utils.config import load_config, project_root
from src.utils.geo import TileSpec, tiles_from_bbox
from src.utils.io import read_json


def _tile_center(tile: TileSpec) -> tuple[float, float]:
    import mercantile

    bounds = mercantile.bounds(tile.x, tile.y, tile.z)
    center_lon = (bounds.west + bounds.east) / 2.0
    center_lat = (bounds.south + bounds.north) / 2.0
    return center_lon, center_lat


def assign_districts(config_path: str, districts_path: Path, max_tiles: int | None = None) -> Path:
    config = load_config(config_path)
    bbox = config["aoi"]["bbox"]
    zoom = config["tiles"]["zoom"]

    districts = read_json(districts_path)
    geometries = []
    names = []
    for feature in districts.get("features", []):
        geom = shape(feature.get("geometry"))
        if geom.is_empty:
            continue
        name = feature.get("properties", {}).get("name") or feature.get("properties", {}).get("district")
        if not name:
            name = f"district_{len(names)}"
        geometries.append(geom)
        names.append(name)

    if not geometries:
        raise ValueError("No district geometries found.")

    index = STRtree(geometries)
    geom_id_to_name = {id(geom): name for geom, name in zip(geometries, names)}

    tiles = tiles_from_bbox(bbox, zoom)
    if max_tiles and max_tiles > 0 and len(tiles) > max_tiles:
        tiles = tiles[:max_tiles]

    rows = []
    for tile in tiles:
        lon, lat = _tile_center(tile)
        point = Point(lon, lat)
        match_name = None
        candidates = index.query(point)
        for candidate in candidates:
            geom = geometries[candidate] if isinstance(candidate, (int, np.integer)) else candidate
            if geom.contains(point):
                match_name = geom_id_to_name.get(id(geom))
                break
        rows.append({"tile_id": tile.id, "district": match_name})

    output_path = project_root() / "data" / "processed" / "tile_districts.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign tiles to district polygons.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--districts", default="data/external/districts.geojson")
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()

    output_path = assign_districts(args.config, Path(args.districts), args.max_tiles)
    print(f"Saved tile districts to {output_path}")


if __name__ == "__main__":
    main()
