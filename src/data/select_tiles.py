from __future__ import annotations

import argparse
from pathlib import Path
import random

import mercantile
import pandas as pd
from shapely.geometry import shape

from src.utils.config import load_config, project_root
from src.utils.geo import TileSpec, tile_polygon, tiles_from_bbox, sample_tiles
from src.utils.io import ensure_dir, read_json


def _load_pool_geometries(osm_path: Path):
    payload = read_json(osm_path)
    geometries = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        geom = shape(geometry)
        if geom.is_empty:
            continue
        geometries.append(geom)
    return geometries


def _positive_tiles(geometries, zoom: int) -> set[TileSpec]:
    positive_tiles: set[TileSpec] = set()
    for geom in geometries:
        minx, miny, maxx, maxy = geom.bounds
        for tile in mercantile.tiles(minx, miny, maxx, maxy, zoom):
            tile_spec = TileSpec(tile.z, tile.x, tile.y)
            if geom.intersects(tile_polygon(tile_spec)):
                positive_tiles.add(tile_spec)
    return positive_tiles


def _select_train_tiles(
    positive_tiles: set[TileSpec],
    all_tiles: list[TileSpec],
    negatives_per_positive: float,
    max_tiles: int | None,
    seed: int,
) -> list[TileSpec]:
    rng = random.Random(seed)
    positive_list = list(positive_tiles)
    rng.shuffle(positive_list)
    pos_count = len(positive_list)

    if max_tiles and max_tiles > 0 and pos_count > max_tiles:
        return positive_list[:max_tiles]

    positive_ids = {tile.id for tile in positive_list}
    negative_tiles = [tile for tile in all_tiles if tile.id not in positive_ids]
    rng.shuffle(negative_tiles)

    target_neg = int(pos_count * max(0.0, negatives_per_positive))
    if max_tiles and max_tiles > 0:
        target_neg = min(target_neg, max_tiles - pos_count)
    target_neg = max(0, min(target_neg, len(negative_tiles)))

    train_tiles = positive_list + negative_tiles[:target_neg]
    rng.shuffle(train_tiles)
    return train_tiles


def _write_tiles_csv(tiles: list[TileSpec], output_path: Path, kind: str) -> None:
    ensure_dir(output_path.parent)
    rows = [{"tile_id": tile.id, "z": tile.z, "x": tile.x, "y": tile.y, "kind": kind} for tile in tiles]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def select_tile_sets(config_path: str) -> tuple[Path, Path]:
    config = load_config(config_path)
    root = project_root()

    osm_path = root / "data" / "raw" / "osm_pools.geojson"
    if not osm_path.exists():
        raise FileNotFoundError(f"Missing OSM pools data: {osm_path}")

    bbox = config["aoi"]["bbox"]
    zoom = config["tiles"]["zoom"]
    all_tiles = tiles_from_bbox(bbox, zoom)

    training_cfg = config.get("training", {})
    negatives_per_positive = training_cfg.get("negatives_per_positive", 1.0)
    max_train_tiles = training_cfg.get("max_tiles")
    train_seed = training_cfg.get("random_seed", config.get("dataset", {}).get("random_seed", 42))

    sampling_cfg = config.get("sampling", {})
    sample_size = sampling_cfg.get("sample_size", 0)
    sample_seed = sampling_cfg.get("random_seed", 42)

    geometries = _load_pool_geometries(osm_path)
    positive_tiles = _positive_tiles(geometries, zoom)
    train_tiles = _select_train_tiles(positive_tiles, all_tiles, negatives_per_positive, max_train_tiles, train_seed)

    sample_tiles_list = sample_tiles(all_tiles, sample_size, seed=sample_seed) if sample_size else all_tiles

    output_dir = root / "data" / "interim" / "tiles"
    train_path = output_dir / "train_tiles.csv"
    sample_path = output_dir / "sample_tiles.csv"
    _write_tiles_csv(train_tiles, train_path, "train")
    _write_tiles_csv(sample_tiles_list, sample_path, "sample")
    return train_path, sample_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select training and estimation tile sets.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    train_path, sample_path = select_tile_sets(args.config)
    print(f"Saved training tiles to {train_path}")
    print(f"Saved sample tiles to {sample_path}")


if __name__ == "__main__":
    main()
