from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

from src.utils.config import load_config, project_root
from src.utils.geo import TileSpec, bbox_in_tile_pixels, tile_polygon, yolo_box_from_pixels
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


def _assign_splits(tiles: list[TileSpec], ratios: dict, seed: int) -> dict[str, list[TileSpec]]:
    rng = random.Random(seed)
    tiles = tiles[:]
    rng.shuffle(tiles)
    total = len(tiles)

    train_count = int(total * ratios["train"])
    val_count = int(total * ratios["val"])
    test_count = total - train_count - val_count

    splits = {
        "train": tiles[:train_count],
        "val": tiles[train_count:train_count + val_count],
        "test": tiles[train_count + val_count:train_count + val_count + test_count],
    }
    return splits


def make_dataset(config_path: str, min_box_area: int | None = None, tiles_csv: str | None = None) -> Path:
    config = load_config(config_path)
    root = project_root()

    tiles_csv_path = Path(tiles_csv) if tiles_csv else root / "data" / "raw" / "tiles" / "tiles.csv"
    if not tiles_csv_path.is_absolute():
        tiles_csv_path = root / tiles_csv_path
    if not tiles_csv_path.exists():
        raise FileNotFoundError(f"Missing tiles metadata: {tiles_csv_path}")

    osm_path = root / "data" / "raw" / "osm_pools.geojson"
    if not osm_path.exists():
        raise FileNotFoundError(f"Missing OSM pools data: {osm_path}")

    tiles_df = pd.read_csv(tiles_csv_path)
    tiles_df = tiles_df[tiles_df["status"].isin(["downloaded", "cached"])].copy()

    tiles = [TileSpec(int(row.z), int(row.x), int(row.y)) for row in tiles_df.itertuples()]
    path_by_id = dict(zip(tiles_df["tile_id"], tiles_df["path"]))

    ratios = {
        "train": config["dataset"]["train_ratio"],
        "val": config["dataset"]["val_ratio"],
        "test": config["dataset"]["test_ratio"],
    }
    seed = config["dataset"].get("random_seed", 42)
    splits = _assign_splits(tiles, ratios, seed)

    min_box_area = min_box_area or config["dataset"].get("min_box_area_px", 20)
    tile_size = config["tiles"]["size"]

    pools = _load_pool_geometries(osm_path)
    pool_index = STRtree(pools) if pools else None

    yolo_root = root / "data" / "processed" / "yolo"
    for split in splits:
        ensure_dir(yolo_root / "images" / split)
        ensure_dir(yolo_root / "labels" / split)

    rows = []
    for split, split_tiles in splits.items():
        for tile in split_tiles:
            tile_id = tile.id
            tile_poly = tile_polygon(tile)
            labels = []

            candidates = pool_index.query(tile_poly) if pool_index else []
            for candidate in candidates:
                geom = pools[candidate] if isinstance(candidate, (int, np.integer)) else candidate
                if not geom.intersects(tile_poly):
                    continue
                intersection = geom.intersection(tile_poly)
                if intersection.is_empty:
                    continue
                box_px = bbox_in_tile_pixels(intersection, tile, tile_size)
                if box_px is None:
                    continue
                x1, y1, x2, y2 = box_px
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue
                cx, cy, w, h = yolo_box_from_pixels(box_px, tile_size)
                labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            src_path = Path(path_by_id[tile_id])
            if not src_path.exists():
                continue

            dst_image = yolo_root / "images" / split / f"{tile_id}.jpg"
            dst_label = yolo_root / "labels" / split / f"{tile_id}.txt"
            shutil.copyfile(src_path, dst_image)

            ensure_dir(dst_label.parent)
            dst_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

            rows.append(
                {
                    "tile_id": tile_id,
                    "split": split,
                    "z": tile.z,
                    "x": tile.x,
                    "y": tile.y,
                    "label_count": len(labels),
                }
            )

    metadata_path = root / "data" / "processed" / "tiles.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)

    dataset_yaml = yolo_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: pool",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from tiles + OSM pools.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--min-box-area", type=int, default=None)
    parser.add_argument("--tiles-csv", default=None)
    args = parser.parse_args()

    dataset_yaml = make_dataset(args.config, args.min_box_area, args.tiles_csv)
    print(f"Saved dataset config to {dataset_yaml}")


if __name__ == "__main__":
    main()
