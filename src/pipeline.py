from __future__ import annotations

import argparse

from src.data.download_tiles import download_tiles
from src.data.fetch_osm import fetch_osm_pools
from src.data.make_dataset import make_dataset
from src.data.select_tiles import select_tile_sets
from src.models.train import train_model
from src.models.baseline import run_baseline
from src.models.predict import run_inference
from src.analysis.estimate import estimate
from src.visualization.map import build_map
from src.utils.config import project_root


def run_pipeline(config_path: str, step: str) -> None:
    root = project_root()
    interim_tiles = root / "data" / "interim" / "tiles"
    raw_tiles = root / "data" / "raw" / "tiles"
    train_tiles_list = interim_tiles / "train_tiles.csv"
    sample_tiles_list = interim_tiles / "sample_tiles.csv"
    train_tiles_meta = raw_tiles / "train_tiles.csv"
    sample_tiles_meta = raw_tiles / "sample_tiles.csv"

    if step in ("osm", "all"):
        fetch_osm_pools(config_path)
    if step in ("tile-sets", "all"):
        select_tile_sets(config_path)
    if step in ("download", "all"):
        if train_tiles_list.exists():
            download_tiles(config_path, tiles_csv=str(train_tiles_list), output_csv=str(train_tiles_meta))
        else:
            download_tiles(config_path)
        if sample_tiles_list.exists():
            download_tiles(config_path, tiles_csv=str(sample_tiles_list), output_csv=str(sample_tiles_meta))
    if step in ("dataset", "all"):
        tiles_csv = str(train_tiles_meta) if train_tiles_meta.exists() else None
        make_dataset(config_path, tiles_csv=tiles_csv)
    if step in ("train", "all"):
        train_model(config_path)
    if step in ("predict", "all"):
        tiles_csv = str(sample_tiles_meta) if sample_tiles_meta.exists() else None
        run_inference(config_path, weights=None, source=None, tiles_csv=tiles_csv)
    if step in ("estimate", "all"):
        estimate(config_path, predictions_path=None, strata_path=None)
    if step in ("map", "all"):
        build_map(config_path, predictions_path=None)
    if step == "baseline":
        tiles_csv = str(sample_tiles_meta) if sample_tiles_meta.exists() else None
        predictions_path = run_baseline(config_path, source=None, threshold=None, tiles_csv=tiles_csv)
        estimate(config_path, predictions_path=predictions_path, strata_path=None)
        build_map(config_path, predictions_path=predictions_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--step",
        default="all",
        choices=["download", "osm", "tile-sets", "dataset", "train", "predict", "estimate", "map", "baseline", "all"],
    )
    args = parser.parse_args()

    run_pipeline(args.config, args.step)


if __name__ == "__main__":
    main()
