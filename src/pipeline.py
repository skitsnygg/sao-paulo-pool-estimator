from __future__ import annotations

import argparse

from src.data.download_tiles import download_tiles
from src.data.fetch_osm import fetch_osm_pools
from src.data.make_dataset import make_dataset
from src.models.train import train_model
from src.models.baseline import run_baseline
from src.models.predict import run_inference
from src.analysis.estimate import estimate
from src.visualization.map import build_map


def run_pipeline(config_path: str, step: str) -> None:
    if step in ("download", "all"):
        download_tiles(config_path)
    if step in ("osm", "all"):
        fetch_osm_pools(config_path)
    if step in ("dataset", "all"):
        make_dataset(config_path)
    if step in ("train", "all"):
        train_model(config_path)
    if step in ("predict", "all"):
        run_inference(config_path, weights=None, source=None)
    if step in ("estimate", "all"):
        estimate(config_path, predictions_path=None, strata_path=None)
    if step in ("map", "all"):
        build_map(config_path, predictions_path=None)
    if step == "baseline":
        predictions_path = run_baseline(config_path, source=None, threshold=None)
        estimate(config_path, predictions_path=predictions_path, strata_path=None)
        build_map(config_path, predictions_path=predictions_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--step",
        default="all",
        choices=["download", "osm", "dataset", "train", "predict", "estimate", "map", "baseline", "all"],
    )
    args = parser.parse_args()

    run_pipeline(args.config, args.step)


if __name__ == "__main__":
    main()
