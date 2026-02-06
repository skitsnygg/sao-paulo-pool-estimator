from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.utils.config import load_config, project_root
from src.utils.geo import tiles_from_bbox


def _bootstrap_total(counts: np.ndarray, total_tiles: int, iters: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iters):
        sample = rng.choice(counts, size=len(counts), replace=True)
        estimates.append(sample.mean() * total_tiles)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def estimate_uniform(predictions: pd.DataFrame, total_tiles: int, iters: int, seed: int) -> dict:
    counts = predictions["count"].to_numpy()
    mean_count = counts.mean() if len(counts) else 0.0
    estimate = mean_count * total_tiles
    ci_low, ci_high = _bootstrap_total(counts, total_tiles, iters, seed) if len(counts) else (0.0, 0.0)
    return {
        "method": "uniform",
        "sample_tiles": int(len(counts)),
        "total_tiles": int(total_tiles),
        "mean_count_per_tile": float(mean_count),
        "estimate": float(estimate),
        "ci_95": [float(ci_low), float(ci_high)],
    }


def estimate_stratified(predictions: pd.DataFrame, strata_path: Path, iters: int, seed: int) -> dict:
    strata = pd.read_csv(strata_path)
    merged = predictions.merge(strata, on="tile_id", how="left")
    if merged["district"].isna().all():
        raise ValueError("No strata assignment found for predictions.")

    total_by_stratum = strata.groupby("district").size().rename("total_tiles").to_frame().reset_index()
    sample_by_stratum = merged.groupby("district")["count"].agg(["mean", "size"]).reset_index()
    merged_stats = total_by_stratum.merge(sample_by_stratum, on="district", how="left").fillna(0)
    merged_stats["estimate"] = merged_stats["mean"] * merged_stats["total_tiles"]

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iters):
        total = 0.0
        for district in merged_stats["district"]:
            sample_counts = merged.loc[merged["district"] == district, "count"].to_numpy()
            total_tiles = merged_stats.loc[merged_stats["district"] == district, "total_tiles"].values[0]
            if len(sample_counts) == 0:
                continue
            boot = rng.choice(sample_counts, size=len(sample_counts), replace=True).mean() * total_tiles
            total += boot
        estimates.append(total)

    result = {
        "method": "stratified",
        "sample_tiles": int(len(predictions)),
        "total_tiles": int(strata.shape[0]),
        "estimate": float(merged_stats["estimate"].sum()),
        "ci_95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "by_district": merged_stats.to_dict(orient="records"),
    }
    return result


def estimate(config_path: str, predictions_path: Path | None, strata_path: Path | None) -> Path:
    config = load_config(config_path)
    root = project_root()
    if predictions_path is None:
        predictions_path = root / "data" / "processed" / "predictions.csv"
    predictions = pd.read_csv(predictions_path)

    bbox = config["aoi"]["bbox"]
    zoom = config["tiles"]["zoom"]
    total_tiles = len(tiles_from_bbox(bbox, zoom))

    sampling = config.get("sampling", {})
    iters = sampling.get("bootstrap_iters", 1000)
    seed = sampling.get("random_seed", 42)

    if strata_path:
        result = estimate_stratified(predictions, strata_path, iters, seed)
    else:
        result = estimate_uniform(predictions, total_tiles, iters, seed)

    output_path = root / "reports" / "logs" / "estimate.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate total pools from sample predictions.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--predictions", default="data/processed/predictions.csv")
    parser.add_argument("--strata", default=None)
    args = parser.parse_args()

    output_path = estimate(args.config, Path(args.predictions), Path(args.strata) if args.strata else None)
    print(f"Saved estimate to {output_path}")


if __name__ == "__main__":
    main()
