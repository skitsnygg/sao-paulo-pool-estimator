# How Many Pools in Sao Paulo?

Estimate the number of swimming pools in Sao Paulo using aerial imagery + ML, with a reproducible pipeline from data collection to city-wide estimates.

## What is included

- Tile downloader for imagery
- OSM pool polygon fetcher
- Dataset builder (YOLO format)
- YOLOv8 training + inference scripts
- Sampling + bootstrap estimator
- Folium density map
- Quickstart baseline to validate the pipeline without training

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional for training:
# pip install -r requirements-train.txt
```

## Quickstart (no training)

Runs the data pipeline on a small AOI and uses a naive blue-pixel baseline for predictions (threshold tuned for the quickstart AOI).

Threshold can be adjusted via `baseline.blue_ratio_threshold` in `configs/quickstart.yaml`.

```bash
./scripts/quickstart.sh
```

Artifacts:
- `data/raw/tiles/tiles.csv`
- `data/raw/osm_pools.geojson`
- `data/processed/yolo/`
- `data/processed/predictions_baseline.csv`
- `reports/logs/estimate.json`
- `reports/figures/quickstart_pool_density.html`

## Full pipeline (training)

```bash
python -m src.data.download_tiles --config configs/project.yaml
python -m src.data.fetch_osm --config configs/project.yaml
python -m src.data.make_dataset --config configs/project.yaml
python -m src.models.train --config configs/project.yaml
python -m src.models.predict --config configs/project.yaml --weights path/to/best.pt
python -m src.analysis.estimate --config configs/project.yaml --predictions data/processed/predictions.csv
python -m src.visualization.map --config configs/project.yaml --predictions data/processed/predictions.csv
```

Training will copy weights to `checkpoints/best.pt` and `checkpoints/last.pt`.

Or run everything in order:

```bash
./scripts/run_pipeline.sh
```

## Sampling strategy

- AOI is split into Web Mercator tiles at a fixed zoom.
- A uniform random sample of tiles is selected (configurable).
- Pool counts are predicted per sampled tile.
- The city total is estimated as:
  - `mean_count_per_tile * total_tiles`
- Bootstrap resampling provides a 95% confidence interval.

For district-wise estimates:

```bash
python -m src.analysis.assign_districts --districts data/external/districts.geojson
python -m src.analysis.estimate --strata data/processed/tile_districts.csv
```

## Assumptions

- OSM pool polygons are a reasonable proxy for ground truth during training.
- The sampling frame is the full set of tiles covering the AOI at the chosen zoom.
- Predictions are unbiased across sampled tiles (or within strata when stratified).

## Data sources

- Imagery: ESRI World Imagery tile service
- Labels: OpenStreetMap swimming_pool geometries

Check source licenses and terms of use before distribution.

## Repository structure

```
configs/        # YAML configs
src/            # pipeline code
scripts/        # runnable shell scripts
data/           # local data (kept empty in git)
reports/        # logs and figures
checkpoints/    # model weights
```

## Notes on quality

The baseline is a quick sanity check only. For reasonable detection quality (>0.65 mAP), use YOLO training with a larger AOI sample and validate on a held-out set.
