#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

$PYTHON -m src.data.download_tiles --config configs/quickstart.yaml
$PYTHON -m src.data.fetch_osm --config configs/quickstart.yaml
$PYTHON -m src.data.make_dataset --config configs/quickstart.yaml
$PYTHON -m src.models.baseline --config configs/quickstart.yaml --source data/processed/yolo/images --threshold 0.003
$PYTHON -m src.analysis.estimate --config configs/quickstart.yaml --predictions data/processed/predictions_baseline.csv
$PYTHON -m src.visualization.map --config configs/quickstart.yaml --predictions data/processed/predictions_baseline.csv --output reports/figures/quickstart_pool_density.html
