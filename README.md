# São Paulo Pool Estimator

This project estimates the number of swimming pools in São Paulo, Brazil using computer vision and geospatial data.

## System Overview

- YOLOv8 segmentation model trained on São Paulo imagery.
- Pipeline steps:
  - Download tiles from OpenStreetMap.
  - Train the model on São Paulo pool data.
  - Run inference on tiles.
  - Estimate pool density across the AOI.

## Current Status

- Model runs and produces predictions.
- Quickstart tiles are outside São Paulo.
- Zero pools in quickstart is expected.

## Expected Behavior

- Trained model detects pools in São Paulo imagery.
- Baseline method detects some pools.
- Estimates are reasonable for the AOI.

## Running the Pipeline

- Full pipeline:
  ```bash
  bash scripts/quickstart.sh
  ```
- Individual steps:
  ```bash
  # Download tiles
  python -m src.pipeline --step download

  # Run inference
  python -m src.pipeline --step predict

  # Run baseline
  python -m src.models.baseline --config configs/quickstart.yaml --source data/processed/yolo/images --threshold 0.003
  ```

## Quickstart Inference Demo

- Quickstart tiles may contain no pools.
- Demo mode reruns on bundled São Paulo samples if needed.
- If the model still returns zero, demo mode falls back to labeled sample boxes.

- Normal run (may yield 0 pools):
  ```bash
  python -m src.models.predict --config configs/quickstart.yaml --source data/processed/yolo/images/test
  ```

- Demo run (forces a positive example if needed):
  ```bash
  python -m src.models.predict --config configs/quickstart.yaml --source data/processed/yolo/images/test --demo-mode
  ```

## Batch Inference for São Paulo Neighborhoods

- Input should be a top-level directory with one subfolder per neighborhood.
- Outputs are written under `data/processed/batch/<neighborhood_name>/`.

- Basic run:
  ```bash
  python scripts/batch_inference.py --input-dir ~/Downloads/sao_paulo_tiles
  ```

- Run with area filtering:
  ```bash
  python scripts/batch_inference.py --input-dir ~/Downloads/sao_paulo_tiles --min-area-px 50 --max-area-px 50000
  ```

## Outputs

- `data/processed/predictions.csv` - Model predictions for each tile.
- `data/processed/predictions/pool_counts.csv` - Pool counts per tile.
- `data/processed/predictions/detections.csv` - Pool detections per tile.
- `data/processed/predictions_baseline.csv` - Baseline predictions for each tile.
- `reports/logs/estimate.json` - Final pool estimate for São Paulo.
- `reports/figures/quickstart_pool_density.html` - Interactive map of pool density.

## Notes

- Quickstart uses a small non-São Paulo sample.
- Use the full São Paulo dataset for real detection results.
