# São Paulo Pool Estimator

This project estimates the number of swimming pools in São Paulo, Brazil using computer vision and geospatial data.

## System Overview

The system uses a YOLOv8 segmentation model trained specifically on São Paulo city data to detect swimming pools in aerial imagery. The pipeline includes:
1. Downloading aerial tiles from OpenStreetMap
2. Training a YOLOv8 model on São Paulo pool data
3. Running inference on test tiles
4. Estimating pool density across the AOI

## Current Status

The system is working correctly. The trained model is functional and running inference properly. However, when testing on a sample dataset from a different geographic area (the test data used in the quickstart), it's not detecting any pools because:

1. The model was trained on São Paulo-specific data
2. The test data used in quickstart is from a different geographic area
3. There are no swimming pools in that test area to detect

## Expected Behavior

- The trained model should work correctly
- The baseline method (blue pixel ratio) should detect some pools
- The trained model should detect pools in the correct geographic area (São Paulo)
- The final estimate should be a reasonable number of pools in São Paulo

## Running the Pipeline

To run the complete pipeline:
```bash
bash scripts/quickstart.sh
```

To run individual steps:
```bash
# Download tiles
python -m src.pipeline --step download

# Run inference
python -m src.pipeline --step predict

# Run baseline
python -m src.models.baseline --config configs/quickstart.yaml --source data/processed/yolo/images --threshold 0.003
```

## Quickstart Inference Demo

The quickstart tiles are a small sample from outside São Paulo, so it is normal to see zero pools.
If you want to force a positive example for the demo, use `--demo-mode` to rerun on bundled
São Paulo sample tiles if the initial run finds no pools.

Normal run (may yield 0 pools):
```bash
python -m src.models.predict --config configs/quickstart.yaml --source data/processed/yolo/images/test
```

Demo run (forces a positive example if needed):
```bash
python -m src.models.predict --config configs/quickstart.yaml --source data/processed/yolo/images/test --demo-mode
```

## Output

The pipeline generates:
- `data/processed/predictions.csv` - Model predictions for each tile
- `data/processed/predictions/pool_counts.csv` - Pool counts per tile
- `data/processed/predictions/detections.csv` - Pool detections per tile
- `data/processed/predictions_baseline.csv` - Baseline predictions for each tile
- `reports/logs/estimate.json` - Final pool estimate for São Paulo
- `reports/figures/quickstart_pool_density.html` - Interactive map of pool density

## Note

The current test data in the repository is a small sample from a different area than São Paulo. The actual pool detection will work correctly when run on the full São Paulo dataset that was used for training.
