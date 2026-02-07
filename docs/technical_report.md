# Technical Report - Sao Paulo Pool Estimator

## Objective

Estimate the number of swimming pools in Sao Paulo by training a pool detector on aerial imagery, then extrapolating predictions using statistical sampling.

## Data acquisition

- Imagery: Web Mercator tiles from ESRI World Imagery.
- Labels: OpenStreetMap polygons tagged as `leisure=swimming_pool` or `amenity=swimming_pool`.

The pipeline queries OSM for pool polygons within the AOI, downloads imagery tiles, and labels tiles by intersecting pool polygons with tile boundaries.

## Tile selection

- Training tiles include all tiles that intersect OSM pool polygons plus a configurable number of random negative tiles.
- Estimation tiles are a uniform random sample of all tiles in the AOI.
- This separation keeps training data pool-heavy while preserving unbiased city-wide estimates.

## Dataset construction

- Each training tile is treated as an image chip.
- For each tile, pool polygons that intersect the tile are clipped to tile boundaries and converted to bounding boxes.
- Images + labels are exported in YOLO format with a single class: `pool`.
- Train/val/test splits are random and reproducible via a seed in `configs/project.yaml`.

## Model

- Architecture: YOLOv8 (default: `yolov8n`).
- Training parameters are controlled in `configs/project.yaml`.
- Output checkpoints and metrics are written to `reports/` and `checkpoints/`.

## Evaluation

- mAP@0.5 and related metrics are produced via YOLO's internal validation routine.
- Thresholds for inference (confidence / IoU) are configurable.

## Estimation strategy

Let:
- `N` = total tiles in the AOI at the selected zoom
- `n` = sampled tiles with predictions
- `y_i` = predicted pool count on tile `i`

Predictions are generated on the estimation tile sample (not the training split).

Uniform estimator:

```
mean_count = (1 / n) * sum(y_i)
Total pools estimate = mean_count * N
```

95% confidence intervals are computed via bootstrap resampling of tile counts.

For stratified estimation, tiles are assigned to districts and the estimator is computed per stratum:

```
Estimate = sum_h (mean_count_h * N_h)
```

## Assumptions and limitations

- OSM pool annotations are incomplete and contain noise.
- Tile-based sampling assumes that detection error is not strongly biased across the AOI.
- Imagery timestamps are not uniform across the city.
- The baseline method is not designed to meet mAP requirements and is included for quick validation only.

## Reproducibility

- All parameters are stored in YAML configs.
- The pipeline is deterministic given seeds.
- Scripts can be run end-to-end via `scripts/run_pipeline.sh`.

## Experiment evidence

- Example quickstart outputs are stored under `reports/`.
- Training logs and metrics are written to `reports/logs/`.
- Folium maps are written to `reports/figures/`.
