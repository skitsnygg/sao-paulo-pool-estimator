# Sao Paulo Pool Estimator

## 1. Project Overview

YOLOv8 segmentation pipeline for swimming-pool detection in São Paulo imagery.

Current baseline is **Google z21** tiles and datasets (for both training and citywide inference).  
**GeoSampa** remains in the repository as a **legacy/older imagery workflow** for historical datasets and back-comparisons.

## 2. Current Workflow

1. Prepare annotation batches and import reviewed CVAT masks/exports into YOLO segmentation format (`class id 0 = pool`).
2. Audit image/label integrity and split consistency before training.
3. Train YOLOv8 segmentation on the Google z21 baseline dataset.
4. Run tiled inference and export pool polygons plus tile-level summaries.
5. Run Folium-based QC and optional deduplication/selection loops.

## 3. Datasets

### Google z21 (current baseline)

- Baseline training dataset in this repo: `data/datasets/google_z21_v7/dataset.yaml`
- YAML fields: `nc: 1`, `names: {0: pool}`, `train: images/train`, `val: images/val`
- Baseline city imagery root used by inference: `data/raw/google/sp_city_2020_rebuild_google_z21`

### GeoSampa (legacy workflow)

- Legacy dataset families remain under `data/datasets/geosampa_*`
- Legacy workflow scripts are still available (for older imagery lineage), e.g.:
  - `tools/build_geosampa_master_dataset.py`
  - `tools/geosampa_2020_official_pipeline.py`
  - `tools/geosampa_ortho_pipeline.py`

## 4. Key Scripts

- `tools/import_mask_dataset_to_yolo.py`: import mask datasets to YOLO-seg format.
- `tools/import_cvat_positives_zip_to_yolo.py`: import positive CVAT zip exports into an existing YOLO dataset.
- `tools/merge_yolo_datasets.py`: merge YOLO datasets while preserving split structure.
- `tools/audit_fix_yolo_seg_dataset.py`: audit/fix mismatches, malformed labels, and split issues.
- `tools/predict_tiles_to_geojson.py`: tiled inference to `GeoJSON`, per-tile CSV/JSONL, and `pools_stats.json`.
- `tools/dedupe_geojson_polygons.py`: IoU-based post-inference deduplication.
- `tools/folium_view.py`: Folium QC viewer over imagery + predictions.
- `tools/compare_yolo_runs.py`: compare train runs using run artifacts.

## 5. Training

### Example training command (Google z21 baseline pattern)

```bash
.venv/bin/yolo segment train \
  model=runs/segment/runs/segment/z21_v720260409_234054/weights/best.pt \
  data=data/datasets/google_z21_v7/dataset.yaml \
  imgsz=1024 \
  epochs=120 \
  batch=8 \
  optimizer=AdamW \
  lr0=0.0005 \
  lrf=0.01 \
  cos_lr=True \
  device=mps \
  workers=0 \
  project=runs/segment \
  name=z21_v7_<timestamp>
```

### Current baseline training run (exact)

- Run directory: `runs/segment/runs/segment/z21_v7_20260414_165212`
- Metric source files:
  - `runs/segment/runs/segment/z21_v7_20260414_165212/args.yaml`
  - `runs/segment/runs/segment/z21_v7_20260414_165212/results.csv`
  - `runs/segment/runs/segment/z21_v7_20260414_165212/weights/best.pt`
  - `runs/segment/runs/segment/z21_v7_20260414_165212/weights/last.pt`
- Dataset: `data/datasets/google_z21_v7/dataset.yaml`
- Initialization checkpoint: `/Users/admin/sao-paulo-pool-estimator/runs/segment/runs/segment/z21_v720260409_234054/weights/best.pt`
- Image size: `1024`
- Batch size: `8`
- Epochs configured/completed: `120` / `120` (from `results.csv`)
- Optimizer + LR details: `AdamW`, `lr0=0.0005`, `lrf=0.01`, `cos_lr=true`, `warmup_epochs=3.0`

Best checkpoint metrics (from `weights/best.pt` -> `train_metrics`, corresponding to epoch `102` in `results.csv`):

| Metric | Box (B) | Mask (M) |
|---|---:|---:|
| Precision | 0.85322 | 0.85322 |
| Recall | 0.83516 | 0.83516 |
| mAP50 | 0.90649 | 0.90649 |
| mAP50-95 | 0.75377 | 0.75530 |

- Best epoch (by Ultralytics fitness): `102` (from `results.csv`)
- Fitness: `1.50907` (from `weights/best.pt`, equals `mAP50-95(B) + mAP50-95(M)`)
- Final epoch (`120`) metrics from `results.csv`:
  - Box: `precision=0.85811`, `recall=0.83414`, `mAP50=0.90747`, `mAP50-95=0.75477`
  - Mask: `precision=0.86035`, `recall=0.83477`, `mAP50=0.90890`, `mAP50-95=0.75249`
- Weights:
  - `runs/segment/runs/segment/z21_v7_20260414_165212/weights/best.pt`
  - `runs/segment/runs/segment/z21_v7_20260414_165212/weights/last.pt`

## 6. Inference Outputs

Latest verified citywide inference run with full output artifacts:

- Run: `runs/segment/z21_v7_infer_20260410_054950`
- Stats source: `runs/segment/z21_v7_infer_20260410_054950/pools_stats.json`
- Model: `/Users/admin/sao-paulo-pool-estimator/checkpoints/pools_google_z21_v7_best_20260410_054420.pt`

Key metrics from `pools_stats.json`:

- `tiles_processed`: `163366`
- `tiles_with_predictions`: `23906`
- `tiles_with_predictions_rate`: `0.14633399850642118`
- `polys_total`: `60157`
- `polys_kept`: `50696`
- `features_31983`: `50696`
- `features_3857`: `50696`
- `polys_dropped_mask_area`: `186`
- `polys_dropped_invalid_geom`: `2423`
- `polys_dropped_area_m2_threshold`: `6852`

Per-run output files:

- `pools.geojson`: pool polygons in `EPSG:31983`
- `pools_3857.geojson`: reprojected output in `EPSG:3857`
- `pools_tiles.csv`: per-tile summary for mining/QC
- `pools_tiles.jsonl`: detailed per-tile records
- `pools_stats.json`: run counters + thresholds + derived metrics
- `pools_dedup.geojson`: optional deduplicated polygons (present in this run)

## 7. Visualization / QC

Use Folium QC over Google z21 tiles:

```bash
.venv/bin/python tools/folium_view.py \
  --tiles-dir data/raw/google/sp_city_2020_rebuild_google_z21 \
  --name z21_v7_qc \
  --pred-geojson runs/segment/z21_v7_infer_20260410_054950/pools.geojson \
  --assume-pred-crs EPSG:31983 \
  --base google_sat
```

## 8. Repo Structure

```text
.
├── tools/                  # Dataset prep/import/audit, inference, QC, utilities
├── checkpoints/            # Promoted checkpoints for reproducible runs
├── data/
│   ├── raw/                # Source imagery (do not modify in place)
│   └── datasets/           # Versioned YOLO segmentation datasets
└── runs/segment/           # Inference outputs and many training run directories
```
