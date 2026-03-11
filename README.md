# São Paulo Pool Detection
This repository contains a computer vision pipeline for detecting swimming pools in São Paulo satellite/orthophoto imagery using YOLOv8 segmentation. It includes data preparation utilities, CVAT export conversion scripts, training artifacts, and geospatial post-processing for city-scale inference outputs.

The current workflow is built around GeoSampa 2020 orthophotos and newer 2023/2024 imagery, with polygon predictions exported to GeoJSON and deduplicated for stable downstream use.

## Project Overview
The goal is to produce reliable pool polygons across São Paulo from tiled aerial imagery. The project uses YOLOv8 segmentation (single class: `pool`) and a geospatial inference pipeline that:

1. runs tile-level model inference,
2. converts masks to map coordinates (`EPSG:3857` and `EPSG:31983`),
3. deduplicates overlapping polygons for clean citywide inventories.

## Dataset
### GeoSampa 2020 orthophotos
- Raw 2020 imagery and rebuild outputs live under `data/raw/geosampa_ortho/`.
- Official rebuild/coverage tooling is implemented in `tools/geosampa_2020_official_pipeline.py` and `tools/rebuild_sp_city_geosampa_2020_official.sh`.

### GeoSampa 2024 imagery
- The repo’s 2024-era imagery is stored under `data/raw/idesp_ortho/FEHIDRO_ORTOMOSAICO_IGC_RMSP_2023_2024_3857_jpg/` and is used for transfer/fine-tune experiments and inference.

### CVAT annotation workflow
- CVAT segmentation exports are stored in `data/cvat_exports_2020_fresh/` (`Jardins`, `Moema`, `Pinheiros`, `batch_2000_600`).
- Each export contains `ImageSets/Segmentation/default.txt`, `SegmentationClass/*.png`, `labelmap.txt`, and optional XML polygons.

### Mask -> YOLO segmentation conversion
- `tools/build_geosampa_2020_cvat_dataset.py` converts CVAT segmentation masks to YOLOv8 segmentation labels (`class x1 y1 x2 y2 ...`).
- It creates deterministic train/val splits, writes `dataset.yaml`, and outputs QA artifacts (`build_report.json`, `invalid_masks.csv`).

Current dataset snapshots in `data/datasets/`:
- `geosampa_2020_cvat_all`: 900 images (697 train / 203 val), 337 positive labels.
- `geosampa_2020_cvat_all_v2`: 1500 images (1173 train / 327 val), 416 positive labels.
- `geosampa_2020_cvat_fresh_all`: 337 images (295 train / 42 val), all labeled positive.

## Repository Structure
```text
.
├── tools/                         # Data prep, conversion, inference, dedupe, QA utilities
├── checkpoints/                   # Exported model checkpoints (.pt)
├── runs/                          # Training runs, inference outputs, folium maps, audits
├── data/
│   ├── raw/                       # Raw imagery and downloaded tiles
│   │   ├── geosampa_ortho/
│   │   ├── idesp_ortho/
│   │   └── tiles/
│   ├── datasets/                  # YOLO-ready datasets with images/labels + dataset.yaml
│   │   ├── geosampa_2020_cvat_all/
│   │   ├── geosampa_2020_cvat_all_v2/
│   │   └── geosampa_2020_cvat_fresh_all/
│   ├── annotations/               # Neighborhood image/mask workspaces
│   │   ├── jardins_seg/
│   │   ├── moema_seg/
│   │   ├── pinheiros_seg/
│   │   └── brooklin_seg/
│   └── cvat_exports_2020_fresh/   # CVAT segmentation exports (mask format)
│       ├── Jardins/
│       ├── Moema/
│       ├── Pinheiros/
│       └── batch_2000_600/
├── docs/                          # Pipeline docs and notes
├── src/                           # Legacy/auxiliary Python package modules
└── tests/                         # Unit tests
```

## Training Pipeline
1. Download imagery: use GeoSampa/official pipelines (`tools/geosampa_2020_official_pipeline.py`) and 2024 imagery acquisition scripts (`tools/fetch_idesp_wms_tiles.py`).
2. Annotate tiles in CVAT: label pools as segmentation masks.
3. Export masks: CVAT segmentation mask export (`SegmentationClass/*.png`).
4. Build YOLO dataset: run `tools/build_geosampa_2020_cvat_dataset.py` to produce train/val images and YOLO labels.
5. Train model: run YOLOv8 segmentation training on the generated `dataset.yaml`.
6. Run citywide inference: predict on rebuilt city tiles using `tools/predict_tiles_to_geojson.py`.
7. Deduplicate polygons: run `tools/dedupe_geojson_polygons.py` (and optional `tools/audit_geojson_overlaps.py`).

## Model Training
Command used for the current GeoSampa 2020 best run (`geosampa_2020_all_neighborhoods_v28`):

```bash
.venv/bin/yolo segment train \
  model=checkpoints/pools_idesp_v26_best.pt \
  data=data/datasets/geosampa_2020_cvat_all/dataset.yaml \
  imgsz=1024 \
  epochs=150 \
  batch=8 \
  patience=30 \
  device=mps \
  workers=4 \
  project=runs/segment \
  name=geosampa_2020_all_neighborhoods_v28
```

## Running Inference
Example citywide command with `tools/predict_tiles_to_geojson.py`:

```bash
.venv/bin/python tools/predict_tiles_to_geojson.py \
  --model checkpoints/pools_geosampa_2020_v28_best.pt \
  --tiles-dir data/raw/geosampa_ortho/sp_city_2020_rebuild_official \
  --imgsz 1024 \
  --conf 0.10 \
  --iou 0.50 \
  --min-mask-area-px 30 \
  --min-area-m2 6.0 \
  --worldfile-crs EPSG:31983 \
  --out-geojson runs/geosampa_2020_best_model_predict/pools.geojson
```

Optional dedupe pass:

```bash
.venv/bin/python tools/dedupe_geojson_polygons.py \
  --in-geojson runs/geosampa_2020_best_model_predict/pools.geojson \
  --out-geojson runs/geosampa_2020_best_model_predict/pools_dedup.geojson \
  --iou 0.35 \
  --precision 7 \
  --stats
```

## Dataset Builder
`tools/build_geosampa_2020_cvat_dataset.py` builds a clean YOLO segmentation dataset from CVAT exports by:

- reading export manifests (`default.txt`),
- resolving source tiles from `data/raw/geosampa_ortho/...`,
- converting `SegmentationClass` RGB masks into YOLO polygons (OpenCV contours),
- falling back to CVAT XML polygons when present,
- performing deterministic split assignment,
- generating `dataset.yaml`, `build_report.json`, and `invalid_masks.csv`.

Example:

```bash
.venv/bin/python tools/build_geosampa_2020_cvat_dataset.py \
  --exports-root data/cvat_exports_2020_fresh \
  --raw-root data/raw/geosampa_ortho \
  --out-dir data/datasets/geosampa_2020_cvat_all \
  --clean
```

## Current Best Model
- Requested path in this README spec: `checkpoints/pools_geosampa_2020_best.pt` (this file is not currently present).
- Current checkpoint artifact in the repository: `checkpoints/pools_geosampa_2020_v28_best.pt`.
- That file matches `runs/segment/runs/segment/geosampa_2020_all_neighborhoods_v28/weights/best.pt` and was trained:
  - from `checkpoints/pools_idesp_v26_best.pt`,
  - on `data/datasets/geosampa_2020_cvat_all/dataset.yaml`,
  - with `imgsz=1024`, `batch=8`, `epochs=150` (early stopped after 33 epochs).

## Results
Validation (GeoSampa 2020 best run, `geosampa_2020_all_neighborhoods_v28`):
- Best mask `mAP50`: **0.6726**
- Best mask `mAP50-95`: **0.4716**
- Precision: **0.5966**
- Recall: **0.8162**

Citywide inference outputs currently in `runs/`:
- `runs/geosampa_2020_best_model_predict/pools.geojson`: 56,513 polygons (raw).
- `runs/geosampa_2020_best_model_predict/pools_dedup.geojson`: 52,307 polygons after dedupe (~7.4% removed).
- `runs/geosampa_2024_from_moema_best2_predict/pools.geojson`: 49,686 polygons (raw), 46,611 deduped (~6.2% removed).

Expected performance: recall-oriented citywide predictions with a required dedupe/audit post-processing step for production-quality polygon inventories.

## Future Improvements
- Expand citywide annotation coverage and add more hard-negative samples.
- Scale model experiments (larger backbones, augmentation/hyperparameter sweeps).
- Improve polygon deduplication heuristics with stronger confidence/shape priors.
- Train jointly across multiple years (2020 + 2024) for better temporal robustness.

## License
This repository currently does not include a `LICENSE` file.

## Author
Brian Migliore
