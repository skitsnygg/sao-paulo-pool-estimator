# São Paulo Pool Estimator

YOLOv8 segmentation pipeline for city-scale swimming pool detection in São Paulo.

## Project Overview

This repository is currently focused on **2020 São Paulo citywide pool detection**.

- Primary imagery for training and citywide inference: `data/raw/geosampa_ortho/sp_city_2020_rebuild_official`
- Comparison imagery now in use: `data/raw/google/sp_city_2020_rebuild_google_z19`
- Google z21 rebuild path exists and is being prepared: `data/raw/google/sp_city_2020_rebuild_google_z21`
- Primary class: `pool` (class id `0`)
- Main model lineage currently used in full-city runs:
  - `checkpoints/pools_2020_2024_after_2020_unsure_round1.pt`
  - `checkpoints/pools_2020_2024_after_missed_round2.pt`

Recent citywide outputs indicate GeoSampa 2020 runs around **~60k** detections depending on threshold
(e.g., `59,819` at conf `0.12`; `62,289` at conf `0.10`).

## Current Pipeline (Practical)

The active loop is:

1. rebuild/download tiles
2. run prediction to georeferenced GeoJSON
3. select uncertain/missed/false-positive candidates
4. annotate or correct in CVAT
5. import masks back into YOLO format
6. merge into a **new** dataset version
7. audit dataset integrity
8. retrain
9. rerun full-city inference and compare

## Repository Structure

```text
.
├── tools/                 # Main operational scripts (rebuild, inference, CVAT prep, merge, audit)
├── data/
│   ├── raw/               # Source/rebuilt imagery (GeoSampa, Google, IDESP)
│   └── datasets/          # Versioned YOLO segmentation datasets
├── checkpoints/           # Named model checkpoints used for training/inference
├── runs/
│   ├── segment/           # Training runs + full-city inference outputs
│   └── folium/            # Generated map viewers
├── docs/                  # Additional notes
└── src/                   # Legacy/auxiliary modules
```

## Key Scripts

- `tools/geosampa_2020_official_pipeline.py`
  - Official GeoSampa 2020 rebuild/index/coverage pipeline (subcommands).
- `tools/rebuild_sp_city_geosampa_2020_official.sh`
  - Convenience wrapper to run a full GeoSampa 2020 city rebuild.
- `tools/predict_tiles_to_geojson.py`
  - Core inference script: tile predictions -> GeoJSON (`EPSG:31983` + `EPSG:3857`) + tile summaries + run stats.
- `tools/prepare_cvat_tiles.py`
  - Builds deterministic CVAT upload sets with unique filenames and manifests.
- `tools/geojson_to_coco_cvat.py`
  - Converts geospatial prediction polygons to COCO polygons in image pixel space for CVAT import.
- `tools/import_mask_dataset_to_yolo.py`
  - Converts CVAT segmentation-mask exports back into YOLO segmentation labels.
- `tools/merge_yolo_datasets.py`
  - Deterministic multi-dataset merge with collision-safe renaming and merge manifests.
- `tools/audit_fix_yolo_seg_dataset.py`
  - Dataset integrity audit (and optional quarantine/smoke check).
- `tools/select_uncertain_tiles.py`
  - Samples uncertain predictions (confidence band) for active learning.
- `tools/select_missed_pool_tiles.py`
  - Ranks likely missed pools from per-tile prediction summaries.

## Recommended Workflow (Current)

### 0) Environment

```bash
source .venv/bin/activate
```

### 1) Build/Rebuild imagery

GeoSampa 2020 official rebuild (wrapper):

```bash
SOURCE_VRT=/path/to/source_ortho_2020.vrt \
OUT_ROOT=data/raw/geosampa_ortho/sp_city_2020_rebuild_official \
bash tools/rebuild_sp_city_geosampa_2020_official.sh
```

If no VRT exists, set `SOURCES_ROOT` so the script can build one first.

Google imagery rebuilds are currently maintained under:

- `data/raw/google/sp_city_2020_rebuild_google_z19`
- `data/raw/google/sp_city_2020_rebuild_google_z21` (in progress)

### 2) Full-city inference -> GeoJSON

GeoSampa 2020 (example):

```bash
.venv/bin/python tools/predict_tiles_to_geojson.py \
  --model checkpoints/pools_2020_2024_after_missed_round2.pt \
  --tiles-dir data/raw/geosampa_ortho/sp_city_2020_rebuild_official \
  --out-geojson runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full.geojson \
  --out-geojson-3857 runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full_3857.geojson \
  --out-tile-summary-csv runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full_tiles.csv \
  --out-tile-summary-jsonl runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full_tiles.jsonl \
  --out-stats-json runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full_stats.json \
  --imgsz 1024 \
  --conf 0.12 \
  --iou 0.7 \
  --min-area-px 120 \
  --min-mask-area-px 120 \
  --min-area-m2 4.0 \
  --worldfile-crs EPSG:31983 \
  --progress-every 500
```

Google z19 comparison (example):

```bash
.venv/bin/python tools/predict_tiles_to_geojson.py \
  --model checkpoints/pools_2020_2024_after_missed_round2.pt \
  --tiles-dir data/raw/google/sp_city_2020_rebuild_google_z19 \
  --out-geojson runs/segment/google_2020_z19_after_missed_round2_conf25/pools_google_2020_z19.geojson \
  --out-geojson-3857 runs/segment/google_2020_z19_after_missed_round2_conf25/pools_google_2020_z19_3857.geojson \
  --out-tile-summary-csv runs/segment/google_2020_z19_after_missed_round2_conf25/pools_google_2020_z19_tiles.csv \
  --out-tile-summary-jsonl runs/segment/google_2020_z19_after_missed_round2_conf25/pools_google_2020_z19_tiles.jsonl \
  --out-stats-json runs/segment/google_2020_z19_after_missed_round2_conf25/pools_google_2020_z19_stats.json \
  --imgsz 1024 \
  --conf 0.25 \
  --iou 0.7 \
  --min-area-px 120 \
  --min-mask-area-px 120 \
  --min-area-m2 4.0 \
  --worldfile-crs EPSG:31983 \
  --progress-every 500
```

### 3) Select active-learning candidates

Uncertain-confidence sampling:

```bash
.venv/bin/python tools/select_uncertain_tiles.py \
  --tiles-csv runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full_tiles.csv \
  --min-conf 0.20 \
  --max-conf 0.45 \
  --num-tiles 2000 \
  --out runs/segment/2020_uncertain_candidates.csv \
  --seed 42
```

Likely missed-pool ranking:

```bash
.venv/bin/python tools/select_missed_pool_tiles.py \
  --tiles-csv runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full_tiles.csv \
  --num-tiles 2000 \
  --out runs/segment/2020_missed_candidates.csv
```

### 4) Prepare CVAT package (optional pre-annotations included)

Prepare deterministic CVAT image package:

```bash
.venv/bin/python tools/prepare_cvat_tiles.py \
  --tiles-root data/raw/geosampa_ortho/sp_city_2020_rebuild_official \
  --tiles-csv runs/segment/2020_uncertain_candidates.csv \
  --out-dir data/annotations/2020_uncertain_round2_for_cvat \
  --max-tiles 2000 \
  --symlink-images
```

Convert predicted polygons to COCO for CVAT pre-annotation import:

```bash
.venv/bin/python tools/geojson_to_coco_cvat.py \
  --geojson runs/segment/geosampa_2020_full_after_missed_round2_c12/pools_geosampa_2020_full.geojson \
  --manifest-csv data/annotations/2020_uncertain_round2_for_cvat/manifest.csv \
  --images-dir data/annotations/2020_uncertain_round2_for_cvat/JPEGImages \
  --out data/annotations/2020_uncertain_round2_for_cvat/predictions_coco.json \
  --worldfile-crs EPSG:31983 \
  --min-confidence 0.25
```

### 5) Import CVAT mask exports back to YOLO

```bash
.venv/bin/python tools/import_mask_dataset_to_yolo.py \
  --src data/cvat_exports_2020_fresh/Jardins \
  --dataset data/datasets/2020_unsure_round1 \
  --split train \
  --class-id 0 \
  --min-area-px 10 \
  --overwrite
```

Repeat per export and split as needed.

### 6) Merge datasets into a new version and audit

Current canonical lineage target: `data/datasets/geosampa_master_2020_plus_2024_v6_rebalanced`

```bash
.venv/bin/python tools/merge_yolo_datasets.py \
  --input-dirs \
    data/datasets/geosampa_master_2020_plus_2024_v6_clean \
    data/datasets/2020_missed_round2 \
  --out-dir data/datasets/geosampa_master_2020_plus_2024_v6_rebalanced \
  --overwrite
```

```bash
.venv/bin/python tools/audit_fix_yolo_seg_dataset.py \
  --dataset data/datasets/geosampa_master_2020_plus_2024_v6_rebalanced \
  --strict
```

### 7) Train

```bash
.venv/bin/yolo segment train \
  model=checkpoints/pools_2020_2024_after_missed_round2.pt \
  data=data/datasets/geosampa_master_2020_plus_2024_v6_rebalanced/dataset.yaml \
  imgsz=1024 \
  epochs=30 \
  batch=2 \
  device=mps \
  workers=8 \
  optimizer=AdamW \
  lr0=0.0005 \
  patience=100 \
  project=runs/segment \
  name=train_next
```

## Annotation Workflow (CVAT)

1. Run full-city inference and generate per-tile summary CSV.
2. Select uncertain/missed candidates using `select_uncertain_tiles.py` and `select_missed_pool_tiles.py`.
3. Build CVAT upload package with `prepare_cvat_tiles.py` (`manifest.csv` + `manifest.jsonl`).
4. Optionally import model polygons as COCO using `geojson_to_coco_cvat.py`.
5. Annotate/correct in CVAT and export segmentation masks.
6. Import masks back to YOLO with `import_mask_dataset_to_yolo.py`.
7. Merge and audit before any training.

## Dataset Hygiene and Audit Guidance

- Keep raw imagery read-only (`data/raw/*`).
- Always produce a **new dataset version directory** when merging or importing.
- Use `merge_manifest.csv` and `merge_stats.json` from `merge_yolo_datasets.py` to track provenance.
- Run `audit_fix_yolo_seg_dataset.py --strict` before training.
- Keep class index stable: `pool = 0`.
- Confirm `dataset.yaml` points to the intended dataset version before each training run.

## GeoSampa vs Google Imagery Notes

- GeoSampa 2020 is currently the most stable source for citywide 2020 inference.
- Google z19 runs currently show over-detection relative to GeoSampa (example with same checkpoint):
  - GeoSampa 2020 conf `0.12`: `59,819` features (`runs/segment/geosampa_2020_full_after_missed_round2_c12`)
  - Google z19 conf `0.25`: `69,935` features (`runs/segment/google_2020_z19_after_missed_round2_conf25`)
  - Google z19 conf `0.15`: `109,399` features (`runs/segment/google_2020_z19_after_missed_round2_conf15`)
- Practical implication: Google-specific threshold tuning and hard-negative annotation are required before trusting citywide counts.

## Known Issues / Caveats

- Threshold portability across imagery sources is limited (GeoSampa vs Google behavior differs).
- CRS discipline matters:
  - training/inference georeferencing commonly uses `EPSG:31983`
  - map/export consumers often need `EPSG:3857` or `EPSG:4326`
- Full polygon Folium maps can become very large and slow to load.
- Some historical artifacts under `runs/segment/` are experiments; do not treat all run outputs as training-ready data.

## Current Best Practices

- Never train from `runs/segment` directories.
- Always merge into a **new** dataset version.
- Always audit merged datasets before training.
- Clear `train.cache` / `val.cache` after dataset changes.
- Use `batch=2` on MPS unless you are deliberately testing higher values.

Cache clear example:

```bash
rm -f data/datasets/geosampa_master_2020_plus_2024_v6_rebalanced/labels/train.cache \
      data/datasets/geosampa_master_2020_plus_2024_v6_rebalanced/labels/val.cache
```

## Model Iteration History (Recent)

- **2020 unsure rounds**:
  - focused on borderline detections and annotation uncertainty cleanup
  - checkpoint lineage includes `checkpoints/pools_2020_2024_after_2020_unsure_round1.pt`
- **2020 missed-pool rounds**:
  - targeted low-prediction neighborhoods and neighbor-ranked candidates
  - checkpoint lineage includes `checkpoints/pools_2020_2024_after_missed_round2.pt`
- **Google cleanup direction**:
  - current z19 inference overfires
  - next loop is Google-specific false-positive mining + hard-negative annotation + retraining

## License

No `LICENSE` file is currently present in this repository.
