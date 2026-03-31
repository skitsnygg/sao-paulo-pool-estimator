# Sao Paulo Pool Estimator

YOLOv8 segmentation pipeline for swimming pool detection on Sao Paulo aerial imagery.

## Overview

- Primary target: `pool` segmentation (`class id = 0`).
- Main operational code is in `tools/` (most `src/` and `scripts/` code is legacy/auxiliary).
- Current work mixes:
  - GeoSampa 2020 citywide tiles
  - Google z21 tiles (active corrective loop)
  - IDESP 2023/2024 imagery tooling for recall checks and transfer
- Immediate training dataset: `data/datasets/geosampa_z21_v1`
- Typical loop:
  1. Build/import dataset version
  2. Train (YOLOv8 seg)
  3. Run tiled inference to GeoJSON + tile summaries
  4. Build annotation batches (uncertain/missed/hard-neg/hard-pos/corrective)
  5. Import reviewed labels and retrain

## Repo Layout

```text
.
├── tools/                  # Main operational scripts (dataset, train helpers, inference, annotation, QA)
├── data/
│   ├── raw/                # Source imagery (GeoSampa, Google, IDESP)
│   ├── datasets/           # Versioned YOLO datasets
│   └── annotations/        # CVAT packaging and exports
├── runs/
│   ├── segment/            # Training runs + inference artifacts
│   ├── annotation_batches/ # Targeted/corrective annotation exports
│   └── folium/             # HTML map outputs
├── checkpoints/            # Named promoted checkpoints
├── docs/                   # Pipeline notes (GeoSampa rebuild, inference recipes)
└── src/                    # Legacy modules
```

## Data Sources

### GeoSampa 2020 (primary citywide base)

- Current rebuilt root in repo:
  - `data/raw/geosampa_ortho/sp_city_2020_rebuild_official`
- Official rebuild pipeline docs:
  - `docs/geosampa_2020_official_pipeline.md`
- Wrapper script:
  - `tools/rebuild_sp_city_geosampa_2020_official.sh`

Example:

```bash
SOURCE_VRT=/abs/path/to/source_ortho_2020.vrt \
OUT_ROOT=data/raw/geosampa_ortho/sp_city_2020_rebuild_official \
bash tools/rebuild_sp_city_geosampa_2020_official.sh
```

### Google tiles (current z21 corrective focus)

- Active roots:
  - `data/raw/google/sp_city_2020_rebuild_google_z21`
  - `data/raw/google/sp_city_2020_rebuild_google_z19` (older comparison runs)
- Recent inference runs are in `runs/segment/z21_*`.

### IDESP 2023/2024 tooling (still relevant)

- Raw imagery root:
  - `data/raw/idesp_ortho/FEHIDRO_ORTOMOSAICO_IGC_RMSP_2023_2024_3857_jpg`
- Fetch/missing-tile repair tooling:
  - `tools/fetch_idesp_wms_tiles.py`
- Rebalance scripts for IDESP YOLO datasets exist, but are currently hardcoded:
  - `tools/rebuild_balanced_idesp_dataset.py`
  - `tools/rebuild_balanced_idesp_dataset_grouped_by_cell.py`

## Dataset Preparation

Current immediate train base is `data/datasets/geosampa_z21_v1`.

### 1) Build a clean base dataset from CVAT zip exports

Use when assembling master dataset from nested raw CVAT exports:

```bash
.venv/bin/python tools/build_geosampa_master_dataset.py \
  --archive /abs/path/to/cvat_exports_bundle.zip \
  --out-dir data/datasets/geosampa_master_2020_plus_2024_v6_clean \
  --clean-out \
  --fail-on-suspicious \
  --fail-on-missing-mask
```

### 2) Import mask-based reviewed annotations into YOLO format

```bash
.venv/bin/python tools/import_mask_dataset_to_yolo.py \
  --src data/annotations/my_round_export \
  --dataset data/datasets/geosampa_z21_v1 \
  --split train \
  --class-id 0 \
  --overwrite
```

### 3) Merge dataset versions

```bash
.venv/bin/python tools/merge_yolo_datasets.py \
  --input-dirs \
    data/datasets/geosampa_z21_v1 \
    data/datasets/<new_round_dataset> \
  --out-dir data/datasets/geosampa_z21_v2 \
  --overwrite
```

### 4) Audit before training

```bash
.venv/bin/python tools/audit_fix_yolo_seg_dataset.py \
  --dataset data/datasets/geosampa_z21_v2 \
  --strict
```

Optional split/content rebalance tools:

- `tools/rebuild_geosampa_split_by_content_v2.py`
- `tools/rebuild_geosampa_split_by_content_v3.py`

## Training

Primary method is Ultralytics CLI (`segment train`) with dataset YAML under `data/datasets/.../dataset.yaml`.

Example (z21-style finetune on MPS):

```bash
.venv/bin/yolo segment train \
  model=checkpoints/pools_2020_2024_after_missed_round2.pt \
  data=data/datasets/geosampa_z21_v1/dataset.yaml \
  imgsz=1024 \
  epochs=80 \
  batch=8 \
  device=mps \
  workers=8 \
  cache=True \
  name=z21_finetune_v13
```

Outputs:

- `runs/segment/<name>/weights/best.pt`
- `runs/segment/<name>/weights/last.pt`
- `runs/segment/<name>/results.csv`
- `runs/segment/<name>/args.yaml`

Checkpoint handling convention:

- Promote selected weights into `checkpoints/` with explicit names (for inference reproducibility).
- Recent example in repo:
  - `checkpoints/pools_z21_v4_targeted_fix_best.pt`

Example promotion:

```bash
cp runs/segment/<train_run>/weights/best.pt checkpoints/pools_<tag>_best.pt
```

## Inference

### Core script

- `tools/predict_tiles_to_geojson.py`

It writes:

- `*.geojson` in `EPSG:31983`
- `*_3857.geojson`
- per-tile summary `*_tiles.csv`
- per-tile JSONL `*_tiles.jsonl`
- run stats `*_stats.json`

### Example (Google z21 run layout)

```bash
.venv/bin/python tools/predict_tiles_to_geojson.py \
  --model checkpoints/pools_z21_v4_targeted_fix_best.pt \
  --tiles-dir data/raw/google/sp_city_2020_rebuild_google_z21 \
  --out-geojson runs/segment/z21_ft_v4_targeted_fix_google_z21_c10_20260330_012450/pools.geojson \
  --out-geojson-3857 runs/segment/z21_ft_v4_targeted_fix_google_z21_c10_20260330_012450/pools_3857.geojson \
  --out-tile-summary-csv runs/segment/z21_ft_v4_targeted_fix_google_z21_c10_20260330_012450/pools_tiles.csv \
  --out-tile-summary-jsonl runs/segment/z21_ft_v4_targeted_fix_google_z21_c10_20260330_012450/pools_tiles.jsonl \
  --out-stats-json runs/segment/z21_ft_v4_targeted_fix_google_z21_c10_20260330_012450/pools_stats.json \
  --imgsz 1024 \
  --conf 0.10 \
  --iou 0.7 \
  --min-area-px 120 \
  --min-mask-area-px 120 \
  --min-area-m2 6.0 \
  --worldfile-crs EPSG:31983 \
  --device mps \
  --progress-every 500
```

### Threshold and recall notes

- `--conf`, `--iou`, `--min-area-px`, `--min-mask-area-px`, and `--min-area-m2` are the main precision/recall levers.
- `--recall-profile-2024` applies recall-first overrides (very low conf, lower area thresholds, higher max-det), intended for 2024 imagery probing.
- Use `--device mps` on Apple Silicon when available.

### Tile georeferencing expectations

- GeoTIFF tiles: CRS/geotransform read from file.
- PNG/JPG tiles with worldfiles (`.pgw`, `.jgw`, `.wld`): use `--worldfile-crs` (default `EPSG:31983`).
- XYZ tiles without worldfiles require `--z`.

## Annotation / Active Learning

### Candidate selection from inference summaries

Uncertain predictions:

```bash
.venv/bin/python tools/select_uncertain_tiles.py \
  --tiles-csv runs/segment/<run>/pools_tiles.csv \
  --min-conf 0.20 \
  --max-conf 0.45 \
  --num-tiles 500 \
  --out runs/segment/<run>/uncertain_candidates.csv
```

Likely misses:

```bash
.venv/bin/python tools/select_missed_pool_tiles.py \
  --tiles-csv runs/segment/<run>/pools_tiles.csv \
  --num-tiles 500 \
  --out runs/segment/<run>/missed_candidates.csv
```

### Build CVAT package + optional pre-annotations

```bash
.venv/bin/python tools/prepare_cvat_tiles.py \
  --tiles-root data/raw/google/sp_city_2020_rebuild_google_z21 \
  --tiles-csv runs/segment/<run>/uncertain_candidates.csv \
  --out-dir data/annotations/<round_name> \
  --max-tiles 500
```

```bash
.venv/bin/python tools/geojson_to_coco_cvat.py \
  --geojson runs/segment/<run>/pools.geojson \
  --manifest-csv data/annotations/<round_name>/manifest.csv \
  --images-dir data/annotations/<round_name>/JPEGImages \
  --out data/annotations/<round_name>/predictions_coco.json \
  --worldfile-crs EPSG:31983 \
  --min-confidence 0.20
```

### Hard negative / hard positive mining

- `tools/mine_hard_negatives.py`
- `tools/mine_hard_positives_from_misses.py`

Example hard-negative mining:

```bash
.venv/bin/python tools/mine_hard_negatives.py \
  --existing-label-roots data/datasets/geosampa_z21_v1/labels/train data/datasets/geosampa_z21_v1/labels/val \
  --prediction-label-roots runs/hardneg_predict/<cell_run>/labels \
  --tile-roots data/raw/google/sp_city_2020_rebuild_google_z21 \
  --out-dir runs/review/hard_negatives_z21 \
  --max-candidates 150
```

### Targeted/corrective Google z21 batches

- `tools/build_google_z21_targeted_annotation_batch.py`
  - profiles: `round_next`, `v5_corrective`
  - output: `runs/annotation_batches/<prefix>_<timestamp>/...`

Example (`v5_corrective`):

```bash
.venv/bin/python tools/build_google_z21_targeted_annotation_batch.py \
  --profile v5_corrective \
  --run-dir runs/segment/z21_ft_v3_real_google_z21_c18_20260328_134904 \
  --tiles-root data/raw/google/sp_city_2020_rebuild_google_z21 \
  --batch-prefix google_z21_v5_corrective \
  --tree-shadows-count 72 \
  --pool-edge-shadows-count 40 \
  --clean-positive-anchors-count 24 \
  --mixed-random-small-count 15
```

## Evaluation / Comparison

- Run ranking / checkpoint recommendation:

```bash
.venv/bin/python tools/compare_yolo_runs.py
```

- A/B tile recall check:

```bash
.venv/bin/python tools/ab_tile_eval.py \
  --model checkpoints/pools_z21_v4_targeted_fix_best.pt \
  --tiles-dir /tmp/tiles_eval \
  --z 21 \
  --out-geojson runs/segment/ab_eval/z21_candidate.geojson
```

- Folium visualization:

```bash
.venv/bin/python tools/folium_view.py \
  --tiles-dir data/raw/google/sp_city_2020_rebuild_google_z21 \
  --name z21_example \
  --pred-geojson runs/segment/<run>/pools.geojson \
  --assume-pred-crs EPSG:31983 \
  --base google_sat
```

## Common Pitfalls / Notes

- `tools/compare_yolo_runs.py` scans both:
  - `runs/segment`
  - `runs/segment/runs/segment`
  This nested path exists from older batch wrappers (`tools/run_pgw_geojson_batch.sh` defaults can create it).

- Some downstream tools expect strict filenames:
  - `tools/build_google_z21_targeted_annotation_batch.py` expects `<run-dir>/pools.geojson` and `<run-dir>/tiles.csv` exactly.
  - If inference wrote `pools_tiles.csv`, rename/symlink it to `tiles.csv` before using that script.

- CRS/worldfile assumptions are strict:
  - Most pipelines expect `EPSG:31983`.
  - `build_google_z21_targeted_annotation_batch.py` assumes axis-aligned worldfiles (rotated transforms are skipped).

- Keep raw imagery immutable:
  - Do not edit `data/raw/*` in place.

- Keep class mapping stable:
  - `pool = 0`.

- Full polygon Folium maps can be very heavy; use `--centroids` or `--max-features` for faster map loads when needed.

- Legacy code exists in `src/` and `scripts/`; current operational workflow is centered on `tools/`.
