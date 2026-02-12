# São Paulo Pool Estimator

Estimate swimming pool polygons in São Paulo using tile-based YOLOv8 segmentation and GeoJSON post-processing.

## What's in this repo
- **Segmentation models** (YOLOv8). The current production base model is:
  `runs/segment/jardins_seg_v2_1024_s2/weights/best.pt`.
- **Tile inference pipeline** to generate GeoJSON polygons from z18 tiles.
- **Geometry pipeline** for dedupe + overlap audit to ensure clean, stable output.

## Production GeoJSON pipeline (z18 pools)
The production path is:
1) **Inference** -> 2) **Dedupe** -> 3) **Overlap audit**.

Full, copy/paste commands live in `docs/production_inference.md`.

### Quick commands (recall-first)
```bash
.venv/bin/python tools/predict_tiles_to_geojson.py \
  --model runs/segment/jardins_seg_v2_1024_s2/weights/best.pt \
  --tiles-dir data/raw/tiles/18 \
  --z 18 \
  --imgsz 1024 \
  --conf 0.05 \
  --iou 0.5 \
  --min-area-px 30 \
  --precision 7 \
  --out-geojson runs/segment/predict_xyz_z18/pools_conf05_iou05_area30.geojson

.venv/bin/python tools/dedupe_geojson_polygons.py \
  --in-geojson runs/segment/predict_xyz_z18/pools_conf05_iou05_area30.geojson \
  --out-geojson runs/segment/predict_xyz_z18/prod_dedup.geojson \
  --iou 0.35 \
  --precision 7 \
  --stats

.venv/bin/python tools/audit_geojson_overlaps.py \
  --in-geojson runs/segment/predict_xyz_z18/prod_dedup.geojson \
  --iou 0.35 \
  --top-k 10
```

## Model status (important)
- The pseudo-label fine-tune (`checkpoints/pools_best.pt`) **collapsed recall** on real tiles. Do **not** ship it.
- Use the tile-strong base model at `runs/segment/jardins_seg_v2_1024_s2/weights/best.pt`.

## Tile recall sanity harness
Use `tools/ab_tile_eval.py` to compare models on pool-heavy tiles.

```bash
.venv/bin/python tools/ab_tile_eval.py \
  --model runs/segment/jardins_seg_v2_1024_s2/weights/best.pt \
  --tiles-dir /tmp/tiles18_poolheavy_200 \
  --z 18 \
  --imgsz 1024 \
  --conf 0.05 \
  --iou 0.5 \
  --min-area-px 30 \
  --out-geojson runs/segment/ab_eval/base_poolheavy.geojson
```

## Geometry tools
- `tools/dedupe_geojson_polygons.py`: STRtree-based dedupe with robust geometry repair and stable output.
- `tools/audit_geojson_overlaps.py`: audits remaining overlaps; exits nonzero if any IoU >= threshold.

## Inference details
- `tools/predict_tiles_to_geojson.py` reads tiles named `{x}_{y}.jpg` under `data/raw/tiles/18`.
- **Area filtering is based on mask pixel area** (from `masks.data`) when available, falling back to polygon area.
- Output polygons are rounded (`--precision`), CCW-oriented, and repaired if needed.

## Tests / CI
- Unit tests live in `tests/`.
- CI runs:
  - `python -m py_compile tools/*.py tests/*.py`
  - `python -m unittest discover -s tests -p "test_*.py" -v`
  - dedupe + audit on a small fixture `tests/fixtures/overlap_fixture.geojson`

## Legacy quickstart (older pipeline)
```bash
bash scripts/quickstart.sh
```
Note: quickstart tiles are outside São Paulo; zero pools can be expected.

## Outputs (legacy pipeline)
- `data/processed/predictions.csv` - Model predictions per tile
- `reports/logs/estimate.json` - Final estimate for São Paulo
- `reports/figures/quickstart_pool_density.html` - Pool density map

## Dependencies
- Python 3.10+ recommended
- `ultralytics`, `shapely`, `Pillow`

See `requirements.txt` for the full set.
