# Production Inference (z18 Pools)

This is the production recipe for generating stable, deduped pool polygons at z18.

## Recommended parameters (recall-first)
- `conf=0.05`, `iou=0.5` (keeps recall)
- `min-area-px=30` (mask pixel area threshold; earlier runs showed many small pools around ~60px², so 30 keeps recall without letting through tiny specks)
- `precision=7` (stable coordinate rounding)
- dedupe `iou=0.35`

## A/B tile-recall sanity (pool-heavy tiles)
Use the evaluation harness to compare the tile-strong base model vs the collapsed pseudo-label model.

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

```bash
.venv/bin/python tools/ab_tile_eval.py \
  --model checkpoints/pools_best.pt \
  --tiles-dir /tmp/tiles18_poolheavy_200 \
  --z 18 \
  --imgsz 1024 \
  --conf 0.05 \
  --iou 0.5 \
  --min-area-px 30 \
  --out-geojson runs/segment/ab_eval/pseudo_poolheavy.geojson
```

## Production pipeline (full tiles)
1) **Inference** (tile-strong base model):
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
```

2) **Dedupe**:
```bash
.venv/bin/python tools/dedupe_geojson_polygons.py \
  --in-geojson runs/segment/predict_xyz_z18/pools_conf05_iou05_area30.geojson \
  --out-geojson runs/segment/predict_xyz_z18/prod_dedup.geojson \
  --iou 0.35 \
  --precision 7 \
  --stats
```

3) **Overlap audit**:
```bash
.venv/bin/python tools/audit_geojson_overlaps.py \
  --in-geojson runs/segment/predict_xyz_z18/prod_dedup.geojson \
  --iou 0.35 \
  --top-k 10
```

If the audit exits with code 0, the output is ready to ship.
