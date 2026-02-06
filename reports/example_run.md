# Example Run

This file captures a small quickstart run using `configs/quickstart.yaml` and the baseline predictor.

Steps:

```bash
./scripts/quickstart.sh
```

Artifacts:
- `data/raw/tiles/tiles.csv`
- `data/raw/osm_pools.geojson`
- `data/processed/yolo/`
- `data/processed/predictions_baseline.csv`
- `reports/logs/estimate.json`
- `reports/figures/quickstart_pool_density.html`

Run summary:
- Baseline blue ratio threshold: 0.003
- Sample tiles: 60
- Mean pools per tile (baseline): 0.1
- Estimated pools in AOI (uniform): 254.8
