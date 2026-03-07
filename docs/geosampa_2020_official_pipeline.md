# GeoSampa 2020 Official Download Pipeline

This pipeline avoids brute-force WMS chip fetching and rebuilds chips from locally downloaded **official 2020 orthophoto files**.

It keeps compatibility with the existing layout and inference flow:
- `cell_xxxx_yyyy/r####_c####.png`
- matching `.pgw`
- `chips.csv` per cell
- `chips_manifest.csv` and coverage outputs
- CRS preserved as `EPSG:31983`

## Metadata/API anchors (no scraping)

Use GeoSampa metadata records as the source of truth:

- Records API base: `https://metadados.geosampa.prefeitura.sp.gov.br/geonetwork/srv/api/records`
- 2020 orthophoto record UUID: `892c862e-f564-4b1c-a3d0-9d28052e5d58`
- Mosaic record UUID: `b0046671-e190-46a8-ba1d-19f39c5bf4c8`
- Articulation record UUID: `b78ccbcd-d4d2-419f-80f9-28f6dde6f007`

Extract online resources from `.../records/<UUID>/related` and prioritize:

- WFS: `https://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs`
- WMS: `https://wms.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms`
- Raster WMS: `https://raster.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms`

Official 2020 download workflow entrypoint (manual download by faixa code/map selection):

- `https://geosampa.prefeitura.sp.gov.br/PaginasPublicas/_SBC.aspx`
- In the UI: `Pesquisar` -> `Download Imagens/MDC` -> `Ortofotos 2020 - RGB`

2020 imagery/index layers validated against capabilities:

- Raster imagery: `ORTO_RGB_2020`
- Raster mosaic: `MOSAICO_ORTO_RGB_10CM_20CM`
- Articulation WFS typeName: `geoportal:quadricula_orto_2020` (`cd_quadricula` as code field)
- Articulation WMS layer: `Articulacao_Orto_2020`

All of the above are compatible with `EPSG:31983`.

## 1) Fetch official articulation index (faixa grid)

```bash
.venv/bin/python tools/geosampa_2020_official_pipeline.py fetch-articulation \
  --out-geojson data/external/geosampa_2020/quadricula_orto_2020.geojson \
  --out-csv data/external/geosampa_2020/quadricula_orto_2020.csv \
  --out-codes data/external/geosampa_2020/quadricula_orto_2020_codes.txt
```

## 2) Select faixa codes for your AOI

`--aoi` should be a geometry in/convertible to EPSG:31983.

```bash
.venv/bin/python tools/geosampa_2020_official_pipeline.py select-faixas \
  --articulation-geojson data/external/geosampa_2020/quadricula_orto_2020.geojson \
  --aoi data/external/sp_city_boundary_epsg31983.geojson \
  --out-geojson data/external/geosampa_2020/selected_faixas.geojson \
  --out-csv data/external/geosampa_2020/selected_faixas.csv \
  --out-codes data/external/geosampa_2020/selected_faixas_codes.txt
```

Use `selected_faixas_codes.txt` as your area/faixa checklist when downloading from the official GeoSampa interface.

## 3) Put official downloaded rasters locally

Place extracted official files (GeoTIFF/JP2/etc.) under a local root, for example:

`data/raw/geosampa_ortho/sp_city_2020_official_sources`

## 4) Index local official sources

```bash
.venv/bin/python tools/geosampa_2020_official_pipeline.py index-sources \
  --sources-root data/raw/geosampa_ortho/sp_city_2020_official_sources \
  --out-csv data/external/geosampa_2020/sources_index.csv \
  --out-geojson-31983 data/external/geosampa_2020/sources_footprints_31983.geojson \
  --out-geojson-4326 data/external/geosampa_2020/sources_footprints_4326.geojson
```

## 5) Check missing faixa coverage before chipping

```bash
.venv/bin/python tools/geosampa_2020_official_pipeline.py compare-faixas \
  --expected-codes data/external/geosampa_2020/selected_faixas_codes.txt \
  --sources-index-csv data/external/geosampa_2020/sources_index.csv \
  --out-missing-codes data/external/geosampa_2020/missing_faixas.txt \
  --out-extra-codes data/external/geosampa_2020/extra_faixas.txt \
  --out-report-json data/external/geosampa_2020/faixa_comparison_report.json
```

If `missing_faixas.txt` is non-empty, complete those official downloads first.

## 5b) Bulk download only missing faixas via API (no UI/captcha)

Use WFS articulation + raster WMS to download only the remaining faixa codes directly in `EPSG:31983`.

```bash
.venv/bin/python tools/download_missing_faixas_geosampa_2020.py \
  --codes-file data/external/geosampa_2020/missing_faixas.txt \
  --out-root data/raw/geosampa_ortho/sp_city_2020_missing_faixas \
  --workers 12
```

Notes:
- This avoids the GeoSampa UI captcha flow entirely.
- If your missing list comes from spatial comparison, pass `missing_faixas_spatial.txt` instead.
- Outputs include `manifest.csv` and `summary.json` under `--out-root`.

If source filenames do not include faixa codes, use spatial validation instead:

```bash
.venv/bin/python tools/geosampa_2020_official_pipeline.py compare-faixas-spatial \
  --articulation-geojson data/external/geosampa_2020/selected_faixas.geojson \
  --sources-footprints-geojson data/external/geosampa_2020/sources_footprints_31983.geojson \
  --expected-codes data/external/geosampa_2020/selected_faixas_codes.txt \
  --out-detail-csv data/external/geosampa_2020/faixa_spatial_detail.csv \
  --out-missing-codes data/external/geosampa_2020/missing_faixas_spatial.txt \
  --out-report-json data/external/geosampa_2020/faixa_spatial_report.json
```

## 6) Build a VRT mosaic from downloaded official files

```bash
.venv/bin/python tools/geosampa_2020_official_pipeline.py build-vrt \
  --sources-index-csv data/external/geosampa_2020/sources_index.csv \
  --out-vrt data/raw/geosampa_ortho/sp_city_2020_rebuild_official/_source_ortho_2020.vrt \
  --overwrite
```

The command tries `gdalbuildvrt` first. If unavailable/broken, it falls back to an internal VRT writer for uniform north-up rasters.

Optional: filter by selected faixas.

```bash
.venv/bin/python tools/geosampa_2020_official_pipeline.py build-vrt \
  --sources-index-csv data/external/geosampa_2020/sources_index.csv \
  --filter-codes data/external/geosampa_2020/selected_faixas_codes.txt \
  --out-vrt data/raw/geosampa_ortho/sp_city_2020_rebuild_official/_source_ortho_2020.vrt \
  --overwrite
```

## 7) Rebuild chips from local VRT (no WMS)

```bash
tools/rebuild_sp_city_geosampa_2020_official.sh
```

Common env overrides:

```bash
SOURCES_ROOT=/abs/path/to/official_downloads \
OUT_ROOT=/Users/admin/sao-paulo-pool-estimator/data/raw/geosampa_ortho/sp_city_2020_rebuild_official \
WORKERS=8 \
tools/rebuild_sp_city_geosampa_2020_official.sh
```

Or provide an existing VRT directly:

```bash
SOURCE_VRT=/abs/path/to/source_ortho_2020.vrt \
tools/rebuild_sp_city_geosampa_2020_official.sh
```

Outputs:
- `data/raw/geosampa_ortho/sp_city_2020_rebuild_official/chips_manifest.csv`
- `data/raw/geosampa_ortho/sp_city_2020_rebuild_official/_coverage/coverage_report.json`
- coverage layers for Folium in `_coverage` (`cell_coverage_4326.geojson`, `coverage_union_4326.geojson`, `missing_tiles_4326.geojson`)

## 8) Rerun model inference on rebuilt dataset

After coverage is complete/acceptable:

```bash
.venv/bin/python tools/predict_full_ortho_unique_labels.py \
  --model checkpoints/pools_moema_ft_best.pt \
  --tiles-root data/raw/geosampa_ortho/sp_city_2020_rebuild_official \
  --out-dir runs/segment/citywide_2020_moemaft_best_conf010_rebuilt
```

Adjust inference args to your preferred confidence/IOU settings.
