#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

OUT_ROOT="${OUT_ROOT:-${ROOT}/data/raw/geosampa_ortho/sp_city_2020_rebuild}"
COVERAGE_OUT="${COVERAGE_OUT:-${OUT_ROOT}/_coverage}"
MANIFEST_CSV="${MANIFEST_CSV:-${OUT_ROOT}/chips_manifest.csv}"

WMS="${WMS:-https://raster.geosampa.prefeitura.sp.gov.br/geoserver/wms}"
LAYER="${LAYER:-geoportal:ORTO_RGB_2020}"
CRS="${CRS:-EPSG:31983}"
WORKERS="${WORKERS:-10}"
BLOCK_ROWS="${BLOCK_ROWS:-4}"
BLOCK_COLS="${BLOCK_COLS:-4}"
TIMEOUT="${TIMEOUT:-60}"
REQUEST_RETRIES="${REQUEST_RETRIES:-4}"
RETRY_DELAY="${RETRY_DELAY:-2.0}"
MAX_ROUNDS="${MAX_ROUNDS:-8}"
ROUND_SLEEP="${ROUND_SLEEP:-2.0}"
MIN_BYTES="${MIN_BYTES:-4096}"
STATUSES="${STATUSES:-pending,failed,missing}"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python: ${PY}" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_CSV}" ]]; then
  echo "Missing manifest CSV: ${MANIFEST_CSV}" >&2
  exit 1
fi

cmd=(
  "${PY}" "${ROOT}/tools/rebuild_sp_city_geosampa_2020.py" retry-until-complete
  --out-root "${OUT_ROOT}"
  --manifest-csv "${MANIFEST_CSV}"
  --wms "${WMS}"
  --layer "${LAYER}"
  --crs "${CRS}"
  --workers "${WORKERS}"
  --block-rows "${BLOCK_ROWS}"
  --block-cols "${BLOCK_COLS}"
  --timeout "${TIMEOUT}"
  --request-retries "${REQUEST_RETRIES}"
  --retry-delay "${RETRY_DELAY}"
  --max-rounds "${MAX_ROUNDS}"
  --round-sleep "${ROUND_SLEEP}"
  --min-bytes "${MIN_BYTES}"
  --statuses "${STATUSES}"
)

echo "[retry] out=${OUT_ROOT}"
echo "[retry] workers=${WORKERS} block=${BLOCK_ROWS}x${BLOCK_COLS}"
echo "+ ${cmd[*]}"
PYTHONPATH=. "${cmd[@]}"

validate_cmd=(
  "${PY}" "${ROOT}/tools/rebuild_sp_city_geosampa_2020.py" validate
  --out-root "${OUT_ROOT}"
  --manifest-csv "${MANIFEST_CSV}"
  --coverage-out "${COVERAGE_OUT}"
  --crs "${CRS}"
  --dst-crs "EPSG:4326"
  --full-chip-count 50
)

echo "+ ${validate_cmd[*]}"
PYTHONPATH=. "${validate_cmd[@]}"

echo "[done] coverage report: ${COVERAGE_OUT}/coverage_report.json"
