#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

GRID_DIR="${GRID_DIR:-${ROOT}/data/external/sp_city_grid_2km_epsg31983}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/raw/geosampa_ortho/sp_city_2020_rebuild}"
COVERAGE_OUT="${COVERAGE_OUT:-${OUT_ROOT}/_coverage}"

CRS="${CRS:-EPSG:31983}"
CHIP_SIZE="${CHIP_SIZE:-1024}"
METERS_PER_PIXEL="${METERS_PER_PIXEL:-0.10}"

WMS="${WMS:-https://raster.geosampa.prefeitura.sp.gov.br/geoserver/wms}"
LAYER="${LAYER:-geoportal:ORTO_RGB_2020}"
WORKERS="${WORKERS:-10}"
BLOCK_ROWS="${BLOCK_ROWS:-4}"
BLOCK_COLS="${BLOCK_COLS:-4}"
TIMEOUT="${TIMEOUT:-60}"
REQUEST_RETRIES="${REQUEST_RETRIES:-4}"
RETRY_DELAY="${RETRY_DELAY:-2.0}"
MAX_ROUNDS="${MAX_ROUNDS:-12}"
ROUND_SLEEP="${ROUND_SLEEP:-2.0}"
MIN_BYTES="${MIN_BYTES:-4096}"

# Default keeps old/broken dataset untouched by rebuilding into a new folder.
CLEAN="${CLEAN:-1}"
REUSE_EXISTING_FILES="${REUSE_EXISTING_FILES:-0}"
PRESERVE_STATUS="${PRESERVE_STATUS:-0}"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python: ${PY}" >&2
  exit 1
fi

if [[ ! -d "${GRID_DIR}" ]]; then
  echo "Missing grid dir: ${GRID_DIR}" >&2
  exit 1
fi

cmd=(
  "${PY}" "${ROOT}/tools/rebuild_sp_city_geosampa_2020.py" full-rebuild
  --grid-dir "${GRID_DIR}"
  --out-root "${OUT_ROOT}"
  --coverage-out "${COVERAGE_OUT}"
  --crs "${CRS}"
  --chip-size "${CHIP_SIZE}"
  --meters-per-pixel "${METERS_PER_PIXEL}"
  --wms "${WMS}"
  --layer "${LAYER}"
  --workers "${WORKERS}"
  --block-rows "${BLOCK_ROWS}"
  --block-cols "${BLOCK_COLS}"
  --timeout "${TIMEOUT}"
  --request-retries "${REQUEST_RETRIES}"
  --retry-delay "${RETRY_DELAY}"
  --max-rounds "${MAX_ROUNDS}"
  --round-sleep "${ROUND_SLEEP}"
  --min-bytes "${MIN_BYTES}"
  --dst-crs "EPSG:4326"
  --full-chip-count 50
)

if [[ "${CLEAN}" == "1" ]]; then
  cmd+=(--clean)
fi
if [[ "${REUSE_EXISTING_FILES}" == "1" ]]; then
  cmd+=(--reuse-existing-files)
fi
if [[ "${PRESERVE_STATUS}" == "1" ]]; then
  cmd+=(--preserve-status)
fi

echo "[rebuild] grid=${GRID_DIR}"
echo "[rebuild] out=${OUT_ROOT}"
echo "[rebuild] coverage_out=${COVERAGE_OUT}"
echo "[rebuild] wms=${WMS} layer=${LAYER}"
echo "[rebuild] workers=${WORKERS} block=${BLOCK_ROWS}x${BLOCK_COLS}"
echo "+ ${cmd[*]}"

PYTHONPATH=. "${cmd[@]}"

echo "[done] manifest: ${OUT_ROOT}/chips_manifest.csv"
echo "[done] coverage report: ${COVERAGE_OUT}/coverage_report.json"
