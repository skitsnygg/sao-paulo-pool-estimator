#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

GRID_DIR="${GRID_DIR:-${ROOT}/data/external/sp_city_grid_2km_epsg31983}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/raw/geosampa_ortho/sp_city_2020_rebuild_official}"
COVERAGE_OUT="${COVERAGE_OUT:-${OUT_ROOT}/_coverage}"

# Either provide SOURCE_VRT directly, or provide SOURCES_ROOT and this script builds a VRT once.
SOURCE_VRT="${SOURCE_VRT:-${OUT_ROOT}/_source_ortho_2020.vrt}"
SOURCES_ROOT="${SOURCES_ROOT:-}"
EXTENSIONS="${EXTENSIONS:-tif,tiff,jp2,j2k,ecw}"

CRS="${CRS:-EPSG:31983}"
CHIP_SIZE="${CHIP_SIZE:-1024}"
METERS_PER_PIXEL="${METERS_PER_PIXEL:-0.10}"

WORKERS="${WORKERS:-6}"
RENDER_RETRIES="${RENDER_RETRIES:-1}"
RETRY_DELAY="${RETRY_DELAY:-1.0}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
ROUND_SLEEP="${ROUND_SLEEP:-1.0}"
BLANK_MAX_VALUE="${BLANK_MAX_VALUE:-0}"
RESAMPLING="${RESAMPLING:-nearest}"

CLEAN="${CLEAN:-1}"
REUSE_EXISTING_FILES="${REUSE_EXISTING_FILES:-0}"
PRESERVE_STATUS="${PRESERVE_STATUS:-0}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

if [[ ! -x "${PY}" ]]; then
  echo "Missing venv python: ${PY}" >&2
  exit 1
fi

if [[ ! -d "${GRID_DIR}" ]]; then
  echo "Missing grid dir: ${GRID_DIR}" >&2
  exit 1
fi

if [[ ! -f "${SOURCE_VRT}" ]]; then
  if [[ -z "${SOURCES_ROOT}" ]]; then
    echo "SOURCE_VRT not found and SOURCES_ROOT is empty." >&2
    echo "Set SOURCE_VRT=<existing.vrt> or SOURCES_ROOT=<official_rasters_dir>." >&2
    exit 1
  fi

  echo "[official] building source VRT from ${SOURCES_ROOT}"
  "${PY}" "${ROOT}/tools/geosampa_2020_official_pipeline.py" build-vrt \
    --sources-root "${SOURCES_ROOT}" \
    --extensions "${EXTENSIONS}" \
    --out-vrt "${SOURCE_VRT}" \
    --overwrite
fi

cmd=(
  "${PY}" "${ROOT}/tools/geosampa_2020_official_pipeline.py" full-rebuild-from-vrt
  --grid-dir "${GRID_DIR}"
  --out-root "${OUT_ROOT}"
  --coverage-out "${COVERAGE_OUT}"
  --source-vrt "${SOURCE_VRT}"
  --crs "${CRS}"
  --chip-size "${CHIP_SIZE}"
  --meters-per-pixel "${METERS_PER_PIXEL}"
  --workers "${WORKERS}"
  --render-retries "${RENDER_RETRIES}"
  --retry-delay "${RETRY_DELAY}"
  --max-rounds "${MAX_ROUNDS}"
  --round-sleep "${ROUND_SLEEP}"
  --blank-max-value "${BLANK_MAX_VALUE}"
  --resampling "${RESAMPLING}"
  --statuses "pending,failed,missing"
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
if [[ "${OVERWRITE_EXISTING}" == "1" ]]; then
  cmd+=(--overwrite-existing)
fi

echo "[official] grid=${GRID_DIR}"
echo "[official] out=${OUT_ROOT}"
echo "[official] coverage_out=${COVERAGE_OUT}"
echo "[official] source_vrt=${SOURCE_VRT}"
echo "+ ${cmd[*]}"

PYTHONPATH=. "${cmd[@]}"

echo "[done] manifest: ${OUT_ROOT}/chips_manifest.csv"
echo "[done] coverage report: ${COVERAGE_OUT}/coverage_report.json"
