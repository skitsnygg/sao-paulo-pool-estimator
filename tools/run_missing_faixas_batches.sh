#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
DL_SCRIPT="${ROOT}/tools/download_missing_faixas_geosampa_2020.py"

CODES_FILE="${1:-}"
OUT_ROOT="${2:-${ROOT}/data/raw/geosampa_ortho/sp_city_2020_missing_faixas}"
WORKERS="${WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-200}"
TIMEOUT="${TIMEOUT:-90}"
RETRIES="${RETRIES:-2}"
TMP_DIR="${TMP_DIR:-/tmp/geosampa_missing_batches}"

if [[ -z "${CODES_FILE}" ]]; then
  echo "Usage: $0 <codes_file> [out_root]" >&2
  exit 1
fi
if [[ ! -f "${CODES_FILE}" ]]; then
  echo "Codes file not found: ${CODES_FILE}" >&2
  exit 1
fi
if [[ ! -x "${PY}" ]]; then
  echo "Missing python venv: ${PY}" >&2
  exit 1
fi
if [[ ! -f "${DL_SCRIPT}" ]]; then
  echo "Missing downloader script: ${DL_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${TMP_DIR}"
rm -f "${TMP_DIR}"/batch_*

# Keep only valid faixa codes and unique-sort once.
rg -o "[0-9]{4}-[0-9]{3}" "${CODES_FILE}" | sort -u > "${TMP_DIR}/codes_clean.txt"
TOTAL_CODES="$(wc -l < "${TMP_DIR}/codes_clean.txt" | tr -d ' ')"

split -l "${BATCH_SIZE}" -d -a 4 "${TMP_DIR}/codes_clean.txt" "${TMP_DIR}/batch_"

LOG="${OUT_ROOT}/batch_download.log"
: > "${LOG}"

echo "[start] total_codes=${TOTAL_CODES} batch_size=${BATCH_SIZE} workers=${WORKERS}" | tee -a "${LOG}"

batch_idx=0
for batch in "${TMP_DIR}"/batch_*; do
  [[ -f "${batch}" ]] || continue
  batch_idx=$((batch_idx + 1))
  batch_codes="$(wc -l < "${batch}" | tr -d ' ')"
  echo "[batch ${batch_idx}] codes=${batch_codes} file=${batch}" | tee -a "${LOG}"

  env PYTHONUNBUFFERED=1 "${PY}" "${DL_SCRIPT}" \
    --codes-file "${batch}" \
    --out-root "${OUT_ROOT}" \
    --workers "${WORKERS}" \
    --request-retries "${RETRIES}" \
    --timeout "${TIMEOUT}" \
    2>&1 | tee -a "${LOG}"

  echo "[batch ${batch_idx}] done" | tee -a "${LOG}"
done

echo "[done] all batches complete" | tee -a "${LOG}"
