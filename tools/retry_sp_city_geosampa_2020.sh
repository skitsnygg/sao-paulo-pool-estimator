#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRY_PATHS="${ROOT}/data/raw/geosampa_ortho/sp_city_2020/_retry_paths.txt"
GRID_DIR="${ROOT}/data/external/aoi_sp_grid_2km"
OUT_ROOT="${ROOT}/data/raw/geosampa_ortho/sp_city_2020"

if [[ ! -f "${RETRY_PATHS}" ]]; then
  echo "Missing retry list: ${RETRY_PATHS}" >&2
  exit 1
fi

tmp_cells="$(mktemp)"
fail_log="${OUT_ROOT}/_retry_failures.txt"
touch "${fail_log}"

PYTHONPATH=. "${ROOT}/.venv/bin/python" - <<'PY' "${RETRY_PATHS}" "${tmp_cells}"
from pathlib import Path
import re
import sys

src = Path(sys.argv[1]).read_text().splitlines()
cells = set()
for line in src:
    m = re.search(r"(cell_\d+_\d+)", line)
    if not m:
        continue
    cells.add(m.group(1))
cells = sorted(cells)
Path(sys.argv[2]).write_text("\\n".join(cells) + "\\n", encoding="utf-8")
print("cells:", len(cells))
PY

while read -r cell; do
  [[ -z "${cell}" ]] && continue
  aoi="${GRID_DIR}/${cell}.geojson"
  out="${OUT_ROOT}/${cell}"
  if [[ ! -f "${aoi}" ]]; then
    echo "[skip] ${cell} (missing AOI: ${aoi})" | tee -a "${fail_log}"
    continue
  fi
  mkdir -p "${out}"
  echo "[retry] ${cell}"
  if ! PYTHONPATH=. "${ROOT}/.venv/bin/python" src/data/fetch_geosampa_ortho.py \
      --aoi-geojson "${aoi}" \
      --aoi-crs "EPSG:31983" \
      --crs "EPSG:31983" \
      --chip-size 1024 \
      --meters-per-pixel 0.10 \
      --out-dir "${out}" \
      --timeout 60 \
      --sleep 0.1; then
    echo "[error] ${cell}" | tee -a "${fail_log}"
    continue
  fi
done < "${tmp_cells}"

rm -f "${tmp_cells}"
echo "[done] failures logged to ${fail_log}"
