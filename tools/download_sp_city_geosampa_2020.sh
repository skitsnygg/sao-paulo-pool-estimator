#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AOI="${ROOT}/data/external/aoi_sao_paulo_municipality_epsg31983.geojson"
GRID_DIR="${ROOT}/data/external/sp_city_grid_2km_epsg31983"
OUT_ROOT="${ROOT}/data/raw/geosampa_ortho/sp_city_2020"
FAIL_LOG="${OUT_ROOT}/_failures.txt"

STEP_METERS="${STEP_METERS:-2000}"
CHIP_SIZE="${CHIP_SIZE:-1024}"
METERS_PER_PIXEL="${METERS_PER_PIXEL:-0.10}"
TIMEOUT_S="${TIMEOUT_S:-60}"
SLEEP_S="${SLEEP_S:-0.10}"
MAX_CHIPS="${MAX_CHIPS:-0}"   # 0 = no limit

PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv python: $PY" >&2
  exit 1
fi

if [[ ! -f "${AOI}" ]]; then
  echo "Missing AOI GeoJSON: ${AOI}" >&2
  exit 1
fi

mkdir -p "${GRID_DIR}" "${OUT_ROOT}"
: > "${FAIL_LOG}"

echo "[grid] building ${STEP_METERS}m grid from ${AOI} -> ${GRID_DIR}"

AOI="${AOI}" GRID_DIR="${GRID_DIR}" STEP="${STEP_METERS}" PYTHONPATH=. "${PY}" - <<'PY'
import json
import os
from pathlib import Path

from shapely.geometry import box, shape, mapping

aoi_path = Path(os.environ["AOI"])
grid_dir = Path(os.environ["GRID_DIR"])
step = float(os.environ.get("STEP", "2000"))

grid_dir.mkdir(parents=True, exist_ok=True)

payload = json.loads(aoi_path.read_text(encoding="utf-8"))
gtype = payload.get("type")

geoms = []
if gtype == "FeatureCollection":
    for f in payload.get("features", []):
        g = f.get("geometry")
        if g:
            geoms.append(shape(g))
elif gtype in ("Polygon", "MultiPolygon"):
    geoms = [shape(payload)]
else:
    raise SystemExit(f"Unsupported GeoJSON type: {gtype}")

if not geoms:
    raise SystemExit("AOI has no geometries")

# Union (avoid unary_union import; shapely geometry has union)
geom = geoms[0]
for g in geoms[1:]:
    geom = geom.union(g)

minx, miny, maxx, maxy = geom.bounds

written = 0
ix = 0
y = miny
while y < maxy:
    x = minx
    jx = 0
    while x < maxx:
        cell_poly = box(x, y, x + step, y + step)
        inter = cell_poly.intersection(geom)
        if (not inter.is_empty) and (inter.area > 1.0):
            cell_id = f"cell_{ix:04d}_{jx:04d}"
            out = grid_dir / f"{cell_id}.geojson"
            fc = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"cell_id": cell_id},
                    "geometry": mapping(inter),
                }],
            }
            out.write_text(json.dumps(fc), encoding="utf-8")
            written += 1
        x += step
        jx += 1
    y += step
    ix += 1

print(f"cells_written: {written}")
PY

echo "[grid] cells: $(ls -1 "${GRID_DIR}"/*.geojson 2>/dev/null | wc -l | tr -d ' ')"

# Download each cell
for aoi in "${GRID_DIR}"/*.geojson; do
  [[ -f "$aoi" ]] || continue
  cell_id="$(basename "$aoi" .geojson)"
  out_dir="${OUT_ROOT}/${cell_id}"

  # skip if already done
  if [[ -s "${out_dir}/chips.csv" ]]; then
    echo "[skip] ${cell_id} (chips.csv exists)"
    continue
  fi

  mkdir -p "${out_dir}"
  echo "[download] ${cell_id}"

  extra_max=()
  if [[ "${MAX_CHIPS}" != "0" ]]; then
    extra_max=(--max-chips "${MAX_CHIPS}")
  fi

  if ! PYTHONPATH=. "${PY}" -m src.data.fetch_geosampa_ortho \
      --aoi-geojson "${aoi}" \
      --aoi-crs "EPSG:31983" \
      --crs "EPSG:31983" \
      --chip-size "${CHIP_SIZE}" \
      --meters-per-pixel "${METERS_PER_PIXEL}" \
      --out-dir "${out_dir}" \
      --timeout "${TIMEOUT_S}" \
      --sleep "${SLEEP_S}" \
      "${extra_max[@]}"
  then
    echo "[error] ${cell_id}" | tee -a "${FAIL_LOG}"
    continue
  fi
done

echo "[done] failures logged to ${FAIL_LOG}"
