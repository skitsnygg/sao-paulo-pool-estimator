#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model path}"
CONF="${2:-0.15}"
IOU="${3:-0.7}"
DEDUPE_IOU="${4:-0.35}"
OUT_DIR="${5:-runs/predict}"
shift 5 || true

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model.pt> [conf] [iou] [dedupe_iou] [out_dir] <tiles_dir1> [tiles_dir2 ...]" >&2
  exit 2
fi

dedupe_tag="$(python - <<PY
v=float("$DEDUPE_IOU")
print(f"{int(round(v*100)):03d}")
PY
)"
conf_tag="$(python - <<PY
v=float("$CONF")
print(f"{int(round(v*100)):03d}")
PY
)"

echo "MODEL=$MODEL"
echo "CONF=$CONF (tag=$conf_tag) IOU=$IOU"
echo "DEDUPE_IOU=$DEDUPE_IOU (tag=$dedupe_tag)"
echo "OUT_DIR=$OUT_DIR"
echo "TILES_DIRS=$#"

for TILES_DIR in "$@"; do
  base="$(basename "$TILES_DIR")"
  name="${base}_conf${conf_tag}"
  out_subdir="${OUT_DIR}/${name}"
  final_geojson="${out_subdir}/pools_dedup_iou${dedupe_tag}_clean.geojson"
  latest_link="${OUT_DIR}/${base}_latest.geojson"

  echo
  echo "=== $TILES_DIR -> $out_subdir ==="

  if [[ ! -d "$TILES_DIR" ]]; then
    echo "skip: tiles dir not found: $TILES_DIR" >&2
    continue
  fi

  if [[ -f "$final_geojson" ]]; then
    echo "skip: already have $final_geojson"
  else
    PYTHONPATH=. .venv/bin/python tools/predict_pgw_tiles_to_geojson.py \
      --model "$MODEL" \
      --tiles-dir "$TILES_DIR" \
      --out-dir "$OUT_DIR" \
      --name "$name" \
      --imgsz 1024 \
      --conf "$CONF" \
      --iou "$IOU" \
      --dedupe \
      --dedupe-iou "$DEDUPE_IOU" \
      --dedupe-buffer 0
  fi

  ln -sf "${name}/pools_dedup_iou${dedupe_tag}_clean.geojson" "$latest_link" || true
  echo "latest: $latest_link -> ${name}/pools_dedup_iou${dedupe_tag}_clean.geojson"
done
