#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  tools/run_pgw_geojson_batch.sh <model.pt> [conf] [iou] [dedupe_iou] [out_dir] <tiles_dir1> [tiles_dir2 ...]

Args:
  model.pt      Path to YOLOv8-seg weights (e.g. runs/segment/.../weights/best.pt)
  conf          YOLO confidence threshold (default: 0.15)
  iou           YOLO NMS IoU for prediction (default: 0.7)
  dedupe_iou    Dedupe IoU threshold in geojson stage (default: 0.35)
  out_dir       Output directory root (default: segment/runs)

Notes on out_dir (normalization):
  - If out_dir is relative and does NOT start with "runs/", we treat it as a suffix under "runs/".
    Example default "segment/runs" => "runs/segment/runs"
  - If out_dir starts with "runs/" or is absolute (/...), we use it as-is.

Examples:
  tools/run_pgw_geojson_batch.sh runs/segment/moema_finetune_prod_split/weights/best.pt \
    0.05 0.7 0.35 segment/runs \
    data/raw/geosampa_ortho/sp_city_2020/cell_0026_0007 \
    data/raw/geosampa_ortho/sp_city_2020/cell_0026_0008
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

MODEL="${1:?model path}"
CONF="${2:-0.15}"
IOU="${3:-0.7}"
DEDUPE_IOU="${4:-0.35}"
OUT_DIR_RAW="${5:-segment/runs}"
shift 5 || true

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

# Normalize OUT_DIR:
# - absolute paths stay absolute
# - "runs/..." stays as-is
# - otherwise prefix with "runs/"
if [[ "$OUT_DIR_RAW" == /* ]]; then
  OUT_DIR="$OUT_DIR_RAW"
elif [[ "$OUT_DIR_RAW" == runs/* ]]; then
  OUT_DIR="$OUT_DIR_RAW"
else
  OUT_DIR="runs/$OUT_DIR_RAW"
fi

# Format tags like 0.35 -> 035, 0.05 -> 005
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
echo "OUT_DIR_RAW=$OUT_DIR_RAW"
echo "OUT_DIR=$OUT_DIR"
echo "TILES_DIRS=$#"

# Ensure out root exists (and so "latest" links have a home)
mkdir -p "$OUT_DIR"

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
    # NOTE: we pass OUT_DIR as already-normalized ("runs/segment/runs" by default),
    # so downstream tools should not prepend another "runs/segment" layer.
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

  # Stable symlink pointing at the latest output for this base cell
  ln -sf "${name}/pools_dedup_iou${dedupe_tag}_clean.geojson" "$latest_link" || true
  echo "latest: $latest_link -> ${name}/pools_dedup_iou${dedupe_tag}_clean.geojson"
done