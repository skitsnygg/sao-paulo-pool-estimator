#!/usr/bin/env bash
set -euo pipefail

cd ~/sao-paulo-pool-estimator || exit 1

MODEL="/Users/admin/sao-paulo-pool-estimator/runs/segment/geosampa_2020_reviewed_empties_blacklist_prunedmoema2/weights/last.pt"
TILE_BASE="/Users/admin/sao-paulo-pool-estimator/data/raw/geosampa_ortho/sp_city_2020_rebuild_official"
PRED_BASE="/Users/admin/sao-paulo-pool-estimator/runs/hardneg_predict"
OUT_DIR="/Users/admin/sao-paulo-pool-estimator/runs/review/hard_negatives_batch_auto"

EXISTING_LABEL_ROOTS=(
  "/Users/admin/sao-paulo-pool-estimator/data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned/labels/train"
  "/Users/admin/sao-paulo-pool-estimator/data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned/labels/val"
)

NUM_CELLS=20
CONF="0.15"
IMGSZ="1024"
MAX_CANDIDATES="150"
SEED="42"

# Set to 1 to export files after the dry-run.
RUN_EXPORT=1

# Set to 1 if you want to delete and re-run prediction folders that already exist.
RERUN_EXISTING_PREDICTIONS=0

# These are included first if present, then the script fills the rest randomly.
PRIORITY_CELLS=(
  "cell_0026_0007"
  "cell_0018_0002"
  "cell_0024_0014"
  "cell_0021_0015"
  "cell_0022_0013"
)

EXCLUDE_CELLS=(
  "cell_0004_0006"
  "cell_0006_0002"
  "cell_0009_0006"
  "cell_0018_0002"
  "cell_0018_0004"
  "cell_0019_0005"
  "cell_0021_0015"
  "cell_0022_0013"
  "cell_0023_0008"
  "cell_0024_0014"
  "cell_0024_0020"
  "cell_0025_0010"
  "cell_0026_0005"
  "cell_0026_0007"
  "cell_0028_0008"
  "cell_0029_0017"
  "cell_0030_0010"
  "cell_0032_0001"
  "cell_0033_0001"
)
EXCLUDE_CELLS_FILE="$(mktemp)"
for c in "${EXCLUDE_CELLS[@]}"; do
  echo "$TILE_BASE/$c" >> "$EXCLUDE_CELLS_FILE"
done

echo "Repo: $(pwd)"
echo "Model: $MODEL"
echo "Tile base: $TILE_BASE"
echo "Prediction base: $PRED_BASE"
echo "Output dir: $OUT_DIR"
echo

if [ ! -f "$MODEL" ]; then
  echo "Model not found: $MODEL"
  exit 1
fi

if [ ! -d "$TILE_BASE" ]; then
  echo "Tile base not found: $TILE_BASE"
  exit 1
fi

mkdir -p "$PRED_BASE"
mkdir -p "$OUT_DIR"

ALL_CELLS_FILE="$(mktemp)"
RANDOM_CELLS_FILE="$(mktemp)"
SELECTED_CELLS_FILE="$(mktemp)"
trap 'rm -f "$ALL_CELLS_FILE" "$RANDOM_CELLS_FILE" "$SELECTED_CELLS_FILE"' EXIT

find "$TILE_BASE" -mindepth 1 -maxdepth 1 -type d -name 'cell_*' | sort > "$ALL_CELLS_FILE.tmp"
grep -Fvx -f "$EXCLUDE_CELLS_FILE" "$ALL_CELLS_FILE.tmp" > "$ALL_CELLS_FILE"
rm -f "$ALL_CELLS_FILE.tmp"

TOTAL_CELLS=$(wc -l < "$ALL_CELLS_FILE" | tr -d ' ')
if [ "$TOTAL_CELLS" -eq 0 ]; then
  echo "No cell_* directories found under: $TILE_BASE"
  exit 1
fi

echo "Found $TOTAL_CELLS total cells."
echo

# Add priority cells first.
for cell_name in "${PRIORITY_CELLS[@]}"; do
  cell_path="$TILE_BASE/$cell_name"
  if [ -d "$cell_path" ]; then
    if ! grep -Fxq "$cell_path" "$SELECTED_CELLS_FILE" 2>/dev/null; then
      echo "$cell_path" >> "$SELECTED_CELLS_FILE"
    fi
  fi
done

# Shuffle all cells using Python so this works on macOS without shuf.
python - "$ALL_CELLS_FILE" "$SEED" > "$RANDOM_CELLS_FILE" <<'PY'
import random
import sys
from pathlib import Path

src = Path(sys.argv[1])
seed = int(sys.argv[2])

items = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
random.Random(seed).shuffle(items)
for item in items:
    print(item)
PY

CURRENT_COUNT=$(wc -l < "$SELECTED_CELLS_FILE" | tr -d ' ')
while IFS= read -r cell_path; do
  [ -n "$cell_path" ] || continue
  if [ "$CURRENT_COUNT" -ge "$NUM_CELLS" ]; then
    break
  fi
  if ! grep -Fxq "$cell_path" "$SELECTED_CELLS_FILE" 2>/dev/null; then
    echo "$cell_path" >> "$SELECTED_CELLS_FILE"
    CURRENT_COUNT=$((CURRENT_COUNT + 1))
  fi
done < "$RANDOM_CELLS_FILE"

SELECTED_COUNT=$(wc -l < "$SELECTED_CELLS_FILE" | tr -d ' ')
if [ "$SELECTED_COUNT" -eq 0 ]; then
  echo "No cells selected."
  exit 1

echo "Selected $SELECTED_COUNT cells:"
sed 's/^/  /' "$SELECTED_CELLS_FILE"
echo

PRED_LABEL_ARGS_FILE="$(mktemp)"
trap 'rm -f "$ALL_CELLS_FILE" "$RANDOM_CELLS_FILE" "$SELECTED_CELLS_FILE" "$PRED_LABEL_ARGS_FILE" "$EXCLUDE_CELLS_FILE"' EXIT
: > "$PRED_LABEL_ARGS_FILE"

while IFS= read -r CELL_DIR; do
  [ -n "$CELL_DIR" ] || continue

  CELLNAME="$(basename "$CELL_DIR")"
  DEST_DIR="$PRED_BASE/$CELLNAME"
  LABEL_DIR="$DEST_DIR/labels"

  echo "=== Processing $CELLNAME ==="

  if [ -d "$LABEL_DIR" ] && [ "$RERUN_EXISTING_PREDICTIONS" -ne 1 ]; then
    echo "Prediction labels already exist, skipping inference: $LABEL_DIR"
  else
    rm -rf "$DEST_DIR"

    yolo predict \
      model="$MODEL" \
      source="$CELL_DIR" \
      task=segment \
      imgsz="$IMGSZ" \
      conf="$CONF" \
      save=True \
      save_txt=True \
      save_conf=True \
      project="$PRED_BASE" \
      name="$CELLNAME" \
      exist_ok=True
  fi

  if [ -d "$LABEL_DIR" ]; then
    label_count=$(find "$LABEL_DIR" -type f -name '*.txt' | wc -l | tr -d ' ')
    echo "Label txt files: $label_count"
    if [ "$label_count" -gt 0 ]; then
      echo "$LABEL_DIR" >> "$PRED_LABEL_ARGS_FILE"
    fi
  else
    echo "No labels dir found: $LABEL_DIR"
  fi

  echo
done < "$SELECTED_CELLS_FILE"

PRED_ROOT_COUNT=$(wc -l < "$PRED_LABEL_ARGS_FILE" | tr -d ' ')
if [ "$PRED_ROOT_COUNT" -eq 0 ]; then
  echo "No prediction label directories with txt files were found."
  exit 1
fi

echo "Prediction label roots to mine:"
sed 's/^/  /' "$PRED_LABEL_ARGS_FILE"
echo

DRY_RUN_CMD=(
  python tools/mine_hard_negatives.py
  --existing-label-roots
)

for p in "${EXISTING_LABEL_ROOTS[@]}"; do
  DRY_RUN_CMD+=( "$p" )
done

DRY_RUN_CMD+=( --prediction-label-roots )
while IFS= read -r p; do
  [ -n "$p" ] || continue
  DRY_RUN_CMD+=( "$p" )
done < "$PRED_LABEL_ARGS_FILE"

DRY_RUN_CMD+=( --tile-roots "$TILE_BASE" )
DRY_RUN_CMD+=( --out-dir "$OUT_DIR" )
DRY_RUN_CMD+=( --max-candidates "$MAX_CANDIDATES" )
DRY_RUN_CMD+=( --seed "$SEED" )
DRY_RUN_CMD+=( --write-pred-label-copies )
DRY_RUN_CMD+=( --dry-run )

echo "=== Dry-run mining ==="
printf '%q ' "${DRY_RUN_CMD[@]}"
echo
echo

"${DRY_RUN_CMD[@]}"

echo
echo "Dry-run complete."
echo

if [ "$RUN_EXPORT" -eq 1 ]; then
  EXPORT_CMD=(
    python tools/mine_hard_negatives.py
    --existing-label-roots
  )

  for p in "${EXISTING_LABEL_ROOTS[@]}"; do
    EXPORT_CMD+=( "$p" )
  done

  EXPORT_CMD+=( --prediction-label-roots )
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    EXPORT_CMD+=( "$p" )
  done < "$PRED_LABEL_ARGS_FILE"

  EXPORT_CMD+=( --tile-roots "$TILE_BASE" )
  EXPORT_CMD+=( --out-dir "$OUT_DIR" )
  EXPORT_CMD+=( --max-candidates "$MAX_CANDIDATES" )
  EXPORT_CMD+=( --seed "$SEED" )
  EXPORT_CMD+=( --write-pred-label-copies )

  echo "=== Exporting mined candidates ==="
  printf '%q ' "${EXPORT_CMD[@]}"
  echo
  echo

  "${EXPORT_CMD[@]}"

  echo
  echo "Finished."
  echo "Review folder: $OUT_DIR"
  echo "Images: $OUT_DIR/images"
  echo "Prediction labels: $OUT_DIR/pred_labels"
  echo "Manifest: $OUT_DIR/manifest.csv"
  echo "Summary: $OUT_DIR/summary.json"
else
  echo "RUN_EXPORT=0, so only the dry-run was executed."
fi