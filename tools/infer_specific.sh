MODEL="/Users/admin/sao-paulo-pool-estimator/runs/segment/geosampa_2020_reviewed_empties_blacklist_prunedmoema2/weights/last.pt"
TILE_BASE="/Users/admin/sao-paulo-pool-estimator/data/raw/geosampa_ortho/sp_city_2020_rebuild_official"
PRED_BASE="/Users/admin/sao-paulo-pool-estimator/runs/hardneg_predict"
OUT_DIR="/Users/admin/sao-paulo-pool-estimator/runs/review/hard_negatives_batch2"

CELLS=(
    "cell_0025_0020",
    "cell_0029_0015",
    "cell_0033_0003",
    "cell_0029_0002",
    "cell_0025_0008",
    "cell_0026_0014",
    "cell_0023_0021",
    "cell_0023_0010",
    "cell_0022_0005"
)

CONF="0.15"
IMGSZ="1024"
MAX_CANDIDATES="150"

PRED_LABEL_ARGS=()

for CELLNAME in "${CELLS[@]}"; do
  CELL_DIR="$TILE_BASE/$CELLNAME"

  if [ ! -d "$CELL_DIR" ]; then
    echo "Skipping missing cell dir: $CELL_DIR"
    continue
  fi

  echo
  echo "=== Predicting $CELLNAME ==="

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

  LABEL_DIR="$PRED_BASE/$CELLNAME/labels"
  if [ -d "$LABEL_DIR" ]; then
    PRED_LABEL_ARGS+=( "$LABEL_DIR" )
    echo "Added label dir: $LABEL_DIR"
    find "$LABEL_DIR" -type f -name '*.txt' | wc -l
  else
    echo "No labels dir found for $CELLNAME: $LABEL_DIR"
  fi
done

echo
echo "Prediction label roots:"
printf '  %s\n' "${PRED_LABEL_ARGS[@]}"

if [ "${#PRED_LABEL_ARGS[@]}" -eq 0 ]; then
  echo "No prediction label dirs found. Exiting."
  exit 1
fi

echo
echo "=== Dry-run mining ==="

python tools/mine_hard_negatives.py \
  --existing-label-roots \
    data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned/labels/train \
    data/datasets/geosampa_master_2020_with_reviewed_empties_v1_blacklist_pruned/labels/val \
  --prediction-label-roots \
    "${PRED_LABEL_ARGS[@]}" \
  --tile-roots \
    "$TILE_BASE" \
  --out-dir \
    "$OUT_DIR" \
  --max-candidates \
    "$MAX_CANDIDATES" \
  --write-pred-label-copies \
  --dry-run