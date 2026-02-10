#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

RUN_OUTPUT="$(PYTHONPATH=. "$PYTHON" -m src.models.predict \
  --config configs/quickstart.yaml \
  --source data/processed/yolo/images/test \
  --demo-mode 2>&1)"
printf '%s\n' "$RUN_OUTPUT"

FALLBACK_TYPE="$(printf '%s\n' "$RUN_OUTPUT" | sed -n 's/.*demo_fallback=\([^ ]*\).*/\1/p' | tail -n1)"
FALLBACK_TYPE="${FALLBACK_TYPE:-none}"

if [[ "$FALLBACK_TYPE" == "samples" || "$FALLBACK_TYPE" == "labels" ]]; then
  BASE_DIR="data/processed/predictions/demo"
else
  BASE_DIR="data/processed/predictions"
fi

POOL_COUNTS="$BASE_DIR/pool_counts.csv"
DETECTIONS="$BASE_DIR/detections.csv"

if [ ! -f "$POOL_COUNTS" ]; then
  echo "Missing $POOL_COUNTS" >&2
  exit 1
fi

if [ ! -f "$DETECTIONS" ]; then
  echo "Missing $DETECTIONS" >&2
  exit 1
fi

"$PYTHON" - << PY
import csv
from pathlib import Path

pool_counts_path = Path("$POOL_COUNTS")
fallback_type = "$FALLBACK_TYPE"
with pool_counts_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

if not rows:
    raise SystemExit("pool_counts.csv is empty")

if not fallback_type:
    fallback_type = (rows[0].get("demo_fallback_type") or "none").strip()
if fallback_type in {"samples", "labels"}:
    nonzero = any(int(row.get("pool_count", "0")) > 0 for row in rows)
    if not nonzero:
        raise SystemExit("Expected nonzero pool_count with demo fallback")
PY
