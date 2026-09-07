#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/base_datasets/TARS-RLHF-V}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/base_datasets}"
NUM_SAMPLES="${NUM_SAMPLES:-4800}"
SEED="${SEED:-42}"
PY="${PY:-python}"
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT"
huggingface-cli download openbmb/RLHF-V-Dataset --repo-type dataset --local-dir "$DATA_ROOT"
PARQUET="${PARQUET:-$(find "$DATA_ROOT" -type f -name '*.parquet' | head -1)}"
[[ -n "$PARQUET" ]] || { echo "RLHF-V parquet not found in $DATA_ROOT" >&2; exit 2; }
"$PY" "$ROOT/base_operations/make_tars_opadpo_dataset.py" --parquet "$PARQUET" \
  --output-root "$OUTPUT_ROOT" --num-samples "$NUM_SAMPLES" --seed "$SEED"
