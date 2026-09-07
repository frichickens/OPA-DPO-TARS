#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; WORK_ROOT="${WORK_ROOT:-$ROOT/work}"
ENV_NAME="${ENV_NAME:-TARS_DPO}"; MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
DATA_ROOT="${DATA_ROOT:-$WORK_ROOT/datasets/OPA-DPO-7B}"; AMBER_ROOT="${AMBER_ROOT:-$WORK_ROOT/datasets/AMBER}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/checkpoints/tars_on_opadpo_7b}"; EVAL_ROOT="${EVAL_ROOT:-$WORK_ROOT/results/AMBER}"
bash "$ROOT/run/setup_tars_code.sh"
conda run -n "$ENV_NAME" env MODEL_ROOT="$MODEL_ROOT" bash "$ROOT/run/setup_base_model_7b.sh"
conda run -n "$ENV_NAME" env DATA_ROOT="$DATA_ROOT" bash "$ROOT/run/setup_opadpo_pairs.sh"
conda run -n "$ENV_NAME" env AMBER_ROOT="$AMBER_ROOT" bash "$ROOT/run/setup_amber.sh"
conda run -n "$ENV_NAME" env MODEL_DIR="$MODEL_ROOT/llava-v1.5-7b" VISION_TOWER_DIR="$MODEL_ROOT/vision_tower-clip336" \
 DATA_PATH="$DATA_ROOT/opadpo_7b_pairs_tars.parquet" OUTPUT_DIR="$OUTPUT_DIR" GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
 NPROC="${NPROC:-8}" GRAD_ACC="${GRAD_ACC:-8}" bash "$ROOT/run/train_tars_on_opadpo.sh"
conda run -n "$ENV_NAME" env MODEL_PATH="$OUTPUT_DIR" AMBER_ROOT="$AMBER_ROOT" EVAL_ROOT="$EVAL_ROOT" bash "$ROOT/run/eval_amber_tars.sh"
