#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$ROOT/work_tars_to_amber}"
MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/llava-v1.5-7b}"
VISION_TOWER_DIR="${VISION_TOWER_DIR:-$MODEL_ROOT/vision_tower-clip336}"
TARS_DOWNLOAD_DIR="${TARS_DOWNLOAD_DIR:-$WORK_ROOT/datasets/RLHF-V-Dataset}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-$WORK_ROOT/datasets/converted_tars}"
AMBER_ROOT="${AMBER_ROOT:-$WORK_ROOT/datasets/AMBER}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$WORK_ROOT/training}"
EVAL_ROOT="${EVAL_ROOT:-$WORK_ROOT/evaluation/AMBER}"
NUM_SAMPLES="${NUM_SAMPLES:-4800}"
SEED="${SEED:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

command -v conda >/dev/null || { echo "conda is required" >&2; exit 2; }
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
PY
mkdir -p "$WORK_ROOT" "$TRAIN_OUTPUT_ROOT" "$EVAL_ROOT"
MODEL_ROOT="$MODEL_ROOT" MODEL_DIR="$MODEL_DIR" VISION_TOWER_DIR="$VISION_TOWER_DIR" bash "$ROOT/run/setup_base_model_7b.sh"
DATA_ROOT="$TARS_DOWNLOAD_DIR" OUTPUT_ROOT="$TRAIN_DATA_ROOT" NUM_SAMPLES="$NUM_SAMPLES" SEED="$SEED" bash "$ROOT/run/setup_tars_dataset.sh"
AMBER_ROOT="$AMBER_ROOT" bash "$ROOT/run/setup_amber.sh"
python -c 'import en_core_web_lg' 2>/dev/null || python -m spacy download en_core_web_lg

OPA_DATA_DIR="$TRAIN_DATA_ROOT/opa_training_data-tars-7B" \
DPO_DATA_DIR="$TRAIN_DATA_ROOT/opadpo_training_data-tars-7B" MODEL_DIR="$MODEL_DIR" \
OPA_OUTPUT_DIR="$TRAIN_OUTPUT_ROOT/llava7b_opa" DPO_OUTPUT_DIR="$TRAIN_OUTPUT_ROOT/llava7b_opadpo" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" GPUS_PER_NODE="$GPUS_PER_NODE" \
  bash "$ROOT/run/train_opa_dpo_on_tars.sh"

MODEL_DIR="$MODEL_DIR" \
LORA_DIR="$TRAIN_OUTPUT_ROOT/llava7b_opadpo/checkpoint-final/adapter_model/lora_policy" \
AMBER_ROOT="$AMBER_ROOT" EVAL_ROOT="$EVAL_ROOT" bash "$ROOT/run/eval_amber_tars_opadpo.sh"
