#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export DATA_DIR="${OPA_DATA_DIR:-$ROOT/base_datasets/opa_training_data-tars-7B}"
export DPO_DATA_DIR="${DPO_DATA_DIR:-$ROOT/base_datasets/opadpo_training_data-tars-7B}"
export MODEL_DIR="${MODEL_DIR:-$ROOT/base_models/llava-v1.5-7b}"
export OPA_OUTPUT_DIR="${OPA_OUTPUT_DIR:-$ROOT/output/tars_corpus_llava7b_opa_model}"
export DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-$ROOT/output/tars_corpus_llava7b_opadpo_model}"
export POLICY_LORA_DIR="${POLICY_LORA_DIR:-$OPA_OUTPUT_DIR/checkpoint-final}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

echo "Stage 1/2: official OPA LoRA-SFT"
DATA_DIR="$DATA_DIR" MODEL_DIR="$MODEL_DIR" OUTPUT_DIR="$OPA_OUTPUT_DIR" \
  bash run/train_opa.sh
echo "Stage 2/2: OPA-DPO on TARS pairs"
# TARS lacks OPA's GPT-4V granular report and has only one positive response.
DATA_DIR="$DPO_DATA_DIR" MODEL_DIR="$MODEL_DIR" POLICY_LORA_DIR="$POLICY_LORA_DIR" \
OUTPUT_DIR="$DPO_OUTPUT_DIR" DETAILED_REPORT=False RESPONSE_SCORE=False \
RESPONSE_IMAGE_RELATION=False STANDARD_PAIR_COEF=0.0 AI_PAIR_COEF=1.0 \
  bash run/train_opa_dpo.sh
