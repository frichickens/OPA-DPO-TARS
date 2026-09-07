#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/base_models/llava-v1.5-7b}"
LORA_DIR="${LORA_DIR:-$ROOT/output/tars_corpus_llava7b_opadpo_model/checkpoint-final/adapter_model/lora_policy}"
AMBER_ROOT="${AMBER_ROOT:-$ROOT/base_datasets/AMBER}"
AMBER_IMAGE_DIR="${AMBER_IMAGE_DIR:-$(cat "$AMBER_ROOT/.image_dir" 2>/dev/null || true)}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/output/evaluation/tars_corpus_opadpo_amber}"
ANSWERS="$EVAL_ROOT/answers.jsonl"
mkdir -p "$EVAL_ROOT"
[[ -f "$MODEL_DIR/config.json" ]] || { echo "Missing model: $MODEL_DIR" >&2; exit 2; }
[[ -f "$LORA_DIR/adapter_config.json" ]] || { echo "Missing adapter: $LORA_DIR" >&2; exit 2; }
[[ -d "$AMBER_IMAGE_DIR" ]] || { echo "Missing AMBER image directory: $AMBER_IMAGE_DIR" >&2; exit 2; }
[[ ! -e "$ANSWERS" ]] || { echo "Output exists; choose a new EVAL_ROOT or remove: $ANSWERS" >&2; exit 2; }
cd "$ROOT"
# These are the released OPA-DPO AMBER generation and scoring programs/arguments.
python ./eval_llava_rlhf_coco/AMBER_generate.py --model-path "$MODEL_DIR" \
  --use-qlora True --qlora-path "$LORA_DIR" --temperature 0.0 \
  --answers-file "$ANSWERS" --image-file "$AMBER_IMAGE_DIR" \
  --image_aspect_ratio pad --test-prompt ''
python ./eval_llava_rlhf_coco/AMBER_eval.py --inference_data "$ANSWERS" --evaluation_type g \
  2>&1 | tee "$EVAL_ROOT/metrics.log"
