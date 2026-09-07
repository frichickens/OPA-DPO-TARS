#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT/output/tars_on_opadpo_7b}"
AMBER_ROOT="${AMBER_ROOT:-$ROOT/base_datasets/AMBER}"
AMBER_IMAGE_DIR="${AMBER_IMAGE_DIR:-$(cat "$AMBER_ROOT/.image_dir" 2>/dev/null || true)}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/output/evaluation/tars_on_opadpo_amber}"
ANSWERS="$EVAL_ROOT/answers.jsonl"; mkdir -p "$EVAL_ROOT"
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing TARS model: $MODEL_PATH" >&2; exit 2; }
[[ -d "$AMBER_IMAGE_DIR" ]] || { echo "Missing AMBER images: $AMBER_IMAGE_DIR" >&2; exit 2; }
[[ ! -e "$ANSWERS" ]] || { echo "Output exists: $ANSWERS" >&2; exit 2; }
cd "$ROOT"
python ./eval_llava_rlhf_coco/AMBER_generate.py --model-path "$MODEL_PATH" --temperature 0.0 \
 --answers-file "$ANSWERS" --image-file "$AMBER_IMAGE_DIR" --image_aspect_ratio pad --test-prompt ''
python ./eval_llava_rlhf_coco/AMBER_eval.py --inference_data "$ANSWERS" --evaluation_type g 2>&1 | tee "$EVAL_ROOT/metrics.log"
