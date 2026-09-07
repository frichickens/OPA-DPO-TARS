#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/base_models}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/llava-v1.5-7b}"
VISION_TOWER_DIR="${VISION_TOWER_DIR:-$MODEL_ROOT/vision_tower-clip336}"
mkdir -p "$MODEL_ROOT"
huggingface-cli download openai/clip-vit-large-patch14-336 --repo-type model --local-dir "$VISION_TOWER_DIR"
huggingface-cli download liuhaotian/llava-v1.5-7b --repo-type model --local-dir "$MODEL_DIR"
python "$ROOT/base_operations/modify_base_model_config.py" \
  --model "$MODEL_DIR/config.json" --vision-tower "$VISION_TOWER_DIR"
echo "MODEL_DIR=$MODEL_DIR"
echo "VISION_TOWER_DIR=$VISION_TOWER_DIR"
