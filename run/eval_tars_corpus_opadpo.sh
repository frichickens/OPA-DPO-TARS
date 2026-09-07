#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MODEL_BASE="${MODEL_BASE:-$ROOT/base_models/llava-v1.5-7b}"
export MODEL_LORA_BASE="${MODEL_LORA_BASE:-$ROOT/output/tars_corpus_llava7b_opadpo_model/checkpoint-final/adapter_model/lora_policy}"
export MODEL_SUFFIX="${MODEL_SUFFIX:-tars_corpus_llava7b_opadpo}"
for required in "$MODEL_BASE" "$MODEL_LORA_BASE"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done
# Calls the official OPA-DPO evaluation pipeline unchanged; only paths are supplied.
bash run/eval_all_metrics.sh
