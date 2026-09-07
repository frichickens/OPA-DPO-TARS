#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -d llava ]]; then
  [[ -d llava_setup/LLaVA ]] || git clone https://github.com/haotian-liu/LLaVA llava_setup/LLaVA
  git -C llava_setup/LLaVA checkout 817a4af
  (cd llava_setup && patch -N -p1 < llava_modifications.patch)
  mv llava_setup/LLaVA/llava ./llava
fi
echo "OPA-DPO code setup complete. Create the official environment with:"
echo "  conda env create -f environment.yaml && conda activate OPA_DPO && pip install flash-attn==2.5.3"
