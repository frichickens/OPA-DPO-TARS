#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-TARS_DPO}"
TARS_SRC="${TARS_SRC:-$ROOT/tars_src}"
bash "$ROOT/run/setup_tars_code.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -n "$ENV_NAME" python=3.10 -y
conda run -n "$ENV_NAME" python -m pip install -e "$TARS_SRC"
conda run -n "$ENV_NAME" python -m pip install gdown spacy nltk
conda run -n "$ENV_NAME" python -m pip install flash-attn==2.5.9.post1 --no-build-isolation
conda run -n "$ENV_NAME" python -m spacy download en_core_web_lg
