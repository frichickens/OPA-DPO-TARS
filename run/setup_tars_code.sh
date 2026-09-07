#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARS_SRC="${TARS_SRC:-$ROOT/tars_src}"
TARS_COMMIT="${TARS_COMMIT:-b2e62881bb39dd28a0d12b81d5bcbdf033f7fa9d}"
if [[ ! -d "$TARS_SRC/.git" ]]; then git clone https://github.com/KejiaZhang-Robust/TARS.git "$TARS_SRC"; fi
git -C "$TARS_SRC" fetch origin
git -C "$TARS_SRC" checkout "$TARS_COMMIT"
git -C "$TARS_SRC" restore --source "$TARS_COMMIT" -- llava/train/llava_trainer.py
git -C "$TARS_SRC" apply "$ROOT/patches/tars_initialize_original_tokens.patch"
echo "TARS_SRC=$TARS_SRC commit=$(git -C "$TARS_SRC" rev-parse HEAD)"
