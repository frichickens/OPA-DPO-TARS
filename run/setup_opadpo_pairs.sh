#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/base_datasets/OPA-DPO-7B}"
PY="${PY:-python}"
mkdir -p "$DATA_ROOT"
"$PY" -c 'import gdown' 2>/dev/null || "$PY" -m pip install gdown
LIST="$(mktemp)"
trap 'unlink "$LIST" 2>/dev/null || true' EXIT
"$PY" -m gdown --folder --json 'https://drive.google.com/drive/folders/1Xmrb43zIbk3IzLLRQx65iBf7J4SHXa9j' > "$LIST"
"$PY" - "$LIST" "$DATA_ROOT" <<'PY' | xargs -0 -n2 -P4 bash -c 'mkdir -p "$(dirname "$1")"; [[ -s "$1" ]] || gdown --continue "$0" -O "$1"'
import json,os,sys
items=json.load(open(sys.argv[1])); root=sys.argv[2]
selected=[x for x in items if x['path'].startswith('llava7b_online_generation_subset1/rollouts/') or x['path'].startswith('llava7b_online_generation_subset2/rollouts/')]
if len(selected)!=616: raise SystemExit(f'Expected 616 rollout files, found {len(selected)}')
for x in selected: print(x['url'],os.path.join(root,x['path']),sep='\0',end='\0')
PY
"$PY" "$ROOT/base_operations/make_opadpo_tars_dataset.py" --rollouts-root "$DATA_ROOT" \
  --output "$DATA_ROOT/opadpo_7b_pairs_tars.parquet" --manifest "$DATA_ROOT/dataset_manifest.json"
