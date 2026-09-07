#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMBER_ROOT="${AMBER_ROOT:-$ROOT/base_datasets/AMBER}"
ARCHIVE="$AMBER_ROOT/amber_images.zip"
mkdir -p "$AMBER_ROOT"
python -c 'import gdown' 2>/dev/null || python -m pip install gdown
if [[ ! -f "$ARCHIVE" ]]; then
  python -m gdown --fuzzy 'https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing' -O "$ARCHIVE"
fi
if ! find "$AMBER_ROOT" -type f -name 'AMBER_1.jpg' -print -quit | grep -q .; then
  unzip -q -o "$ARCHIVE" -d "$AMBER_ROOT"
fi
FIRST_IMAGE="$(find "$AMBER_ROOT" -type f -name 'AMBER_1.jpg' -print -quit)"
[[ -n "$FIRST_IMAGE" ]] || { echo "AMBER_1.jpg not found after extraction under $AMBER_ROOT" >&2; exit 2; }
AMBER_IMAGE_DIR="$(dirname "$FIRST_IMAGE")"
COUNT="$(find "$AMBER_IMAGE_DIR" -maxdepth 1 -type f -name 'AMBER_*.jpg' | wc -l)"
[[ "$COUNT" -ge 1004 ]] || { echo "Expected >=1004 AMBER images, found $COUNT in $AMBER_IMAGE_DIR" >&2; exit 2; }
printf '%s\n' "$AMBER_IMAGE_DIR" > "$AMBER_ROOT/.image_dir"
echo "AMBER_IMAGE_DIR=$AMBER_IMAGE_DIR ($COUNT images)"
