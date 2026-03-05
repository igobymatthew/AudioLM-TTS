#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/small_debug.yaml}"

if [[ ! -f src/data/__init__.py || ! -f src/data/manifest.py ]]; then
  echo "Missing src/data package bootstrap files. Expected src/data/__init__.py and src/data/manifest.py" >&2
  exit 1
fi

MANIFEST=$(python - <<PY
from src.utils.config import load_config
print(load_config('${CFG}')['paths']['manifest'])
PY
)
OUT_MANIFEST=$(python - <<PY
from src.utils.config import load_config
print(load_config('${CFG}')['paths']['token_manifest'])
PY
)

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Run: python -m src.data.manifest --data_dir <wav_txt_dir> --out $MANIFEST" >&2
  exit 1
fi

python -m src.codec.encode --manifest "$MANIFEST" --out_dir data/tokenized/cache --out_manifest "$OUT_MANIFEST"
