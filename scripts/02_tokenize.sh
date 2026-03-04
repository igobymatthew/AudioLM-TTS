#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/small_debug.yaml}"
python - <<PY
from src.utils.config import load_config
cfg=load_config('${CFG}')
print(cfg['paths']['manifest'])
print(cfg['paths']['token_manifest'])
PY
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
python -m src.codec.encode --manifest "$MANIFEST" --out_dir data/tokenized/cache --out_manifest "$OUT_MANIFEST"
