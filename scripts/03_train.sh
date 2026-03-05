#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/small_debug.yaml}"

TOKEN_MANIFEST=$(python - <<PY
from src.utils.config import load_config
print(load_config('${CFG}')['paths']['token_manifest'])
PY
)

if [[ ! -f "$TOKEN_MANIFEST" ]]; then
  echo "Token manifest missing: $TOKEN_MANIFEST" >&2
  echo "Run scripts/02_tokenize.sh first (which depends on src.data.manifest output)." >&2
  exit 1
fi

python train.py --config "$CFG"
