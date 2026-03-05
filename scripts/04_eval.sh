#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/small_debug.yaml}"
CKPT="${2:-checkpoints/latest.pt}"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  echo "Run scripts/03_train.sh first." >&2
  exit 1
fi

python eval.py --config "$CFG" --ckpt "$CKPT" --n_samples 2
