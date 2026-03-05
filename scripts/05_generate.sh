#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/small_debug.yaml}"
CKPT="${2:-checkpoints/latest.pt}"
TEXT="${3:-Hello world}"
SPK="${4:-0}"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  echo "Run scripts/03_train.sh first." >&2
  exit 1
fi

python inference.py --config "$CFG" --ckpt "$CKPT" --text "$TEXT" --speaker_id "$SPK" --out_wav outputs/demo.wav
