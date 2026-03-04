#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/small_debug.yaml}"
CKPT="${2:-checkpoints/latest.pt}"
TEXT="${3:-Hello world}"
SPK="${4:-0}"
python inference.py --config "$CFG" --ckpt "$CKPT" --text "$TEXT" --speaker_id "$SPK" --out_wav outputs/demo.wav
