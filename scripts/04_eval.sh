#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/small_debug.yaml}"
CKPT="${2:-checkpoints/latest.pt}"
python eval.py --config "$CFG" --ckpt "$CKPT" --n_samples 2
