#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/small_debug.yaml}"
python train.py --config "$CFG"
