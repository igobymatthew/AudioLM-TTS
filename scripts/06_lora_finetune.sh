#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/lora_finetune.yaml}"
python train.py --config "$CFG" --max_steps 10
