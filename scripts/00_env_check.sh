#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import torch
print('python ok')
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
PY
