#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-sample}"

if [[ "$DATASET" == "sample" ]]; then
  mkdir -p data/raw/sample
  python - <<'PY'
import math
import struct
import wave
from pathlib import Path

sr = 24000
secs = 1.0
freq = 440.0
base = Path('data/raw/sample')
base.mkdir(parents=True, exist_ok=True)

wav_path = base / 'sample.wav'
with wave.open(str(wav_path), 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    for i in range(int(sr * secs)):
        v = int(32767 * 0.2 * math.sin(2 * math.pi * freq * i / sr))
        w.writeframesraw(struct.pack('<h', v))

(base / 'sample.txt').write_text('hello world from sample data', encoding='utf-8')
print('created sample wav/txt pair at data/raw/sample/')
print('next step: python -m src.data.manifest --data_dir data/raw/sample --out data/raw/sample/manifest.jsonl --default_speaker 0')
PY
else
  echo "Dataset '$DATASET' is instructions-only placeholder."
  echo "Please download manually and run:"
  echo "  python -m src.data.manifest --data_dir <wav_txt_dir> --out data/raw/manifest.jsonl"
fi
