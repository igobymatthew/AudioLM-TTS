#!/usr/bin/env bash
set -euo pipefail
docker build -t audiolm-tts:local .
docker run --rm -p 8000:8000 audiolm-tts:local
