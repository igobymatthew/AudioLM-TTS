# IMPLEMENT.md — Build plan (source of truth)

Milestone 0 — Scaffold
- Create repo structure (src/, scripts/, configs/, tests/)
- Add .gitignore for data/, checkpoints/, outputs/, mlruns/
Validation:
- `python -c "print('ok')"`

Milestone 1 — Dependencies
- Create requirements.txt
- Create minimal import checks for torch, torchaudio, transformers, encodec, mlflow, fastapi
Validation:
- `pip install -r requirements.txt`
- `python -c "import torch, torchaudio, transformers, encodec, mlflow, fastapi; print('imports ok')"`

Milestone 2 — Config system
- Implement src/utils/config.py to load YAML configs + CLI overrides
- Add configs/local_4070ti_16gb.yaml, small_debug.yaml, lora_finetune.yaml
Validation:
- `python -c "from src.utils.config import load_config; print(load_config('configs/small_debug.yaml')['seed'])"`

Milestone 3 — Dataset manifests
- Implement src/data/manifest.py to build a manifest JSONL:
  - fields: audio_path, text_path, speaker_id (optional), sample_rate
- Implement scripts/01_get_data.sh as a placeholder downloader or instructions-only (no auto huge downloads)
Validation:
- `python -m src.data.manifest --help`

Milestone 4 — Codec tokenization cache
- Implement src/codec/encode.py CLI:
  - input: manifest.jsonl
  - output: data/tokenized/<name>/*.pt + updated manifest w token paths
  - use EnCodec 24kHz, bandwidth configurable
- Implement src/codec/decode.py to decode tokens back to wav
- Implement scripts/02_tokenize.sh
Validation:
- Create a tiny sample wav + txt in data/raw/sample/
- Run tokenize and decode to outputs/sample_roundtrip.wav

Milestone 5 — Model + vocab
- Implement src/models/vocab.py:
  - text tokenizer (GPT-2) + special tokens
  - audio token vocab mapping (from codec)
  - combined vocab strategy (document it)
- Implement src/models/transformer.py:
  - decoder-only GPT-style model (HF GPT2LMHeadModel ok)
- Implement src/models/conditioning.py:
  - speaker embedding option (on/off)
Validation:
- `python -c "from src.models.transformer import build_model; m=build_model(10000); print(sum(p.numel() for p in m.parameters()))"`

Milestone 6 — Training loop
- Implement src/training/trainer.py and train.py
  - bf16/fp16, grad accumulation, checkpointing, logging to MLflow
  - train on token sequences from cache
- Implement scripts/03_train.sh
Validation:
- `python train.py --config configs/small_debug.yaml --max_steps 20`
- Confirm checkpoint saved + mlflow run created

Milestone 7 — Inference
- Implement src/inference/generate.py and inference.py:
  - text → tokens → decode wav
- Implement scripts/05_generate.sh
Validation:
- Generate `outputs/demo.wav` from a checkpoint

Milestone 8 — Evaluation
- Implement eval.py and src/evaluation/*:
  - Whisper WER roundtrip
  - basic latency benchmark (tokens/sec)
  - MOS proxy placeholder (optional, document if stubbed)
- Implement scripts/04_eval.sh
Validation:
- `python eval.py --config configs/small_debug.yaml --ckpt checkpoints/latest.pt --n_samples 2`

Milestone 9 — LoRA fine-tuning
- Implement src/models/lora.py (PEFT optional) + scripts/06_lora_finetune.sh
Validation:
- Run a short finetune (10 steps) on tiny sample dataset

Milestone 10 — Serving + Docker
- Implement src/serving/api.py and serve.py (FastAPI)
- Implement scripts/07_serve_api.sh
- Add Dockerfile + scripts/08_docker_build_run.sh
Validation:
- Start server locally and hit /health
- Docker builds successfully