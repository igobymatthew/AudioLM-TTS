# TASKS.md

- [x] Milestone 0 — Scaffold
- [x] Milestone 1 — Dependencies
- [x] Milestone 2 — Config system
- [x] Milestone 3 — Dataset manifests
- [x] Milestone 4 — Codec tokenization cache
- [x] Milestone 5 — Model + vocab
- [x] Milestone 6 — Training loop
- [x] Milestone 7 — Inference
- [x] Milestone 8 — Evaluation
- [x] Milestone 9 — LoRA fine-tuning
- [x] Milestone 10 — Serving + Docker

## 2026-03-05 bootstrap ordering fix

- [x] Ensure `src/data` is a package before script imports.
- [x] Add `src/data/manifest.py` CLI that builds `manifest.jsonl` from wav/txt pairs.
- [x] Update `scripts/01_get_data.sh` to avoid importing project modules.
- [x] Add script guards for manifest/token-manifest dependent flows.
- [x] Re-ran milestone validation commands (Docker build skipped in this environment because Docker is unavailable).
