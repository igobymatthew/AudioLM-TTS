# AGENTS.md — AudioLM-TTS build rules

You are Codex running in Agent mode. You may read/write files and run commands.  [oai_citation:2‡OpenAI Developers](https://developers.openai.com/codex/quickstart/?utm_source=chatgpt.com)

GOAL
- Implement the repository described by README.md as a runnable project.
- Prioritize correctness + reproducibility over “extra features”.

SCOPE
- Implement the Minimal Runnable Skeleton (single GPU local), plus:
  - Tokenization (EnCodec), training loop, inference, eval, FastAPI serving, Docker, scripts, configs.
- Do NOT attempt multi-GPU/FSDP unless explicitly requested.

WORKFLOW
- Always read README.md first.
- Follow IMPLEMENT.md milestone-by-milestone.
- After each milestone:
  - run the specified validation commands
  - fix failures immediately
  - update README only if behavior diverged from README.

QUALITY BAR
- Every script in scripts/ must run on macOS/Linux (bash).
- Python code must have clear CLIs with argparse.
- Make conservative defaults for RTX 4070 Ti Super 16GB.
- Keep codebase small and understandable.

SAFETY
- Do not download huge datasets automatically without user invoking scripts.
- Use gitignored folders: data/, checkpoints/, outputs/, mlruns/.

DELIVERABLES
- Must produce: requirements.txt, Dockerfile, configs/*.yaml, scripts/*.sh, src/*, train.py, eval.py, inference.py, serve.py
- Must include smoke tests in tests/ (minimal).