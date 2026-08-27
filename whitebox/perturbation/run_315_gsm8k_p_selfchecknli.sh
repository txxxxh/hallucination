#!/usr/bin/env bash
set -euo pipefail
cd /home/tong56/whitebox/perturbation
PY=/home/tong56/venvs/whitebox/bin/python
ROOT=/home/tong56/whitebox/perturbation/runs/315_gsm8k_p_selfchecknli
export HF_HOME=/tmp/selfcheckgpt_hf
export TRITON_CACHE_DIR=/tmp/triton_selfcheckgpt
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_selfcheckgpt
"$PY" -u 284_selfcheckgpt_nli_paper.py sample gsm8k --batch 12 --resume --out "$ROOT"
"$PY" -u 284_selfcheckgpt_nli_paper.py score gsm8k --nli-batch 32 --resume --out "$ROOT" --cache /tmp/selfcheckgpt_hf
"$PY" -u 315_fuse_gsm8k_p_selfchecknli.py --scores "$ROOT/gsm8k/scores.jsonl" --out "$ROOT/fusion"
