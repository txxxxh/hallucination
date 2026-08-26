#!/usr/bin/env bash
set -euo pipefail
cd /home/tong56/whitebox/perturbation
PY=/home/tong56/venvs/whitebox/bin/python
ROOT=/home/tong56/whitebox/perturbation/runs/316_full_scientist_p_selfchecknli
export HF_HOME=/tmp/selfcheckgpt_hf
export TRITON_CACHE_DIR=/tmp/triton_selfcheckgpt
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_selfcheckgpt
export TORCH_DISABLE_NATIVE_JIT=1
"$PY" -u 316_full_scientist_selfcheckgpt_nli.py sample --batch 12 --resume --out "$ROOT"
"$PY" -u 316_full_scientist_selfcheckgpt_nli.py score --nli-batch 32 --resume --out "$ROOT" --cache /tmp/selfcheckgpt_hf
"$PY" -u 316_fuse_full_scientist_p_selfchecknli.py --scores "$ROOT/scientist/scores.jsonl" --out "$ROOT/fusion"
