#!/usr/bin/env bash
set -euo pipefail
cd /home/tong56/whitebox
export CPATH=/home/tong56/.local/python310-dev/usr/include/python3.10:/home/tong56/.local/python310-dev/usr/include
export TRITON_CACHE_DIR=/tmp/triton_selfcheckgpt
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_selfcheckgpt
export HF_HOME=/tmp/selfcheckgpt_hf
PY=/home/tong56/venvs/whitebox/bin/python
SCRIPT=perturbation/284_selfcheckgpt_nli_paper.py
OUT=/tmp/selfcheckgpt_original_benchmarks
for ds in scientist trivia gsm8k drop; do
  "$PY" "$SCRIPT" sample "$ds" --batch 2 --resume --out "$OUT"
  "$PY" "$SCRIPT" score "$ds" --nli-batch 32 --resume --out "$OUT" --cache /tmp/selfcheckgpt_hf
  mkdir -p "perturbation/runs/284_selfcheckgpt_nli_paper/$ds"
  cp "$OUT/$ds/report.json" "perturbation/runs/284_selfcheckgpt_nli_paper/$ds/report.json"
done
