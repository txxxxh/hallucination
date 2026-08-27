#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/tong56/venvs/whitebox/bin/python}"
MODEL_PATH="${MODEL_PATH:-/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/sese_official_original_benchmarks}"
SCRIPT=perturbation/285_sese_official_original_benchmarks.py

export HF_HOME="${HF_HOME:-/tmp/sese_hf}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/sese_triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/sese_torchinductor}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sese_matplotlib}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

for dataset in scientist trivia gsm8k drop; do
  "$PYTHON_BIN" "$SCRIPT" sample "$dataset" --model "$MODEL_PATH" \
    --batch 4 --resume --local-files-only --out "$OUTPUT_ROOT"
done

# Generation and scoring are separate so failures resume cleanly; scoring loads the official DeBERTa-v2-xlarge and embedding
# models after generation exits.
for dataset in scientist trivia gsm8k drop; do
  "$PYTHON_BIN" "$SCRIPT" score "$dataset" --resume --local-files-only --out "$OUTPUT_ROOT"
  mkdir -p "perturbation/runs/285_sese_official/$dataset"
  cp "$OUTPUT_ROOT/$dataset/report.json" "perturbation/runs/285_sese_official/$dataset/report.json"
done
