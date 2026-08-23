#!/usr/bin/env bash
set -euo pipefail
command_name="${1:-help}"
python_bin="${PYTHON_BIN:-/tmp/quco-rag-venv/bin/python}"

# Keep model and library downloads off the space-constrained home filesystem.
export HF_HOME="${HF_HOME:-/tmp/quco-rag-cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/tmp/quco-rag-cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/quco-rag-cache/xdg}"
case "$command_name" in
  check)
    "$python_bin" -m py_compile src/data.py src/main_quco.py src/main_baseline.py src/evaluate_reallife.py
    "$python_bin" -c 'import json; from pathlib import Path; p=Path("/home/tong56/whitebox/question_and_result.json"); assert len(json.loads(p.read_text())) == 500; print("real-life rows: 500")'
    ;;
  wo-rag) (cd src && "$python_bin" main_quco.py -c ../config/Qwen2.5-7B-Instruct/RealLifeChoice/wo-RAG.json) ;;
  quco-rag) (cd src && "$python_bin" main_quco.py -c ../config/Qwen2.5-7B-Instruct/RealLifeChoice/QuCo-RAG.json) ;;
  *) echo "usage: $0 {check|wo-rag|quco-rag}" >&2; exit 2 ;;
esac
