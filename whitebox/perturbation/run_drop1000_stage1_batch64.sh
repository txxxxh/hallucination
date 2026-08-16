#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export SPANATTR_DISABLE_NATIVE_BMM=1 TRITON_CACHE_DIR=/tmp/perturb_drop167
PY=/home/tong56/venvs/whitebox/bin/python
mkdir -p runs/167_drop1000_logs "$TRITON_CACHE_DIR"
$PY 167_collect_drop1000_fast_stage1.py --method exact --out-dir runs/167_drop1000_exact --batch 64 --resume >>runs/167_drop1000_logs/exact.log 2>&1
$PY 167_collect_drop1000_fast_stage1.py --method attention_maxhead --out-dir runs/167_drop1000_attention_maxhead --batch 64 --resume >>runs/167_drop1000_logs/attention.log 2>&1
$PY 167_collect_drop1000_fast_stage1.py --method gradient_sentence --out-dir runs/167_drop1000_gradient_sentence --batch 64 --resume >>runs/167_drop1000_logs/gradient.log 2>&1
$PY 168_eval_drop1000_fast_stage1.py >>runs/167_drop1000_logs/evaluate.log 2>&1
