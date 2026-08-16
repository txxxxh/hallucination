#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/tong56/venvs/whitebox/bin/python"
LOG_DIR="$HERE/runs/167_drop1000_logs"
mkdir -p "$LOG_DIR" /tmp/perturb_drop167

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export SPANATTR_DISABLE_NATIVE_BMM=1
export TRITON_CACHE_DIR=/tmp/perturb_drop167

run_collect() {
  local method="$1" out_dir="$2" batch="$3" log="$4"
  "$PYTHON_BIN" "$HERE/167_collect_drop1000_fast_stage1.py" \
    --method "$method" --out-dir "$HERE/runs/$out_dir" \
    --batch "$batch" --resume >>"$LOG_DIR/$log" 2>&1
}

date -u +'%FT%TZ exact start' >>"$LOG_DIR/pipeline.log"
run_collect exact 167_drop1000_exact 192 exact.log
date -u +'%FT%TZ attention start' >>"$LOG_DIR/pipeline.log"
run_collect attention_maxhead 167_drop1000_attention_maxhead 128 attention.log
date -u +'%FT%TZ gradient start' >>"$LOG_DIR/pipeline.log"
run_collect gradient_sentence 167_drop1000_gradient_sentence 128 gradient.log
date -u +'%FT%TZ evaluation start' >>"$LOG_DIR/pipeline.log"
"$PYTHON_BIN" "$HERE/168_eval_drop1000_fast_stage1.py" \
  >>"$LOG_DIR/evaluate.log" 2>&1
date -u +'%FT%TZ complete' >>"$LOG_DIR/pipeline.log"
