#!/usr/bin/env bash
# Resumable 4-model x 4-dataset x 2-method paper experiment matrix.
set -uo pipefail
# INVALIDATED: this legacy runner reuses Llama labels across backbones.
echo "ERROR: invalid cross-model protocol. Use run_paper4_self_matrix.sh" >&2
exit 64

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/tong56/venvs/whitebox/bin/python}"
OUTPUT_ROOT="${PAPER4_OUTPUT_ROOT:-$HERE/runs/paper4_matrix}"
BATCH="${PAPER4_BATCH:-24}"
LIMIT="${PAPER4_LIMIT:-0}"
SHARD="${PAPER4_SHARD:-all}"

if [[ "$SHARD" != "all" && "$SHARD" != "0" && "$SHARD" != "1" ]]; then
  printf 'PAPER4_SHARD must be all, 0, or 1 (got %s)\n' "$SHARD" >&2
  exit 2
fi

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export SPANATTR_DISABLE_NATIVE_BMM=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/perturb_paper4_triton}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status" "$TRITON_CACHE_DIR"

MODELS=(llama mistral qwen falcon3)
MODEL_IDS=(
  "NousResearch/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "Qwen/Qwen2.5-7B-Instruct"
  "tiiuae/Falcon3-7B-Instruct"
)
DATASETS=(scientist multidomain trivia gsm8k)
METHODS=(exact attention)

run_one() {
  local model_name="$1" model_id="$2" dataset="$3" method="$4"
  local task="${model_name}__${dataset}__${method}"
  local out="$OUTPUT_ROOT/features/$model_name/$dataset/$method"
  local log="$OUTPUT_ROOT/logs/$task.log"
  local done_file="$OUTPUT_ROOT/status/$task.done"
  local failed_file="$OUTPUT_ROOT/status/$task.failed"
  if [[ -f "$done_file" ]]; then
    printf '[SKIP] %s already complete\n' "$task" | tee -a "$OUTPUT_ROOT/matrix.log"
    return 0
  fi
  rm -f "$failed_file"
  printf '[START] %s %s\n' "$(date -u +%FT%TZ)" "$task" | tee -a "$OUTPUT_ROOT/matrix.log" "$log"
  local cmd=("$PYTHON_BIN" "$HERE/158_collect_paper4_matrix.py"
    --dataset "$dataset" --method "$method" --model "$model_id"
    --out-dir "$out" --batch "$BATCH" --limit "$LIMIT" --resume)
  if "${cmd[@]}" >>"$log" 2>&1; then
    printf '%s\n' "$(date -u +%FT%TZ)" >"$done_file"
    printf '[DONE] %s %s\n' "$(date -u +%FT%TZ)" "$task" | tee -a "$OUTPUT_ROOT/matrix.log" "$log"
  else
    local rc=$?
    printf '%s rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >"$failed_file"
    printf '[FAIL] %s %s rc=%s (continuing)\n' "$(date -u +%FT%TZ)" "$task" "$rc" | tee -a "$OUTPUT_ROOT/matrix.log" "$log"
  fi
}

for i in "${!MODELS[@]}"; do
  if [[ "$SHARD" != "all" && $((i / 2)) -ne "$SHARD" ]]; then
    continue
  fi
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_one "${MODELS[$i]}" "${MODEL_IDS[$i]}" "$dataset" "$method"
    done
  done
done

printf '[MATRIX-END] %s\n' "$(date -u +%FT%TZ)" | tee -a "$OUTPUT_ROOT/matrix.log"
