#!/usr/bin/env bash
# Reproduce the audited GSM8K 0.743 protocol for 3 backbones x 3 methods.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/tong56/venvs/whitebox/bin/python}"
ROOT="${CLEAN_GSM8K_ROOT:-$HERE/runs/paper3_gsm8k_clean942}"
SOURCE="${CLEAN_GSM8K_SOURCE:-$HERE/runs/140_gsm8k_natural/natural_balanced_n942.jsonl}"
BATCH="${CLEAN_GSM8K_BATCH:-16}"
SHARD="${CLEAN_GSM8K_SHARD:-all}"

if [[ ! "$SHARD" =~ ^(all|0|1)$ ]]; then
  echo "CLEAN_GSM8K_SHARD must be all, 0, or 1" >&2
  exit 2
fi

mkdir -p "$ROOT/manifests" "$ROOT/features" "$ROOT/logs" "$ROOT/status"
"$PY" "$HERE/267_build_clean_gsm8k_manifests.py" \
  --source "$SOURCE" --output-root "$ROOT/manifests"

names=(llama mistral qwen)
ids=(NousResearch/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct)
methods=(exact attention gradient)
task_index=0

run() {
  local tag="$1"
  shift
  local done_file="$ROOT/status/$tag.done"
  local log_file="$ROOT/logs/$tag.log"
  if [[ -f "$done_file" ]]; then
    echo "[SKIP] $tag"
    return
  fi
  echo "[START] $(date -u +%FT%TZ) $tag" | tee -a "$ROOT/run.log"
  if "$@" >>"$log_file" 2>&1; then
    date -u +%FT%TZ >"$done_file"
    echo "[DONE] $tag" | tee -a "$ROOT/run.log"
  else
    echo "[FAIL] $tag (see $log_file)" | tee -a "$ROOT/run.log"
    return 1
  fi
}

for i in "${!names[@]}"; do
  name="${names[$i]}"
  model="${ids[$i]}"
  for method in "${methods[@]}"; do
    if [[ "$SHARD" == all || $((task_index % 2)) -eq "$SHARD" ]]; then
      run "$name.gsm8k.$method.clean942" \
        "$PY" "$HERE/158_collect_paper4_matrix.py" \
        --dataset gsm8k \
        --method "$method" \
        --model "$model" \
        --manifest "$ROOT/manifests/$name/gsm8k.jsonl" \
        --out-dir "$ROOT/features/$name/gsm8k/$method" \
        --layer14-pooling last \
        --batch "$BATCH" \
        --resume
    fi
    task_index=$((task_index + 1))
  done
done

echo "[END] $(date -u +%FT%TZ)" | tee -a "$ROOT/run.log"
