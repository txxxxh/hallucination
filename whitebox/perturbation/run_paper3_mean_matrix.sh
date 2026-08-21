#!/usr/bin/env bash
# Final 3-backbone collection: frozen self-manifests, answer-token mean layer14.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="${PYTHON_BIN:-/home/tong56/venvs/whitebox/bin/python}"; SOURCE_ROOT="${PAPER3_SOURCE_ROOT:-$HERE/runs/paper4_self_matrix_v2}"; ROOT="${PAPER3_MEAN_ROOT:-$HERE/runs/paper3_mean_matrix}"; SHARD="${PAPER3_SHARD:-all}"; BATCH="${PAPER3_BATCH:-16}"; LIMIT="${PAPER3_LIMIT:-0}"; ONLY_MODEL="${PAPER3_ONLY_MODEL:-}"
[[ "$SHARD" =~ ^(all|0|1)$ ]] || { echo 'PAPER4_SHARD must be all, 0, or 1'; exit 2; }
MODEL_CACHE_ROOT="${PAPER4_MODEL_CACHE_ROOT:-/home/tong56/.cache/huggingface}"
INTERMEDIATE_ROOT="${PAPER4_INTERMEDIATE_ROOT:-/tmp/tong56_paper4_intermediate}"
mkdir -p "$ROOT/logs" "$ROOT/status" "$MODEL_CACHE_ROOT/hub" "$MODEL_CACHE_ROOT/transformers" "$MODEL_CACHE_ROOT/datasets" "$INTERMEDIATE_ROOT"
export HF_HOME="$MODEL_CACHE_ROOT" HF_HUB_CACHE="$MODEL_CACHE_ROOT/hub" TRANSFORMERS_CACHE="$MODEL_CACHE_ROOT/transformers" HF_DATASETS_CACHE="$MODEL_CACHE_ROOT/datasets"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 TORCH_DISABLE_NATIVE_JIT=1 SPANATTR_DISABLE_NATIVE_BMM=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tong56_paper4_triton}"
mkdir -p "$TRITON_CACHE_DIR"
names=(llama mistral qwen); ids=(NousResearch/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct)
run(){ local tag="$1"; shift; local done="$ROOT/status/$tag.done" log="$ROOT/logs/$tag.log"; [[ -f "$done" ]] && { echo "[SKIP] $tag"; return; }; echo "[START] $(date -u +%FT%TZ) $tag" | tee -a "$ROOT/run.log"; if "$@" >>"$log" 2>&1; then date -u +%FT%TZ >"$done"; echo "[DONE] $tag" | tee -a "$ROOT/run.log"; else echo "[FAIL] $tag (see $log)" | tee -a "$ROOT/run.log"; fi; }
task_index=0
for i in "${!names[@]}"; do
 n="${names[$i]}"; model="${ids[$i]}"; d="$SOURCE_ROOT/models/$n"
 [[ -d "$d/manifests" ]] || { echo "missing frozen manifests: $d/manifests"; exit 1; }
 for ds in scientist multidomain trivia gsm8k; do
 [[ -n "$ONLY_MODEL" && "$n" != "$ONLY_MODEL" ]] && { task_index=$((task_index + 12)); continue; }
  for method in exact attention gradient; do
   if [[ "$SHARD" == all || $((task_index % 2)) -eq "$SHARD" ]]; then
    run "$n.$ds.$method.mean" "$PY" "$HERE/158_collect_paper4_matrix.py" --dataset "$ds" --method "$method" --model "$model" --manifest "$d/manifests/$ds.jsonl" --out-dir "$ROOT/features/$n/$ds/$method" --layer14-pooling mean --batch "$BATCH" --limit "$LIMIT" --resume
   fi
   task_index=$((task_index + 1))
  done
 done
done
echo "[END] $(date -u +%FT%TZ)" | tee -a "$ROOT/run.log"
