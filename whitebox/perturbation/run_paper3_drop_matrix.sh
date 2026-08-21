#!/usr/bin/env bash
# Model-specific DROP generation, then unified mean-pooling detector collection.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/tong56/venvs/whitebox/bin/python}"
ROOT="${PAPER3_MEAN_ROOT:-$HERE/runs/paper3_mean_matrix}"
SOURCE="$ROOT/drop_manifests"; ITEMS="${DROP_ITEMS:-$HERE/runs/166_drop1000/items.jsonl}"
SHARD="${PAPER3_SHARD:-all}"; BATCH="${PAPER3_BATCH:-16}"
MODEL_CACHE_ROOT="${PAPER4_MODEL_CACHE_ROOT:-/tmp/tong56_huggingface_models}"
[[ "$SHARD" =~ ^(all|0|1)$ ]] || { echo 'PAPER3_SHARD must be all, 0, or 1'; exit 2; }
mkdir -p "$ROOT/logs" "$ROOT/status" "$SOURCE" "$MODEL_CACHE_ROOT/hub" "$MODEL_CACHE_ROOT/transformers"
export HF_HOME="$MODEL_CACHE_ROOT" HF_HUB_CACHE="$MODEL_CACHE_ROOT/hub" TRANSFORMERS_CACHE="$MODEL_CACHE_ROOT/transformers"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 SPANATTR_DISABLE_NATIVE_BMM=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tong56_paper3_drop_triton}"
mkdir -p "$TRITON_CACHE_DIR"
names=(llama mistral qwen)
ids=(NousResearch/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct)
run(){ local tag="$1"; shift; local done="$ROOT/status/$tag.done" log="$ROOT/logs/$tag.log"; [[ -f "$done" ]] && { echo "[SKIP] $tag"; return; }; echo "[START] $(date -u +%FT%TZ) $tag" | tee -a "$ROOT/run.log"; if "$@" >>"$log" 2>&1; then date -u +%FT%TZ >"$done"; echo "[DONE] $tag" | tee -a "$ROOT/run.log"; else echo "[FAIL] $tag (see $log)" | tee -a "$ROOT/run.log"; return 1; fi; }

# Each manifest has one owner; the other shard waits for its completion marker.
for i in "${!names[@]}"; do
  n="${names[$i]}"; model="${ids[$i]}"
  if [[ "$SHARD" == all || $((i % 2)) -eq "$SHARD" ]]; then
    run "$n.drop.manifest" "$PY" "$HERE/169_prepare_drop_self_manifest.py" --model "$model" --items "$ITEMS" --out-dir "$SOURCE/$n" --batch "$BATCH" --resume
  fi
done
task=0
for i in "${!names[@]}"; do
  n="${names[$i]}"; model="${ids[$i]}"; manifest="$SOURCE/$n/drop.jsonl"
  for method in exact attention gradient; do
    if [[ "$SHARD" == all || $((task % 2)) -eq "$SHARD" ]]; then
      while [[ ! -f "$SOURCE/$n/manifest.done" ]]; do sleep 30; done
      run "$n.drop.$method.mean" "$PY" "$HERE/158_collect_paper4_matrix.py" --dataset drop --method "$method" --model "$model" --manifest "$manifest" --out-dir "$ROOT/features/$n/drop/$method" --layer14-pooling mean --batch "$BATCH" --resume
    fi
    task=$((task + 1))
  done
done
echo "[END DROP] $(date -u +%FT%TZ)" | tee -a "$ROOT/run.log"
