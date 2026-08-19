#!/usr/bin/env bash
# Correct protocol: every backbone generates, is graded, and is knowledge-filtered on itself.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="${PYTHON_BIN:-/home/tong56/venvs/whitebox/bin/python}"; ROOT="${PAPER4_SELF_ROOT:-$HERE/runs/paper4_self_matrix_v2}"; SHARD="${PAPER4_SHARD:-all}"; BATCH="${PAPER4_BATCH:-16}"; LIMIT="${PAPER4_LIMIT:-0}"
[[ "$SHARD" =~ ^(all|0|1)$ ]] || { echo 'PAPER4_SHARD must be all, 0, or 1'; exit 2; }
MODEL_CACHE_ROOT="${PAPER4_MODEL_CACHE_ROOT:-/tmp/tong56_huggingface}"
INTERMEDIATE_ROOT="${PAPER4_INTERMEDIATE_ROOT:-/tmp/tong56_paper4_intermediate}"
mkdir -p "$ROOT/logs" "$ROOT/status" "$MODEL_CACHE_ROOT/hub" "$MODEL_CACHE_ROOT/transformers" "$MODEL_CACHE_ROOT/datasets" "$INTERMEDIATE_ROOT"
export HF_HOME="$MODEL_CACHE_ROOT" HF_HUB_CACHE="$MODEL_CACHE_ROOT/hub" TRANSFORMERS_CACHE="$MODEL_CACHE_ROOT/transformers" HF_DATASETS_CACHE="$MODEL_CACHE_ROOT/datasets"
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 TORCH_DISABLE_NATIVE_JIT=1 SPANATTR_DISABLE_NATIVE_BMM=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tong56_paper4_triton}"
mkdir -p "$TRITON_CACHE_DIR"
names=(llama mistral qwen falcon3); ids=(NousResearch/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct tiiuae/Falcon3-7B-Instruct)
run(){ local tag="$1"; shift; local done="$ROOT/status/$tag.done" log="$ROOT/logs/$tag.log"; [[ -f "$done" ]] && { echo "[SKIP] $tag"; return; }; echo "[START] $(date -u +%FT%TZ) $tag" | tee -a "$ROOT/run.log"; if "$@" >>"$log" 2>&1; then date -u +%FT%TZ >"$done"; echo "[DONE] $tag" | tee -a "$ROOT/run.log"; else echo "[FAIL] $tag (see $log)" | tee -a "$ROOT/run.log"; fi; }
for i in "${!names[@]}"; do
 [[ "$SHARD" != all && $((i/2)) -ne "$SHARD" ]] && continue
 n="${names[$i]}"; model="${ids[$i]}"; d="$ROOT/models/$n"; mkdir -p "$d"
 hidden="$d/scientist_answers/hidden"; tmp_hidden="$INTERMEDIATE_ROOT/$n/scientist_answers/hidden"
 mkdir -p "$d/scientist_answers" "$tmp_hidden"
 [[ -e "$hidden" || -L "$hidden" ]] || ln -s "$tmp_hidden" "$hidden"
 run "$n.prepare.scientist_answers" "$PY" "$HERE/../tool_gate_correctness_stratification.py" collect --data "$HERE/../shuffled_prepend_names_question.json" --output "$d/scientist_answers" --model "$model" --batch-size "$BATCH" --limit "$LIMIT" --resume
 run "$n.prepare.scientist_probes" "$PY" "$HERE/77_run_closedbook_fact_probes.py" --model "$model" --batch-size 64 --output "$d/scientist_probes.jsonl" --summary "$d/scientist_probes_summary.json"
 run "$n.prepare.multidomain" "$PY" "$HERE/../athlete_qa/eval_multidomain_llama.py" --data-root "$HERE/../athlete_qa/multidomain_v5" --out "$d/multidomain" --model "$model" --name-batch "$BATCH"
 run "$n.prepare.trivia" "$PY" "$HERE/162_generate_trivia_self.py" --model "$model" --source "$HERE/runs/127_triviaqa_balanced_n1000.jsonl" --output "$d/trivia_answers.jsonl" --batch "$BATCH" --limit "$LIMIT" --resume
 run "$n.prepare.gsm8k" "$PY" "$HERE/160_generate_self_outputs.py" --model "$model" --source-manifest "$HERE/runs/140_gsm8k_natural/natural_balanced_n942.jsonl" --out-dir "$d/gsm8k" --batch "$BATCH" --limit "$LIMIT" --target-wrong 0 --resume
 run "$n.prepare.manifests" "$PY" "$HERE/161_build_self_manifests.py" --model "$model" --model-dir "$d"
 for ds in scientist multidomain trivia gsm8k; do for method in exact attention; do run "$n.$ds.$method" "$PY" "$HERE/158_collect_paper4_matrix.py" --dataset "$ds" --method "$method" --model "$model" --manifest "$d/manifests/$ds.jsonl" --out-dir "$ROOT/features/$n/$ds/$method" --batch "$BATCH" --limit "$LIMIT" --resume; done; done
done
echo "[END] $(date -u +%FT%TZ)" | tee -a "$ROOT/run.log"
