#!/usr/bin/env bash
# Hallu-Diagnose 无人值守全流程（Z3 等待人工标注，故明确跳过）。
set -Eeuo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/tong56/venvs/whitebox/bin/python}
EVAL_MODEL=${EVAL_MODEL:-unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit}
GEN_MODEL=${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA_LIMIT=${DATA_LIMIT:-1000}
Z4_FULL_BUDGET=${Z4_FULL_BUDGET:-8192}
Z4_CUT_RATIO=${Z4_CUT_RATIO:-0.3}
HALLU_BATCH_SIZE=${HALLU_BATCH_SIZE:-1}
INCLUDE_Z4=${INCLUDE_Z4:-1}
RESUME=${RESUME:-1}
DRY_RUN=0

usage() {
  echo "Usage: $0 [--dry-run] [--detach] [--force]"
  echo "Environment: DATA_LIMIT, EVAL_MODEL, GEN_MODEL, JUDGE_MODEL, HALLU_BATCH_SIZE"
}

if [[ ${1:-} == "--detach" ]]; then
  shift
  mkdir -p "$ROOT_DIR/logs"
  launch_log="$ROOT_DIR/logs/launcher_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup "$0" "$@" >"$launch_log" 2>&1 < /dev/null &
  echo "Pipeline started: pid=$! log=$launch_log"
  exit 0
fi

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --force) RESUME=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT_DIR"
mkdir -p logs data/run_state .runtime/matplotlib
LOG_FILE="$ROOT_DIR/logs/full_pipeline_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export HALLU_BATCH_SIZE
export MPLCONFIGDIR="$ROOT_DIR/.runtime/matplotlib"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

exec 9>"$ROOT_DIR/data/run_state/pipeline.lock"
if ! flock -n 9; then
  echo "Another full pipeline is already running." >&2
  exit 1
fi

current_step=preflight
trap 'code=$?; echo "[FAILED] step=$current_step exit=$code log=$LOG_FILE"; exit $code' ERR

run_step() {
  local name=$1
  shift
  local marker="$ROOT_DIR/data/run_state/${name}.done"
  current_step=$name
  if [[ $RESUME == 1 && -f $marker ]]; then
    echo "[SKIP] $name (completed marker exists)"
    return
  fi
  echo "[$(date -u +%FT%TZ)] START $name"
  if [[ $DRY_RUN == 1 ]]; then
    printf '  DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
  touch "$marker"
  echo "[$(date -u +%FT%TZ)] DONE  $name"
}

require_jsonl() {
  local path=$1
  if [[ $DRY_RUN == 1 ]]; then
    return
  fi
  [[ -s $path ]] || { echo "Missing or empty JSONL: $path" >&2; return 1; }
  "$PYTHON_BIN" -c 'import json,sys; p=sys.argv[1]; rows=[json.loads(x) for x in open(p) if x.strip()]; assert rows, p; print(f"[validate] {p}: {len(rows)} rows")' "$path"
}

echo "=== Hallu-Diagnose full pipeline (Z3 skipped) ==="
echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "eval_model=$EVAL_MODEL"
echo "gen_model=$GEN_MODEL"
echo "judge_model=$JUDGE_MODEL"
echo "per_source_limit=$DATA_LIMIT batch_size=$HALLU_BATCH_SIZE"
echo "log=$LOG_FILE"

[[ -x $PYTHON_BIN ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 1; }
run_step preflight "$PYTHON_BIN" -c 'import sys,torch; from transformers import AutoConfig; [AutoConfig.from_pretrained(m, local_files_only=True, trust_remote_code=True) for m in sys.argv[1:]]; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' "$EVAL_MODEL" "$GEN_MODEL" "$JUDGE_MODEL"
run_step download_data bash download_data.sh

run_step build_z1 "$PYTHON_BIN" scripts/01_build_z1.py --limit "$DATA_LIMIT"
require_jsonl data/processed/z1_pool.jsonl build_z1
run_step build_z2 "$PYTHON_BIN" scripts/02_build_z2.py --gen_model "$GEN_MODEL" --limit "$DATA_LIMIT"
require_jsonl data/processed/z2_pool.jsonl build_z2
if [[ $INCLUDE_Z4 == 1 ]]; then
  run_step build_z4 "$PYTHON_BIN" scripts/04_build_z4.py --model "$EVAL_MODEL" --n_pool "$DATA_LIMIT" --full_budget "$Z4_FULL_BUDGET" --cut_ratio "$Z4_CUT_RATIO"
  require_jsonl data/processed/z4_pool.jsonl build_z4
else
  echo "[SKIP] build_z4 (INCLUDE_Z4=0)"
fi
run_step build_z6 "$PYTHON_BIN" scripts/05_build_z6.py --limit "$DATA_LIMIT"
require_jsonl data/processed/z6_pool.jsonl build_z6

echo "[INFO] Z3 omitted: scripts/03_build_z3.py requires reviewed keep labels."
run_step screen_z1 "$PYTHON_BIN" scripts/10_screen.py --stressor z1 --model "$EVAL_MODEL"
require_jsonl data/processed/z1_final.jsonl screen_z1
run_step screen_z2 "$PYTHON_BIN" scripts/10_screen.py --stressor z2 --model "$EVAL_MODEL"
require_jsonl data/processed/z2_final.jsonl screen_z2
if [[ $INCLUDE_Z4 == 1 ]]; then
  run_step screen_z4 "$PYTHON_BIN" scripts/10_screen.py --stressor z4 --model "$EVAL_MODEL"
  require_jsonl data/processed/z4_final.jsonl screen_z4
else
  echo "[SKIP] screen_z4 (INCLUDE_Z4=0)"
fi
run_step screen_z6 "$PYTHON_BIN" scripts/10_screen.py --stressor z6 --model "$EVAL_MODEL"
require_jsonl data/processed/z6_final.jsonl screen_z6

if [[ $INCLUDE_Z4 == 1 ]]; then
  STRESSORS=(z1 z2 z4 z6)
else
  STRESSORS=(z1 z2 z6)
fi
run_step symptom_traces "$PYTHON_BIN" scripts/11_annotate_symptom.py --gen --model "$EVAL_MODEL" --stressors "${STRESSORS[@]}"
run_step symptom_judge "$PYTHON_BIN" scripts/11_annotate_symptom.py --judge --judge_model "$JUDGE_MODEL" --stressors "${STRESSORS[@]}"
run_step symptom_stats "$PYTHON_BIN" scripts/11_annotate_symptom.py --stats --stressors "${STRESSORS[@]}"

run_step matrix "$PYTHON_BIN" scripts/21_run_matrix.py --model "$EVAL_MODEL" --stressors "${STRESSORS[@]}"
RESULT_FILE="data/results/matrix_${EVAL_MODEL##*/}.jsonl"
require_jsonl "$RESULT_FILE" matrix
run_step matrix_stats "$PYTHON_BIN" scripts/30_stats.py --result "$RESULT_FILE"

if [[ $DRY_RUN == 1 ]]; then
  echo "=== DRY RUN COMPLETE ==="
  exit 0
fi
echo "=== PIPELINE COMPLETE ==="
echo "result=$ROOT_DIR/$RESULT_FILE"
echo "log=$LOG_FILE"
