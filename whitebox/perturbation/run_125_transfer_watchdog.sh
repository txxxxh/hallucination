#!/usr/bin/env bash
set -u

cd /home/tong56/whitebox/perturbation || exit 1
PY=/home/tong56/venvs/whitebox/bin/python
LOG=runs/125_transfer_watchdog.log
STATUS=runs/125_transfer_watchdog_status.json

# This workload only needs eager inference.  Avoid torch.compile/Inductor and
# Triton compilation caches, which are unreliable on the current filesystem.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export SPANATTR_DISABLE_NATIVE_BMM=1
export TRITON_CACHE_DIR=/tmp/perturb_transfer_triton_cache

run_retry() {
  local name="$1"
  shift
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    printf '{"stage":"%s","state":"running","attempt":%d,"time":"%s"}\n' "$name" "$attempt" "$(date -u +%FT%TZ)" > "$STATUS"
    printf '[%s] START %s attempt=%d\n' "$(date -u +%FT%TZ)" "$name" "$attempt" >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
      printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$name" >> "$LOG"
      return 0
    fi
    code=$?
    printf '[%s] RETRY %s exit=%d\n' "$(date -u +%FT%TZ)" "$name" "$code" >> "$LOG"
    sleep 10
  done
}

run_retry trivia "$PY" 125_collect_current_three_benchmarks.py trivia --resume
run_retry halueval "$PY" 125_collect_current_three_benchmarks.py halueval --resume
run_retry reallife "$PY" 125_collect_current_three_benchmarks.py reallife --resume
run_retry evaluation "$PY" 126_eval_current_four_benchmarks.py trivia halueval reallife

printf '{"stage":"all","state":"complete","time":"%s"}\n' "$(date -u +%FT%TZ)" > "$STATUS"
printf '[%s] ALL COMPLETE\n' "$(date -u +%FT%TZ)" >> "$LOG"
