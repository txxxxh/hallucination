#!/usr/bin/env bash
set -u

current_pid=2759
output_file="/home/tong56/router/hidden-repr-bench/out_dist_shard1/pairs_curve.jsonl"
log_file="/home/tong56/router/hidden-repr-bench/out_dist_shard1/auto_resume.log"
target_count=33

while kill -0 "${current_pid}" 2>/dev/null; do
  sleep 30
done

cd /home/tong56/router/hidden-repr-bench || exit 1

while true; do
  count=0
  if [[ -f "${output_file}" ]]; then
    count=$(wc -l < "${output_file}")
  fi
  if (( count >= target_count )); then
    printf '%s complete: %d/%d\n' "$(date -u +%FT%TZ)" "${count}" "${target_count}" >> "${log_file}"
    exit 0
  fi

  printf '%s retry start: %d/%d\n' "$(date -u +%FT%TZ)" "${count}" "${target_count}" >> "${log_file}"
  CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/tmp/hf_budget_meta \
    HF_HUB_OFFLINE=1 \
    TORCH_DISABLE_NATIVE_JIT=1 \
    TRITON_CACHE_DIR=/tmp/triton_budget_shard1 \
    /home/tong56/venvs/whitebox/bin/python distraction_budget_reduction.py \
      --stage collect \
      --output-dir out_dist_shard1 \
      --gsm-ic-dir /home/tong56/router/hallu-diagnose-claude/data/raw/gsm_ic \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --max-pairs 200 \
      --num-shards 3 \
      --shard-id 1 \
      --resume >> "${log_file}" 2>&1
  status=$?
  printf '%s retry exit=%d\n' "$(date -u +%FT%TZ)" "${status}" >> "${log_file}"
  sleep 10
done
