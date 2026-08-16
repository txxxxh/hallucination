#!/usr/bin/env bash
set -u

cd /home/tong56/whitebox/perturbation
source ../activate_whitebox.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TARGET=1500
OUT=runs/97_profiles_oracle_top11_n1500.jsonl
LOG=runs/97_profiles_detector_n1500.log
BATCHES=(12 8 4 2)
attempt=0

while true; do
    done_count=0
    if [[ -f "$OUT" ]]; then
        done_count=$(wc -l < "$OUT")
    fi
    if (( done_count >= TARGET )); then
        echo "[$(date -u +%FT%TZ)] oracle complete: ${done_count}/${TARGET}" >> "$LOG"
        exit 0
    fi

    idx=$attempt
    if (( idx >= ${#BATCHES[@]} )); then
        idx=$((${#BATCHES[@]} - 1))
    fi
    batch=${BATCHES[$idx]}
    echo "[$(date -u +%FT%TZ)] resume ${done_count}/${TARGET} with batch=${batch}" >> "$LOG"

    python 70_oracle_topk_detector.py collect \
        --source runs/97_profiles_contextual_gt05_n1500.jsonl \
        --data runs/97_profiles_question_only_data.json \
        --records ../tool_gate_correctness_profiles_llama31_8b/records.jsonl \
        --out "$OUT" \
        --model NousResearch/Meta-Llama-3.1-8B-Instruct \
        --dtype bfloat16 --batch "$batch" --topk 11 --resume >> "$LOG" 2>&1
    status=$?
    new_count=$(wc -l < "$OUT")
    echo "[$(date -u +%FT%TZ)] worker exit=${status}; progress=${new_count}/${TARGET}" >> "$LOG"
    attempt=$((attempt + 1))
    sleep 10
done
