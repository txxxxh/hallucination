#!/usr/bin/env bash
set -euo pipefail

cd /home/tong56/whitebox/perturbation
source ../activate_whitebox.sh

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
ITEMS="data/items_n128_generation_flip.json"
EXPECTED=30

line_count() {
    local path="$1"
    if [[ -f "$path" ]]; then
        wc -l < "$path"
    else
        echo 0
    fi
}

complete() {
    [[ "$(line_count "$1")" -ge "$EXPECTED" ]]
}

if [[ -n "${WATCH_PID:-}" ]]; then
    echo "[$(date -u +%FT%TZ)] waiting for existing run_all PID $WATCH_PID"
    while kill -0 "$WATCH_PID" 2>/dev/null; do
        sleep 30
    done
fi

echo "[$(date -u +%FT%TZ)] watchdog checking stage outputs"

if ! complete runs/61.jsonl; then
    echo "[$(date -u +%FT%TZ)] running stage 61 from scratch"
    python 61_grad_span_proposal.py --items "$ITEMS" --model "$MODEL" \
        --out runs/61.jsonl --m 12 --ig_steps 32 --null_draws 24
fi

if ! complete runs/62.jsonl; then
    echo "[$(date -u +%FT%TZ)] running stage 62 from scratch"
    python 62_interaction_matrix.py --in runs/61.jsonl --model "$MODEL" \
        --out runs/62.jsonl --null_pairs 24
fi

if ! complete runs/63.jsonl; then
    echo "[$(date -u +%FT%TZ)] running stage 63 from scratch"
    python 63_subset_select.py --in61 runs/61.jsonl --in62 runs/62.jsonl \
        --model "$MODEL" --out runs/63.jsonl --k 3 --n_gen 20
fi

if ! complete runs/64a.jsonl; then
    echo "[$(date -u +%FT%TZ)] running stage 64a from scratch"
    python 64a_vocab_decode.py --in61 runs/61.jsonl --in63 runs/63.jsonl \
        --model "$MODEL" --out runs/64a.jsonl --strategy second_order --topn 2 --limit 30
fi

if ! complete runs/64b.jsonl; then
    echo "[$(date -u +%FT%TZ)] running stage 64b from scratch"
    python 64b_vocab_recovery_generation.py --in61 runs/61.jsonl \
        --in64 runs/64a.jsonl --items "$ITEMS" --model "$MODEL" \
        --out runs/64b.jsonl --n_gen 3 --temperature 1.0
fi

echo "[$(date -u +%FT%TZ)] all stages complete"
