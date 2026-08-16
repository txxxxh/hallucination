#!/usr/bin/env bash
set -euo pipefail

cd /home/tong56/whitebox/perturbation
source ../activate_whitebox.sh

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
ITEMS="data/items_n128_generation_flip.json"
BASIS="runs/81_q0000_active_basis.pt"

python 82_zo_active_keywords.py \
    --items "$ITEMS" --basis "$BASIS" --rank 32 \
    --out runs/82_active_n30_r32_q4.jsonl \
    --model "$MODEL" --steps 1 --directions 2 --topk 5 --limit 30

python 88_tokenwise_active_projection.py \
    --in82 runs/82_active_n30_r32_q4.jsonl \
    --items "$ITEMS" --basis "$BASIS" \
    --out runs/88_tokenwise_active_n30.jsonl \
    --model "$MODEL" --top_spans 3
