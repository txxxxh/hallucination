#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
ITEMS="${ITEMS:-data/items_example.json}"
python 84_active_vocab_decode.py --in82 runs/82_ex2_active_r16_q4.jsonl --items "$ITEMS" --basis runs/81_ex1_active_basis.pt --model "$MODEL"
python 85_active_word_generation.py --in84 runs/84_active_vocab_decode.jsonl --items "$ITEMS" --model "$MODEL"
python 86_compare_active_mean_words.py
