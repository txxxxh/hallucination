#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
ITEMS="${ITEMS:-data/items_example.json}"
python 81_zo_span_keywords.py --items "$ITEMS" --model "$MODEL" --out runs/81_zo_span_keywords.jsonl
python 82_compare_zo_mean.py --in81 runs/81_zo_span_keywords.jsonl --out runs/82_zo_mean_comparison.json
