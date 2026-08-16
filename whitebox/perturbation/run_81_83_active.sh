#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
ITEMS="${ITEMS:-data/items_example.json}"
python 81_active_subspace_diagnosis.py --items "$ITEMS" --item_id ex1 --model "$MODEL"
python 82_zo_active_keywords.py --items "$ITEMS" --item_id ex2 --basis runs/81_active_basis.pt --model "$MODEL"
python 83_compare_zo_subspaces.py
