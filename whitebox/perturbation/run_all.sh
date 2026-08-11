#!/usr/bin/env bash
# End-to-end pipeline. Edit MODEL / ITEMS then run: bash run_all.sh
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
ITEMS="${ITEMS:-data/items_example.json}"
K="${K:-3}"
M="${M:-12}"

echo "### torch-free tests"
python -m spanattr.selftest
python tests/test_contracts.py

echo "### stage 1: span proposal"
python 61_grad_span_proposal.py --items "$ITEMS" --model "$MODEL" \
    --out runs/61.jsonl --m "$M" --ig_steps 32 --null_draws 24

echo "### stage 2: interaction matrix"
python 62_interaction_matrix.py --in runs/61.jsonl --model "$MODEL" \
    --out runs/62.jsonl --null_pairs 24

echo "### stage 3: head-to-head selection"
python 63_subset_select.py --in61 runs/61.jsonl --in62 runs/62.jsonl \
    --model "$MODEL" --out runs/63.jsonl --k "$K" --n_gen 20

echo "### stage 4: vocabulary decoding"
python 64a_vocab_decode.py --in61 runs/61.jsonl --in63 runs/63.jsonl \
    --model "$MODEL" --out runs/64a.jsonl --strategy second_order --topn 2 --limit 30

echo "### stage 4b: recovered-word sampled generation"
python 64b_vocab_recovery_generation.py --in61 runs/61.jsonl --in64 runs/64a.jsonl \
    --items "$ITEMS" --model "$MODEL" --out runs/64b.jsonl --n_gen 3 --temperature 1.0
