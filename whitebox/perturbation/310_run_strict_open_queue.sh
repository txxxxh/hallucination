#!/usr/bin/env bash
set -euo pipefail
cd /home/tong56/whitebox/perturbation
export CUDA_VISIBLE_DEVICES=0
PY=/home/tong56/venvs/whitebox/bin/python
MODEL=/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77
LOG=runs/308_scientist_full_open_p_r/queue.log
mkdir -p "$(dirname "$LOG")"
"$PY" 308_scientist_full_open_p_r.py all --resume --batch 32 --model "$MODEL" 2>&1 | tee -a "$LOG"
"$PY" 309_trivia_gsm_open_p_r.py trivia gsm8k drop 2>&1 | tee -a "$LOG"
echo "STRICT_OPEN_QUEUE_COMPLETE" | tee -a "$LOG"
