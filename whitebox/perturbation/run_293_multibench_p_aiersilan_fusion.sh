#!/usr/bin/env bash
set -euo pipefail

cd /home/tong56
PY=/home/tong56/venvs/whitebox/bin/python
COLLECTOR=whitebox/perturbation/282_aiersilan_exact_original_benchmarks.py
WORK=whitebox/perturbation/runs/293_multibench_p_aiersilan_fusion
LOG="$WORK/run.log"

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

date -u
"$PY" "$COLLECTOR" collect trivia --work "$WORK" --resume
"$PY" "$COLLECTOR" collect gsm8k --work "$WORK" --resume
"$PY" "$COLLECTOR" collect drop --work "$WORK" --resume
"$PY" whitebox/perturbation/293_multibench_p_aiersilan_fusion.py trivia gsm8k drop
date -u
