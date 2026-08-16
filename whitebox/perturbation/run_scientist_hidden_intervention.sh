#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/home/tong56/venvs/whitebox/bin/python"
cd "$HERE"

# Pilot: change both 100 values to 0 for the complete 1084-row experiment.
"$PY" 170_collect_scientist_hidden_intervention.py \
  --layers 4 12 20 28 --batch 32 --limit 100 --resume
"$PY" 171_eval_scientist_hidden_intervention.py --require 100
