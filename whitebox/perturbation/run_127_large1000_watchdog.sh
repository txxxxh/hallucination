#!/usr/bin/env bash
set -u
cd /home/tong56/whitebox/perturbation || exit 1
PY=/home/tong56/venvs/whitebox/bin/python
LOG=runs/127_large1000_watchdog.log
STATUS=runs/127_large1000_status.json
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export SPANATTR_DISABLE_NATIVE_BMM=1 TRITON_CACHE_DIR=/tmp/perturb_large1000_triton
mkdir -p "$TRITON_CACHE_DIR"
retry(){ local n="$1";shift;local a=0;while true;do a=$((a+1));printf '{"stage":"%s","state":"running","attempt":%d,"time":"%s"}\n' "$n" "$a" "$(date -u +%FT%TZ)" > "$STATUS";printf '[%s] START %s attempt=%d\n' "$(date -u +%FT%TZ)" "$n" "$a" >> "$LOG";if "$@" >> "$LOG" 2>&1;then printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$n" >> "$LOG";return;fi;printf '[%s] RETRY %s\n' "$(date -u +%FT%TZ)" "$n" >> "$LOG";sleep 10;done;}
retry trivia_generate "$PY" 97_prepare_triviaqa_pilot.py generate --n 1800 --items runs/127_triviaqa_items_n1800.jsonl --generations runs/127_triviaqa_generations_n1800.jsonl --model NousResearch/Meta-Llama-3.1-8B-Instruct --batch 16 --resume
retry trivia_balance "$PY" 98_build_triviaqa_balanced.py --input runs/127_triviaqa_generations_n1800.jsonl --out runs/127_triviaqa_balanced_n1000.jsonl --model NousResearch/Meta-Llama-3.1-8B-Instruct --batch 16 --per-class 500
retry trivia_features "$PY" 125_collect_current_three_benchmarks.py trivia --trivia-manifest runs/127_triviaqa_balanced_n1000.jsonl --out-dir runs/127_trivia1000_current127 --resume
retry halueval_features "$PY" 125_collect_current_three_benchmarks.py halueval --questions 500 --out-dir runs/127_halueval1000_current127 --resume
retry evaluation "$PY" 128_eval_large1000.py
printf '{"stage":"all","state":"complete","time":"%s"}\n' "$(date -u +%FT%TZ)" > "$STATUS"
