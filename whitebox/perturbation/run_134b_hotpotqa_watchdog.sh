#!/usr/bin/env bash
set -u
cd /home/tong56/whitebox/perturbation || exit 1
PY=/home/tong56/venvs/whitebox/bin/python
LOG=runs/134_hotpotqa_watchdog.log
STATUS=runs/134_hotpotqa_status.json
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export SPANATTR_DISABLE_NATIVE_BMM=1 TRITON_CACHE_DIR=/tmp/perturb_hotpotqa_triton
mkdir -p "$TRITON_CACHE_DIR"
retry(){ local n="$1";shift;local a=0;while true;do a=$((a+1));printf '{"stage":"%s","state":"running","attempt":%d,"time":"%s"}\n' "$n" "$a" "$(date -u +%FT%TZ)" > "$STATUS";printf '[%s] START %s attempt=%d\n' "$(date -u +%FT%TZ)" "$n" "$a" >> "$LOG";if "$@" >> "$LOG" 2>&1;then printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$n" >> "$LOG";return;fi;printf '[%s] RETRY %s\n' "$(date -u +%FT%TZ)" "$n" >> "$LOG";sleep 10;done;}
retry generate "$PY" 130b_generate_hotpotqa.py --resume
retry decoys_balance "$PY" 131a_hotpot_decoys_balance.py
retry features "$PY" 132_collect_hotpotqa.py --resume
retry evaluation "$PY" 133_eval_hotpotqa.py
printf '{"stage":"all","state":"complete","time":"%s"}\n' "$(date -u +%FT%TZ)" > "$STATUS"
