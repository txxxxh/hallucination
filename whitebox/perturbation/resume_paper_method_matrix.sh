#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
source activate_whitebox.sh

export TORCH_DISABLE_NATIVE_JIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNS="$HERE/runs"
SE_ROOT="$RUNS/266_semantic_entropy_paper"
LOG_ROOT="$SE_ROOT/logs"
mkdir -p "$LOG_ROOT"

expected_count() {
  case "$1" in
    scientist) echo 1084 ;;
    trivia) echo 1000 ;;
    gsm8k) echo 942 ;;
    drop) echo 1000 ;;
  esac
}

sample_batch() {
  case "$1" in
    gsm8k|drop) echo 1 ;;
    *) echo 8 ;;
  esac
}

valid_jsonl_count() {
  local path="$1"
  [[ -f "$path" ]] || { echo 0; return; }
  python - "$path" <<'PY'
import json, sys
n = 0
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        json.loads(line)
        n += 1
print(n)
PY
}

run_semantic_entropy() {
  local ds="$1" expected batch samples report count
  expected="$(expected_count "$ds")"
  batch="$(sample_batch "$ds")"
  samples="$SE_ROOT/$ds/samples.jsonl"
  report="$SE_ROOT/$ds/report.json"

  if [[ -f "$report" ]]; then
    echo "[semantic-entropy] $ds already complete"
    return
  fi

  count="$(valid_jsonl_count "$samples")"
  if (( count < expected )); then
    echo "[semantic-entropy] sampling $ds: $count/$expected (batch=$batch)"
    python perturbation/266_semantic_entropy_paper.py sample "$ds" \
      --batch "$batch" --resume 2>&1 | tee -a "$LOG_ROOT/${ds}_sample.log"
  fi

  count="$(valid_jsonl_count "$samples")"
  if (( count != expected )); then
    echo "[semantic-entropy] $ds incomplete after sampling: $count/$expected" >&2
    return 1
  fi

  echo "[semantic-entropy] scoring $ds"
  python perturbation/266_semantic_entropy_paper.py score "$ds" \
    2>&1 | tee -a "$LOG_ROOT/${ds}_score.log"
  [[ -f "$report" ]]
}

for dataset in scientist trivia drop gsm8k; do
  run_semantic_entropy "$dataset"
done

# These reports are prerequisites for the expanded table.  They are already
# complete in the current workspace; fail loudly if a restored environment is
# missing any of them.
required_reports=(
  "$RUNS/263_representation_0770_protocol/trivia/report.json"
  "$RUNS/263_representation_0770_protocol/gsm8k/report.json"
  "$RUNS/263_representation_0770_protocol/drop/report.json"
  "$RUNS/264_saplma_paper/scientist/report.json"
  "$RUNS/264_saplma_paper/trivia/report.json"
  "$RUNS/264_saplma_paper/gsm8k/report.json"
  "$RUNS/264_saplma_paper/drop/report.json"
  "$RUNS/265_icr_probe_paper/scientist/report.json"
  "$RUNS/265_icr_probe_paper/trivia/report.json"
  "$RUNS/265_icr_probe_paper/gsm8k/report.json"
  "$RUNS/265_icr_probe_paper/drop/report.json"
)
for report in "${required_reports[@]}"; do
  [[ -f "$report" ]] || { echo "Missing prerequisite: $report" >&2; exit 1; }
done

echo "All currently implemented paper-method runs are complete."
echo "Next: run the AAAI-26 adaptive Bayesian SE implementation, then rebuild"
echo "$RUNS/262_paper_sota_completed_matrix.md with all method rows."
