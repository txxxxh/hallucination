#!/usr/bin/env bash
set -euo pipefail

run_pid="${1:?usage: $0 SCREEN_PID}"
old_dir="data/processed/real_life_screened"
new_dir="data/processed/real_life_screened_251_1000"
combined_dir="data/processed/real_life_screened_1_1000"
resolve_dir="data/processed/real_life_1_1000"

while kill -0 "$run_pid" 2>/dev/null; do
  sleep 30
done

for z in z1 z2 z6; do
  test -f "$new_dir/${z}_final.jsonl"
done
test -f "$old_dir/z4_final.jsonl"

mkdir -p "$combined_dir"
for z in z1 z2 z6; do
  cp "$new_dir/${z}_final.jsonl" "$combined_dir/${z}_final.jsonl"
  if test -s "$old_dir/${z}_final.jsonl"; then
    sed -i "\$r $old_dir/${z}_final.jsonl" "$combined_dir/${z}_final.jsonl"
  fi
done
cp "$old_dir/z4_final.jsonl" "$combined_dir/z4_final.jsonl"

/home/tong56/venvs/whitebox/bin/python scripts/06_build.py resolve \
  --screened-dir "$combined_dir" \
  --out-dir "$resolve_dir" \
  --merge-final

eval_model="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
judge_model="Qwen/Qwen2.5-7B-Instruct"
stressors=(z1 z2 z4 z6)

/home/tong56/venvs/whitebox/bin/python scripts/11_annotate_symptom.py \
  --gen --model "$eval_model" --stressors "${stressors[@]}"
/home/tong56/venvs/whitebox/bin/python scripts/11_annotate_symptom.py \
  --judge --judge_model "$judge_model" --stressors "${stressors[@]}"
/home/tong56/venvs/whitebox/bin/python scripts/11_annotate_symptom.py \
  --stats --stressors "${stressors[@]}"
