#!/usr/bin/env python3
"""Deterministic correctness-stratified sample from contextual profiles >0.5."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "runs"
SOURCE = ROOT / "97_profiles_contextual_gt05_n2863.jsonl"
OUT = ROOT / "97_profiles_contextual_gt05_n1500.jsonl"
SEED = 42
N = 1500

rows = [json.loads(x) for x in open(SOURCE) if x.strip()]
rng = random.Random(SEED)
positive = [x for x in rows if x["correct"]]
negative = [x for x in rows if not x["correct"]]
rng.shuffle(positive)
rng.shuffle(negative)
n_positive = round(N * len(positive) / len(rows))
selected = positive[:n_positive] + negative[:N - n_positive]
rng.shuffle(selected)
with open(OUT, "w") as fh:
    for row in selected:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(selected)}: correct={sum(x['correct'] for x in selected)}, "
      f"incorrect={sum(not x['correct'] for x in selected)}, seed={SEED}")
