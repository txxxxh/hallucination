#!/usr/bin/env python3
"""Complete multi-layer perturbation trajectories for full Scientist.

The historical 118 cache covers the old 1084 subset.  This collector targets
the complementary rows in 135 and reproduces exactly its current127 top-5 span
selection, while retaining six layers for every perturbation endpoint.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from spanattr.core import Item, SpanAttributor, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OUT = RUNS / "275_full_scientist_perturbation_trajectory"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--layers", type=int, nargs="+",
                   default=[10, 14, 18, 22, 26, 30])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    set_seed(42)

    current = importlib.import_module("125_collect_current_three_benchmarks")
    hidden_for = importlib.import_module(
        "116_collect_dual_candidate_hidden").hidden_for
    data = {str(x["key"]): x for x in json.load(
        (ROOT / "shuffled_prepend_names_question.json").open())}
    records = {x["key"]: x for x in map(json.loads, (
        ROOT / "tool_gate_correctness_names_llama31_8b" /
        "records.jsonl").open())}
    source_keys = sorted(fp.stem for fp in
                         (RUNS / "135_scientist_full_current127").glob("*.npz"))
    OUT.mkdir(parents=True, exist_ok=True)

    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=a.batch)
    todo = source_keys[:a.limit or None]
    for number, key in enumerate(todo, 1):
        target = OUT / f"{key}.npz"
        if target.exists() and a.resume:
            continue
        raw, record = data[key], records[key]
        pred = str(record["parsed_answer"])
        right, wrong = str(raw["rgt_ans"]), str(raw["wrg_ans"])
        other = wrong if pred == right else right
        item = Item.from_dict(dict(raw, pred=pred, gold=other))
        item.pred, item.gold = pred, other
        prep = att.prepare(item)
        spans, _ = current.spans(att, prep)
        pred_score, other_score = current.scan(att, prep, spans)
        influence = ((pred_score[0] - pred_score[1:]) -
                     (other_score[0] - other_score[1:]))
        ids = np.argsort(-np.abs(influence))[:min(5, len(influence))]

        import torch
        zero = torch.zeros(prep.prompt_ids.shape[0], device=att.device)
        alphas = torch.stack(
            [zero, *[att.alpha_from_spans(prep, [int(i)]) for i in ids]])
        pred_hidden = hidden_for(att, prep, alphas,
                                 prep.pred_variant_ids[0], a.layers)
        other_hidden = hidden_for(att, prep, alphas,
                                  prep.gold_variant_ids[0], a.layers)
        np.savez_compressed(
            target, key=np.asarray(key), top_ids=ids.astype(np.int32),
            layers=np.asarray(a.layers, np.int16),
            pred_u=(pred_score[0]-pred_score[1:])[ids].astype(np.float32),
            other_u=(other_score[0]-other_score[1:])[ids].astype(np.float32),
            pred_hidden=pred_hidden.astype(np.float16),
            other_hidden=other_hidden.astype(np.float16))
        print(f"[{number}/{len(todo)}] {key}", flush=True)


if __name__ == "__main__":
    main()
