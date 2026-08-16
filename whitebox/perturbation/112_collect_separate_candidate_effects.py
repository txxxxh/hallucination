#!/usr/bin/env python3
"""Collect top-span effects for predicted and alternative candidates separately."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from spanattr.core import Item, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=RUNS/"88_known_gt05_n1084.jsonl")
    p.add_argument("--oracle", type=Path, default=RUNS/"88_oracle_top11_known_gt05.jsonl")
    p.add_argument("--data", type=Path, default=HERE.parent/"shuffled_prepend_names_question.json")
    p.add_argument("--records", type=Path, default=HERE.parent/"tool_gate_correctness_names_llama31_8b"/"records.jsonl")
    p.add_argument("--out-dir", type=Path, default=RUNS/"112_separate_candidate_top5")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); set_seed(a.seed)
    source = [json.loads(x) for x in a.source.open() if x.strip()]
    oracle = {x["key"]: x for x in map(json.loads, a.oracle.open())}
    data = {str(x["key"]): x for x in json.load(a.data.open())}
    records = {x["key"]: x for x in map(json.loads, a.records.open())}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, a.dtype, a.device)
    att = SpanAttributor(model, tok, device=a.device, baseline="mean",
                         length_norm=True, max_rows=a.batch)
    todo = source[:a.limit or None]
    for number, src in enumerate(todo, 1):
        key = src["key"]; target = a.out_dir/f"{key}.npz"
        if target.exists() and a.resume: continue
        raw, record = data[key], records[key]
        pred = str(record["parsed_answer"]); right = str(raw["rgt_ans"]); wrong = str(raw["wrg_ans"])
        other = wrong if pred == right else right
        item = Item.from_dict(dict(raw, pred=pred, gold=other)); item.pred, item.gold = pred, other
        prep = att.prepare(item)
        spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
        old_u = np.asarray(oracle[key]["u"], np.float32)
        top_ids = np.argsort(-np.abs(old_u))[:a.topk]
        zero = torch.zeros(prep.prompt_ids.shape[0], device=a.device)
        gates = [att.alpha_from_spans(prep, [int(i)]) for i in top_ids]
        alphas = torch.stack([zero, *gates])
        pred_scores, other_scores = att.class_scores_batched(prep, alphas)
        pred_scores, other_scores = pred_scores.numpy(), other_scores.numpy()
        np.savez_compressed(
            target, key=np.asarray(key), group=np.asarray(src["group"]),
            correct=np.asarray(int(src["correct"])), top_ids=top_ids,
            span_text=np.asarray([spans[int(i)].text for i in top_ids]),
            pred_name=np.asarray(pred), other_name=np.asarray(other),
            pred_scores=pred_scores, other_scores=other_scores,
            pred_u=pred_scores[0]-pred_scores[1:],
            other_u=other_scores[0]-other_scores[1:])
        print(f"[{number}/{len(todo)}] {key}", flush=True)


if __name__ == "__main__": main()
