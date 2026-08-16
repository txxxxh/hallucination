#!/usr/bin/env python3
"""Collect exact, max-head-attention, or sentence-gradient Stage-1 DROP features."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from spanattr.core import Item, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
fast = importlib.import_module("164_collect_trivia1000_fast_stage1")
paper = importlib.import_module("158_collect_paper4_matrix")
gradient = importlib.import_module("159_scientist_classgrad_sentence_current127")
collector = importlib.import_module("125_collect_current_three_benchmarks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=("exact", "attention_maxhead", "gradient_sentence"))
    parser.add_argument("--manifest", type=Path,
                        default=RUNS / "166_drop1000/drop_balanced_n1000.jsonl")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open() if line.strip()]
    if len(rows) != 1000 and not args.limit:
        raise RuntimeError(f"expected DROP pool of 1000, got {len(rows)}")
    if args.limit:
        rows = rows[:args.limit]
    set_seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "DROP validation natural generations", "expected": len(rows),
        "method": args.method, "model": args.model, "stage": "stage1_only",
        "detector": "fixed 108d = stage1 curves32 + hidden4x8 + layer14 PCA44",
        "attention": {"blocks": 12, "keep": 7},
        "gradient": {"saliency_mass": .75, "candidate_cap": .60,
                     "sentence_topk": 3},
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    loader = importlib.import_module("61_grad_span_proposal")
    model, tokenizer = loader.load_model(args.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tokenizer, device="cuda", baseline="mean",
                         length_norm=True, max_rows=args.batch)

    for number, row in enumerate(rows, 1):
        path = args.out_dir / f"{row['key']}.npz"
        if args.resume and path.exists():
            continue
        item = Item(row["key"], row["context"], row["question"],
                    row["other_answer"], row["generation"])
        prep = att.prepare(item)
        spans, _ = collector.spans(att, prep)
        if args.method == "exact":
            pool = list(range(len(spans)))
        elif args.method == "attention_maxhead":
            pool = fast.maxhead_shortlist(att, prep, spans)
        else:
            pool = gradient.sentence_shortlist(
                att, prep, spans, saliency_mass=.75,
                max_candidate_fraction=.60, topk=3)
        pred, other = paper.scan_subset(att, prep, pool)
        effect = (pred[0] - pred[1:]) - (other[0] - other[1:])
        local = np.argsort(-np.abs(effect))[:min(5, len(effect))]
        selected = np.asarray([pool[i] for i in local], dtype=int)
        pred_hidden, other_hidden, layer14 = collector.selected_hidden(
            att, prep, selected)
        np.savez_compressed(
            path, key=np.asarray(row["key"]), group=np.asarray(row["group"]),
            correct=np.asarray(row["correct"]),
            generation_words=np.asarray(row["generation_words"]),
            stage1_pred=np.r_[pred[0], pred[1:][local]],
            stage1_other=np.r_[other[0], other[1:][local]],
            pred_hidden=pred_hidden.astype(np.float16),
            other_hidden=other_hidden.astype(np.float16),
            layer14=layer14.astype(np.float16),
            stage1_candidates=np.asarray(len(pool)), stage1_full=np.asarray(len(spans)),
        )
        print(f"[{number}/{len(rows)}] {row['key']} q={len(pool)}/{len(spans)}",
              flush=True)


if __name__ == "__main__":
    main()
