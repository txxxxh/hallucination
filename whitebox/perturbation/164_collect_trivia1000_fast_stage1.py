#!/usr/bin/env python3
"""Collect stage-1 max-head-attention or sentence-gradient features on TriviaQA-1000."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from spanattr.core import SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
paper = importlib.import_module("158_collect_paper4_matrix")
gradient = importlib.import_module("159_scientist_classgrad_sentence_current127")
collector = importlib.import_module("125_collect_current_three_benchmarks")


def maxhead_shortlist(att, prep, spans, blocks=12, keep=7):
    import torch

    answer = prep.pred_variant_ids[0]
    prompt_len = len(prep.prompt_ids)
    ids = torch.cat([prep.prompt_ids, answer]).unsqueeze(0)
    with torch.inference_mode():
        output = att.model(input_ids=ids, output_attentions=True, use_cache=False)
    # Match experiment 155: average answer-query positions and layers, retain
    # heads, then use the strongest head at each prompt token.
    heads = torch.stack([
        layer[0, :, prompt_len - 1:prompt_len + len(answer) - 1, :prompt_len]
        .float().mean(1) for layer in output.attentions
    ]).mean(0).cpu().numpy()
    score = heads.max(0)
    del output
    edges = np.linspace(prep.ctx_start, prep.ctx_end, blocks + 1).round().astype(int)
    regions = [(edges[i], edges[i + 1]) for i in range(blocks)
               if edges[i] < edges[i + 1]]
    region_score = np.asarray([score[start:end].sum() for start, end in regions])
    chosen = np.argsort(-region_score)[:min(keep, len(regions))]
    return [i for i, span in enumerate(spans)
            if any(span.end > regions[j][0] and span.start < regions[j][1]
                   for j in chosen)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=("attention_maxhead", "gradient_sentence"))
    parser.add_argument("--manifest", type=Path,
                        default=RUNS / "127_triviaqa_balanced_n1000.jsonl")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    rows = paper.trivia_rows(args.manifest)
    if len(rows) != 1000 and not args.limit:
        raise RuntimeError(f"expected TriviaQA pool of 1000, got {len(rows)}")
    if args.limit:
        rows = rows[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "trivia", "expected": len(rows), "method": args.method,
        "model": args.model, "stage": "stage1_only",
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
        item = paper.make_item(row)
        prep = att.prepare(item)
        spans, _ = collector.spans(att, prep)
        if args.method == "attention_maxhead":
            pool = maxhead_shortlist(att, prep, spans)
        else:
            pool = gradient.sentence_shortlist(
                att, prep, spans, saliency_mass=.75,
                max_candidate_fraction=.60, topk=3,
            )
        pred, other = paper.scan_subset(att, prep, pool)
        effect = (pred[0] - pred[1:]) - (other[0] - other[1:])
        local = np.argsort(-np.abs(effect))[:min(5, len(effect))]
        selected = np.asarray([pool[i] for i in local], dtype=int)
        pred_hidden, other_hidden, layer14 = collector.selected_hidden(
            att, prep, selected)
        np.savez_compressed(
            path, key=np.asarray(row["key"]), group=np.asarray(row["group"]),
            correct=np.asarray(row["correct"]),
            stage1_pred=np.r_[pred[0], pred[1:][local]],
            stage1_other=np.r_[other[0], other[1:][local]],
            pred_hidden=pred_hidden.astype(np.float16),
            other_hidden=other_hidden.astype(np.float16),
            layer14=layer14.astype(np.float16),
            stage1_candidates=np.asarray(len(pool)),
            stage1_full=np.asarray(len(spans)),
        )
        print(f"[{number}/{len(rows)}] {row['key']} q={len(pool)}/{len(spans)}",
              flush=True)


if __name__ == "__main__":
    main()
