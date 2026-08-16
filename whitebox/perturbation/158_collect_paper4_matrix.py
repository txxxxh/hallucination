#!/usr/bin/env python3
"""Resumable exact or attention-pruned current127 collection for paper4 pools."""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np

from spanattr.core import Item, SpanAttributor, set_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"


def scientist_rows():
    base = importlib.import_module("152_scientist_attention_pruned_current127")
    return [dict(key=k, group=g, correct=y, context=p, question="",
                 pred=pred, other=other, prompt_mode=True)
            for k, g, y, p, pred, other in base.jobs()]


def trivia_rows(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [dict(key=r["key"], group=r["key"], correct=int(r["correct"]),
                 context=r["context"], question=r["question"],
                 pred=r["generation"], other=r["other_answer"], prompt_mode=False)
            for r in rows]


def gsm8k_rows(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [dict(key=r["key"], group=r["group"], correct=int(r["correct"]),
                 context=r["question"],
                 question="Provide the complete solution to this math problem.",
                 pred=r["generation"], other=r["reference_solution"], prompt_mode=False)
            for r in rows]


def multidomain_rows():
    mod = importlib.import_module("149_multidomain_scientist_frozen_transfer")
    return [dict(key=r["key"], group=r["domain"], correct=int(r["correct"]),
                 context=r["prompt"], question="", pred=r["pred"],
                 other=r["other"], prompt_mode=True) for r in mod.rows()]


def load_rows(args):
    if args.dataset == "scientist":
        return scientist_rows()
    if args.dataset == "trivia":
        return trivia_rows(args.manifest or RUNS / "127_triviaqa_balanced_n1000.jsonl")
    if args.dataset == "gsm8k":
        return gsm8k_rows(args.manifest or RUNS / "140_gsm8k_natural/natural_balanced_n942.jsonl")
    return multidomain_rows()


def attention_shortlist(att, prep, spans, blocks, keep):
    import torch
    prompt_len = len(prep.prompt_ids)
    maps = []
    for answer in (prep.pred_variant_ids[0], prep.gold_variant_ids[0]):
        ids = torch.cat([prep.prompt_ids, answer]).unsqueeze(0)
        with torch.inference_mode():
            output = att.model(input_ids=ids, output_attentions=True, use_cache=False)
        if not output.attentions or any(x is None for x in output.attentions):
            raise RuntimeError("model did not return attentions; load it with eager attention")
        layers = [a[0, :, prompt_len - 1:prompt_len + len(answer) - 1, :prompt_len]
                  .float().mean((0, 1)).cpu().numpy() for a in output.attentions]
        maps.append(np.mean(layers, axis=0))
        del output
    score = maps[0] + np.abs(maps[0] - maps[1])
    edges = np.linspace(prep.ctx_start, prep.ctx_end, blocks + 1).round().astype(int)
    regions = [(edges[i], edges[i + 1]) for i in range(blocks) if edges[i] < edges[i + 1]]
    block_score = np.array([score[a:b].sum() for a, b in regions])
    chosen = np.argsort(-block_score)[:min(keep, len(regions))]
    return [i for i, span in enumerate(spans)
            if any(span.end > regions[j][0] and span.start < regions[j][1] for j in chosen)]


def scan_subset(att, prep, ids):
    import torch
    zero = torch.zeros(len(prep.prompt_ids), device=att.device)
    alpha = torch.stack([zero, *[att.alpha_from_spans(prep, [i]) for i in ids]])
    pred, other = att.class_scores_batched(prep, alpha)
    return pred.numpy(), other.numpy()


def delete_span(row, item, chars, top):
    start, end = chars[top]
    text = re.sub(r"[ \t]+", " ", item.context[:start] + item.context[end:])
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def make_item(row):
    if row["prompt_mode"]:
        return Item.from_dict({"key": row["key"], "prompt": row["context"],
                               "pred": row["pred"], "gold": row["other"]})
    return Item(row["key"], row["context"], row["question"],
                row["other"], row["pred"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=("scientist", "trivia", "gsm8k", "multidomain"))
    parser.add_argument("--method", required=True, choices=("exact", "attention"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    if args.limit:
        rows = rows[:args.limit]
    collector = importlib.import_module("125_collect_current_three_benchmarks")
    loader = importlib.import_module("61_grad_span_proposal")
    model, tokenizer = loader.load_model(args.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tokenizer, device="cuda", baseline="mean",
                         length_norm=True, max_rows=args.batch)

    config = dict(dataset=args.dataset, method=args.method, model=args.model,
                  expected=len(rows), blocks=args.blocks, keep=args.keep)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for number, row in enumerate(rows, 1):
        path = args.out_dir / f"{row['key']}.npz"
        if args.resume and path.exists():
            continue
        item = make_item(row)
        prep = att.prepare(item)
        spans, chars = collector.spans(att, prep)
        pool = list(range(len(spans))) if args.method == "exact" else attention_shortlist(
            att, prep, spans, args.blocks, args.keep)
        pred1, other1 = scan_subset(att, prep, pool)
        effect1 = (pred1[0] - pred1[1:]) - (other1[0] - other1[1:])
        local = np.argsort(-np.abs(effect1))[:min(5, len(effect1))]
        selected = np.asarray([pool[i] for i in local], dtype=int)
        pred_hidden, other_hidden, layer14 = collector.selected_hidden(att, prep, selected)
        deleted = delete_span(row, item, chars, int(selected[0]))
        second = att.prepare(Item(row["key"] + "_d", deleted, item.question,
                                  row["other"], row["pred"],
                                  context_prefix=item.context_prefix))
        spans2, _ = collector.spans(att, second)
        pool2 = list(range(len(spans2))) if args.method == "exact" else attention_shortlist(
            att, second, spans2, args.blocks, args.keep)
        pred2, other2 = scan_subset(att, second, pool2)
        effect2 = (pred2[0] - pred2[1:]) - (other2[0] - other2[1:])
        local2 = np.argsort(-np.abs(effect2))[:min(5, len(effect2))]
        np.savez_compressed(
            path, key=np.asarray(row["key"]), group=np.asarray(row["group"]),
            correct=np.asarray(row["correct"]), deleted_text=np.asarray(spans[selected[0]].text),
            stage1_pred=np.r_[pred1[0], pred1[1:][local]],
            stage1_other=np.r_[other1[0], other1[1:][local]],
            stage2_pred=np.r_[pred2[0], pred2[1:][local2]],
            stage2_other=np.r_[other2[0], other2[1:][local2]],
            pred_hidden=pred_hidden.astype(np.float16),
            other_hidden=other_hidden.astype(np.float16), layer14=layer14.astype(np.float16),
            stage1_candidates=np.asarray(len(pool)), stage1_full=np.asarray(len(spans)),
            stage2_candidates=np.asarray(len(pool2)), stage2_full=np.asarray(len(spans2)))
        print(f"[{number}/{len(rows)}] {row['key']} q={len(pool)}/{len(spans)}+"
              f"{len(pool2)}/{len(spans2)}", flush=True)


if __name__ == "__main__":
    main()
