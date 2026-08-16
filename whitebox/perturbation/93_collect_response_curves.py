#!/usr/bin/env python3
"""Collect intermediate-alpha response curves for the known>0.5 top-5 spans."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from spanattr.core import Item, SpanAttributor, set_seed


def hidden_last(att, prep, alphas, layer):
    import torch
    ans = prep.pred_variant_ids[0]
    chunks = []
    for start in range(0, len(alphas), att.max_rows):
        alpha = alphas[start:start + att.max_rows]
        pe = att._embeds(prep, alpha)
        batch = pe.shape[0]
        ae = att.emb_layer(ans).detach().unsqueeze(0).expand(batch, -1, -1)
        seq = torch.cat([pe, ae.to(pe.dtype)], 1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=att.device)
        with torch.inference_mode():
            out = att.model(inputs_embeds=seq, attention_mask=mask,
                            output_hidden_states=True, use_cache=False)
        chunks.append(out.hidden_states[layer][:, pe.shape[1] + len(ans) - 1].float().cpu())
        del out, seq, pe
    return torch.cat(chunks).numpy()


def main():
    import torch
    p = argparse.ArgumentParser()
    root = HERE / "runs"
    p.add_argument("--source", default=root / "88_known_gt05_n1084.jsonl", type=Path)
    p.add_argument("--oracle", default=root / "88_oracle_top11_known_gt05.jsonl", type=Path)
    p.add_argument("--data", default=HERE.parent / "shuffled_prepend_names_question.json", type=Path)
    p.add_argument("--records", default=HERE.parent / "tool_gate_correctness_names_llama31_8b/records.jsonl", type=Path)
    p.add_argument("--out-dir", default=root / "93_response_curve_top5", type=Path)
    p.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--mid-alphas", type=float, nargs="+", default=[.25, .5, .75])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    source = [json.loads(x) for x in args.source.open() if x.strip()]
    oracle = {x["key"]: x for x in map(json.loads, args.oracle.open())}
    data = {str(x["key"]): x for x in json.load(args.data.open())}
    records = {x["key"]: x for x in map(json.loads, args.records.open())}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_model = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=args.batch)
    todo = source[:args.limit or None]
    for number, src in enumerate(todo, 1):
        key = src["key"]; target = args.out_dir / f"{key}.npz"
        if target.exists() and args.resume:
            continue
        raw, rr = data[key], records[key]
        pred = str(rr["parsed_answer"]); right, wrong = str(raw["rgt_ans"]), str(raw["wrg_ans"])
        other = wrong if pred == right else right
        item = Item.from_dict(dict(raw, pred=pred, gold=other)); item.pred, item.gold = pred, other
        prep = att.prepare(item)
        spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
        u = np.asarray(oracle[key]["u"], np.float32)
        top_ids = np.argsort(-np.abs(u))[:args.topk]
        gates = [att.alpha_from_spans(prep, [int(i)]) for i in top_ids]
        alphas = torch.stack([a * level for a in gates for level in args.mid_alphas])
        hidden = hidden_last(att, prep, alphas, args.layer).reshape(args.topk, len(args.mid_alphas), -1)
        margins = att.S_batched(prep, alphas).numpy().reshape(args.topk, len(args.mid_alphas))
        np.savez_compressed(target, key=np.asarray(key), group=np.asarray(src["group"]),
                            correct=np.asarray(int(src["correct"])), layer=np.asarray(args.layer),
                            top_ids=top_ids, top_u=u[top_ids], mid_alphas=np.asarray(args.mid_alphas, np.float32),
                            answer_last_mid=hidden.astype(np.float16), margin_mid=margins.astype(np.float32))
        print(f"[{number}/{len(todo)}] {key}", flush=True)


if __name__ == "__main__":
    main()
