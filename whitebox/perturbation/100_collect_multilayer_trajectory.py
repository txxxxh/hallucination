#!/usr/bin/env python3
"""Collect compact SOTA-style depth trajectories for scientist or TriviaQA.

This is deliberately a single unperturbed teacher-forced pass per example.
The resulting cache is fused later with the already collected perturbation
features, so the expensive span-response computation is not repeated.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def _scientist_rows(source_mode="known"):
    source = [json.loads(x) for x in (RUNS / "88_known_gt05_n1084.jsonl").open() if x.strip()]
    raw = {str(x["key"]): x for x in json.load(open(HERE.parent / "shuffled_prepend_names_question.json"))}
    records = {x["key"]: x for x in map(json.loads, open(HERE.parent / "tool_gate_correctness_names_llama31_8b" / "records.jsonl"))}
    if source_mode == "all":
        manifest = {x["key"]: x for x in map(json.loads, (RUNS / "76_closedbook_fact_probe_manifest.jsonl").open())}
        source = [dict(key=k, group=manifest[k]["right_qid"], correct=int(rec["correct"]))
                  for k, rec in records.items() if rec.get("parse_valid", True)]
    out = []
    for s in source:
        r, rec = raw[s["key"]], records[s["key"]]
        pred = str(rec["parsed_answer"])
        right, wrong = str(r["rgt_ans"]), str(r["wrg_ans"])
        out.append(dict(key=s["key"], group=s["group"], correct=int(s["correct"]),
                        context=r.get("context", r.get("prompt")), question=r.get("question", ""),
                        pred=pred, gold=wrong if pred == right else right, raw=r))
    return out


def _trivia_rows():
    mod = importlib.import_module("99_triviaqa_response_pilot")
    rows = mod.selected(RUNS / "98_triviaqa_balanced_n238.jsonl")
    return [dict(key=r["key"], group=r["key"], correct=int(r["correct"]),
                 context=r["context"], question=r["question"], pred=r["generation"],
                 gold=r["other_answer"], raw=r) for r in rows]


def _halueval_rows():
    rows = [json.loads(x) for x in open("/home/tong56/other_bench/qa_data (2).json") if x.strip()][:128]
    out = []
    for qi, r in enumerate(rows):
        group = f"hq{qi:05d}"
        out += [
            dict(key=group + "_right", group=group, correct=1, context=r["knowledge"],
                 question=r["question"], pred=r["right_answer"], gold=r["hallucinated_answer"], raw=r),
            dict(key=group + "_hall", group=group, correct=0, context=r["knowledge"],
                 question=r["question"], pred=r["hallucinated_answer"], gold=r["right_answer"], raw=r),
        ]
    return out


def _stats(x):
    """Coordinate-distribution statistics used by MultiHaluDet, plus PR."""
    import torch
    xf = x.float()
    mean = xf.mean(-1)
    centered = xf - mean[..., None]
    var = centered.square().mean(-1)
    std = var.sqrt().clamp_min(1e-8)
    z = centered / std[..., None]
    med = xf.median(-1).values
    mad = (xf - med[..., None]).abs().median(-1).values
    h2 = xf.square()
    pr = h2.sum(-1).square() / (h2.square().sum(-1) + 1e-8)
    return torch.stack([
        xf.norm(dim=-1), mean, std, xf.amin(-1), xf.amax(-1), med, mad,
        z.pow(4).mean(-1), (xf.abs() < 1e-3).float().mean(-1),
        (xf > 0).float().mean(-1), pr / xf.shape[-1],
    ], -1)


def collect(args):
    import torch
    from spanattr.core import Item, SpanAttributor

    rows = (_scientist_rows(args.scientist_source) if args.dataset == "scientist" else
            {"trivia": _trivia_rows, "halueval": _halueval_rows}[args.dataset]())
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    load_model = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean", length_norm=True,
                         max_rows=1)
    n_layers = int(model.config.num_hidden_layers)
    layers = np.unique(np.rint(np.linspace(1, n_layers, args.layers)).astype(int)).tolist()

    for number, row in enumerate(rows[:args.limit or None], 1):
        target = cache / f"{row['key']}.npz"
        if target.exists() and args.resume:
            continue
        if args.dataset == "scientist":
            item = Item.from_dict(dict(row["raw"], pred=row["pred"], gold=row["gold"]))
            item.pred, item.gold = row["pred"], row["gold"]
        else:
            item = Item(row["key"], row["context"], row["question"], row["gold"], row["pred"])
        prep = att.prepare(item)
        ans = prep.pred_variant_ids[0]
        pe = prep.E.unsqueeze(0)
        ae = att.emb_layer(ans).detach().unsqueeze(0).to(pe.dtype)
        seq = torch.cat([pe, ae], 1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            out = model(inputs_embeds=seq, attention_mask=mask,
                        output_hidden_states=True, use_cache=False)
        plen, alen = pe.shape[1], len(ans)
        last = torch.stack([out.hidden_states[l][0, plen + alen - 1] for l in layers])
        mean = torch.stack([out.hidden_states[l][0, plen:plen + alen].mean(0) for l in layers])
        # Teacher-forced token confidence features; answer-length normalized.
        logits = out.logits[0, plen - 1:plen + alen - 1].float()
        lp = logits.log_softmax(-1)
        tok_lp = lp.gather(1, ans[:, None]).squeeze(1)
        probs = lp.exp()
        entropy = -(probs * lp).sum(-1)
        top2 = logits.topk(2, dim=-1).values
        logit_features = torch.stack([
            tok_lp.mean(), tok_lp.amin(), tok_lp.std(unbiased=False),
            entropy.mean(), entropy.amax(), (top2[:, 0] - top2[:, 1]).mean(),
            torch.tensor(float(alen), device=logits.device),
        ])
        np.savez_compressed(
            target, key=np.asarray(row["key"]), group=np.asarray(row["group"]),
            correct=np.asarray(row["correct"]), layers=np.asarray(layers),
            last=last.float().cpu().numpy().astype(np.float16),
            mean=mean.float().cpu().numpy().astype(np.float16),
            last_stats=_stats(last).cpu().numpy().astype(np.float32),
            mean_stats=_stats(mean).cpu().numpy().astype(np.float32),
            logits=logit_features.cpu().numpy().astype(np.float32),
        )
        del out, seq, pe, ae
        print(f"[{number}/{min(len(rows), args.limit or len(rows))}] {row['key']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["scientist", "trivia", "halueval"])
    p.add_argument("--model", default="/tmp/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--scientist-source", choices=["known", "all"], default="known")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache", type=Path)
    a = p.parse_args()
    if a.cache is None:
        a.cache = RUNS / f"100_{a.dataset}_trajectory_l{a.layers}"
    collect(a)


if __name__ == "__main__":
    main()
