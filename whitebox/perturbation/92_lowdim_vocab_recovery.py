#!/usr/bin/env python3
"""Low-dimensional active-coordinate vocabulary recovery with residual control."""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, SpanAttributor, set_seed

p87 = importlib.import_module("87_projection_aware_decode")
p88 = importlib.import_module("88_tokenwise_active_projection")


def projected_tables(att, prep, span, basis, delta, pool, lambdas, chunk):
    """Retrieve tokens by active-coordinate target distance plus orthogonal residual."""
    import torch

    W = att.emb_layer.weight.detach()
    special = set(getattr(att.tok, "all_special_ids", []) or [])
    tables = []
    B = basis.float()
    for local, pos in enumerate(range(span.start, span.end)):
        e = prep.E[pos].float()
        target_z = delta[local].float() @ B
        heaps = {lam: [] for lam in lambdas}
        for lo in range(0, W.shape[0], chunk):
            hi = min(W.shape[0], lo + chunk)
            D = W[lo:hi].float() - e
            Z = D @ B
            active_dist2 = (Z - target_z).square().sum(1)
            residual2 = (D.square().sum(1) - Z.square().sum(1)).clamp_min(0)
            for lam in lambdas:
                metric = active_dist2 / B.shape[1] + lam * residual2 / (B.shape[0] - B.shape[1])
                k = min(pool, hi - lo)
                vals, ids = torch.topk(-metric, k)
                heaps[lam].extend((-float(v), lo + int(i),
                                   float(active_dist2[int(i)].sqrt()),
                                   float(residual2[int(i)].sqrt()))
                                  for v, i in zip(vals.cpu(), ids.cpu()))
        old = int(prep.prompt_ids[pos])
        per_lambda = {}
        for lam, rows in heaps.items():
            out, seen = [], set()
            for metric, vid, ad, residual in sorted(rows):
                if vid in seen or vid == old or vid in special or not att.tok.decode([vid]).strip():
                    continue
                seen.add(vid)
                out.append({"pos": pos, "id": vid, "tok": att.tok.decode([vid]),
                            "orig": att.tok.decode([old]), "metric": metric,
                            "active_distance": ad, "orthogonal_residual": residual,
                            "lambda": lam})
                if len(out) >= pool:
                    break
            per_lambda[str(lam)] = out
        tables.append(per_lambda)
    return tables


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--in82", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--out", default="runs/92_lowdim_vocab_recovery.jsonl")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--top_spans", type=int, default=3)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--directions", type=int, default=4)
    p.add_argument("--mu", type=float, default=.25)
    p.add_argument("--lr", type=float, default=.35)
    p.add_argument("--pool", type=int, default=3)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    p.add_argument("--vocab_chunk", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    set_seed(a.seed)
    items = {x.item_id: x for x in (Item.from_dict(d) for d in json.load(open(a.items)))}
    rows = [json.loads(x) for x in open(a.in82) if x.strip()][:a.limit]
    loader = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = loader(a.model, a.dtype, a.device)
    att = SpanAttributor(model, tok, device=a.device, baseline="mean", length_norm=True, max_rows=1)
    saved = torch.load(a.basis, map_location="cpu", weights_only=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    with open(a.out, "w") as fh:
        for ni, row in enumerate(rows):
            item = items[row["item_id"]]
            prep = att.prepare(item)
            spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
            prep.spans = spans
            rank = int(row["rank"])
            B = saved["basis"][:, :rank].to(a.device, dtype=att.emb_layer.weight.dtype)
            s0 = att.S0(prep)
            results = []
            for sid in row["selection"]["active"][:a.top_spans]:
                span = spans[sid]
                budget = float((prep.Ebar[span.start:span.end] - prep.E[span.start:span.end]).float().norm())
                cont = p88.optimize_tokenwise(att, prep, span, B, s0, budget,
                                              a.steps, a.directions, a.mu, a.lr,
                                              a.seed + 100003 * ni + sid)
                tables = projected_tables(att, prep, span, B, cont["delta"],
                                          a.pool, a.lambdas, a.vocab_chunk)
                recovered = {}
                for lam in a.lambdas:
                    choices = [x[str(lam)] for x in tables]
                    recovered[str(lam)] = p87.score_combos(att, prep, s0, choices, top=5, cap=512)
                results.append({"span_id": sid, "span_text": span.text,
                                "continuous_u": cont["continuous_u"],
                                "recovered": recovered})
            out = {"item_id": item.item_id, "S0": s0, "rank": rank,
                   "results": results, "config": vars(a)}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            summary = {str(lam): max((r["recovered"][str(lam)][0]["u_realized"]
                                      for r in results if r["recovered"][str(lam)]), default=0.0)
                       for lam in a.lambdas}
            print(f"[{ni + 1}/{len(rows)}] {item.item_id}: {summary}", flush=True)


if __name__ == "__main__":
    main()
