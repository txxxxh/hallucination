#!/usr/bin/env python3
"""Rank spans by ZO proposals after exact nearest-vocabulary projection.

Every objective evaluation is a realizable token sequence: token-wise active
coefficients are proposed, converted to target embeddings, projected to exact
vocabulary nearest neighbours, and only then scored by the true margin.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, SpanAttributor, nms_disjoint, set_seed

p88 = importlib.import_module("88_tokenwise_active_projection")


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--in82", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--out", default="runs/91_discrete_constrained_n30.jsonl")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top_each", type=int, default=5,
                   help="candidate pool is union of active and mean top-k")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--directions", type=int, default=8)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--vocab_chunk", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    set_seed(a.seed)

    items = {x.item_id: x for x in (Item.from_dict(d) for d in json.load(open(a.items)))}
    rows = [json.loads(x) for x in open(a.in82) if x.strip()]
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
            if [s.text for s in spans] != row["span_text"]:
                raise ValueError("span reconstruction drift")
            rank = int(row["rank"])
            basis = saved["basis"][:, :rank].to(a.device, dtype=att.emb_layer.weight.dtype)
            s0 = att.S0(prep)
            active_ids = row["selection"]["active"][:a.top_each]
            mean_ids = row["selection"]["mean"][:a.top_each]
            candidate_ids = list(dict.fromkeys(active_ids + mean_ids))
            results = []
            for sid in candidate_ids:
                span = spans[sid]
                budget = float((prep.Ebar[span.start:span.end] - prep.E[span.start:span.end]).float().norm())
                proposals = p88.quantized_tokenwise(
                    att, prep, span, basis, s0, budget,
                    a.directions, a.rounds, a.scales,
                    a.seed + 700001 * ni + sid, a.vocab_chunk,
                )
                best = proposals[0] if proposals else None
                results.append({
                    "span_id": sid,
                    "span_text": span.text,
                    "from_active": sid in active_ids,
                    "from_mean": sid in mean_ids,
                    "best": best,
                    "n_projected_queries": a.directions * a.rounds * len(a.scales),
                })
            utility = np.array([
                max(0.0, r["best"]["u_realized"]) if r["best"] else 0.0
                for r in results
            ])
            local_spans = [spans[r["span_id"]] for r in results]
            selected_local = nms_disjoint(utility, local_spans, min(a.topk, len(results)))
            selected = [results[i]["span_id"] for i in selected_local]
            out = {
                "item_id": item.item_id,
                "S0": s0,
                "rank": rank,
                "candidate_span_ids": candidate_ids,
                "results": results,
                "selection": selected,
                "keywords": [spans[i].text for i in selected],
                "config": vars(a),
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            best_score = min((r["best"]["score"] for r in results if r["best"]), default=s0)
            print(f"[{ni + 1}/{len(rows)}] {item.item_id}: candidates={len(results)} best_u={s0-best_score:+.3f}", flush=True)


if __name__ == "__main__":
    main()
