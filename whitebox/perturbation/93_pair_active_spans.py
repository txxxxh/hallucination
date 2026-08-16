#!/usr/bin/env python3
"""Exhaustively score pairs from top-3 active spans and top-3 edits/span."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in88", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--out", default="runs/93_pair_active_spans_n30.jsonl")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top_spans", type=int, default=3)
    p.add_argument("--top_edits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    set_seed(a.seed)

    items = {x.item_id: x for x in (Item.from_dict(d) for d in json.load(open(a.items)))}
    rows = [json.loads(x) for x in open(a.in88) if x.strip()]
    loader = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = loader(a.model, a.dtype, a.device)
    att = SpanAttributor(model, tok, device=a.device, baseline="mean", length_norm=True, max_rows=1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    with open(a.out, "w") as fh:
        for ni, row in enumerate(rows):
            item = items[row["item_id"]]
            prep = att.prepare(item)
            s0 = att.S0(prep)
            spans = row["results"][:a.top_spans]
            combos = []
            for left, right in itertools.combinations(spans, 2):
                left_edits = left.get("margin_oracle", [])[:a.top_edits]
                right_edits = right.get("margin_oracle", [])[:a.top_edits]
                for le, re in itertools.product(left_edits, right_edits):
                    lpos = {int(x["pos"]) for x in le["substitutions"]}
                    rpos = {int(x["pos"]) for x in re["substitutions"]}
                    if lpos & rpos:
                        continue
                    combos.append((left, right, le, re))
            if combos:
                ids = prep.prompt_ids.unsqueeze(0).repeat(len(combos), 1)
                for j, (_, _, le, re) in enumerate(combos):
                    for sub in le["substitutions"] + re["substitutions"]:
                        ids[j, int(sub["pos"])] = int(sub["id"])
                scores = att.score_ids_batched(prep, ids).numpy()
                order = np.argsort(scores)
                ranked = []
                for j in order[:min(10, len(order))]:
                    left, right, le, re = combos[int(j)]
                    ranked.append({
                        "score": float(scores[j]),
                        "u_realized": float(s0 - scores[j]),
                        "span_ids": [left["span_id"], right["span_id"]],
                        "span_text": [left["span_text"], right["span_text"]],
                        "substitutions": le["substitutions"] + re["substitutions"],
                    })
            else:
                ranked = []
            out = {"item_id": item.item_id, "S0": s0,
                   "n_combinations": len(combos), "pair_top": ranked,
                   "config": vars(a)}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            best = ranked[0]["score"] if ranked else s0
            print(f"[{ni + 1}/{len(rows)}] {item.item_id}: n={len(combos)} "
                  f"u={s0-best:+.3f} crossed={s0>0 and best<0}", flush=True)


if __name__ == "__main__":
    main()
