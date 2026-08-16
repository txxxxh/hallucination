#!/usr/bin/env python3
"""Generate answers from the best token-wise active margin candidate per item."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, SpanAttributor, bootstrap_ci, set_seed


def generate(att, ids, n, temperature, max_new_tokens, seed):
    import torch

    answers = []
    for k in range(n):
        torch.manual_seed(seed + k)
        with torch.inference_mode():
            out = att.model.generate(
                input_ids=ids.unsqueeze(0),
                attention_mask=torch.ones_like(ids).unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=getattr(att.tok, "pad_token_id", 0) or 0,
            )
        answers.append(att.tok.decode(out[0, ids.shape[0]:].tolist()).strip())
    return answers


def flags(att, generations, targets):
    return [bool(att.match_rate([g], targets) > 0) for g in generations]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in88", default="runs/88_tokenwise_active_n30.jsonl")
    p.add_argument("--items", required=True)
    p.add_argument("--out", default="runs/90_active_tokenwise_generation_n30.jsonl")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_gen", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    set_seed(a.seed)

    raw_items = json.load(open(a.items))
    raw_by_id = {str(x.get("item_id", x.get("key"))): x for x in raw_items}
    items = {i.item_id: i for i in (Item.from_dict(x) for x in raw_items)}
    rows = [json.loads(x) for x in open(a.in88) if x.strip()]
    loader = importlib.import_module("61_grad_span_proposal").load_model
    model, tok = loader(a.model, a.dtype, a.device)
    att = SpanAttributor(model, tok, device=a.device, baseline="mean", length_norm=True, max_rows=16)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    paired, rises, after = [], [], []
    with open(a.out, "w") as fh:
        for idx, row in enumerate(rows):
            item = items[row["item_id"]]
            raw = raw_by_id[item.item_id]
            prep = att.prepare(item)
            eval_gold = str(raw.get("eval_gold", raw.get("rgt_ans", item.gold)))
            gold = [eval_gold] + list(raw.get("eval_gold_variants", []))
            pred = [item.pred] + item.pred_variants
            seed = a.seed + idx * 10000
            baseline = generate(att, prep.prompt_ids, a.n_gen, a.temperature, a.max_new_tokens, seed)
            bg, bp = flags(att, baseline, gold), flags(att, baseline, pred)

            candidates = [c for result in row["results"] for c in result.get("margin_oracle", [])]
            best = min(candidates, key=lambda x: x["score"])
            ids = prep.prompt_ids.clone()
            for sub in best["substitutions"]:
                ids[int(sub["pos"])] = int(sub["id"])
            generations = generate(att, ids, a.n_gen, a.temperature, a.max_new_tokens, seed)
            gm, pm = flags(att, generations, gold), flags(att, generations, pred)
            correction = float(np.mean([(not bg[k]) and gm[k] for k in range(a.n_gen)]))
            output = {
                "item_id": item.item_id,
                "source_score": best["score"],
                "source_u_realized": best["u_realized"],
                "substitutions": best["substitutions"],
                "baseline": {"generations": baseline, "gold_match": bg, "pred_match": bp,
                             "p_gold": float(np.mean(bg)), "p_pred": float(np.mean(bp))},
                "edit": {"generations": generations, "gold_match": gm, "pred_match": pm,
                         "p_gold": float(np.mean(gm)), "p_pred": float(np.mean(pm)),
                         "rise_p_gold": float(np.mean(gm) - np.mean(bg)),
                         "drop_p_pred": float(np.mean(bp) - np.mean(pm)),
                         "correction_rate_paired": correction},
            }
            fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            fh.flush()
            paired.append(correction)
            rises.append(output["edit"]["rise_p_gold"])
            after.append(output["edit"]["p_gold"])
            print(f"[{idx + 1}/{len(rows)}] {item.item_id}: p_gold={after[-1]:.3f} rise={rises[-1]:+.3f}", flush=True)

    for label, values in (("P(gold) after active edit", after),
                          ("rise in P(gold)", rises),
                          ("paired wrong-to-gold rate", paired)):
        lo, hi = bootstrap_ci(values, seed=a.seed)
        print(f"{label}: mean={np.mean(values):.3f} 95%CI=[{lo:.3f},{hi:.3f}] n={len(values)}")


if __name__ == "__main__":
    main()
