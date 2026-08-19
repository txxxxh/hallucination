#!/usr/bin/env python3
"""Single-keyword binding experiment restricted to likelihood errors.

Unlike 204/207, selection is made by the unperturbed, length-normalized
teacher-forced name likelihood: wrong_name - right_name > 0.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
base = importlib.import_module("204_scientist_binding_override_pilot")
attrs = importlib.import_module("206_scientist_attribute_binding_pilot")
strict = importlib.import_module("207_scientist_strict_attribute_binding_pilot")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=1084,
                   help="Number of Scientist-known rows to screen")
    p.add_argument("--out", type=Path,
                   default=RUNS / "211_likelihood_error_binding")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    import torch
    from spanattr.core import Item, SpanAttributor, set_seed

    set_seed(42)
    a.out.mkdir(parents=True, exist_ok=True)
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, "bfloat16", "cuda")
    tok.padding_side = "left"
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=a.batch)
    scanmod = importlib.import_module("125_collect_current_three_benchmarks")
    facts = attrs.full_profile_attributes()
    jobs = importlib.import_module(
        "152_scientist_attention_pruned_current127").jobs()[:a.limit]

    selected = []
    screened = []
    for n, (key, group, generation_correct, prompt, pred, other) in enumerate(jobs, 1):
        # In jobs(), `other` is the right answer exactly when generation was wrong;
        # when generation was right, pred/other must be swapped for a stable
        # right/wrong definition independent of generation.
        if generation_correct:
            right, wrong = pred, other
        else:
            right, wrong = other, pred
        fp = a.out / f"{key}.json"
        if fp.exists() and a.resume:
            rec = json.loads(fp.read_text())
            screened.append(rec)
            if rec["likelihood_error"]:
                selected.append(rec)
            continue

        prep = att.prepare(Item.from_dict({"key": key, "prompt": prompt,
                                           "pred": wrong, "gold": right}))
        spans, _ = attrs.sliding_spans(att, prep)
        pr, ot = scanmod.scan(att, prep, spans)
        margin = float(pr[0] - ot[0])
        rec = {"key": key, "group": group, "right": right, "wrong": wrong,
               "generation_correct": bool(generation_correct),
               "base_margin_wrong_minus_right": margin,
               "likelihood_error": bool(margin > 0), "top_signed": [],
               "entity": None}
        if margin > 0:
            u = (pr[0] - pr[1:]) - (ot[0] - ot[1:])
            ranked = []
            for i in np.argsort(-u):
                fact = strict.strict_match(spans[int(i)].text, facts.get(key, []))
                words = base.toks(spans[int(i)].text)
                ranked.append({"rank_positive": len(ranked) + 1,
                               "text": spans[int(i)].text, "u": float(u[i]),
                               "logic": bool(words & base.LOGIC), "fact": fact})
            rec["top_signed"] = ranked[:10]
            rec["entity"] = next((x for x in ranked
                                  if x["u"] > 0 and x["fact"] is not None), None)
            selected.append(rec)
        fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        screened.append(rec)
        print(f"screen {n}/{len(jobs)} {key} likelihood_error={margin > 0}",
              flush=True)
        torch.cuda.empty_cache()

    probes, meta = [], []
    for n, rec in enumerate(selected):
        if rec["entity"] is None:
            continue
        keyword = rec["entity"]["fact"]["value"]
        nonce = f"ZORP-{100+n}"
        for condition in base.binding_prompts(rec, keyword, nonce):
            meta.append((rec, condition))
            probes.append(condition["prompt"])
    lp = base.score_ab(model, tok, probes, a.batch) if probes else np.zeros((0, 2))
    by = {}
    for (rec, condition), value in zip(meta, lp):
        gold = condition["gold"]
        condition = {**condition,
                     "correct_margin": float(value[gold] - value[1-gold])}
        by.setdefault(rec["key"], []).append(condition)

    rows = []
    for rec in selected:
        conditions = by.get(rec["key"])
        if not conditions:
            continue
        means = {(cue, owner): np.mean([q["correct_margin"] for q in conditions
                                        if q["cue"] == cue and q["owner"] == owner])
                 for cue in ("real", "nonce") for owner in ("right", "wrong")}
        real = means["real", "wrong"] - means["real", "right"]
        null = means["nonce", "wrong"] - means["nonce", "right"]
        rows.append({"key": rec["key"],
                     "keyword": rec["entity"]["fact"]["value"],
                     "field": rec["entity"]["fact"]["field"],
                     "base_margin_wrong_minus_right": rec["base_margin_wrong_minus_right"],
                     "perturb_u": rec["entity"]["u"],
                     "keyword_rank": rec["entity"]["rank_positive"],
                     "real_override_asymmetry": float(real),
                     "nonce_asymmetry": float(null),
                     "binding_effect": float(real-null), "conditions": conditions})
    with (a.out / "binding_items.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    d = np.asarray([r["binding_effect"] for r in rows])
    report = {"screened": len(screened),
              "generation_errors": sum(not r["generation_correct"] for r in screened),
              "likelihood_errors": len(selected),
              "likelihood_error_rate": len(selected) / max(1, len(screened)),
              "matched_positive_keyword": len(rows)}
    if len(rows):
        from scipy.stats import spearmanr, wilcoxon
        w = wilcoxon(d, alternative="greater")
        s = spearmanr(d, [r["perturb_u"] for r in rows])
        report.update(mean_binding_effect=float(d.mean()),
                      median_binding_effect=float(np.median(d)),
                      fraction_positive=float(np.mean(d > 0)),
                      flip_to_correct_fraction=float(np.mean([
                          r["base_margin_wrong_minus_right"] - r["perturb_u"] < 0
                          for r in rows])),
                      wilcoxon_greater_p=float(w.pvalue),
                      binding_vs_perturb_spearman={"rho": float(s.statistic),
                                                   "p": float(s.pvalue)})
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
