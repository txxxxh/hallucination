#!/usr/bin/env python3
"""Binding cards for up to three highest-ranked matched attributes per question."""
from __future__ import annotations
import argparse, importlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
SRC = HERE / "runs/209_strict_attribute_binding_full"
OUT = HERE / "runs/220_top3_matched_attribute_binding"
card = importlib.import_module("204_scientist_binding_override_pilot")


def candidates():
    out = []
    for fp in sorted(SRC.glob("question_*.json")):
        q = json.loads(fp.read_text())
        ranked = list(q.get("top_signed", []))
        ent = q.get("entity")
        if ent and all(z.get("rank_positive") != ent.get("rank_positive") for z in ranked):
            ranked.append(ent)
        seen, selected = set(), []
        for z in sorted(ranked, key=lambda x: x.get("rank_positive", 10**9)):
            fact = z.get("fact")
            if not fact or z.get("u", 0) <= 0:
                continue
            k = (fact["field"], fact["value"].casefold().strip())
            if k in seen:
                continue
            seen.add(k); selected.append(z)
            if len(selected) == 3:
                break
        for j, z in enumerate(selected):
            out.append({"item_id": f"{q['key']}::a{j}", "key": q["key"], "right": q["right"],
                        "wrong": q["wrong"], "rank": z["rank_positive"], "span": z["text"],
                        "perturb_u": z["u"], "field": z["fact"]["field"],
                        "keyword": z["fact"]["value"]})
    return out


def cluster_stats(rows, rng, boot=10000):
    byq = defaultdict(list)
    for r in rows: byq[r["key"]].append(r["binding_effect"])
    qvals = np.array([np.mean(v) for v in byq.values()])
    raw = np.array([r["binding_effect"] for r in rows])
    means = np.mean(rng.choice(qvals, (boot, len(qvals)), replace=True), axis=1)
    try: p = float(wilcoxon(qvals, alternative="greater").pvalue)
    except ValueError: p = None
    return {"n_keywords": len(rows), "n_questions": len(byq), "keyword_mean": float(raw.mean()),
            "keyword_fraction_positive": float(np.mean(raw > 0)),
            "question_cluster_mean": float(qvals.mean()),
            "question_cluster_fraction_positive": float(np.mean(qvals > 0)),
            "cluster_bootstrap_ci95": [float(np.quantile(means,.025)), float(np.quantile(means,.975))],
            "question_wilcoxon_greater_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = candidates(); prompts, meta = [], []
    for n, r in enumerate(rows):
        for cond in card.binding_prompts(r, r["keyword"], f"ZORP-{10000+n}"):
            prompts.append(cond["prompt"]); meta.append((r["item_id"], cond))
    loader = importlib.import_module("61_grad_span_proposal")
    model, tok = loader.load_model(a.model, "bfloat16", "cuda"); tok.padding_side = "left"
    lp = card.score_ab(model, tok, prompts, a.batch)
    conditions = defaultdict(list)
    for (iid, cond), v in zip(meta, lp):
        m = float(v[cond["gold"]] - v[1-cond["gold"]])
        conditions[iid].append({**cond, "correct_margin": m})
    scored = []
    for r in rows:
        z = conditions[r["item_id"]]
        means = {(cue, owner): np.mean([x["correct_margin"] for x in z
                 if x["cue"] == cue and x["owner"] == owner])
                 for cue in ("real", "nonce") for owner in ("right", "wrong")}
        real = means["real", "wrong"] - means["real", "right"]
        null = means["nonce", "wrong"] - means["nonce", "right"]
        scored.append({**r, "real_override_asymmetry": float(real), "nonce_asymmetry": float(null),
                       "binding_effect": float(real-null), "conditions": z})
    with (a.out/"items.jsonl").open("w") as f:
        for r in scored: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    rng = np.random.default_rng(42)
    fields = sorted(set(r["field"] for r in scored))
    report = {"selection": "up to 3 highest-ranked distinct positive matched attributes per question",
              "source_questions": len(list(SRC.glob("question_*.json"))),
              "all": cluster_stats(scored, rng),
              "by_field": {fld: cluster_stats([r for r in scored if r["field"] == fld], rng)
                           for fld in fields}}
    (a.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
