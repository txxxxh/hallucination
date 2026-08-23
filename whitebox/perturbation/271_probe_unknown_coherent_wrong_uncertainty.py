#!/usr/bin/env python3
"""Split Scientist probe-unknown items into missing vs coherent-wrong knowledge.

The phenotype is defined only from paired closed-book fact probes, independently
of generated-answer uncertainty and correctness.  Each fact contributes

    P(Yes | fact paired with its true owner)
      - P(Yes | the same fact paired with the distractor).

Small absolute margins operationalize missing/indecisive knowledge.  Large,
consistently negative margins over at least two facts operationalize confident,
self-consistent but wrong ownership knowledge.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "271_probe_unknown_coherent_wrong_uncertainty"
SEED = 20260822


def read_jsonl(path):
    return [json.loads(x) for x in path.open() if x.strip()]


def originally_known(x):
    # Exact split used by 215 (strict > .5 matters for one/two-fact items).
    return bool(x["n_discriminative_facts"] >= 1 and
                x["binary_accuracy"] > .5 and
                x["pairwise_owner_accuracy"] > .5)


def fact_margins(x):
    out = []
    for fact_id in range(x["n_discriminative_facts"]):
        pair = [p for p in x["probes"]
                if p["probe_id"].split("::")[1] == f"f{fact_id}"]
        owner = next(p for p in pair if p["gold_yes"])
        other = next(p for p in pair if not p["gold_yes"])
        out.append(owner["p_yes"] - other["p_yes"])
    return np.asarray(out, dtype=float)


def component_map(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for x in rows:
        union(x["right_qid"], x["wrong_qid"])
    return {x["key"]: find(x["right_qid"]) for x in rows}


def clustered_ci(y, score, groups, draws=3000):
    y = np.asarray(y); score = np.asarray(score); groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(draws):
        selected = rng.choice(unique, len(unique), replace=True)
        take = np.concatenate([members[g] for g in selected])
        if np.unique(y[take]).size == 2:
            aucs.append(roc_auc_score(y[take], score[take]))
    return [float(x) for x in np.quantile(aucs, [.025, .975])]


def compare(rows, positive, negative, signal, high_cut):
    selected = [x for x in rows if x["phenotype"] in (positive, negative)]
    y = np.asarray([x["phenotype"] == positive for x in selected])
    s = np.asarray([x[signal] for x in selected])
    g = [x["component"] for x in selected]
    pos = s[y]; neg = s[~y]
    return {
        "positive": positive, "negative": negative,
        "n_positive": int(y.sum()), "n_negative": int((~y).sum()),
        "auroc": float(roc_auc_score(y, s)),
        "auroc_cluster_bootstrap_95ci": clustered_ci(y, s, g),
        "auprc": float(average_precision_score(y, s)),
        "mean_positive": float(pos.mean()), "mean_negative": float(neg.mean()),
        "median_positive": float(np.median(pos)), "median_negative": float(np.median(neg)),
        "high_uncertainty_rate_positive": float(np.mean(pos >= high_cut)),
        "high_uncertainty_rate_negative": float(np.mean(neg >= high_cut)),
    }


def analyze(rows, signal, high_cut):
    result = {}
    contrasts = [("true_missing", "confident_coherent_wrong"),
                 ("true_missing", "probe_known"),
                 ("confident_coherent_wrong", "probe_known")]
    for positive, negative in contrasts:
        for suffix, subset in (("all", rows),
                               ("generated_errors_only", [x for x in rows if x["error"]])):
            usable = [x for x in subset if x["phenotype"] in (positive, negative)]
            if len({x["phenotype"] for x in usable}) == 2:
                result[f"{positive}_vs_{negative}__{suffix}"] = compare(
                    usable, positive, negative, signal, high_cut)
    return result


def main():
    probes = read_jsonl(RUNS / "77_closedbook_fact_probe_results.jsonl")
    manifest = {str(x["key"]): x for x in read_jsonl(
        RUNS / "76_closedbook_fact_probe_manifest.jsonl")}
    uncertainty = {str(x["key"]): x for x in read_jsonl(
        RUNS / "215_scientist_uncertainty_known_unknown_predictions.jsonl")}
    components = component_map([manifest[str(x["key"])] for x in probes])
    rows = []
    for x in probes:
        key = str(x["key"])
        if key not in uncertainty or x["n_discriminative_facts"] < 1:
            continue
        margins = fact_margins(x)
        rows.append({
            "key": key, "component": components[key],
            "n_facts": int(len(margins)), "known": originally_known(x),
            "mean_signed_margin": float(margins.mean()),
            "mean_absolute_margin": float(np.abs(margins).mean()),
            "wrong_direction_fraction": float(np.mean(margins < 0)),
            "all_fact_margins": margins.tolist(),
            "error": int(uncertainty[key]["error"]),
            **{k: float(v) for k, v in uncertainty[key].items()
               if k not in {"key", "known", "error"}},
        })

    unknown = [x for x in rows if not x["known"]]
    strengths = np.asarray([x["mean_absolute_margin"] for x in unknown])
    main_lo, main_hi = np.quantile(strengths, [.30, .70])

    def assign(lo, hi):
        for x in rows:
            if x["known"]:
                x["phenotype"] = "probe_known"
            elif x["mean_absolute_margin"] <= lo:
                x["phenotype"] = "true_missing"
            elif (x["n_facts"] >= 2 and x["mean_absolute_margin"] >= hi and
                  x["mean_signed_margin"] < 0 and
                  x["wrong_direction_fraction"] >= .75):
                x["phenotype"] = "confident_coherent_wrong"
            else:
                x["phenotype"] = "ambiguous_probe_unknown"

    signals = ["mean_token_nll", "worst_token_nll", "mean_token_entropy",
               "max_token_entropy", "negative_top2_margin",
               "logit_uncertainty_lr_oof"]
    # A common, label-free operating point: upper 30% of the full fixed population.
    high_cuts = {s: float(np.quantile([x[s] for x in rows], .70)) for s in signals}
    assign(main_lo, main_hi)
    main_results = {s: analyze(rows, s, high_cuts[s]) for s in signals}
    counts = {}
    for name in ("probe_known", "true_missing", "confident_coherent_wrong",
                 "ambiguous_probe_unknown"):
        part = [x for x in rows if x["phenotype"] == name]
        counts[name] = {"n": len(part), "errors": sum(x["error"] for x in part),
                        "error_rate": float(np.mean([x["error"] for x in part]))}

    sensitivity = {}
    for q in (.20, .30, .40):
        lo, hi = np.quantile(strengths, [q, 1-q])
        assign(lo, hi)
        sensitivity[str(q)] = {
            "cutoffs": [float(lo), float(hi)],
            "counts": {p: sum(x["phenotype"] == p for x in rows) for p in
                       ("true_missing", "confident_coherent_wrong")},
            "mean_token_nll": analyze(rows, "mean_token_nll", high_cuts["mean_token_nll"]),
            "mean_token_entropy": analyze(rows, "mean_token_entropy", high_cuts["mean_token_entropy"]),
        }
    assign(main_lo, main_hi)

    report = {
        "protocol": {
            "population": "Scientist items used by experiment 215 with >=1 discriminative fact",
            "independence": "phenotypes use only paired closed-book probe probabilities; no generation correctness or uncertainty",
            "probe_unknown": "exact experiment-215 rule complement",
            "true_missing_proxy": "bottom 30% mean absolute owner-minus-distractor fact margin among probe-unknown",
            "confident_coherent_wrong_proxy": "top 30% margin strength, >=2 facts, negative mean margin, and >=75% facts prefer wrong owner",
            "high_uncertainty": "top 30% on the full fixed population",
            "uncertainty_orientation": "larger means more uncertain/error-like",
        },
        "n": len(rows), "unknown_strength_cutoffs": [float(main_lo), float(main_hi)],
        "counts": counts, "high_uncertainty_cutoffs": high_cuts,
        "results": main_results, "threshold_sensitivity": sensitivity,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "items.jsonl").open("w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
