#!/usr/bin/env python3
"""Compare unperturbed uncertainty on Scientist known vs unknown errors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CACHE = RUNS / "141_scientist_all_trajectory_l8"
OUT = RUNS / "215_scientist_uncertainty_known_unknown.json"
PREDS = RUNS / "215_scientist_uncertainty_known_unknown_predictions.jsonl"


def knowledge(p):
    return bool(p["n_discriminative_facts"] >= 1 and
                p["binary_accuracy"] > .5 and
                p["pairwise_owner_accuracy"] > .5)


def components(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for row in rows:
        union(row["right_qid"], row["wrong_qid"])
    return np.asarray([find(row["right_qid"]) for row in rows])


def metric(y, score):
    return {"n": int(len(y)), "errors": int(y.sum()),
            "error_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def bootstrap_delta(y, known, score, draws=2000):
    rng = np.random.default_rng(20260818)
    indices = {k: np.flatnonzero(known == k) for k in (False, True)}
    values = []
    for _ in range(draws):
        auc = {}
        for k, idx in indices.items():
            take = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(y[take])) < 2:
                break
            auc[k] = roc_auc_score(y[take], score[take])
        if len(auc) == 2:
            values.append(auc[False] - auc[True])
    lo, hi = np.quantile(values, [.025, .975])
    return {"unknown_minus_known_auroc": float(
                roc_auc_score(y[~known], score[~known]) -
                roc_auc_score(y[known], score[known])),
            "bootstrap_95ci": [float(lo), float(hi)], "draws": len(values)}


def main():
    probes = {x["key"]: x for x in map(json.loads,
        (RUNS / "77_closedbook_fact_probe_results.jsonl").open())}
    manifest = {x["key"]: x for x in map(json.loads,
        (RUNS / "76_closedbook_fact_probe_manifest.jsonl").open())}
    records = {x["key"]: x for x in map(json.loads,
        (HERE.parent / "tool_gate_correctness_names_llama31_8b" / "records.jsonl").open())}
    rows = []
    for file in sorted(CACHE.glob("*.npz")):
        with np.load(file, allow_pickle=True) as z:
            key = str(z["key"].item())
            if key not in probes or key not in manifest or key not in records:
                continue
            if not records[key].get("parse_valid", True):
                continue
            rows.append({**manifest[key], "key": key,
                         "error": int(not records[key]["correct"]),
                         "known": knowledge(probes[key]),
                         "x": z["logits"].astype(np.float64)})
    y = np.asarray([r["error"] for r in rows])
    known = np.asarray([r["known"] for r in rows], dtype=bool)
    x = np.stack([r["x"] for r in rows])
    groups = components(rows)
    # logits = mean token LP, minimum token LP, LP std, mean entropy,
    # max entropy, mean top1-top2 logit margin, answer token count.
    signals = {
        "mean_token_nll": -x[:, 0], "worst_token_nll": -x[:, 1],
        "mean_token_entropy": x[:, 3], "max_token_entropy": x[:, 4],
        "negative_top2_margin": -x[:, 5],
    }
    probabilities = []
    for seed in (42, 43, 44):
        p = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(x, y, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=.03, max_iter=5000, class_weight="balanced",
                solver="liblinear", random_state=seed))
            model.fit(x[train], y[train])
            p[test] = model.predict_proba(x[test])[:, 1]
        probabilities.append(p)
    signals["logit_uncertainty_lr_oof"] = np.mean(probabilities, axis=0)
    results = {}
    for name, score in signals.items():
        results[name] = {
            "all": metric(y, score),
            "unknown": metric(y[~known], score[~known]),
            "known": metric(y[known], score[known]),
            "gap": bootstrap_delta(y, known, score),
        }
    report = {
        "protocol": ("Unperturbed teacher-forced generated-answer uncertainty. "
                     "Known/unknown is independently defined by closed-book fact probes. "
                     "LR uses 3x5-fold candidate-component-grouped OOF; no perturbation features."),
        "n": len(y), "components": len(set(groups)),
        "counts": {"known_correct": int(np.sum(known & (y == 0))),
                   "known_error": int(np.sum(known & (y == 1))),
                   "unknown_correct": int(np.sum(~known & (y == 0))),
                   "unknown_error": int(np.sum(~known & (y == 1)))},
        "features": ["mean_token_logprob", "minimum_token_logprob", "token_logprob_std",
                     "mean_token_entropy", "max_token_entropy", "mean_top2_logit_margin",
                     "answer_token_count"],
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    with PREDS.open("w") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({"key": row["key"], "known": bool(known[i]),
                                     "error": int(y[i]), **{k: float(v[i])
                                     for k, v in signals.items()}}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
