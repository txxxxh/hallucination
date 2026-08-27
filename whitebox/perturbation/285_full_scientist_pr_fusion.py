#!/usr/bin/env python3
"""Evaluate feature-level P+R fusion on all 2,894 Scientist rows.

Uses the exact population, right-person grouped 3x5 OOF folds, fold-local
scaling/PCA, and logistic-regression settings from experiment 272.
"""
from __future__ import annotations

import importlib
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


base = importlib.import_module("272_full_scientist_standard_upr_tables")
OUT = base.RUNS / "285_full_scientist_pr_fusion"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load()
    y = np.asarray([x["error"] for x in rows])
    groups = np.asarray([x["right_qid"] for x in rows])

    r_blocks = [np.stack([x["r_last"] for x in rows]),
                np.stack([x["r_mean"] for x in rows])]
    p_blocks = [np.stack([x["p_scalar"] for x in rows])]
    p_blocks += [np.stack([x["p_hidden"][j] for x in rows])
                 for j in range(4)]
    p_blocks += [np.stack([x["p_layer"] for x in rows])]

    configs = {
        "P": (p_blocks, [None, 8, 8, 8, 8, 48]),
        "R": (r_blocks, [8, 8]),
        "P_plus_R": (p_blocks + r_blocks,
                     [None, 8, 8, 8, 8, 48, 8, 8]),
    }
    scores = {name: [] for name in configs}
    fold_details = []
    for seed in base.SEEDS:
        per_seed = {name: np.zeros(len(y)) for name in configs}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(
                cv.split(r_blocks[0], y, groups), 1):
            detail = {"seed": seed, "fold": fold,
                      "n_train": int(len(train)), "n_test": int(len(test))}
            for name, (blocks, dims) in configs.items():
                train_x, test_x, used = base.transform_blocks(
                    blocks, train, test, dims, seed)
                per_seed[name][test] = base.error_probability(
                    train_x, test_x, y, train, seed)
                detail[name + "_dims"] = used
            fold_details.append(detail)
        for name in configs:
            scores[name].append(per_seed[name])

    results = {}
    mean_scores = {}
    for name, seed_scores in scores.items():
        mean_score = np.mean(seed_scores, axis=0)
        mean_scores[name] = mean_score
        results[name] = {
            "mean_probability": base.binary_metrics(
                y, mean_score >= .5, mean_score),
            "per_seed": [base.binary_metrics(y, s >= .5, s)
                         for s in seed_scores],
        }

    report = {
        "protocol": (
            "2894 parse-valid full Scientist; right-person grouped 3x5 OOF; "
            "fold-local scaler/PCA; LR C=.03 class_weight=balanced; P and R "
            "feature-level early fusion; no closed-book probe features"),
        "n": len(y), "hallucination": int(y.sum()),
        "correct": int((1-y).sum()), "groups": len(set(groups)),
        "feature_definitions": {
            "P": "exact-current127 scalars + four perturbation hidden PCA8 + perturbation layer14 PCA48",
            "R": "unperturbed answer layer14 last+mean, PCA8 each",
            "P_plus_R": "concatenation of fold-local transformed P and R blocks",
        },
        "results": results,
        "fold_details": fold_details,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "predictions.jsonl").open("w") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({
                "key": row["key"], "error": int(y[i]),
                **{name + "_error_probability": float(score[i])
                   for name, score in mean_scores.items()},
            }) + "\n")
    print(json.dumps({"n": len(y), "groups": len(set(groups)),
                      "results": {k: v["mean_probability"]
                                  for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
