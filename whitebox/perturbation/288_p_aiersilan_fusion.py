#!/usr/bin/env python3
"""Evaluate P, Aiersilan layer-14, and their early fusion on Scientist.

The comparison uses the exact P protocol's right-person grouped 3x5 OOF
splits.  Every scaler and PCA is fitted inside the outer training fold.
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


base = importlib.import_module("272_full_scientist_standard_upr_tables")
OUT = base.RUNS / "288_p_aiersilan_fusion"
AIERSILAN = base.RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt"
LAYER = 14


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load()
    y = np.asarray([row["error"] for row in rows])
    groups = np.asarray([row["right_qid"] for row in rows])

    saved = torch.load(AIERSILAN, map_location="cpu")
    a_by_key = {
        key: saved["hidden_states"][i, LAYER].float().numpy()
        for i, key in enumerate(saved["keys"])
    }
    row_keys = [row["key"] for row in rows]
    missing = [key for key in row_keys if key not in a_by_key]
    if missing:
        raise RuntimeError(f"missing {len(missing)} Aiersilan keys; first={missing[0]}")
    a_blocks = [np.stack([a_by_key[key] for key in row_keys])]

    p_blocks = [np.stack([row["p_scalar"] for row in rows])]
    p_blocks += [np.stack([row["p_hidden"][j] for row in rows])
                 for j in range(4)]
    p_blocks += [np.stack([row["p_layer"] for row in rows])]
    p_dims = [None, 8, 8, 8, 8, 48]

    configs = {
        "P": (p_blocks, p_dims),
        "Aiersilan": (a_blocks, [48]),
        "P_plus_Aiersilan": (p_blocks + a_blocks, p_dims + [48]),
    }
    scores = {name: [] for name in configs}
    fold_details = []
    for seed in base.SEEDS:
        per_seed = {name: np.zeros(len(y)) for name in configs}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(a_blocks[0], y, groups), 1):
            detail = {"seed": seed, "fold": fold,
                      "n_train": int(len(train)), "n_test": int(len(test))}
            for name, (blocks, dims) in configs.items():
                train_x, test_x, used = base.transform_blocks(
                    blocks, train, test, dims, seed)
                per_seed[name][test] = base.error_probability(
                    train_x, test_x, y, train, seed)
                detail[name + "_dims"] = used
            fold_details.append(detail)
            print(f"seed={seed} fold={fold}/5", flush=True)
        for name in configs:
            scores[name].append(per_seed[name])

    results = {}
    mean_scores = {}
    for name, seed_scores in scores.items():
        mean_score = np.mean(seed_scores, axis=0)
        mean_scores[name] = mean_score
        results[name] = {
            "mean_probability": base.binary_metrics(y, mean_score >= .5,
                                                     mean_score),
            "per_seed": [base.binary_metrics(y, score >= .5, score)
                         for score in seed_scores],
        }

    report = {
        "protocol": (
            "2894 parse-valid full Scientist; right-person grouped 3x5 OOF; "
            "fold-local scaling/PCA; LR C=.03 class_weight=balanced"
        ),
        "n": len(y), "hallucination": int(y.sum()),
        "correct": int((1-y).sum()), "groups": len(set(groups)),
        "feature_definitions": {
            "P": "exact-current127 P blocks; PCA dimensions [None,8,8,8,8,48]",
            "Aiersilan": "official-extractor candidate-last-token hidden state at layer 14; PCA48",
            "P_plus_Aiersilan": "fold-local transformed P and Aiersilan blocks concatenated",
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
    print(json.dumps({name: value["mean_probability"]
                      for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
