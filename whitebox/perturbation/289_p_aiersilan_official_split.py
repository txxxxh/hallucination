#!/usr/bin/env python3
"""P + Aiersilan using Aiersilan's stratified 70/10/20 random splits."""
from __future__ import annotations

import importlib
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split


base = importlib.import_module("272_full_scientist_standard_upr_tables")
OUT = base.RUNS / "289_p_aiersilan_official_split"
AIERSILAN = base.RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt"
LAYER = 14


def split_indices(y, seed):
    indices = np.arange(len(y))
    train_val, test = train_test_split(
        indices, test_size=.2, stratify=y, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=.1/.8, stratify=y[train_val], random_state=seed)
    return np.asarray(train), np.asarray(val), np.asarray(test)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load()
    y = np.asarray([row["error"] for row in rows])
    keys = [row["key"] for row in rows]

    saved = torch.load(AIERSILAN, map_location="cpu")
    a_by_key = {key: saved["hidden_states"][i, LAYER].float().numpy()
                for i, key in enumerate(saved["keys"])}
    missing = [key for key in keys if key not in a_by_key]
    if missing:
        raise RuntimeError(f"missing {len(missing)} Aiersilan keys")
    a_blocks = [np.stack([a_by_key[key] for key in keys])]
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

    results = {name: [] for name in configs}
    predictions = []
    split_details = []
    for seed in base.SEEDS:
        train, val, test = split_indices(y, seed)
        detail = {"seed": seed, "n_train": len(train), "n_val": len(val),
                  "n_test": len(test)}
        for name, (blocks, dims) in configs.items():
            train_x, test_x, used = base.transform_blocks(
                blocks, train, test, dims, seed)
            score = base.error_probability(train_x, test_x, y, train, seed)
            metrics = base.binary_metrics(y[test], score >= .5, score)
            results[name].append(metrics)
            detail[name + "_dims"] = used
            for idx, probability in zip(test, score):
                predictions.append({"seed": seed, "key": keys[idx],
                                    "error": int(y[idx]), "method": name,
                                    "error_probability": float(probability)})
        split_details.append(detail)
        print(f"seed={seed} complete", flush=True)

    summary = {}
    for name, values in results.items():
        summary[name] = {
            metric + "_mean": float(np.mean([value[metric] for value in values]))
            for metric in ("accuracy", "balanced_accuracy", "hallucination_f1",
                           "auroc", "auprc")
        }
        summary[name].update({
            metric + "_std": float(np.std([value[metric] for value in values]))
            for metric in ("accuracy", "balanced_accuracy", "hallucination_f1",
                           "auroc", "auprc")
        })
        summary[name]["per_seed"] = values

    report = {
        "protocol": (
            "Aiersilan stratified random 70/10/20; seeds 42,43,44; "
            "fold-local scaling/PCA; LR C=.03 class_weight=balanced"
        ),
        "n": len(y), "hallucination": int(y.sum()),
        "correct": int((1-y).sum()), "layer": LAYER,
        "feature_definitions": {
            "P": "exact-current127 P blocks; PCA [None,8,8,8,8,48]",
            "Aiersilan": "candidate-last-token layer14 hidden; PCA48",
            "P_plus_Aiersilan": "concatenated transformed P and Aiersilan",
        },
        "summary": summary, "splits": split_details,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({name: {k: v for k, v in values.items()
                             if k.endswith("_mean") or k.endswith("_std")}
                      for name, values in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
