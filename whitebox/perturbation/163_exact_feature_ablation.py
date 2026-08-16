#!/usr/bin/env python3
"""Apply the targeted feature ablations to the exact-enumeration cache."""
from __future__ import annotations

import importlib
import json

import numpy as np


ablation = importlib.import_module("161_current127_feature_ablation")
targeted = importlib.import_module("162_current127_targeted_ablation")
base = importlib.import_module("152_scientist_attention_pruned_current127")
RUNS = ablation.RUNS


def load_exact(_cache):
    meta = {key: (group, label) for key, group, label, *_ in base.jobs()}
    rows = []
    for key, (group, label) in meta.items():
        with np.load(RUNS / "120_physical_delete_rerank" / f"{key}.npz") as data:
            pred1 = data["stage1_pred_scores"]
            other1 = data["stage1_other_scores"]
            pred2 = data["stage2_pred_scores"]
            other2 = data["stage2_other_scores"]
        with np.load(RUNS / "116_dual_candidate_hidden_top5" / f"{key}.npz") as data:
            pred_hidden = data["pred_hidden"]
            other_hidden = data["other_hidden"]
        with np.load(RUNS / "100_scientist_trajectory_l8" / f"{key}.npz") as data:
            layer14 = data["mean"].astype(np.float32)[3]
        curves = np.r_[
            base.ch(pred1), base.ch(other1), base.ch2(pred2), base.ch2(other2),
            pred1[0] - pred2[0], other1[0] - other2[0],
            (pred1[0] - other1[0]) - (pred2[0] - other2[0]),
        ]
        hidden = [
            pred_hidden[0], base.wd(pred_hidden, pred1[0] - pred1[1:]),
            other_hidden[0], base.wd(other_hidden, other1[0] - other1[1:]),
        ]
        rows.append((key, group, label, curves, hidden, layer14))
    keys = np.asarray([x[0] for x in rows])
    groups = np.asarray([x[1] for x in rows])
    labels = np.asarray([x[2] for x in rows])
    curves = np.stack([x[3] for x in rows]).astype(np.float32)
    hidden = [np.stack([x[4][j] for x in rows]).astype(np.float32) for j in range(4)]
    layer14 = np.stack([x[5] for x in rows]).astype(np.float32)
    return keys, groups, labels, curves, hidden, layer14


def main():
    ablation.load = load_exact
    ablation.configs = targeted.targeted_configs
    report = ablation.run(RUNS / "exact_enumeration_virtual_cache")
    output = RUNS / "163_exact_feature_ablation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "results": report["results"][:15]}, indent=2))


if __name__ == "__main__":
    main()
