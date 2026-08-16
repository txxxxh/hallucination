#!/usr/bin/env python3
"""Grouped 3x5 OOF feature ablations for current127 cached detectors."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
base = importlib.import_module("152_scientist_attention_pruned_current127")

CURVES = {
    "full47": np.arange(47),
    # Remove u/abs(baseline), which duplicates the five raw stage-1 effects in
    # each semantic-class channel and can explode when the baseline is near 0.
    "no_norm37": np.r_[0:6, 11:22, 27:47],
    # Stage-1 baselines/raw effects + all stage-2/cross-stage effects.
    "core27": np.r_[0:6, 16:22, 32:47],
    "stage1_full32": np.arange(32),
    "stage1_core12": np.r_[0:6, 16:22],
    "stage2_cross15": np.arange(32, 47),
    "stage2_12": np.arange(32, 44),
    "cross3": np.arange(44, 47),
    "none0": np.asarray([], dtype=int),
}


def load(cache: Path):
    rows = base.load(cache)
    keys = np.asarray([x[0] for x in rows])
    groups = np.asarray([x[1] for x in rows])
    labels = np.asarray([x[2] for x in rows])
    curves = np.stack([x[3] for x in rows]).astype(np.float32)
    hidden = [np.stack([x[4][j] for x in rows]).astype(np.float32) for j in range(4)]
    layer14 = np.stack([x[5] for x in rows]).astype(np.float32)
    return keys, groups, labels, curves, hidden, layer14


def configs():
    out = []

    def add(name, curve="full47", hidden=(0, 1, 2, 3), hdim=8, ldim=48):
        out.append(dict(name=name, curve=curve, hidden=list(hidden),
                        hdim=hdim, ldim=ldim))

    # Block and curve-family diagnosis at the original PCA dimensions.
    add("current127")
    add("drop_layer14", ldim=0)
    add("drop_all_hidden", hidden=(), hdim=0)
    add("hidden_only", curve="none0", ldim=0)
    add("layer14_only", curve="none0", hidden=(), hdim=0)
    add("curves_only", hidden=(), hdim=0, ldim=0)
    for j, label in enumerate(("pred_base", "pred_delta", "other_base", "other_delta")):
        add("drop_hidden_" + label, hidden=tuple(x for x in range(4) if x != j))
    for curve in CURVES:
        if curve != "full47":
            add("curve_" + curve, curve=curve)

    # Dimensionality search, centered on the two plausible curve sets.
    for curve in ("full47", "no_norm37", "core27"):
        for hdim in (0, 2, 4, 6, 8):
            for ldim in (0, 8, 16, 24, 32, 48):
                if hdim == 8 and ldim == 48:
                    continue
                add(f"grid_{curve}_h{hdim}_l{ldim}", curve=curve,
                    hidden=() if hdim == 0 else (0, 1, 2, 3),
                    hdim=hdim, ldim=ldim)
    # Preserve insertion order while removing equivalent configurations.
    seen, unique = set(), []
    for config in out:
        signature = (config["curve"], tuple(config["hidden"]),
                     config["hdim"], config["ldim"])
        if signature not in seen:
            seen.add(signature)
            unique.append(config)
    return unique


def transform_fold(train, test, curves, hidden, layer14, seed):
    transformed = {}
    scaler = StandardScaler().fit(curves[train])
    transformed["curve_train"] = scaler.transform(curves[train])
    transformed["curve_test"] = scaler.transform(curves[test])
    for j, values in enumerate(hidden):
        scaler = StandardScaler().fit(values[train])
        a, b = scaler.transform(values[train]), scaler.transform(values[test])
        pca = PCA(8, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
        transformed[f"h{j}_train"] = pca.transform(a)
        transformed[f"h{j}_test"] = pca.transform(b)
    scaler = StandardScaler().fit(layer14[train])
    a, b = scaler.transform(layer14[train]), scaler.transform(layer14[test])
    pca = PCA(48, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
    transformed["l_train"] = pca.transform(a)
    transformed["l_test"] = pca.transform(b)
    return transformed


def matrix(parts, nrows):
    return np.concatenate(parts, axis=1) if parts else np.zeros((nrows, 0))


def run(cache: Path, selected=None):
    keys, groups, labels, curves, hidden, layer14 = load(cache)
    candidates = configs()
    if selected is not None:
        wanted = set(selected)
        candidates = [x for x in candidates if x["name"] in wanted]
    probabilities = {(c["name"], seed): np.zeros(len(labels))
                     for c in candidates for seed in (42, 43, 44)}
    dimensions = {}
    for seed in (42, 43, 44):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(curves, labels, groups):
            z = transform_fold(train, test, curves, hidden, layer14, seed)
            for config in candidates:
                curve_ids = CURVES[config["curve"]]
                train_parts, test_parts = [], []
                if len(curve_ids):
                    train_parts.append(z["curve_train"][:, curve_ids])
                    test_parts.append(z["curve_test"][:, curve_ids])
                for j in config["hidden"]:
                    train_parts.append(z[f"h{j}_train"][:, :config["hdim"]])
                    test_parts.append(z[f"h{j}_test"][:, :config["hdim"]])
                if config["ldim"]:
                    train_parts.append(z["l_train"][:, :config["ldim"]])
                    test_parts.append(z["l_test"][:, :config["ldim"]])
                a, b = matrix(train_parts, len(train)), matrix(test_parts, len(test))
                dimensions[config["name"]] = a.shape[1]
                classifier = LogisticRegression(
                    C=.03, max_iter=5000, class_weight="balanced",
                    solver="liblinear", random_state=seed,
                ).fit(a, labels[train])
                probabilities[(config["name"], seed)][test] = classifier.predict_proba(b)[:, 1]

    results = []
    for config in candidates:
        per_seed = []
        for seed in (42, 43, 44):
            probability = probabilities[(config["name"], seed)]
            per_seed.append({
                "auroc": float(roc_auc_score(labels, probability)),
                "auprc": float(average_precision_score(labels, probability)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= .5)),
            })
        mean = {key: float(np.mean([x[key] for x in per_seed]))
                for key in per_seed[0]}
        std = {key: float(np.std([x[key] for x in per_seed]))
               for key in per_seed[0]}
        results.append({**config, "dimensions": dimensions[config["name"]],
                        "per_seed": per_seed, "mean": mean, "std": std})
    results.sort(key=lambda x: x["mean"]["auroc"], reverse=True)
    return {
        "cache": str(cache), "n_rows": len(labels),
        "n_groups": int(len(set(groups))), "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=RUNS / "159_scientist_classgrad_sentence_current127")
    parser.add_argument("--output", type=Path,
                        default=RUNS / "161_current127_feature_ablation.json")
    parser.add_argument("--selected", nargs="*")
    args = parser.parse_args()
    report = run(args.cache, args.selected)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "results": report["results"][:15]}, indent=2))


if __name__ == "__main__":
    main()
