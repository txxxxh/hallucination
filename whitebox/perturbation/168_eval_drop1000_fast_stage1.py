#!/usr/bin/env python3
"""Grouped OOF evaluation of fixed Stage-1 detectors on DROP-1000."""
from __future__ import annotations

import importlib
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


features = importlib.import_module("165_eval_trivia1000_fast_stage1")
RUNS = features.RUNS


def metrics(labels, probability):
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= .5)),
    }


def load(cache):
    rows = []
    for path in sorted(cache.glob("*.npz")):
        with np.load(path) as data:
            pred = data["stage1_pred"].astype(np.float32)
            other = data["stage1_other"].astype(np.float32)
            pred_hidden = data["pred_hidden"].astype(np.float32)
            other_hidden = data["other_hidden"].astype(np.float32)
            curves = np.r_[features.curve(pred), features.curve(other)]
            hidden = [
                pred_hidden[0], features.weighted_hidden(pred_hidden, pred[0] - pred[1:]),
                other_hidden[0], features.weighted_hidden(other_hidden, other[0] - other[1:]),
            ]
            rows.append({
                "key": str(data["key"]), "group": str(data["group"]),
                "label": int(data["correct"]), "curves": curves, "hidden": hidden,
                "layer14": data["layer14"].astype(np.float32),
                "length": int(data["generation_words"]),
                "unperturbed": np.asarray([pred[0], other[0], pred[0] - other[0]]),
                "candidates": int(data["stage1_candidates"]),
                "full": int(data["stage1_full"]),
            })
    if len(rows) != 1000:
        raise RuntimeError(f"{cache}: expected 1000 rows, got {len(rows)}")
    return rows


def evaluate(name, cache):
    rows = load(cache)
    labels = np.asarray([x["label"] for x in rows])
    groups = np.asarray([x["group"] for x in rows])
    curves = np.stack([x["curves"] for x in rows])
    hidden = [np.stack([x["hidden"][j] for x in rows]) for j in range(4)]
    layer14 = np.stack([x["layer14"] for x in rows])
    length = np.asarray([x["length"] for x in rows], dtype=float)[:, None]
    unperturbed = np.stack([x["unperturbed"] for x in rows])
    per_seed, controls = [], {"length_only": [], "unperturbed_only": []}
    for seed in (42, 43, 44):
        probability = np.zeros(len(labels))
        control_probability = {key: np.zeros(len(labels)) for key in controls}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(curves, labels, groups):
            train_parts, test_parts = [], []
            for values, dimensions in [(curves, None), *[(x, 8) for x in hidden],
                                       (layer14, 44)]:
                scaler = StandardScaler().fit(values[train])
                a, b = scaler.transform(values[train]), scaler.transform(values[test])
                if dimensions is not None:
                    pca = PCA(dimensions, whiten=True, svd_solver="randomized",
                              random_state=seed).fit(a)
                    a, b = pca.transform(a), pca.transform(b)
                train_parts.append(a); test_parts.append(b)
            classifier = LogisticRegression(C=.03, max_iter=5000,
                class_weight="balanced", solver="liblinear", random_state=seed)
            classifier.fit(np.concatenate(train_parts, 1), labels[train])
            probability[test] = classifier.predict_proba(np.concatenate(test_parts, 1))[:, 1]
            for key, values in (("length_only", length),
                                ("unperturbed_only", unperturbed)):
                scaler = StandardScaler().fit(values[train])
                a, b = scaler.transform(values[train]), scaler.transform(values[test])
                clf = LogisticRegression(C=.03, max_iter=5000,
                    class_weight="balanced", solver="liblinear", random_state=seed)
                clf.fit(a, labels[train])
                control_probability[key][test] = clf.predict_proba(b)[:, 1]
        per_seed.append(metrics(labels, probability))
        for key in controls:
            controls[key].append(metrics(labels, control_probability[key]))
    queries = np.asarray([[x["candidates"], x["full"]] for x in rows])
    return {
        "method": name, "n": len(rows), "groups": len(set(groups)),
        "dimensions": 108, "per_seed": per_seed,
        "mean": {key: float(np.mean([x[key] for x in per_seed])) for key in per_seed[0]},
        "controls": {name: {key: float(np.mean([x[key] for x in values]))
                             for key in values[0]} for name, values in controls.items()},
        "queries": {"candidate_mean": float(queries[:, 0].mean()),
                    "full_mean": float(queries[:, 1].mean()),
                    "reduction": float(1 - queries[:, 0].sum() / queries[:, 1].sum())},
    }


def main():
    methods = {
        "exact": RUNS / "167_drop1000_exact",
        "attention_maxhead": RUNS / "167_drop1000_attention_maxhead",
        "gradient_sentence": RUNS / "167_drop1000_gradient_sentence",
    }
    report = {
        "protocol": "DROP balanced 1000, 416 passage groups; fixed stage1-only 108d; grouped 3x5 OOF",
        "results": [evaluate(name, cache) for name, cache in methods.items()],
    }
    output = RUNS / "168_drop1000_fast_stage1_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
