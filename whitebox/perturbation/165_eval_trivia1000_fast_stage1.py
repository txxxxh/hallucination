#!/usr/bin/env python3
"""Evaluate the fixed 108d stage-1 detector on TriviaQA candidate caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def fixed(values, length=6):
    values = np.asarray(values, dtype=np.float32)
    return np.pad(values[:length], (0, max(0, length - len(values))))


def curve(values):
    values = fixed(values)
    effect = values[0] - values[1:]
    scale = abs(float(values[0])) + 1e-6
    return np.r_[values[0], effect, effect / scale, effect.max(initial=0),
                 effect.min(initial=0), np.abs(effect).mean(), effect.std(),
                 np.mean(effect > 0)]


def weighted_hidden(hidden, effect):
    delta = hidden[1:].astype(np.float32) - hidden[0].astype(np.float32)
    return (delta * effect[:, None]).sum(0) / (np.abs(effect).sum() + 1e-9)


def load(cache):
    rows = []
    for path in sorted(cache.glob("*.npz")):
        with np.load(path) as data:
            pred = data["stage1_pred"].astype(np.float32)
            other = data["stage1_other"].astype(np.float32)
            pred_hidden = data["pred_hidden"].astype(np.float32)
            other_hidden = data["other_hidden"].astype(np.float32)
            curves = np.r_[curve(pred), curve(other)]
            hidden = [
                pred_hidden[0], weighted_hidden(pred_hidden, pred[0] - pred[1:]),
                other_hidden[0], weighted_hidden(other_hidden, other[0] - other[1:]),
            ]
            rows.append((str(data["key"]), int(data["correct"]), curves,
                         hidden, data["layer14"].astype(np.float32),
                         int(data["stage1_candidates"]) if "stage1_candidates" in data else -1,
                         int(data["stage1_full"]) if "stage1_full" in data else -1))
    if len(rows) != 1000:
        raise RuntimeError(f"{cache}: expected 1000 rows, got {len(rows)}")
    labels = np.asarray([x[1] for x in rows])
    curves = np.stack([x[2] for x in rows])
    hidden = [np.stack([x[3][j] for x in rows]) for j in range(4)]
    layer14 = np.stack([x[4] for x in rows])
    return labels, curves, hidden, layer14, rows


def evaluate(name, cache):
    labels, curves, hidden, layer14, rows = load(cache)
    per_seed = []
    for seed in (42, 43, 44):
        probability = np.zeros(len(labels))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(curves, labels):
            parts_train, parts_test = [], []
            for values, dimensions in [(curves, None),
                                       *[(x, 8) for x in hidden],
                                       (layer14, 44)]:
                scaler = StandardScaler().fit(values[train])
                a, b = scaler.transform(values[train]), scaler.transform(values[test])
                if dimensions is not None:
                    pca = PCA(dimensions, whiten=True, svd_solver="randomized",
                              random_state=seed).fit(a)
                    a, b = pca.transform(a), pca.transform(b)
                parts_train.append(a)
                parts_test.append(b)
            classifier = LogisticRegression(
                C=.03, max_iter=5000, class_weight="balanced",
                solver="liblinear", random_state=seed,
            ).fit(np.concatenate(parts_train, 1), labels[train])
            probability[test] = classifier.predict_proba(
                np.concatenate(parts_test, 1))[:, 1]
        per_seed.append({
            "auroc": float(roc_auc_score(labels, probability)),
            "auprc": float(average_precision_score(labels, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= .5)),
        })
    valid = np.asarray([[x[5], x[6]] for x in rows if x[5] >= 0])
    return {
        "method": name, "cache": str(cache), "n": len(labels),
        "dimensions": 108, "per_seed": per_seed,
        "mean": {key: float(np.mean([x[key] for x in per_seed]))
                 for key in per_seed[0]},
        "queries": None if not len(valid) else {
            "candidate_mean": float(valid[:, 0].mean()),
            "full_mean": float(valid[:, 1].mean()),
            "reduction": float(1 - valid[:, 0].sum() / valid[:, 1].sum()),
        },
    }


def main():
    methods = {
        "exact_existing": RUNS / "127_trivia1000_current127",
        "attention_maxhead": RUNS / "164_trivia1000_attention_maxhead",
        "gradient_sentence": RUNS / "164_trivia1000_gradient_sentence",
    }
    report = {
        "protocol": "TriviaQA balanced 1000; fixed stage1-only 108d; stratified 3x5 OOF",
        "results": [evaluate(name, path) for name, path in methods.items()],
    }
    output = RUNS / "165_trivia1000_fast_stage1_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
