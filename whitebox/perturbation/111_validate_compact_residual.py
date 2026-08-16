#!/usr/bin/env python3
"""Repeated grouped-OOF validation of compact nonlinear residual candidates."""
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def met(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))}


def main():
    source = importlib.import_module("109_nonlinear_head_search")
    results = {name: [] for name in ("lr105", "lr95_quad5", "lr95_extra5")}
    for seed in (42, 43, 44):
        pred = {name: None for name in results}; labels = None
        for fold, train, test, xtr, xte, y in source.fold_features(seed):
            labels = y
            for name in pred:
                if pred[name] is None: pred[name] = np.zeros(len(y))
            lr = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                    solver="liblinear", random_state=seed).fit(xtr, y[train])
            linear = lr.predict_proba(xte)[:, 1]
            quad = make_pipeline(
                PolynomialFeatures(2, include_bias=False), StandardScaler(),
                LogisticRegression(C=.01, max_iter=5000, class_weight="balanced",
                                   solver="liblinear", random_state=seed))
            quad.fit(xtr[:, :21], y[train])
            extra = ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=10, max_features=.7,
                class_weight="balanced", n_jobs=-1, random_state=seed)
            extra.fit(xtr[:, :21], y[train])
            pred["lr105"][test] = linear
            pred["lr95_quad5"][test] = .95*linear + .05*quad.predict_proba(xte[:, :21])[:, 1]
            pred["lr95_extra5"][test] = .95*linear + .05*extra.predict_proba(xte[:, :21])[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        for name, values in pred.items(): results[name].append(met(labels, values))
    rows = []
    for name, values in results.items():
        row = {"model": name, "per_seed": values}
        for key in values[0]: row[f"mean_{key}"] = float(np.mean([v[key] for v in values]))
        rows.append(row)
    rows.sort(key=lambda x: x["mean_auroc"], reverse=True)
    report = {"protocol": "Scientist question-grouped 3x5-fold OOF",
              "selection_warning": "candidates selected on seed-42 screen", "results": rows}
    path = Path(__file__).resolve().parent/"runs"/"111_compact_residual_validation.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
