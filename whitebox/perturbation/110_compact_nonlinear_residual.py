#!/usr/bin/env python3
"""Nonlinear residual heads restricted to the 21 interpretable perturbation scalars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from importlib import import_module


RUNS = Path(__file__).resolve().parent / "runs"


def met(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5))}


def main():
    folds = import_module("109_nonlinear_head_search")
    specs = [
        ("mlp21_relu_h4_a1", lambda s: MLPClassifier(
            hidden_layer_sizes=(4,), activation="relu", solver="lbfgs", alpha=1,
            max_iter=3000, random_state=s)),
        ("mlp21_relu_h8_a1", lambda s: MLPClassifier(
            hidden_layer_sizes=(8,), activation="relu", solver="lbfgs", alpha=1,
            max_iter=3000, random_state=s)),
        ("mlp21_tanh_h4_a1", lambda s: MLPClassifier(
            hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs", alpha=1,
            max_iter=3000, random_state=s)),
        ("rbf21_C0.1", lambda s: SVC(C=.1, probability=True, class_weight="balanced",
                                      random_state=s)),
        ("rbf21_C0.3", lambda s: SVC(C=.3, probability=True, class_weight="balanced",
                                      random_state=s)),
        ("hist21_leaf5", lambda s: HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=5, min_samples_leaf=30,
            l2_regularization=5, learning_rate=.04, random_state=s)),
        ("extra21", lambda s: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=10, max_features=.7,
            class_weight="balanced", n_jobs=-1, random_state=s)),
        ("quadratic21_C.003", lambda s: make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(),
            LogisticRegression(C=.003, max_iter=5000, class_weight="balanced",
                               solver="liblinear", random_state=s))),
        ("quadratic21_C.01", lambda s: make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(),
            LogisticRegression(C=.01, max_iter=5000, class_weight="balanced",
                               solver="liblinear", random_state=s))),
    ]
    seed = 42
    pred = {n: None for n, _ in specs}; linear = None; labels = None
    for fold, train, test, xtr, xte, y in folds.fold_features(seed):
        labels = y
        if linear is None: linear = np.zeros(len(y))
        lr = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                solver="liblinear", random_state=seed).fit(xtr, y[train])
        linear[test] = lr.predict_proba(xte)[:, 1]
        for name, make in specs:
            if pred[name] is None: pred[name] = np.zeros(len(y))
            model = make(seed).fit(xtr[:, :21], y[train])
            pred[name][test] = model.predict_proba(xte[:, :21])[:, 1]
        print(f"fold={fold}/5", flush=True)
    rows = [{"model": "lr105", **met(labels, linear)}]
    for name, values in pred.items():
        rows.append({"model": name, "nonlinear_weight": 1.0, **met(labels, values)})
        for w in (.05, .1, .15, .2, .3, .4, .5):
            rows.append({"model": f"lr105+{w:g}*{name}", "nonlinear_weight": w,
                         **met(labels, (1-w)*linear+w*values)})
    rows.sort(key=lambda x: (x["auroc"], x["auprc"]), reverse=True)
    out = {"seed": seed, "warning": "same-OOF screen", "results": rows}
    path = RUNS / "110_compact_nonlinear_residual_screen.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"out": str(path), "top": rows[:20]}, indent=2))


if __name__ == "__main__":
    main()
