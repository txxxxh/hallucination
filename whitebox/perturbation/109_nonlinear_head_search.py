#!/usr/bin/env python3
"""Grouped-OOF search for nonlinear heads on the frozen unified linear features."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
    }


def fold_features(seed: int):
    mod = importlib.import_module("101_fuse_sota_trajectory")
    keys, groups, y, margin, hidden, _, _ = mod.load_response("scientist")
    _, _, last, _ = mod.trajectory("scientist", keys)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(cv.split(margin, y, groups), 1):
        scaler = StandardScaler().fit(margin[train])
        train_parts = [scaler.transform(margin[train])]
        test_parts = [scaler.transform(margin[test])]
        for values in hidden:
            scaler = StandardScaler().fit(values[train])
            scaled = scaler.transform(values[train])
            pca = PCA(12, whiten=True, svd_solver="randomized", random_state=seed).fit(scaled)
            train_parts.append(pca.transform(scaled))
            test_parts.append(pca.transform(scaler.transform(values[test])))
        values = last[:, 3]
        scaler = StandardScaler().fit(values[train])
        scaled = scaler.transform(values[train])
        pca = PCA(48, whiten=True, svd_solver="randomized", random_state=seed).fit(scaled)
        train_parts.append(pca.transform(scaled))
        test_parts.append(pca.transform(scaler.transform(values[test])))
        yield fold, train, test, np.concatenate(train_parts, 1), np.concatenate(test_parts, 1), y


def model_specs(stage: str):
    specs = [("lr_C.03", lambda seed: LogisticRegression(
        C=0.03, max_iter=5000, class_weight="balanced", solver="liblinear",
        random_state=seed))]
    if stage == "screen":
        for activation in ("relu", "tanh"):
            for width in (8, 16, 32, 64):
                for alpha in (0.1, 1.0):
                    name = f"mlp_{activation}_h{width}_a{alpha:g}"
                    specs.append((name, lambda seed, a=activation, w=width, reg=alpha:
                                  MLPClassifier(hidden_layer_sizes=(w,), activation=a,
                                                solver="lbfgs", alpha=reg, max_iter=2000,
                                                random_state=seed)))
        for c in (0.1, 0.3, 1.0, 3.0):
            specs.append((f"rbf_C{c:g}", lambda seed, c=c:
                          SVC(C=c, kernel="rbf", gamma="scale", probability=True,
                              class_weight="balanced", random_state=seed)))
        for leaves in (5, 9, 15):
            specs.append((f"hist_leaf{leaves}", lambda seed, leaves=leaves:
                          HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=leaves,
                                                         l2_regularization=3,
                                                         learning_rate=0.05,
                                                         random_state=seed)))
        specs.append(("extra_trees", lambda seed: ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=6, max_features=0.5,
            class_weight="balanced", n_jobs=-1, random_state=seed)))
    else:
        specs += [
            ("mlp_relu_h8_a1", lambda seed: MLPClassifier(
                hidden_layer_sizes=(8,), activation="relu", solver="lbfgs", alpha=1,
                max_iter=2000, random_state=seed)),
            ("mlp_relu_h16_a1", lambda seed: MLPClassifier(
                hidden_layer_sizes=(16,), activation="relu", solver="lbfgs", alpha=1,
                max_iter=2000, random_state=seed)),
            ("mlp_tanh_h8_a1", lambda seed: MLPClassifier(
                hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=1,
                max_iter=2000, random_state=seed)),
            ("rbf_C0.3", lambda seed: SVC(C=0.3, kernel="rbf", gamma="scale",
                                           probability=True, class_weight="balanced",
                                           random_state=seed)),
            ("hist_leaf5", lambda seed: HistGradientBoostingClassifier(
                max_iter=200, max_leaf_nodes=5, l2_regularization=3,
                learning_rate=0.05, random_state=seed)),
        ]
    return specs


def evaluate(stage: str, seeds: list[int]):
    specs = model_specs(stage)
    scores = {name: [] for name, _ in specs}
    blend_scores = {name: {w: [] for w in (0.25, 0.5, 0.75)}
                    for name, _ in specs if name != "lr_C.03"}
    for seed in seeds:
        pred = {name: None for name, _ in specs}
        y_all = None
        for fold, train, test, x_train, x_test, y in fold_features(seed):
            y_all = y
            for name, make_model in specs:
                if pred[name] is None:
                    pred[name] = np.zeros(len(y), dtype=np.float64)
                model = make_model(seed)
                model.fit(x_train, y[train])
                pred[name][test] = model.predict_proba(x_test)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        assert y_all is not None
        linear = pred["lr_C.03"]
        for name in pred:
            scores[name].append(metrics(y_all, pred[name]))
            if name != "lr_C.03":
                for weight in blend_scores[name]:
                    blend = (1 - weight) * linear + weight * pred[name]
                    blend_scores[name][weight].append(metrics(y_all, blend))
    rows = []
    for name, values in scores.items():
        rows.append({"model": name, "kind": "single", **{
            f"mean_{key}": float(np.mean([x[key] for x in values]))
            for key in values[0]}, "per_seed": values})
    for name, by_weight in blend_scores.items():
        for weight, values in by_weight.items():
            rows.append({"model": f"blend_lr_{1-weight:g}_{name}_{weight:g}",
                         "kind": "blend", "nonlinear_weight": weight, **{
                f"mean_{key}": float(np.mean([x[key] for x in values]))
                for key in values[0]}, "per_seed": values})
    rows.sort(key=lambda x: (x["mean_auroc"], x["mean_auprc"]), reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen", "validate"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = args.seeds or ([42] if args.stage == "screen" else [42, 43, 44])
    output = args.out or RUNS / f"109_nonlinear_head_{args.stage}.json"
    rows = evaluate(args.stage, seeds)
    report = {"stage": args.stage, "seeds": seeds,
              "selection_warning": "screen results are same-OOF model selection",
              "results": rows}
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(output), "top": rows[:15]}, indent=2))


if __name__ == "__main__":
    main()
