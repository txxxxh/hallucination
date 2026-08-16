#!/usr/bin/env python3
"""Grouped OOF evaluation for Scientist layerwise hidden interventions."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_CACHE = RUNS / "170_scientist_hidden_intervention"
DEFAULT_OUT = RUNS / "171_scientist_hidden_intervention_report.json"
scientist = importlib.import_module("152_scientist_attention_pruned_current127")


def fixed_curve(pred, other, topk=5):
    """Compact, fixed-dimensional response geometry for one intervention layer."""
    margin = pred - other
    effect = margin[0] - margin[1:]
    order = np.argsort(-np.abs(effect))[:topk]
    chosen = effect[order]
    pp = pred[0] - pred[1:][order]
    oo = other[0] - other[1:][order]
    def pad(x):
        return np.pad(np.asarray(x, np.float32), (0, topk - len(x)))
    chosen, pp, oo = pad(chosen), pad(pp), pad(oo)
    scale = abs(float(margin[0])) + 1e-6
    stats = np.asarray([
        margin[0], np.max(effect), np.min(effect), np.mean(np.abs(effect)),
        np.std(effect), np.mean(effect > 0), np.max(np.abs(effect)),
    ], np.float32)
    return np.r_[stats, chosen, chosen / scale, pp, oo].astype(np.float32)


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))}


def oof(features, y, groups):
    per_seed = []
    for seed in (42, 43, 44):
        probability = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(features, y, groups):
            scaler = StandardScaler().fit(features[train])
            a, b = scaler.transform(features[train]), scaler.transform(features[test])
            clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                     solver="liblinear", random_state=seed).fit(a, y[train])
            probability[test] = clf.predict_proba(b)[:, 1]
        per_seed.append(metrics(y, probability))
    return {"per_seed": per_seed,
            "mean": {k: float(np.mean([x[k] for x in per_seed]))
                     for k in per_seed[0]}}


def embedding_features(keys):
    """Existing exact embedding-perturbation scalar features for the same keys."""
    out = []
    for key in keys:
        with np.load(RUNS / "120_physical_delete_rerank" / f"{key}.npz",
                     allow_pickle=True) as z:
            p, o = z["stage1_pred_scores"], z["stage1_other_scores"]
            q, r = z["stage2_pred_scores"], z["stage2_other_scores"]
        out.append(np.r_[scientist.ch(p), scientist.ch(o), scientist.ch2(q),
                         scientist.ch2(r), p[0] - q[0], o[0] - r[0],
                         (p[0] - o[0]) - (q[0] - r[0])])
    return np.asarray(out, np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require", type=int, default=0,
                        help="fail unless exactly this many caches exist")
    args = parser.parse_args()
    files = sorted(args.cache.glob("*.npz"))
    if args.require and len(files) != args.require:
        raise RuntimeError(f"expected {args.require} caches, found {len(files)}")
    if len(files) < 50:
        raise RuntimeError("need at least 50 rows for the grouped pilot")
    rows = []
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            rows.append((str(z["key"]), str(z["group"]), int(z["correct"]),
                         z["layers"].astype(int), z["pred_scores"].astype(np.float32),
                         z["other_scores"].astype(np.float32)))
    layers = rows[0][3].tolist()
    if any(x[3].tolist() != layers for x in rows):
        raise RuntimeError("cache contains inconsistent layer sets")
    keys = [x[0] for x in rows]
    groups = np.asarray([x[1] for x in rows])
    y = np.asarray([x[2] for x in rows])
    by_layer = []
    for li in range(len(layers)):
        by_layer.append(np.stack([fixed_curve(x[4][li], x[5][li]) for x in rows]))
    embed = embedding_features(keys)
    experiments = {f"hidden_layer_{layer}": x for layer, x in zip(layers, by_layer)}
    experiments["hidden_all_layers"] = np.concatenate(by_layer, 1)
    experiments["embedding_exact_scalar"] = embed
    experiments["embedding_plus_hidden_all"] = np.concatenate(
        [embed, *by_layer], 1)
    results = {name: {"dimensions": int(x.shape[1]), **oof(x, y, groups)}
               for name, x in experiments.items()}
    report = {
        "protocol": ("Scientist-known; same cached subset for every method; "
                     "candidate-QID grouped 3x5 OOF; fold-local scaling; fixed LR C=.03"),
        "n": len(rows), "groups": len(set(groups)), "correct": int(y.sum()),
        "incorrect": int(len(y) - y.sum()), "layers_zero_based": layers,
        "intervention": "same-row non-target context mean at decoder-block output",
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
