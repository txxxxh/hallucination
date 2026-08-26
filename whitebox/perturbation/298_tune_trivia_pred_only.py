#!/usr/bin/env python3
"""Small, auditable hyperparameter sweep for TriviaQA pred-only features."""
from __future__ import annotations

import importlib
import itertools
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
SOURCE = RUNS / "297_multibench_pred_only" / "trivia"
OUT = RUNS / "298_trivia_pred_only_20seed.json"
SEEDS = tuple(range(42, 62))


def load():
    base = importlib.import_module("297_multibench_pred_only")
    def fixed(values, size=6):
        values = np.asarray(values, np.float32)[:size]
        return np.pad(values, (0, max(0, size - len(values))))

    rows = []
    for path in sorted(SOURCE.glob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            p = z["stage1_pred"].astype(np.float32)
            q = z["stage2_pred"].astype(np.float32)
            h = z["pred_hidden"].astype(np.float32)
            scalar = np.r_[base.ch(fixed(p)), base.ch2(fixed(q)), p[0] - q[0]].astype(np.float32)
            rows.append((int(z["correct"]), scalar, h[0], base.wd(h, p),
                         z["layer14"].astype(np.float32)))
    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 rows, found {len(rows)}")
    return np.asarray([x[0] for x in rows]), [np.stack([x[j] for x in rows]) for j in range(1, 5)]


def evaluate(y, blocks, c, pred_dim, disp_dim, layer_dim, use_scalar=True):
    dims = (None, pred_dim, disp_dim, layer_dim)
    seed_probs, per_seed = [], []
    for seed in SEEDS:
        probability = np.zeros(len(y))
        splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(blocks[0], y)
        for train, test in splits:
            left, right = [], []
            for index, (values, dim) in enumerate(zip(blocks, dims)):
                if index == 0 and not use_scalar:
                    continue
                scaler = StandardScaler().fit(values[train])
                a, b = scaler.transform(values[train]), scaler.transform(values[test])
                if dim:
                    dim = min(dim, len(train) - 1, a.shape[1])
                    pca = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
                    a, b = pca.transform(a), pca.transform(b)
                left.append(a); right.append(b)
            clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                                     solver="liblinear", random_state=seed)
            clf.fit(np.concatenate(left, 1), y[train])
            probability[test] = clf.predict_proba(np.concatenate(right, 1))[:, 1]
        seed_probs.append(probability)
        per_seed.append(float(roc_auc_score(y, probability)))
    mean_probability = np.mean(seed_probs, axis=0)
    return {
        "per_seed_auroc": per_seed,
        "mean_per_seed_auroc": float(np.mean(per_seed)),
        "ensemble_auroc": float(roc_auc_score(y, mean_probability)),
        "ensemble_auprc": float(average_precision_score(y, mean_probability)),
        "ensemble_balanced_accuracy": float(balanced_accuracy_score(y, mean_probability >= .5)),
    }


def main():
    y, blocks = load()
    # Compact grid: enough to test regularization/capacity without a large search.
    grid = itertools.product(
        (0.03,),
        ((8, 8),),
        (48,),
        (True,),
    )
    results = []
    for c, (pred_dim, disp_dim), layer_dim, use_scalar in grid:
        config = {"C": c, "pred_pca": pred_dim, "displacement_pca": disp_dim,
                  "layer14_pca": layer_dim, "use_scalar": use_scalar}
        metrics = evaluate(y, blocks, c, pred_dim, disp_dim, layer_dim, use_scalar)
        results.append({**config, **metrics})
        print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda x: (x["mean_per_seed_auroc"], x["ensemble_auroc"]), reverse=True)
    report = {
        "dataset": "TriviaQA pred-only n=1000",
        "selection_warning": "configuration selected on repeated OOF; confirm with nested OOF or held-out data",
        "ranking_key": "mean_per_seed_auroc",
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "top10": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
