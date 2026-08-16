#!/usr/bin/env python3
"""Leakage-safe OOF evaluation on natural GSM8K correctness labels."""
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
CACHE = RUNS / "141_gsm8k_natural_current127"
OUT = RUNS / "142_gsm8k_natural_current127_report.json"
SEEDS = (42, 43, 44)


def ch(scores):
    u = scores[0] - scores[1:]
    scale = abs(float(scores[0])) + 1e-6
    return np.r_[scores[0], u, u / scale, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def ch2(scores):
    return np.r_[scores[0], scores[0] - scores[1:]]


def wd(hidden, u):
    delta = hidden[1:].astype(np.float32) - hidden[0].astype(np.float32)
    return (delta * u[:, None]).sum(0) / (np.abs(u).sum() + 1e-9)


def load_rows():
    rows = []
    for fp in sorted(CACHE.glob("*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            p, o = z["stage1_pred"], z["stage1_other"]
            q, r = z["stage2_pred"], z["stage2_other"]
            scalar = np.r_[ch(p), ch(o), ch2(q), ch2(r), p[0] - q[0],
                           o[0] - r[0], (p[0] - o[0]) - (q[0] - r[0])]
            ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
            hidden = (ph[0], wd(ph, z["stage1_pred"][0] - z["stage1_pred"][1:]),
                      oh[0], wd(oh, z["stage1_other"][0] - z["stage1_other"][1:]))
            rows.append((str(z["key"].item()), int(z["correct"]), scalar, hidden,
                         z["layer14"].astype(np.float32), int(z["generation_tokens"])))
    if len(rows) != 942:
        raise RuntimeError(f"expected 942 rows, got {len(rows)}")
    return rows


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, p >= .5))}


def fit_blocks(train_blocks, test_blocks, seed):
    left, right = [], []
    for train, test, dim in zip(train_blocks, test_blocks, (None, 8, 8, 8, 8, 48)):
        scaler = StandardScaler().fit(train)
        a, b = scaler.transform(train), scaler.transform(test)
        if dim is not None:
            pca = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
            a, b = pca.transform(a), pca.transform(b)
        left.append(a); right.append(b)
    return np.concatenate(left, 1), np.concatenate(right, 1)


def oof(blocks, y, kind):
    runs, preds = [], []
    for seed in SEEDS:
        pred = np.zeros(len(y))
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(blocks[0], y):
            if kind == "full":
                a, b = fit_blocks([x[train] for x in blocks], [x[test] for x in blocks], seed)
            else:
                scaler = StandardScaler().fit(blocks[0][train])
                a, b = scaler.transform(blocks[0][train]), scaler.transform(blocks[0][test])
            clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                     solver="liblinear", random_state=seed).fit(a, y[train])
            pred[test] = clf.predict_proba(b)[:, 1]
        runs.append(metrics(y, pred)); preds.append(pred)
    ensemble = np.mean(preds, axis=0)
    return {"ensemble": metrics(y, ensemble),
            "mean_across_seeds": {k: float(np.mean([x[k] for x in runs])) for k in runs[0]},
            "per_seed": [{"seed": s, **x} for s, x in zip(SEEDS, runs)]}, ensemble


def main():
    rows = load_rows(); keys = np.asarray([x[0] for x in rows]); y = np.asarray([x[1] for x in rows])
    scalar = np.stack([x[2] for x in rows]); hidden = [np.stack([x[3][j] for x in rows]) for j in range(4)]
    layer14 = np.stack([x[4] for x in rows]); length = np.asarray([x[5] for x in rows])[:, None]
    full, full_pred = oof([scalar, *hidden, layer14], y, "full")
    scalar_only, _ = oof([scalar], y, "simple")
    length_only, _ = oof([length], y, "simple")
    report = {"dataset": "GSM8K train natural free-form greedy generations",
              "n": len(rows), "correct": int(y.sum()), "incorrect": int((1-y).sum()),
              "protocol": "fixed current127 config; 3x5 stratified OOF; all scaling/PCA inside folds",
              "config": "scalar47 + four candidate-hidden PCA8 + layer14 PCA48; LR C=.03",
              "full_detector": full, "scalar_only": scalar_only, "generation_length_only": length_only,
              "per_item": [{"id": k, "correct": bool(v), "oof_score": float(p)}
                           for k, v, p in zip(keys, y, full_pred)]}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "per_item"}, indent=2))


if __name__ == "__main__": main()
