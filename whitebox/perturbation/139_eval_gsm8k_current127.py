#!/usr/bin/env python3
"""Evaluate the fixed current127 correctness detector on GSM8K.

Reports both in-domain stratified OOF performance and a frozen transfer model
whose preprocessing, PCA, and logistic regression are fitted only on
ScientistQA.  GSM8K labels are used only for metrics in the transfer result.
"""
from __future__ import annotations

import importlib
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
CACHE = RUNS / "137_gsm8k_train_current127"
OUT = RUNS / "139_gsm8k_current127_report.json"
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


def load_gsm8k():
    rows = []
    for fp in sorted(CACHE.glob("*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            p, o = z["stage1_pred"], z["stage1_other"]
            q, r = z["stage2_pred"], z["stage2_other"]
            scalar = np.r_[ch(p), ch(o), ch2(q), ch2(r), p[0] - q[0], o[0] - r[0],
                           (p[0] - o[0]) - (q[0] - r[0])]
            ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
            rows.append((str(z["key"].item()), int(z["correct"]), scalar,
                         (ph[0], wd(ph, z["pred_u"]), oh[0], wd(oh, z["other_u"])),
                         z["layer14"].astype(np.float32)))
    if len(rows) != 1500:
        raise RuntimeError(f"expected 1500 GSM8K rows, got {len(rows)}")
    return rows


def source_rows():
    # Reuse the exact ScientistQA feature loader used by the frozen Tennis transfer.
    return importlib.import_module("135_eval_scientist_frozen_on_tennis").source_rows()


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, p >= .5)),
            "mean_score_correct": float(p[y == 1].mean()),
            "mean_score_incorrect": float(p[y == 0].mean())}


def transform_fit(source_blocks, target_blocks, seed):
    fitted_source, fitted_target = [], []
    for source, target, dim in zip(source_blocks, target_blocks, (None, 8, 8, 8, 8, 48)):
        scaler = StandardScaler().fit(source)
        a, b = scaler.transform(source), scaler.transform(target)
        if dim is not None:
            pca = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
            a, b = pca.transform(a), pca.transform(b)
        fitted_source.append(a)
        fitted_target.append(b)
    return np.concatenate(fitted_source, 1), np.concatenate(fitted_target, 1)


def main():
    rows = load_gsm8k()
    keys = np.asarray([x[0] for x in rows])
    y = np.asarray([x[1] for x in rows])
    scalar = np.stack([x[2] for x in rows])
    hidden = [np.stack([x[3][j] for x in rows]) for j in range(4)]
    layer14 = np.stack([x[4] for x in rows])
    blocks = [scalar, *hidden, layer14]

    oof_runs = []
    oof_predictions = []
    for seed in SEEDS:
        pred = np.zeros(len(y))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(scalar, y):
            x_train, x_test = transform_fit([x[train] for x in blocks],
                                            [x[test] for x in blocks], seed)
            clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                     solver="liblinear", random_state=seed).fit(x_train, y[train])
            pred[test] = clf.predict_proba(x_test)[:, 1]
        oof_predictions.append(pred)
        oof_runs.append(metrics(y, pred))

    sy, ss, sh, sl = source_rows()
    source_blocks = [ss, *sh, sl]
    transfer_runs = []
    transfer_predictions = []
    for seed in SEEDS:
        x_source, x_gsm = transform_fit(source_blocks, blocks, seed)
        clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                 solver="liblinear", random_state=seed).fit(x_source, sy)
        pred = clf.predict_proba(x_gsm)[:, 1]
        transfer_predictions.append(pred)
        transfer_runs.append(metrics(y, pred))

    def summarize(runs, predictions):
        ensemble = np.mean(predictions, axis=0)
        return {"ensemble": metrics(y, ensemble),
                "mean_across_seeds": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
                "per_seed": [{"seed": seed, **result} for seed, result in zip(SEEDS, runs)]}, ensemble

    oof, oof_mean = summarize(oof_runs, oof_predictions)
    transfer, transfer_mean = summarize(transfer_runs, transfer_predictions)
    report = {
        "dataset": "GSM8K train deterministic two-choice sample",
        "n": int(len(y)), "correct": int(y.sum()), "incorrect": int((1-y).sum()),
        "fixed_config": "scalar47 + four candidate-hidden PCA8 + layer14 PCA48; LR C=.03",
        "knowledge_probe": "not applicable; no knowledge-probe filtering or slicing used",
        "in_domain_oof": {"protocol": "fixed configuration; 3x5 stratified OOF", **oof},
        "scientist_frozen_transfer": {
            "protocol": "all transforms and LR fit only on 1084 ScientistQA rows; GSM8K labels used only for metrics",
            **transfer},
        "per_item": [{"id": key, "correct": bool(label), "oof_score": float(a),
                      "scientist_transfer_score": float(b)}
                     for key, label, a, b in zip(keys, y, oof_mean, transfer_mean)]}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_item"}, indent=2))


if __name__ == "__main__":
    main()
