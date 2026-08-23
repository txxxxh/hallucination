#!/usr/bin/env python3
"""Honest stacked trajectory residuals for full-2894 Scientist.

The outer test fold is never used to construct its P score or the residual
training set.  P scores for the outer-train rows are themselves inner-OOF.
The representation head uses the complete eight-layer trajectory available at
answer time, but never closed-book probes or profile text as detector inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import importlib

base = importlib.import_module("272_full_scientist_standard_upr_tables")
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "274_full_scientist_trajectory_residual"
SEEDS = (42, 43, 44)
CS = (.003, .01, .03, .1, .3)
ERROR_WEIGHTS = (0., 1., 3.)
GATES = (.10, .20, .30, .50)


def metric(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    pred = p >= .5
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "tn": int(np.sum((y == 0) & ~pred)),
            "fp": int(np.sum((y == 0) & pred)),
            "fn": int(np.sum((y == 1) & ~pred)),
            "tp": int(np.sum((y == 1) & pred))}


def load_trajectory(keys):
    cache = RUNS / "141_scientist_all_trajectory_l8"
    values = {}
    for fp in cache.glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            key = str(z["key"].item())
            # Compact full-trajectory shape, update, spectral, and logit cues.
            ls = z["last_stats"].astype(np.float32)
            ms = z["mean_stats"].astype(np.float32)
            compact = np.r_[ls.ravel(), ms.ravel(),
                            np.diff(ls, axis=0).ravel(),
                            np.diff(ms, axis=0).ravel(),
                            z["logits"].astype(np.float32)]
            # Raw trajectory is represented as initial state plus seven layer
            # updates.  PCA is fitted independently for each block/fold.
            last = z["last"].astype(np.float32)
            mean = z["mean"].astype(np.float32)
            raw = [last[0], mean[0]]
            raw += list(np.diff(last, axis=0))
            raw += list(np.diff(mean, axis=0))
            values[key] = (compact, raw)
    if any(k not in values for k in keys):
        raise RuntimeError("trajectory cache is incomplete")
    compact = np.stack([values[k][0] for k in keys])
    blocks = [np.stack([values[k][1][j] for k in keys]) for j in range(16)]
    return compact, blocks


def fit_transform_trajectory(compact, blocks, train, test, seed):
    scaler = StandardScaler().fit(compact[train])
    a = [scaler.transform(compact[train])]
    b = [scaler.transform(compact[test])]
    for block in blocks:
        s = StandardScaler().fit(block[train])
        x, z = s.transform(block[train]), s.transform(block[test])
        pc = PCA(4, whiten=True, svd_solver="randomized",
                 random_state=seed).fit(x)
        a.append(pc.transform(x)); b.append(pc.transform(z))
    return np.concatenate(a, axis=1), np.concatenate(b, axis=1)


def p_predict(p_blocks, y, train, test, seed):
    a, b, _ = base.transform_blocks(
        p_blocks, train, test, [None, 8, 8, 8, 8, 48], seed)
    return base.error_probability(a, b, y, train, seed)


def inner_oof_p(p_blocks, y, groups, outer_train, seed):
    out = np.zeros(len(outer_train), dtype=np.float32)
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1000)
    local_y, local_g = y[outer_train], groups[outer_train]
    for tr0, va0 in inner.split(local_y, local_y, local_g):
        tr, va = outer_train[tr0], outer_train[va0]
        out[va0] = p_predict(p_blocks, y, tr, va, seed)
    return out


def meta_features(p, trajectory):
    lp = logit(np.clip(p, 1e-5, 1-1e-5))[:, None]
    difficulty = (1 - 2*np.abs(p-.5))[:, None]
    # The interaction lets trajectory cues matter most near the P boundary.
    return np.c_[lp, difficulty, trajectory, trajectory*difficulty]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load()
    keys = [r["key"] for r in rows]
    y = np.asarray([r["error"] for r in rows])
    groups = np.asarray([r["right_qid"] for r in rows])
    p_blocks = [np.stack([r["p_scalar"] for r in rows])]
    p_blocks += [np.stack([r["p_hidden"][j] for r in rows]) for j in range(4)]
    p_blocks += [np.stack([r["p_layer"] for r in rows])]
    compact, trajectory_blocks = load_trajectory(keys)

    names = [f"stack_C{c}_ew{w}" for c in CS for w in ERROR_WEIGHTS]
    names += [f"gate{g}_C{c}_ew{w}" for g in GATES for c in CS
              for w in ERROR_WEIGHTS]
    predictions = {name: [] for name in names}
    p_predictions = []
    fold_rows = []

    for seed in SEEDS:
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        p_all = np.zeros(len(y), dtype=np.float32)
        scores = {name: np.zeros(len(y), dtype=np.float32) for name in names}
        for fold, (train, test) in enumerate(outer.split(y, y, groups), 1):
            p_train = inner_oof_p(p_blocks, y, groups, train, seed*10+fold)
            p_test = p_predict(p_blocks, y, train, test, seed*10+fold)
            p_all[test] = p_test
            t_train, t_test = fit_transform_trajectory(
                compact, trajectory_blocks, train, test, seed*10+fold)
            x_train = meta_features(p_train, t_train)
            x_test = meta_features(p_test, t_test)
            p_wrong = ((p_train >= .5) != y[train]).astype(np.float32)
            for c in CS:
                for ew in ERROR_WEIGHTS:
                    model = LogisticRegression(
                        C=c, max_iter=5000, class_weight="balanced",
                        solver="liblinear", random_state=seed)
                    model.fit(x_train, y[train],
                              sample_weight=1 + ew*p_wrong)
                    q = model.predict_proba(x_test)[:, 1]
                    scores[f"stack_C{c}_ew{ew}"][test] = q
                    for gate in GATES:
                        hard = np.abs(p_test-.5) <= gate
                        mixed = p_test.copy(); mixed[hard] = q[hard]
                        scores[f"gate{gate}_C{c}_ew{ew}"][test] = mixed
            fold_rows.append({"seed": seed, "fold": fold,
                              "n_train": int(len(train)),
                              "n_test": int(len(test)),
                              "inner_p_accuracy": float(accuracy_score(
                                  y[train], p_train >= .5))})
            print(f"seed={seed} fold={fold}/5", flush=True)
        p_predictions.append(p_all)
        for name in names:
            predictions[name].append(scores[name])

    p_mean = np.mean(p_predictions, axis=0)
    results = []
    for name, value in predictions.items():
        mean = np.mean(value, axis=0)
        result = {"name": name, **metric(y, mean),
                  "per_seed": [metric(y, p) for p in value]}
        results.append(result)
    results.sort(key=lambda x: (x["auroc"], -x["log_loss"]), reverse=True)
    by_accuracy = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    report = {
        "protocol": ("full 2894; right-person grouped 3x5 outer OOF; outer-train "
                     "P is 4-fold inner OOF; 8-layer last/mean initial+update "
                     "PCA4 plus trajectory statistics; no probe features"),
        "selection_warning": ("candidate ranking uses repeated OOF; freeze the "
                              "winner before a final held-out confirmation"),
        "n": len(y), "errors": int(y.sum()), "groups": len(set(groups)),
        "p_only": metric(y, p_mean), "top_by_auroc": results[:15],
        "top_by_accuracy": by_accuracy[:15], "folds": fold_rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    best_names = list(dict.fromkeys(
        [x["name"] for x in results[:3] + by_accuracy[:3]]))
    with (OUT / "predictions.jsonl").open("w") as handle:
        for i, key in enumerate(keys):
            row = {"key": key, "error": int(y[i]),
                   "p_error_probability": float(p_mean[i])}
            for name in best_names:
                row[name] = float(np.mean(predictions[name], axis=0)[i])
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"p_only": report["p_only"],
                      "top_by_auroc": results[:5],
                      "top_by_accuracy": by_accuracy[:5]}, indent=2))


if __name__ == "__main__":
    main()
