#!/usr/bin/env python3
"""Evaluate cross-layer perturbation deltas on all 2,894 Scientist rows."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

base = importlib.import_module("272_full_scientist_standard_upr_tables")
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "276_full_scientist_perturbation_trajectory"
SEEDS = (42, 43, 44)


def metrics(y, p):
    p = np.clip(p, 1e-7, 1-1e-7); pred = p >= .5
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def weighted_delta(h, u):
    delta = h[1:].astype(np.float32) - h[0].astype(np.float32)
    return (delta * u[:, None, None]).sum(0) / (np.abs(u).sum()+1e-9)


def cosine(a, b):
    return np.sum(a*b, axis=1) / (
        np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)+1e-9)


def load_delta(keys):
    old = RUNS / "118_dual_candidate_multilayer_top5"
    new = RUNS / "275_full_scientist_perturbation_trajectory"
    rows = {}
    for key in keys:
        fp = new/f"{key}.npz" if (new/f"{key}.npz").exists() else old/f"{key}.npz"
        if not fp.exists():
            raise RuntimeError(f"missing trajectory {key}")
        with np.load(fp, allow_pickle=True) as z:
            ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
            pd = weighted_delta(ph, z["pred_u"].astype(np.float32))
            od = weighted_delta(oh, z["other_u"].astype(np.float32))
            # Per-layer endpoint response and its candidate contrast.
            blocks = [*pd, *od, *(pd-od)]
            # Cross-layer evolution of each response is the requested signal.
            blocks += [*np.diff(pd, axis=0), *np.diff(od, axis=0),
                       *np.diff(pd-od, axis=0)]
            geometry = np.r_[np.linalg.norm(pd, axis=1),
                             np.linalg.norm(od, axis=1), cosine(pd, od),
                             np.linalg.norm(np.diff(pd, axis=0), axis=1),
                             np.linalg.norm(np.diff(od, axis=0), axis=1)]
            rows[key] = (blocks, geometry.astype(np.float32))
    nblocks = len(rows[keys[0]][0])
    return ([np.stack([rows[k][0][j] for k in keys]) for j in range(nblocks)],
            np.stack([rows[k][1] for k in keys]))


def p_predict(blocks, y, train, test, seed):
    a, b, _ = base.transform_blocks(
        blocks, train, test, [None, 8, 8, 8, 8, 48], seed)
    return base.error_probability(a, b, y, train, seed)


def inner_p(blocks, y, groups, outer_train, seed):
    out = np.zeros(len(outer_train))
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=seed+1000)
    for tr0, va0 in cv.split(y[outer_train], y[outer_train], groups[outer_train]):
        tr, va = outer_train[tr0], outer_train[va0]
        out[va0] = p_predict(blocks, y, tr, va, seed)
    return out


def delta_transform(blocks, geometry, train, test, seed, dim):
    a, b = [], []
    s = StandardScaler().fit(geometry[train])
    a.append(s.transform(geometry[train])); b.append(s.transform(geometry[test]))
    for x in blocks:
        s = StandardScaler().fit(x[train]); q = s.transform(x[train])
        z = s.transform(x[test])
        pc = PCA(dim, whiten=True, svd_solver="randomized",
                 random_state=seed).fit(q)
        a.append(pc.transform(q)); b.append(pc.transform(z))
    return np.concatenate(a, 1), np.concatenate(b, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load(); keys = [r["key"] for r in rows]
    y = np.asarray([r["error"] for r in rows])
    groups = np.asarray([r["right_qid"] for r in rows])
    p_blocks = [np.stack([r["p_scalar"] for r in rows])]
    p_blocks += [np.stack([r["p_hidden"][j] for r in rows]) for j in range(4)]
    p_blocks += [np.stack([r["p_layer"] for r in rows])]
    delta_blocks, geometry = load_delta(keys)
    configs = [(d, c) for d in (2, 4) for c in (.003, .01, .03, .1)]
    variants = ["delta", "p_plus_delta"]
    scores = {(v, d, c): [] for v in variants for d, c in configs}
    p_scores = []
    saved = {}
    for seed in SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        pred = {k: np.zeros(len(y)) for k in scores}; pp = np.zeros(len(y))
        for fold, (train, test) in enumerate(cv.split(y, y, groups), 1):
            ptrain = inner_p(p_blocks, y, groups, train, seed*10+fold)
            ptest = p_predict(p_blocks, y, train, test, seed*10+fold); pp[test] = ptest
            for dim in (2, 4):
                a, b = delta_transform(delta_blocks, geometry, train, test,
                                       seed*10+fold, dim)
                for c in (.003, .01, .03, .1):
                    for name, xtr, xte in (
                            ("delta", a, b),
                            ("p_plus_delta", np.c_[ptrain, a], np.c_[ptest, b])):
                        model = LogisticRegression(
                            C=c, max_iter=5000, class_weight="balanced",
                            solver="liblinear", random_state=seed).fit(xtr, y[train])
                        pred[(name, dim, c)][test] = model.predict_proba(xte)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        p_scores.append(pp)
        for key, value in pred.items(): scores[key].append(value)
    results = []
    for (name, dim, c), values in scores.items():
        mean = np.mean(values, axis=0)
        results.append({"variant": name, "pca_per_block": dim, "C": c,
                        **metrics(y, mean),
                        "per_seed": [metrics(y, x) for x in values]})
        saved[(name, dim, c)] = mean
    results.sort(key=lambda x: x["auroc"], reverse=True)
    pmean = np.mean(p_scores, axis=0)
    report = {"protocol": ("full-2894 right-person grouped 3x5 OOF; P outer-train "
                           "scores are inner-OOF; six layers; perturbation-minus-original "
                           "and cross-layer delta; no probe"),
              "n": len(y), "p_only": metrics(y, pmean), "results": results,
              "selection_warning": "freeze selected configuration before confirmation"}
    (OUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    best = results[0]; bk = (best["variant"], best["pca_per_block"], best["C"])
    with (OUT/"predictions.jsonl").open("w") as f:
        for i, key in enumerate(keys):
            f.write(json.dumps({"key": key, "error": int(y[i]),
                                "p": float(pmean[i]),
                                "best_delta": float(saved[bk][i])})+"\n")
    print(json.dumps({"p_only": report["p_only"], "top": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
