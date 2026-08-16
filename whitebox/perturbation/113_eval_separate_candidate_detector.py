#!/usr/bin/env python3
"""Grouped-OOF comparison of margin and separate-candidate response features."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent; RUNS = HERE/"runs"


def channel(score, u):
    scale = abs(float(score)) + 1e-6
    return np.r_[score, u, np.abs(u), u/scale, u.max(initial=0),
                 u.min(initial=0), np.abs(u).mean(), u.std(), np.mean(u > 0)]


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=RUNS/"112_separate_candidate_top5")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out", type=Path, default=RUNS/"113_separate_candidate_detector.json")
    a = p.parse_args()
    mod = importlib.import_module("101_fuse_sota_trajectory")
    keys, groups, y, margin, hidden, _, _ = mod.load_response("scientist")
    _, _, last, _ = mod.trajectory("scientist", keys)
    rows = {}
    for fp in a.cache.glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            pred = channel(z["pred_scores"][0], z["pred_u"])
            other = channel(z["other_scores"][0], z["other_u"])
            rows[str(z["key"].item())] = np.r_[pred, other]
    missing = [k for k in keys if k not in rows]
    if missing: raise RuntimeError(f"missing {len(missing)} candidate caches")
    separate = np.stack([rows[k] for k in keys]).astype(np.float32)
    variants = ("margin_scalar", "separate_scalar", "margin_full",
                "separate_full", "margin_plus_separate_full")
    scores = {name: [] for name in variants}
    for seed in a.seeds:
        pred = {name: np.zeros(len(y)) for name in variants}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(margin, y, groups), 1):
            def scaled(values):
                s = StandardScaler().fit(values[train])
                return s.transform(values[train]), s.transform(values[test])
            mt, mv = scaled(margin); st, sv = scaled(separate)
            ht, hv = [], []
            for values in hidden:
                s = StandardScaler().fit(values[train]); q = s.transform(values[train])
                pc = PCA(12, whiten=True, svd_solver="randomized", random_state=seed).fit(q)
                ht.append(pc.transform(q)); hv.append(pc.transform(s.transform(values[test])))
            values = last[:, 3]; s = StandardScaler().fit(values[train]); q = s.transform(values[train])
            pc = PCA(48, whiten=True, svd_solver="randomized", random_state=seed).fit(q)
            ltr, lte = pc.transform(q), pc.transform(s.transform(values[test]))
            common_train = np.concatenate([*ht, ltr], 1)
            common_test = np.concatenate([*hv, lte], 1)
            sets = {
                "margin_scalar": (mt, mv), "separate_scalar": (st, sv),
                "margin_full": (np.c_[mt, common_train], np.c_[mv, common_test]),
                "separate_full": (np.c_[st, common_train], np.c_[sv, common_test]),
                "margin_plus_separate_full":
                    (np.c_[mt, st, common_train], np.c_[mv, sv, common_test])}
            for name, (xtr, xte) in sets.items():
                clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                         solver="liblinear", random_state=seed).fit(xtr, y[train])
                pred[name][test] = clf.predict_proba(xte)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        for name in variants: scores[name].append(metrics(y, pred[name]))
    results = []
    for name, values in scores.items():
        results.append({"variant": name, **{f"mean_{key}": float(np.mean([v[key] for v in values]))
                                             for key in values[0]}, "per_seed": values})
    results.sort(key=lambda x: x["mean_auroc"], reverse=True)
    report = {"protocol": "Scientist question-grouped 3x5-fold OOF",
              "separate_dim": int(separate.shape[1]), "results": results}
    a.out.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
