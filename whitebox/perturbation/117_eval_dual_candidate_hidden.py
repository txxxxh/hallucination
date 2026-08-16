#!/usr/bin/env python3
"""Grouped OOF evaluation of compact, truly candidate-specific hidden states."""
from __future__ import annotations

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


def candidate_channel(score, u):
    scale = abs(float(score)) + 1e-6
    return np.r_[score, u, u/scale, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def weighted_delta(hidden, u):
    delta = hidden[1:].astype(np.float32) - hidden[0].astype(np.float32)
    return (delta*u[:, None]).sum(0)/(np.abs(u).sum() + 1e-9)


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))}


def main():
    mod = importlib.import_module("101_fuse_sota_trajectory")
    keys, groups, y, margin, hidden, _, _ = mod.load_response("scientist")
    _, _, last, _ = mod.trajectory("scientist", keys)
    margin = np.c_[margin[:, :5], margin[:, 10:]]
    sep_rows, dual_rows = {}, {}
    for fp in (RUNS/"112_separate_candidate_top5").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            key = str(z["key"].item())
            sep_rows[key] = np.r_[candidate_channel(z["pred_scores"][0], z["pred_u"]),
                                  candidate_channel(z["other_scores"][0], z["other_u"])]
    for fp in (RUNS/"116_dual_candidate_hidden_top5").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            key = str(z["key"].item())
            ph = z["pred_hidden"].astype(np.float32); oh = z["other_hidden"].astype(np.float32)
            dual_rows[key] = (ph[0], weighted_delta(ph, z["pred_u"].astype(np.float32)),
                              oh[0], weighted_delta(oh, z["other_u"].astype(np.float32)))
    missing = [k for k in keys if k not in sep_rows or k not in dual_rows]
    if missing: raise RuntimeError(f"missing {len(missing)} rows")
    separate = np.stack([sep_rows[k] for k in keys]).astype(np.float32)
    dual = [[np.stack([dual_rows[k][j] for k in keys]) for j in range(4)]][0]
    variants = ["margin_old_hidden", "separate_old_hidden"]
    for d in (4, 6, 8):
        variants += [f"separate_dual_hidden_pca{d}", f"separate_dual_hidden_pca{d}_plus_trajectory"]
    scores = {v: [] for v in variants}
    for seed in (42, 43, 44):
        prediction = {v: np.zeros(len(y)) for v in variants}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(margin, y, groups), 1):
            def scale(x):
                s = StandardScaler().fit(x[train]); return s.transform(x[train]), s.transform(x[test])
            mt, mv = scale(margin); st, sv = scale(separate)
            old_tr, old_te = [], []
            for x in hidden:
                s = StandardScaler().fit(x[train]); q = s.transform(x[train])
                pc = PCA(12, whiten=True, svd_solver="randomized", random_state=seed).fit(q)
                old_tr.append(pc.transform(q)); old_te.append(pc.transform(s.transform(x[test])))
            x = last[:, 3]; s = StandardScaler().fit(x[train]); q = s.transform(x[train])
            pc = PCA(48, whiten=True, svd_solver="randomized", random_state=seed).fit(q)
            trajectory_tr, trajectory_te = pc.transform(q), pc.transform(s.transform(x[test]))
            old_tr = np.concatenate([*old_tr, trajectory_tr], 1)
            old_te = np.concatenate([*old_te, trajectory_te], 1)
            sets = {"margin_old_hidden": (np.c_[mt, old_tr], np.c_[mv, old_te]),
                    "separate_old_hidden": (np.c_[st, old_tr], np.c_[sv, old_te])}
            dual_pcs = []
            for x in dual:
                s = StandardScaler().fit(x[train]); q = s.transform(x[train])
                pc = PCA(8, whiten=True, svd_solver="randomized", random_state=seed).fit(q)
                dual_pcs.append((pc.transform(q), pc.transform(s.transform(x[test]))))
            for d in (4, 6, 8):
                dt = np.concatenate([x[0][:, :d] for x in dual_pcs], 1)
                dv = np.concatenate([x[1][:, :d] for x in dual_pcs], 1)
                sets[f"separate_dual_hidden_pca{d}"] = (np.c_[st, dt], np.c_[sv, dv])
                sets[f"separate_dual_hidden_pca{d}_plus_trajectory"] = (
                    np.c_[st, dt, trajectory_tr], np.c_[sv, dv, trajectory_te])
            for name, (xtr, xte) in sets.items():
                clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                         solver="liblinear", random_state=seed).fit(xtr, y[train])
                prediction[name][test] = clf.predict_proba(xte)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        for name in variants: scores[name].append(metrics(y, prediction[name]))
    result = []
    for name, vals in scores.items():
        result.append({"variant": name, **{f"mean_{k}": float(np.mean([v[k] for v in vals]))
                                             for k in vals[0]}, "per_seed": vals})
    result.sort(key=lambda x: x["mean_auroc"], reverse=True)
    report = {"protocol": "Scientist question-grouped 3x5-fold OOF",
              "dimensions": {"separate": int(separate.shape[1]),
                             "dual_hidden_pca4": 16, "dual_hidden_pca6": 24,
                             "dual_hidden_pca8": 32,
                             "trajectory": 48}, "results": result}
    path = RUNS/"117_dual_candidate_hidden.json"
    path.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
