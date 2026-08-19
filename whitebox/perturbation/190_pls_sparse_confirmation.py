#!/usr/bin/env python3
"""Fold-local screened PLS representation plus <=4 perturbation scalars."""
from __future__ import annotations
import importlib, json
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

T = importlib.import_module("189_supervised_topk_sparse_confirmation")
M, S, B = T.M, T.S, T.B
OUT = M.RUNS / "190_pls_sparse_confirmation.json"
COMPONENTS = (8, 16, 24, 32)
SCREEN = 1024
SEED = B.SEED

def fold_repr(tr, te, q, y):
    order = T.rank_coordinates(q[tr], y[tr])[:SCREEN]
    sc = StandardScaler().fit(q[tr][:, order])
    a, b = sc.transform(q[tr][:, order]), sc.transform(q[te][:, order])
    pls = PLSRegression(n_components=max(COMPONENTS), scale=False, max_iter=500, tol=1e-6)
    za = pls.fit_transform(a, y[tr])[0]
    zb = pls.transform(b)
    return za.astype(np.float32), zb.astype(np.float32)

def one_fold(tr, te, q, x, y, scalar_ix=()):
    za, zb = fold_repr(tr, te, q, y)
    out = {}
    for k in COMPONENTS:
        aa, bb = za[:, :k], zb[:, :k]
        for m in (0, len(scalar_ix)):
            aa1, bb1 = (np.c_[aa, x[tr][:, scalar_ix]], np.c_[bb, x[te][:, scalar_ix]]) if m else (aa, bb)
            sc = StandardScaler().fit(aa1)
            lr = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                    solver="liblinear", random_state=SEED)
            lr.fit(sc.transform(aa1), y[tr])
            out[(k, m)] = lr.predict_proba(sc.transform(bb1))[:, 1]
    return out

def oof(indices, q, x, y, scalar_ix=()):
    pred = {(k, m): np.zeros(len(indices)) for k in COMPONENTS for m in (0, len(scalar_ix))}
    yy = y[indices]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (a, b) in enumerate(cv.split(np.zeros(len(indices)), yy), 1):
        got = one_fold(indices[a], indices[b], q, x, y, scalar_ix)
        for key, val in got.items(): pred[key][b] = val
        print(f"fold {fold}/5 n={len(indices)}", flush=True)
    return pred

def auc(y, p): return float(roc_auc_score(y, p))

def main():
    y, q, x = T.load()
    di, ci = train_test_split(np.arange(len(y)), train_size=1000, stratify=y, random_state=SEED)
    dp = oof(di, q, x, y)
    dauc = {k: auc(y[di], dp[(k, 0)]) for k in COMPONENTS}
    best = max(COMPONENTS, key=dauc.get)
    chosen = T.select_scalars(x[di], y[di] - dp[(best, 0)])
    scalar_ix = tuple(z[2] for z in chosen)
    da = oof(di, q, x, y, scalar_ix)
    cp = oof(ci, q, x, y, scalar_ix)
    base, aug = auc(y[ci], cp[(best, 0)]), auc(y[ci], cp[(best, 4)])
    report = {
        "n": len(y), "split": "discovery=1000 / confirmation=1894",
        "protocol": "5-fold OOF; screening/scaler/PLS/LR training-fold only",
        "screened_coordinates": SCREEN, "component_budgets": list(COMPONENTS),
        "discovery_baseline_auroc": dauc, "selected_pls_components": best,
        "selected_scalars": [{"name": z[3], "rho_residual": z[1]} for z in chosen],
        "final_model_dimensions": best + 4,
        "discovery": {"baseline_auroc": auc(y[di], da[(best, 0)]),
                      "augmented_auroc": auc(y[di], da[(best, 4)])},
        "confirmation": {"baseline_auroc": base, "augmented_auroc": aug,
                         "delta_auroc": aug-base},
    }
    B.atomic_json(OUT, report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
