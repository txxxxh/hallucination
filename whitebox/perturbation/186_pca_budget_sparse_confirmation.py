#!/usr/bin/env python3
"""Train-fold PCA hidden baseline under a <=200 dimensional budget.

Select the PCA budget and at most four perturbation scalars on discovery=1000,
then freeze both choices and report OOF performance on confirmation=1894.
"""
from __future__ import annotations
import importlib, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

M = importlib.import_module("184_sparse_fullcache_confirmation_fixed")
S, B = M.S, M.B
OUT = M.RUNS / "186_pca_budget_sparse_confirmation.json"
LAYERS = S.P.KEEN_LAYERS
PER_LAYER = (8, 12, 16, 24)       # total hidden dims: 64,96,128,192
MAX_PC = max(PER_LAYER)
SEED = B.SEED

def load():
    rows, *_ = B.load_rows()
    rows = [r for r in rows if (M.D / "features" / (r["key"] + ".npz")).exists()]
    y = np.asarray([r["known"] for r in rows], dtype=np.int64)
    q = np.stack([np.load(B.QUESTION_CACHE / (r["key"] + ".npz"))["hidden"][LAYERS].astype(np.float32) for r in rows])
    x = np.stack([M.compact(np.load(M.D / "features" / (r["key"] + ".npz"))["local_geometry"]) for r in rows])
    return rows, y, q, x

def fit_predict(a, b, q, x, y, scalar_ix=()):
    """One fold. PCA is fitted only on a; a single LR sees all layer PCs."""
    za, zb = [], []
    for li in range(q.shape[1]):
        p = PCA(n_components=MAX_PC, svd_solver="randomized", random_state=SEED + li)
        za.append(p.fit_transform(q[a, li]))
        zb.append(p.transform(q[b, li]))
    out = {}
    for k in PER_LAYER:
        aa = np.concatenate([z[:, :k] for z in za], axis=1)
        bb = np.concatenate([z[:, :k] for z in zb], axis=1)
        for m in (0, len(scalar_ix)):
            if m:
                aa1 = np.c_[aa, x[a][:, scalar_ix]]
                bb1 = np.c_[bb, x[b][:, scalar_ix]]
            else:
                aa1, bb1 = aa, bb
            sc = StandardScaler().fit(aa1)
            lr = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                    solver="liblinear", random_state=SEED)
            lr.fit(sc.transform(aa1), y[a])
            out[(k, m)] = lr.predict_proba(sc.transform(bb1))[:, 1]
    return out

def oof(indices, q, x, y, scalar_ix=()):
    pred = {(k, m): np.zeros(len(indices)) for k in PER_LAYER for m in (0, len(scalar_ix))}
    yy = y[indices]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (atr, ate) in enumerate(cv.split(np.zeros(len(indices)), yy), 1):
        got = fit_predict(indices[atr], indices[ate], q, x, y, scalar_ix)
        for key, val in got.items(): pred[key][ate] = val
        print(f"fold {fold}/5 n={len(indices)}", flush=True)
    return pred

def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p))}

def select_scalars(x, y, residual, limit=4):
    ranked = []
    for j, name in enumerate(M.N):
        rho = np.corrcoef(x[:, j], residual)[0, 1]
        if np.isfinite(rho): ranked.append((abs(rho), float(rho), j, name))
    ranked.sort(reverse=True)
    chosen = []
    for z in ranked:
        if all(abs(np.corrcoef(x[:, z[2]], x[:, w[2]])[0, 1]) < .8 for w in chosen):
            chosen.append(z)
            if len(chosen) == limit: break
    return chosen

def main():
    rows, y, q, x = load()
    di, ci = train_test_split(np.arange(len(y)), train_size=1000,
                              stratify=y, random_state=SEED)
    # Discovery chooses only the dimensional budget.
    dp = oof(di, q, x, y)
    dauc = {k: metrics(y[di], dp[(k, 0)])["auroc"] for k in PER_LAYER}
    best = max(PER_LAYER, key=dauc.get)
    chosen = select_scalars(x[di], y[di] - dp[(best, 0)], limit=4)
    scalar_ix = tuple(z[2] for z in chosen)
    # Recompute discovery with fixed scalar set, then untouched confirmation.
    da = oof(di, q, x, y, scalar_ix)
    cp = oof(ci, q, x, y, scalar_ix)
    report = {
        "n": len(y), "split": "discovery=1000 / confirmation=1894",
        "protocol": "5-fold OOF; every PCA/scaler/LR fit training-fold only",
        "budgets": {str(k): 8*k for k in PER_LAYER},
        "discovery_baseline_auroc": dauc,
        "selected_per_layer_pc": best,
        "selected_hidden_dimensions": 8*best,
        "selected_scalars": [{"name": z[3], "rho_residual": z[1]} for z in chosen],
        "total_dimensions_augmented": 8*best + len(chosen),
        "discovery": {
            "baseline": metrics(y[di], da[(best, 0)]),
            "augmented": metrics(y[di], da[(best, len(scalar_ix))]),
        },
        "confirmation": {
            "baseline": metrics(y[ci], cp[(best, 0)]),
            "augmented": metrics(y[ci], cp[(best, len(scalar_ix))]),
        },
    }
    report["confirmation"]["delta_auroc"] = (report["confirmation"]["augmented"]["auroc"] -
                                                report["confirmation"]["baseline"]["auroc"])
    B.atomic_json(OUT, report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
