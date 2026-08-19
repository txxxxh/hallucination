#!/usr/bin/env python3
"""Fold-local supervised coordinate selection under a <=200 feature budget."""
from __future__ import annotations
import importlib, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

M = importlib.import_module("184_sparse_fullcache_confirmation_fixed")
S, B = M.S, M.B
OUT = M.RUNS / "189_supervised_topk_sparse_confirmation.json"
BUDGETS = (64, 96, 128, 144, 192)
SEED = B.SEED

def load():
    rows, *_ = B.load_rows()
    rows = [r for r in rows if (M.D / "features" / (r["key"] + ".npz")).exists()]
    y = np.asarray([r["known"] for r in rows], dtype=np.int64)
    q = np.stack([np.load(B.QUESTION_CACHE / (r["key"] + ".npz"))["hidden"][S.P.KEEN_LAYERS].astype(np.float32) for r in rows])
    q = q.reshape(len(q), -1)
    x = np.stack([M.compact(np.load(M.D / "features" / (r["key"] + ".npz"))["local_geometry"]) for r in rows])
    return y, q, x

def rank_coordinates(q, y):
    """Absolute standardized mean difference, computed on training rows only."""
    a, b = q[y == 0], q[y == 1]
    delta = a.mean(0) - b.mean(0)
    var = a.var(0) + b.var(0) + 1e-8
    score = np.abs(delta) / np.sqrt(var)
    return np.argsort(score)[::-1]

def one_fold(tr, te, q, x, y, scalar_ix=()):
    order = rank_coordinates(q[tr], y[tr])
    out = {}
    for k in BUDGETS:
        ix = order[:k]
        aa, bb = q[tr][:, ix], q[te][:, ix]
        for m in (0, len(scalar_ix)):
            if m:
                aa1, bb1 = np.c_[aa, x[tr][:, scalar_ix]], np.c_[bb, x[te][:, scalar_ix]]
            else:
                aa1, bb1 = aa, bb
            sc = StandardScaler().fit(aa1)
            lr = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                    solver="liblinear", random_state=SEED)
            lr.fit(sc.transform(aa1), y[tr])
            out[(k, m)] = lr.predict_proba(sc.transform(bb1))[:, 1]
    return out

def oof(indices, q, x, y, scalar_ix=()):
    pred = {(k, m): np.zeros(len(indices)) for k in BUDGETS for m in (0, len(scalar_ix))}
    yy = y[indices]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (a, b) in enumerate(cv.split(np.zeros(len(indices)), yy), 1):
        got = one_fold(indices[a], indices[b], q, x, y, scalar_ix)
        for key, val in got.items(): pred[key][b] = val
        print(f"fold {fold}/5 n={len(indices)}", flush=True)
    return pred

def auc(y, p): return float(roc_auc_score(y, p))

def select_scalars(x, residual, limit=4):
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
    y, q, x = load()
    di, ci = train_test_split(np.arange(len(y)), train_size=1000, stratify=y, random_state=SEED)
    dp = oof(di, q, x, y)
    dauc = {k: auc(y[di], dp[(k, 0)]) for k in BUDGETS}
    best = max(BUDGETS, key=dauc.get)
    chosen = select_scalars(x[di], y[di] - dp[(best, 0)])
    scalar_ix = tuple(z[2] for z in chosen)
    da = oof(di, q, x, y, scalar_ix)
    cp = oof(ci, q, x, y, scalar_ix)
    base, aug = auc(y[ci], cp[(best, 0)]), auc(y[ci], cp[(best, 4)])
    report = {
        "n": len(y), "split": "discovery=1000 / confirmation=1894",
        "protocol": "5-fold OOF; coordinate ranking/scaler/LR training-fold only",
        "candidate_hidden_coordinates": q.shape[1], "budgets": list(BUDGETS),
        "discovery_baseline_auroc": dauc, "selected_hidden_dimensions": best,
        "selected_scalars": [{"name": z[3], "rho_residual": z[1]} for z in chosen],
        "total_dimensions_augmented": best + 4,
        "discovery": {"baseline_auroc": auc(y[di], da[(best, 0)]),
                      "augmented_auroc": auc(y[di], da[(best, 4)])},
        "confirmation": {"baseline_auroc": base, "augmented_auroc": aug,
                         "delta_auroc": aug-base},
    }
    B.atomic_json(OUT, report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
