#!/usr/bin/env python3
"""Fold-local hybrid of stable PCA directions and supervised coordinates."""
from __future__ import annotations
import importlib, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

T = importlib.import_module("189_supervised_topk_sparse_confirmation")
M, S, B = T.M, T.S, T.B
OUT = M.RUNS / "191_hybrid_pca_topk_sparse_confirmation.json"
# (PCs per each of 8 layers, supervised raw coordinates)
CONFIGS = ((8, 64), (12, 32), (12, 48), (16, 16))
SEED = B.SEED

def one_fold(tr, te, qflat, x, y, scalar_ix=()):
    q = qflat.reshape(len(qflat), 8, -1)
    za, zb = [], []
    for li in range(8):
        p = PCA(n_components=16, svd_solver="randomized", random_state=SEED+li)
        za.append(p.fit_transform(q[tr, li]))
        zb.append(p.transform(q[te, li]))
    order = T.rank_coordinates(qflat[tr], y[tr])
    out = {}
    for pc, raw in CONFIGS:
        aa = np.c_[np.concatenate([z[:, :pc] for z in za], 1), qflat[tr][:, order[:raw]]]
        bb = np.c_[np.concatenate([z[:, :pc] for z in zb], 1), qflat[te][:, order[:raw]]]
        for m in (0, len(scalar_ix)):
            aa1, bb1 = (np.c_[aa, x[tr][:, scalar_ix]], np.c_[bb, x[te][:, scalar_ix]]) if m else (aa, bb)
            sc = StandardScaler().fit(aa1)
            lr = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                    solver="liblinear", random_state=SEED)
            lr.fit(sc.transform(aa1), y[tr])
            out[((pc, raw), m)] = lr.predict_proba(sc.transform(bb1))[:, 1]
    return out

def oof(ix, q, x, y, scalar_ix=()):
    pred = {(cfg, m): np.zeros(len(ix)) for cfg in CONFIGS for m in (0, len(scalar_ix))}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (a, b) in enumerate(cv.split(np.zeros(len(ix)), y[ix]), 1):
        got = one_fold(ix[a], ix[b], q, x, y, scalar_ix)
        for key, val in got.items(): pred[key][b] = val
        print(f"fold {fold}/5 n={len(ix)}", flush=True)
    return pred

def auc(y,p): return float(roc_auc_score(y,p))

def main():
    y,q,x=T.load();di,ci=train_test_split(np.arange(len(y)),train_size=1000,stratify=y,random_state=SEED)
    dp=oof(di,q,x,y);dauc={f"{a}pcx8+{b}raw":auc(y[di],dp[((a,b),0)]) for a,b in CONFIGS}
    best=max(CONFIGS,key=lambda z:dauc[f"{z[0]}pcx8+{z[1]}raw"])
    chosen=T.select_scalars(x[di],y[di]-dp[(best,0)]);six=tuple(z[2] for z in chosen)
    da=oof(di,q,x,y,six);cp=oof(ci,q,x,y,six);base=auc(y[ci],cp[(best,0)]);aug=auc(y[ci],cp[(best,4)])
    report={"n":len(y),"split":"discovery=1000 / confirmation=1894","protocol":"all PCA/ranking/scaling/LR fold-local","configs":dauc,
      "selected":{"pc_per_layer":best[0],"supervised_coordinates":best[1],"hidden_dimensions":8*best[0]+best[1]},
      "selected_scalars":[z[3] for z in chosen],"total_dimensions_augmented":8*best[0]+best[1]+4,
      "discovery":{"baseline_auroc":auc(y[di],da[(best,0)]),"augmented_auroc":auc(y[di],da[(best,4)])},
      "confirmation":{"baseline_auroc":base,"augmented_auroc":aug,"delta_auroc":aug-base}}
    B.atomic_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
