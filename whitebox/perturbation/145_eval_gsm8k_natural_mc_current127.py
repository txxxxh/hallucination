#!/usr/bin/env python3
"""Leakage-safe current127 OOF evaluation for natural-error GSM8K MC."""
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
CACHE = HERE / "runs/144_gsm8k_natural_mc_current127"
OUT = HERE / "runs/145_gsm8k_natural_mc_current127_report.json"
SEEDS = (42, 43, 44)

def ch(s):
    u = s[0] - s[1:]; scale = abs(float(s[0])) + 1e-6
    return np.r_[s[0], u, u / scale, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]

def ch2(s): return np.r_[s[0], s[0] - s[1:]]

def wd(h, u):
    d = h[1:].astype(np.float32) - h[0].astype(np.float32)
    return (d * u[:, None]).sum(0) / (np.abs(u).sum() + 1e-9)

def load_rows():
    rows = []
    for fp in sorted(CACHE.glob("*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            p, o, q, r = (z[k] for k in ("stage1_pred", "stage1_other", "stage2_pred", "stage2_other"))
            scalar = np.r_[ch(p), ch(o), ch2(q), ch2(r), p[0]-q[0], o[0]-r[0],
                           (p[0]-o[0])-(q[0]-r[0])]
            ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
            hidden = (ph[0], wd(ph, p[0]-p[1:]), oh[0], wd(oh, o[0]-o[1:]))
            rows.append((str(z["key"].item()), int(z["correct"]), scalar, hidden,
                         z["layer14"].astype(np.float32), int(z["choice"]),
                         float(z["p_choice1"]), float(z["p_choice2"])))
    if len(rows) != 471: raise RuntimeError(f"expected 471 rows, got {len(rows)}")
    return rows

def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, p >= .5))}

def transform(train_blocks, test_blocks, seed, full):
    aout, bout = [], []
    dims = (None, 8, 8, 8, 8, 48) if full else (None,)
    for train, test, dim in zip(train_blocks, test_blocks, dims):
        sc = StandardScaler().fit(train); a, b = sc.transform(train), sc.transform(test)
        if dim is not None:
            pc = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
            a, b = pc.transform(a), pc.transform(b)
        aout.append(a); bout.append(b)
    return np.concatenate(aout, 1), np.concatenate(bout, 1)

def oof(blocks, y, full=False):
    runs, allpred = [], []
    for seed in SEEDS:
        pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(blocks[0], y):
            a, b = transform([x[tr] for x in blocks], [x[te] for x in blocks], seed, full)
            clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                     solver="liblinear", random_state=seed).fit(a, y[tr])
            pred[te] = clf.predict_proba(b)[:, 1]
        runs.append(metrics(y, pred)); allpred.append(pred)
    mean = np.mean(allpred, axis=0)
    return {"ensemble": metrics(y, mean),
            "mean_across_seeds": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
            "per_seed": [{"seed": s, **r} for s, r in zip(SEEDS, runs)]}, mean

def main():
    rows = load_rows(); keys = np.asarray([r[0] for r in rows]); y = np.asarray([r[1] for r in rows])
    scalar = np.stack([r[2] for r in rows]); hidden = [np.stack([r[3][j] for r in rows]) for j in range(4)]
    layer14 = np.stack([r[4] for r in rows])
    choice = np.asarray([r[5] for r in rows])[:, None]
    probs = np.asarray([[r[6], r[7], max(r[6], r[7]), abs(r[6]-r[7])] for r in rows])
    full, score = oof([scalar, *hidden, layer14], y, full=True)
    scalar_only, _ = oof([scalar], y)
    confidence_only, _ = oof([probs], y)
    choice_position_only, _ = oof([choice], y)
    report = {"dataset": "GSM8K train natural-error strict two-choice sample",
              "n": len(y), "correct": int(y.sum()), "incorrect": int((1-y).sum()),
              "protocol": "Scientist-format MC; fixed current127; 3x5 stratified OOF; fold-local scaling/PCA",
              "config": "scalar47 + four candidate-hidden PCA8 + layer14 PCA48; LR C=.03",
              "full_detector": full, "scalar_only": scalar_only,
              "unperturbed_choice_confidence_only": confidence_only,
              "choice_position_only": choice_position_only,
              "per_item": [{"id": k, "correct": bool(v), "oof_score": float(p)} for k,v,p in zip(keys,y,score)]}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "per_item"}, indent=2))

if __name__ == "__main__": main()
