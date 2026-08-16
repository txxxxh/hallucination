#!/usr/bin/env python3
"""Leakage-safe HotpotQA tuning on the already collected current127 features."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FEATURES = RUNS / "132_hotpotqa_current127"
OUT = RUNS / "135_hotpotqa_nested_tuning.json"
RECIPES = ((0, 0), (4, 16), (8, 32), (16, 48))
CS = (.003, .01, .03, .1, .3, 1.0, 3.0)


def ch(s):
    u = s[0] - s[1:]
    z = abs(float(s[0])) + 1e-6
    return np.r_[s[0], u, u / z, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def ch2(s):
    return np.r_[s[0], s[0] - s[1:]]


def wd(h, u):
    d = h[1:].astype(np.float32) - h[0].astype(np.float32)
    return (d * u[:, None]).sum(0) / (np.abs(u).sum() + 1e-9)


def load():
    rows = []
    for fp in sorted(FEATURES.glob("*.npz")):
        with np.load(fp, allow_pickle=True) as q:
            p = q["stage1_pred"].astype(np.float32)
            o = q["stage1_other"].astype(np.float32)
            p2 = q["stage2_pred"].astype(np.float32)
            o2 = q["stage2_other"].astype(np.float32)
            ph = q["pred_hidden"].astype(np.float32)
            oh = q["other_hidden"].astype(np.float32)
            scalar = np.r_[ch(p), ch(o), ch2(p2), ch2(o2), p[0] - p2[0],
                           o[0] - o2[0], (p[0] - o[0]) - (p2[0] - o2[0])]
            hidden = (ph[0], wd(ph, p[0] - p[1:]), oh[0], wd(oh, o[0] - o[1:]))
            length = [float(q["generation_words"]), float(q["other_words"])]
            rows.append((int(q["correct"]), scalar, hidden,
                         q["layer14"].astype(np.float32), length))
    y = np.array([r[0] for r in rows])
    return (y, np.stack([r[1] for r in rows]),
            [np.stack([r[2][j] for r in rows]) for j in range(4)],
            np.stack([r[3] for r in rows]), np.stack([r[4] for r in rows]))


def transform(S, H, L, length, tr, te, hd, ld, add_length, seed):
    blocks_tr, blocks_te = [], []
    for x, dim in [(S, None), *[(h, hd) for h in H], (L, ld)]:
        if dim == 0:
            continue
        sc = StandardScaler().fit(x[tr])
        a, b = sc.transform(x[tr]), sc.transform(x[te])
        if dim is not None:
            dim = min(dim, len(tr) - 1, a.shape[1])
            pc = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
            a, b = pc.transform(a), pc.transform(b)
        blocks_tr.append(a)
        blocks_te.append(b)
    if add_length:
        sc = StandardScaler().fit(length[tr])
        blocks_tr.append(sc.transform(length[tr]))
        blocks_te.append(sc.transform(length[te]))
    return np.concatenate(blocks_tr, 1), np.concatenate(blocks_te, 1)


def metric(y, p):
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5))}


def run_variant(y, S, H, L, length, add_length):
    runs = []
    for seed in (42, 43, 44):
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y))
        chosen = []
        for fold, (tr, te) in enumerate(outer.split(S, y)):
            inner = StratifiedKFold(3, shuffle=True, random_state=seed * 10 + fold)
            scores = {(hd, ld, c): [] for hd, ld in RECIPES for c in CS}
            for itr0, iva0 in inner.split(S[tr], y[tr]):
                itr, iva = tr[itr0], tr[iva0]
                for hd, ld in RECIPES:
                    xt, xv = transform(S, H, L, length, itr, iva, hd, ld, add_length, seed)
                    for c in CS:
                        model = LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                                                   solver="liblinear", random_state=seed)
                        p = model.fit(xt, y[itr]).predict_proba(xv)[:, 1]
                        scores[(hd, ld, c)].append(roc_auc_score(y[iva], p))
            best = max(scores, key=lambda k: (np.mean(scores[k]), -k[0] - k[1], -k[2]))
            hd, ld, c = best
            xt, xv = transform(S, H, L, length, tr, te, hd, ld, add_length, seed)
            model = LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                                       solver="liblinear", random_state=seed)
            pred[te] = model.fit(xt, y[tr]).predict_proba(xv)[:, 1]
            chosen.append({"hidden_pca_each": hd, "layer14_pca": ld, "C": c,
                           "inner_auroc": float(np.mean(scores[best]))})
        runs.append({"seed": seed, **metric(y, pred), "chosen": chosen})
    mean = {k: float(np.mean([r[k] for r in runs]))
            for k in ("auroc", "auprc", "balanced_accuracy")}
    selections = Counter((x["hidden_pca_each"], x["layer14_pca"], x["C"])
                         for r in runs for x in r["chosen"])
    return {"mean": mean, "per_seed": runs,
            "selection_counts": [{"hidden_pca_each": k[0], "layer14_pca": k[1],
                                  "C": k[2], "count": v}
                                 for k, v in selections.most_common()]}


def main():
    y, S, H, L, length = load()
    report = {
        "protocol": "3x repeated 5-fold outer CV; 3-fold inner tuning by AUROC; all scaling/PCA fit inside each fold",
        "n": len(y), "correct": int(y.sum()), "incorrect": int((1-y).sum()),
        "search": {"pca_recipes": RECIPES, "C": CS},
        "detector_only": run_variant(y, S, H, L, length, False),
        "detector_plus_answer_length": run_variant(y, S, H, L, length, True),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
