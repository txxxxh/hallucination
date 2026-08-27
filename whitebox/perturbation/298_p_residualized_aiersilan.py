#!/usr/bin/env python3
"""P + residualized Aiersilan on full Scientist with nested model selection."""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

base = importlib.import_module("272_full_scientist_standard_upr_tables")
RUNS = base.RUNS
AIERSILAN = RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt"
OUT = RUNS / "298_p_residualized_aiersilan"


def split_outer(y, seed):
    return tuple(np.asarray(x) for x in train_test_split(
        np.arange(len(y)), test_size=.2, stratify=y, random_state=seed))


def project(values, fit, apply, dim, seed):
    scaler = StandardScaler().fit(values[fit])
    x, z = scaler.transform(values[fit]), scaler.transform(values[apply])
    if dim:
        pca = PCA(min(dim, len(fit)-1, x.shape[1]), whiten=True,
                  svd_solver="randomized", random_state=seed).fit(x)
        x, z = pca.transform(x), pca.transform(z)
    return x.astype(np.float32), z.astype(np.float32)


def blocks(p_values, a_values, fit, apply, seed):
    p = [project(p_values[0], fit, apply, None, seed)]
    p += [project(x, fit, apply, 16, seed) for x in p_values[1:5]]
    p += [project(p_values[5], fit, apply, 96, seed)]
    a = project(a_values, fit, apply, 192, seed)
    return p, a


def p_features(p, side):
    return np.concatenate([x[side] for x in p], axis=1)


def residualize(p, a, alpha):
    px, pz = p_features(p, 0), p_features(p, 1)
    ax, az = a
    reg = Ridge(alpha=alpha).fit(px, ax)
    return ax-reg.predict(px), az-reg.predict(pz)


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def classifier(c, seed):
    return LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                              solver="liblinear", random_state=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46, 47])
    parser.add_argument("--out", type=str, default=str(OUT))
    args = parser.parse_args()
    out = __import__('pathlib').Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = base.load(); keys = [r["key"] for r in rows]
    y = np.asarray([r["error"] for r in rows])
    saved = torch.load(AIERSILAN, map_location="cpu")
    amap = {k: saved["hidden_states"][i, 14].float().numpy()
            for i, k in enumerate(saved["keys"])}
    av = np.stack([amap[k] for k in keys])
    pv = [np.stack([r["p_scalar"] for r in rows])]
    pv += [np.stack([r["p_hidden"][j] for r in rows]) for j in range(4)]
    pv += [np.stack([r["p_layer"] for r in rows])]

    alphas = (.1, 1., 10., 100., 1000.)
    adims = (32, 64, 96, 128, 192)
    cs = (.003, .01, .03, .1)
    configs = [(alpha, adim, c) for alpha in alphas for adim in adims for c in cs]
    reports, predictions = [], []
    for seed in args.seeds:
        dev, test = split_outer(y, seed)
        oof = {cfg: np.zeros(len(dev), np.float32) for cfg in configs}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (fi, vi) in enumerate(cv.split(dev, y[dev]), 1):
            fit, val = dev[fi], dev[vi]
            pb, ab = blocks(pv, av, fit, val, seed+fold)
            px, pz = p_features(pb, 0), p_features(pb, 1)
            for alpha in alphas:
                rx, rz = residualize(pb, ab, alpha)
                for adim in adims:
                    x = np.concatenate([px, rx[:, :adim]], 1)
                    z = np.concatenate([pz, rz[:, :adim]], 1)
                    for c in cs:
                        model = classifier(c, seed).fit(x, y[fit])
                        oof[(alpha, adim, c)][vi] = model.predict_proba(z)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        ranked = [(metrics(y[dev], score)["auroc"],
                   metrics(y[dev], score)["auprc"], cfg)
                  for cfg, score in oof.items()]
        best = max(ranked); alpha, adim, c = best[2]
        pb, ab = blocks(pv, av, dev, test, seed)
        px, pz = p_features(pb, 0), p_features(pb, 1)
        rx, rz = residualize(pb, ab, alpha)
        x = np.concatenate([px, rx[:, :adim]], 1)
        z = np.concatenate([pz, rz[:, :adim]], 1)
        model = classifier(c, seed).fit(x, y[dev])
        score = model.predict_proba(z)[:, 1]; tm = metrics(y[test], score)
        reports.append({"seed": seed,
                        "selected": {"ridge_alpha": alpha,
                                     "residual_aiersilan_dim": adim, "C": c},
                        "cv": {"auroc": best[0], "auprc": best[1]},
                        "test": tm, "n_dev": len(dev), "n_test": len(test)})
        predictions += [{"seed": seed, "key": keys[i], "error": int(y[i]),
                         "score": float(s)} for i, s in zip(test, score)]
        print(f"seed={seed} selected={best[2]} test={tm['auroc']:.6f}", flush=True)
    report = {
        "protocol": "full Scientist 2894; stratified outer 80/20; inner 5-fold selection; fold-local PCA and Ridge P-to-Aiersilan residualization; untouched outer test",
        "candidate_count": len(configs), "per_seed": reports,
        "test_mean": {k: float(np.mean([r["test"][k] for r in reports]))
                      for k in ("auroc", "auprc")},
        "test_std": {k: float(np.std([r["test"][k] for r in reports]))
                     for k in ("auroc", "auprc")}}
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    with (out/"predictions.jsonl").open("w") as f:
        for row in predictions: f.write(json.dumps(row)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
