#!/usr/bin/env python3
"""Nested evaluation of P fused with single- or multi-layer Aiersilan states."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

base = importlib.import_module("272_full_scientist_standard_upr_tables")
RUNS = base.RUNS
AIERSILAN = RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt"
OUT = RUNS / "299_multilayer_aiersilan_p_fusion"


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


def prepare(p_values, a_values, fit, apply, seed):
    p = [project(p_values[0], fit, apply, None, seed)]
    p += [project(x, fit, apply, 16, seed) for x in p_values[1:5]]
    p += [project(p_values[5], fit, apply, 96, seed)]
    # Aiersilan layers 12--16, each with a separate fold-local basis.
    a = {layer: project(a_values[:, layer], fit, apply, 192, seed+layer)
         for layer in range(12, 17)}
    return p, a


def features(p, a, side, layers, dim):
    parts = [x[side] for x in p]
    parts += [a[layer][side][:, :dim] for layer in layers]
    return np.concatenate(parts, axis=1)


def classifier(c, seed):
    return LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                              solver="liblinear", random_state=seed)


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def representation_configs():
    configs = [((14,), d) for d in (64, 96, 128, 192)]
    configs += [(window, d)
                for window in ((12,13,14), (13,14,15), (14,15,16),
                               (12,13,14,15,16))
                for d in (8, 16, 24, 32)]
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46, 47])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = base.load(); keys = [r["key"] for r in rows]
    y = np.asarray([r["error"] for r in rows])
    saved = torch.load(AIERSILAN, map_location="cpu")
    amap = {k: saved["hidden_states"][i].float().numpy()
            for i, k in enumerate(saved["keys"])}
    av = np.stack([amap[k] for k in keys])
    pv = [np.stack([r["p_scalar"] for r in rows])]
    pv += [np.stack([r["p_hidden"][j] for r in rows]) for j in range(4)]
    pv += [np.stack([r["p_layer"] for r in rows])]

    reps = representation_configs(); cs = (.003, .01, .03, .1)
    configs = [(layers, dim, c) for layers, dim in reps for c in cs]
    reports, predictions = [], []
    for seed in args.seeds:
        dev, test = split_outer(y, seed)
        oof = {cfg: np.zeros(len(dev), np.float32) for cfg in configs}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (fi, vi) in enumerate(cv.split(dev, y[dev]), 1):
            fit, val = dev[fi], dev[vi]
            pb, ab = prepare(pv, av, fit, val, seed+fold)
            for layers, dim in reps:
                x = features(pb, ab, 0, layers, dim)
                z = features(pb, ab, 1, layers, dim)
                for c in cs:
                    model = classifier(c, seed).fit(x, y[fit])
                    oof[(layers,dim,c)][vi] = model.predict_proba(z)[:, 1]
            print(f"seed={seed} fold={fold}/5", flush=True)
        ranked = [(metrics(y[dev], s)["auroc"], metrics(y[dev], s)["auprc"], cfg)
                  for cfg, s in oof.items()]
        best = max(ranked); layers, dim, c = best[2]
        pb, ab = prepare(pv, av, dev, test, seed)
        x = features(pb, ab, 0, layers, dim)
        z = features(pb, ab, 1, layers, dim)
        model = classifier(c, seed).fit(x, y[dev])
        score = model.predict_proba(z)[:, 1]; tm = metrics(y[test], score)
        reports.append({"seed": seed,
                        "selected": {"layers": list(layers),
                                     "pca_per_layer": dim, "C": c},
                        "cv": {"auroc": best[0], "auprc": best[1]},
                        "test": tm, "n_dev": len(dev), "n_test": len(test)})
        predictions += [{"seed": seed, "key": keys[i], "error": int(y[i]),
                         "score": float(s)} for i, s in zip(test, score)]
        print(f"seed={seed} selected={best[2]} test={tm['auroc']:.6f}", flush=True)
    report = {
        "protocol": "full Scientist 2894; stratified outer 80/20; inner 5-fold selection; fold-local per-layer PCA; untouched outer test",
        "feature_definition": "fixed P PCA16x4+P-layer96 fused with Aiersilan layer windows 12--16",
        "candidate_count": len(configs), "per_seed": reports,
        "test_mean": {k: float(np.mean([r["test"][k] for r in reports]))
                      for k in ("auroc", "auprc")},
        "test_std": {k: float(np.std([r["test"][k] for r in reports]))
                     for k in ("auroc", "auprc")}}
    (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.out/"predictions.jsonl").open("w") as f:
        for row in predictions: f.write(json.dumps(row)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
