#!/usr/bin/env python3
"""Tune P+Aiersilan by 5-fold CV on the full 80% development split."""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

base = importlib.import_module("272_full_scientist_standard_upr_tables")
OUT = base.RUNS / "291_cv_tune_p_aiersilan_official"
AIERSILAN = base.RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt"


def split_outer(y, seed):
    return tuple(np.asarray(x) for x in train_test_split(
        np.arange(len(y)), test_size=.2, stratify=y, random_state=seed))


def project(values, fit, apply, dim, seed):
    scaler = StandardScaler().fit(values[fit])
    x = scaler.transform(values[fit]); z = scaler.transform(values[apply])
    if dim:
        pca = PCA(min(dim, len(fit)-1, x.shape[1]), whiten=True,
                  svd_solver="randomized", random_state=seed).fit(x)
        x, z = pca.transform(x), pca.transform(z)
    return x.astype(np.float32), z.astype(np.float32)


def max_blocks(p_values, a, fit, apply, seed):
    blocks = [project(p_values[0], fit, apply, None, seed)]
    blocks += [project(x, fit, apply, 16, seed) for x in p_values[1:5]]
    blocks += [project(p_values[5], fit, apply, 96, seed),
               project(a, fit, apply, 192, seed)]
    return blocks


def features(blocks, ph, pl, ad, side):
    parts = [blocks[0][side]]
    parts.extend(x[side][:, :ph] for x in blocks[1:5])
    parts.extend((blocks[5][side][:, :pl], blocks[6][side][:, :ad]))
    return np.concatenate(parts, axis=1)


def lr(c, seed):
    return LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                              solver="liblinear", random_state=seed)


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(base.SEEDS))
    parser.add_argument("--output-name", default=OUT.name)
    args = parser.parse_args()
    out = base.RUNS / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    rows = base.load(); keys = [r["key"] for r in rows]
    y = np.asarray([r["error"] for r in rows])
    saved = torch.load(AIERSILAN, map_location="cpu")
    amap = {k: saved["hidden_states"][i, 14].float().numpy()
            for i, k in enumerate(saved["keys"])}
    a = np.stack([amap[k] for k in keys])
    pv = [np.stack([r["p_scalar"] for r in rows])]
    pv += [np.stack([r["p_hidden"][j] for r in rows]) for j in range(4)]
    pv += [np.stack([r["p_layer"] for r in rows])]

    dims = [(ph, pl, ad) for ph in (4, 8, 16)
            for pl in (24, 48, 72, 96)
            for ad in (16, 32, 48, 64, 96, 128, 192)]
    cs = (.003, .01, .03, .1, .3, 1., 3.)
    configs = [(ph, pl, ad, c) for ph, pl, ad in dims for c in cs]
    reports=[]; predictions=[]
    for seed in args.seeds:
        dev, test = split_outer(y, seed)
        oof = {cfg: np.zeros(len(dev), np.float32) for cfg in configs}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (fi, vi) in enumerate(cv.split(dev, y[dev]), 1):
            fit, val = dev[fi], dev[vi]
            blocks = max_blocks(pv, a, fit, val, seed + fold)
            for ph, pl, ad in dims:
                x, z = features(blocks, ph, pl, ad, 0), features(blocks, ph, pl, ad, 1)
                for c in cs:
                    model = lr(c, seed).fit(x, y[fit])
                    oof[(ph,pl,ad,c)][vi] = model.predict_proba(z)[:, 1]
            print(f"seed={seed} cv_fold={fold}/5", flush=True)
        ranked=[]
        for cfg, score in oof.items():
            m=metrics(y[dev], score); ranked.append((m["auroc"],m["auprc"],cfg))
        best=max(ranked); ph,pl,ad,c=best[2]
        blocks=max_blocks(pv,a,dev,test,seed)
        x,z=features(blocks,ph,pl,ad,0),features(blocks,ph,pl,ad,1)
        model=lr(c,seed).fit(x,y[dev]); score=model.predict_proba(z)[:,1]
        tm=metrics(y[test],score)
        reports.append({"seed":seed,"selected":{"p_hidden":ph,"p_layer":pl,
                        "aiersilan":ad,"C":c},"cv":{"auroc":best[0],"auprc":best[1]},
                        "test":tm,"n_dev":len(dev),"n_test":len(test)})
        for idx,s in zip(test,score):
            predictions.append({"seed":seed,"key":keys[idx],"error":int(y[idx]),
                                "score":float(s)})
        print(f"seed={seed} selected={best[2]} cv={best[0]:.6f} test={tm['auroc']:.6f}",flush=True)
    report={"protocol":"Aiersilan stratified outer 80/20; hyperparameters selected by 5-fold stratified CV across the full outer 80%; refit on full 80%; one evaluation on fixed 20% test",
            "candidate_count":len(configs),"per_seed":reports,
            "test_mean":{k:float(np.mean([r["test"][k] for r in reports])) for k in ("auroc","auprc")},
            "test_std":{k:float(np.std([r["test"][k] for r in reports])) for k in ("auroc","auprc")}}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    with (out/"predictions.jsonl").open("w") as f:
        for r in predictions:f.write(json.dumps(r)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
