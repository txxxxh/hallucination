#!/usr/bin/env python3
"""Leakage-safe OOF score-level stacking of P and Aiersilan on full Scientist."""
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
OUT = RUNS / "300_oof_late_fusion_p_aiersilan"


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


def p_features(pv, fit, apply, ph, pl, seed):
    blocks = [project(pv[0], fit, apply, None, seed)]
    blocks += [project(x, fit, apply, ph, seed) for x in pv[1:5]]
    blocks += [project(pv[5], fit, apply, pl, seed)]
    return np.concatenate([x[0] for x in blocks], 1), np.concatenate([x[1] for x in blocks], 1)


def a_features(av, fit, apply, dim, seed):
    return project(av, fit, apply, dim, seed)


def clf(c, seed):
    return LogisticRegression(C=c, max_iter=5000, class_weight="balanced",
                              solver="liblinear", random_state=seed)


def metric(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def logit(p):
    p = np.clip(p, 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42,43,44,45,46,47])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = base.load(); keys = [r["key"] for r in rows]
    y = np.asarray([r["error"] for r in rows])
    saved = torch.load(AIERSILAN, map_location="cpu")
    amap = {k: saved["hidden_states"][i,14].float().numpy()
            for i,k in enumerate(saved["keys"])}
    av = np.stack([amap[k] for k in keys])
    pv = [np.stack([r["p_scalar"] for r in rows])]
    pv += [np.stack([r["p_hidden"][j] for r in rows]) for j in range(4)]
    pv += [np.stack([r["p_layer"] for r in rows])]
    p_cfgs = [(ph,pl,c) for ph in (8,16) for pl in (48,96)
              for c in (.003,.01,.03,.1)]
    a_cfgs = [(d,c) for d in (48,96,128,192) for c in (.003,.01,.03,.1)]
    reports=[]; predictions=[]
    for seed in args.seeds:
        dev,test=split_outer(y,seed)
        poof={cfg:np.zeros(len(dev),np.float32) for cfg in p_cfgs}
        aoof={cfg:np.zeros(len(dev),np.float32) for cfg in a_cfgs}
        cv=StratifiedKFold(5,shuffle=True,random_state=seed)
        for fold,(fi,vi) in enumerate(cv.split(dev,y[dev]),1):
            fit,val=dev[fi],dev[vi]
            for ph in (8,16):
                for pl in (48,96):
                    x,z=p_features(pv,fit,val,ph,pl,seed+fold)
                    for c in (.003,.01,.03,.1):
                        poof[(ph,pl,c)][vi]=clf(c,seed).fit(x,y[fit]).predict_proba(z)[:,1]
            for d in (48,96,128,192):
                x,z=a_features(av,fit,val,d,seed+fold)
                for c in (.003,.01,.03,.1):
                    aoof[(d,c)][vi]=clf(c,seed).fit(x,y[fit]).predict_proba(z)[:,1]
            print(f"seed={seed} fold={fold}/5",flush=True)
        pcfg=max(p_cfgs,key=lambda q:(metric(y[dev],poof[q])["auroc"],metric(y[dev],poof[q])["auprc"]))
        acfg=max(a_cfgs,key=lambda q:(metric(y[dev],aoof[q])["auroc"],metric(y[dev],aoof[q])["auprc"]))
        pdev,adev=poof[pcfg],aoof[acfg]
        # The level-2 model sees only held-out level-1 predictions.
        meta_x=np.c_[logit(pdev),logit(adev)]
        meta_scaler=StandardScaler().fit(meta_x)
        meta=LogisticRegression(C=1.,max_iter=5000,class_weight="balanced",
                                solver="liblinear",random_state=seed).fit(meta_scaler.transform(meta_x),y[dev])
        ph,pl,pc=pcfg; px,pz=p_features(pv,dev,test,ph,pl,seed)
        ps=clf(pc,seed).fit(px,y[dev]).predict_proba(pz)[:,1]
        ad,ac=acfg; ax,az=a_features(av,dev,test,ad,seed)
        ass=clf(ac,seed).fit(ax,y[dev]).predict_proba(az)[:,1]
        late=meta.predict_proba(meta_scaler.transform(np.c_[logit(ps),logit(ass)]))[:,1]
        avg=(ps+ass)/2
        result={"P":metric(y[test],ps),"Aiersilan":metric(y[test],ass),
                "equal_average":metric(y[test],avg),"OOF_logistic":metric(y[test],late)}
        reports.append({"seed":seed,"selected":{"P":pcfg,"Aiersilan":acfg},
                        "oof":{"P":metric(y[dev],pdev),"Aiersilan":metric(y[dev],adev)},
                        "meta_coefficients":meta.coef_[0].tolist(),"test":result,
                        "n_dev":len(dev),"n_test":len(test)})
        for i,p,a,s in zip(test,ps,ass,late):
            predictions.append({"seed":seed,"key":keys[i],"error":int(y[i]),
                                "p_score":float(p),"aiersilan_score":float(a),
                                "late_score":float(s)})
        print(f"seed={seed} P={result['P']['auroc']:.6f} A={result['Aiersilan']['auroc']:.6f} late={result['OOF_logistic']['auroc']:.6f}",flush=True)
    methods=("P","Aiersilan","equal_average","OOF_logistic")
    report={"protocol":"full Scientist 2894; stratified outer 80/20; level-1 hyperparameters selected by inner 5-fold OOF; level-2 LR trained only on selected OOF logits; untouched outer test",
            "per_seed":reports,"summary":{m:{k+"_mean":float(np.mean([r["test"][m][k] for r in reports])) for k in ("auroc","auprc")}|{k+"_std":float(np.std([r["test"][m][k] for r in reports])) for k in ("auroc","auprc")} for m in methods}}
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.out/"predictions.jsonl").open("w") as f:
        for r in predictions:f.write(json.dumps(r)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
