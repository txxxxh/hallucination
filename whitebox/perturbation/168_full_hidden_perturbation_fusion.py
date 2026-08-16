#!/usr/bin/env python3
"""Exact 150-style five-seed question ensemble plus one nested perturbation score."""
from __future__ import annotations
import argparse, importlib, json, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
PICS=importlib.import_module("163_pics_keen_known_unknown")
GEO=importlib.import_module("164_ic_local_geometry")
SEEDS=(0,5,26,42,63)
RUNS=Path(__file__).resolve().parent/"runs"

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--feature-dir",type=Path,default=RUNS/"164_ic_local_geometry_n2894");ap.add_argument("--order-file",type=Path,default=RUNS/"150_question_layer_ensemble_oof.jsonl");ap.add_argument("--output-dir",type=Path,default=RUNS/"168_full_hidden_perturbation_fusion");a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    order=BASE.read_jsonl(a.order_file);base={r["key"]:r for r in BASE.load_rows()[0]};rows=[base[z["key"]] for z in order if (a.feature_dir/"features"/(z["key"]+".npz")).exists()]
    y=np.asarray([r["known"] for r in rows]);X=np.stack([np.load(a.feature_dir/"features"/(r["key"]+".npz"))["local_geometry"] for r in rows]);Q=np.stack([np.load(BASE.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][PICS.KEEN_LAYERS].astype(np.float32) for r in rows])
    seed_preds=[];per_seed=[];all_weights=[]
    for seed in SEEDS:
        outer=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y));pq=np.zeros(len(y));pg=np.zeros(len(y));pf=np.zeros(len(y));weights=[]
        for fold,(tr,te) in enumerate(outer):
            pq[te]=PICS.fit_q(Q,y,tr,te,seed);pg[te]=GEO.fit_geom(X,y,tr,te,seed)
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(X[tr],y[tr]));meta=np.zeros((len(tr),2))
            for it,iv in inner:
                meta[iv,0]=PICS.fit_q(Q,y,tr[it],tr[iv],seed);meta[iv,1]=GEO.fit_geom(X,y,tr[it],tr[iv],seed)
            test=np.c_[pq[te],pg[te]];sc=StandardScaler().fit(meta);m=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc.transform(meta),y[tr]);pf[te]=m.predict_proba(sc.transform(test))[:,1];weights.append(m.coef_[0].tolist());print(f"seed={seed} fold={fold+1}/5",flush=True)
        seed_preds.append((pq,pg,pf));all_weights.append(weights);per_seed.append({"seed":seed,"strong_question":BASE.metrics(y,pq,.5),"perturbation_score":BASE.metrics(y,pg,.5),"nested_fusion":BASE.metrics(y,pf,.5)})
    pred={"strong_question":np.mean([z[0] for z in seed_preds],0),"perturbation_score":np.mean([z[1] for z in seed_preds],0),"nested_fusion":np.mean([z[2] for z in seed_preds],0)}
    report={"n":len(y),"known":int(y.sum()),"unknown":int(len(y)-y.sum()),"protocol":"exact 150 seeds; stratified outer 5-fold; nested inner 3-fold fusion; all PCA/scaler/base/meta fit training-only; average five seed OOF probabilities","seeds":SEEDS,"entity_leakage_warning":True,"fusion_input_dim":2,"per_seed":per_seed,"meta_coefficients":all_weights,"results":{k:BASE.metrics(y,p,.5) for k,p in pred.items()}}
    BASE.atomic_json(a.output_dir/"evaluation.json",report);BASE.atomic_json(a.output_dir/"config.json",{"seeds":SEEDS,"n":len(y),"model":"NousResearch/Meta-Llama-3.1-8B-Instruct","feature_whitelist":["question_layer_ensemble_probability","training_fold_PCA24_local_geometry_probability"],"fusion_input_dim":2,"training_fold_only":True,"source_feature_dir":str(a.feature_dir)})
    with (a.output_dir/"predictions.jsonl").open("w") as f:
        for i,r in enumerate(rows):f.write(json.dumps({"key":r["key"],"known":int(y[i]),"probabilities":{k:float(p[i]) for k,p in pred.items()}})+"\n")
    (a.output_dir/"summary.md").write_text("# Full hidden + perturbation fusion\n\n"+"\n".join(f"- {k}: AUROC {v['auroc']:.4f}" for k,v in report["results"].items())+"\n");BASE.atomic_json(a.output_dir/"status.json",{"stage":"complete","completed":len(rows),"updated":time.time()});print(json.dumps(report["results"],indent=2))
if __name__=="__main__":main()
