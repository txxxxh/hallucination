#!/usr/bin/env python3
"""Fixed top-64 perturbation probe and leakage-safe two-probability fusion."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
PICS=importlib.import_module("163_pics_keen_known_unknown")
GEO=importlib.import_module("164_ic_local_geometry")
RED=importlib.import_module("165_reduce_local_geometry")
OUT=Path(__file__).resolve().parent/"runs/166_fixed64_fusion"

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source-dir",type=Path,default=GEO.OUT);ap.add_argument("--order-dir",type=Path,default=GEO.SRC);ap.add_argument("--output-dir",type=Path,default=OUT);ap.add_argument("--seed",type=int,default=PICS.SEED);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    order=BASE.read_jsonl(a.order_dir/"predictions.jsonl");base={r["key"]:r for r in BASE.load_rows()[0]};rows=[base[z["key"]] for z in order if (a.source_dir/"features"/(z["key"]+".npz")).exists()]
    y=np.asarray([r["known"] for r in rows]);X=np.stack([np.load(a.source_dir/"features"/(r["key"]+".npz"))["local_geometry"] for r in rows]);Q=np.stack([np.load(BASE.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][PICS.KEEN_LAYERS].astype(np.float32) for r in rows])
    outer=list(StratifiedKFold(5,shuffle=True,random_state=a.seed).split(X,y));pq=np.zeros(len(y));p64=np.zeros(len(y));fusion=np.zeros(len(y));weights=[]
    for fold,(tr,te) in enumerate(outer):
        seed=a.seed+fold;pq[te]=RED.qfit(Q,y,tr,te,seed);p64[te]=RED.weighted_fit_predict(X,y,tr,te,64,seed)
        inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+1).split(X[tr],y[tr]));meta=np.zeros((len(tr),2))
        for it,iv in inner:
            meta[iv,0]=RED.qfit(Q,y,tr[it],tr[iv],seed);meta[iv,1]=RED.weighted_fit_predict(X,y,tr[it],tr[iv],64,seed)
        test=np.c_[pq[te],p64[te]];sc=StandardScaler().fit(meta);m=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc.transform(meta),y[tr]);fusion[te]=m.predict_proba(sc.transform(test))[:,1];weights.append(m.coef_[0].tolist())
    pred={"strong_question":pq,"weighted_64":p64,"nested_fusion_64":fusion};report={"n":len(y),"selected_dim":64,"fusion_input_dim":2,"selection":"absolute standardized LR weights on each training fold only","protocol":"random outer 5-fold; nested inner 3-fold fusion","entity_leakage_warning":True,"meta_coefficients":weights,"results":{k:{"overall":BASE.metrics(y,p,.5),"folds":[BASE.metrics(y[te],p[te],.5) for _,te in outer]} for k,p in pred.items()}}
    BASE.atomic_json(a.output_dir/"evaluation.json",report);BASE.atomic_json(a.output_dir/"config.json",{"seed":a.seed,"selected_dim":64,"fusion_input_dim":2,"feature_whitelist":["local_geometry"],"selection_training_fold_only":True})
    with (a.output_dir/"predictions.jsonl").open("w") as f:
        for i,r in enumerate(rows):f.write(json.dumps({"key":r["key"],"known":int(y[i]),"probabilities":{k:float(p[i]) for k,p in pred.items()}})+"\n")
    (a.output_dir/"summary.md").write_text("# Fixed top-64 fusion\n\n"+"\n".join(f"- {k}: AUROC {v['overall']['auroc']:.4f}" for k,v in report["results"].items())+"\n");BASE.atomic_json(a.output_dir/"status.json",{"stage":"complete","completed":len(rows)});print(json.dumps(report,indent=2))
if __name__=="__main__":main()
