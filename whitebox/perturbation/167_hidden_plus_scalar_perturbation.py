#!/usr/bin/env python3
"""Add eight fixed local-perturbation scalars to the original question-layer ensemble."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
PICS=importlib.import_module("163_pics_keen_known_unknown")
RED=importlib.import_module("165_reduce_local_geometry")
GEO_DIR=Path(__file__).resolve().parent/"runs/164_ic_local_geometry_n500"
ORDER_DIR=Path(__file__).resolve().parent/"runs/161_known_unknown_perturbation_suite_n500_confirm"
OUT=Path(__file__).resolve().parent/"runs/167_hidden_plus_scalar_perturbation_n500"

def scalars(x):
    base=x[:198]
    # Tail schema from 164: odd norms[3], even norms[3], + response[3], - response[3].
    tail=x[1386:]
    return np.r_[base.mean(),base.std(),tail[:6]].astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--feature-dir",type=Path,default=GEO_DIR);ap.add_argument("--order-dir",type=Path,default=ORDER_DIR);ap.add_argument("--output-dir",type=Path,default=OUT);ap.add_argument("--seed",type=int,default=PICS.SEED);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    order=BASE.read_jsonl(a.order_dir/"predictions.jsonl");base={r["key"]:r for r in BASE.load_rows()[0]};rows=[base[z["key"]] for z in order if (a.feature_dir/"features"/(z["key"]+".npz")).exists()]
    y=np.asarray([r["known"] for r in rows]);S=np.stack([scalars(np.load(a.feature_dir/"features"/(r["key"]+".npz"))["local_geometry"]) for r in rows]);Q=np.stack([np.load(BASE.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][PICS.KEEN_LAYERS].astype(np.float32) for r in rows])
    outer=list(StratifiedKFold(5,shuffle=True,random_state=a.seed).split(S,y));pq=np.zeros(len(y));aug=np.zeros(len(y));scalar=np.zeros(len(y));weights=[]
    for fold,(tr,te) in enumerate(outer):
        seed=a.seed+fold;pq[te]=RED.qfit(Q,y,tr,te,seed)
        ss=StandardScaler().fit(S[tr]);sm=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(ss.transform(S[tr]),y[tr]);scalar[te]=sm.predict_proba(ss.transform(S[te]))[:,1]
        inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+1).split(S[tr],y[tr]));meta=np.zeros((len(tr),9))
        for it,iv in inner:meta[iv,0]=RED.qfit(Q,y,tr[it],tr[iv],seed);meta[iv,1:]=S[tr[iv]]
        test=np.c_[pq[te],S[te]];sc=StandardScaler().fit(meta);m=LogisticRegression(C=.03,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc.transform(meta),y[tr]);aug[te]=m.predict_proba(sc.transform(test))[:,1];weights.append(m.coef_[0].tolist())
    pred={"strong_question":pq,"scalar_8_only":scalar,"hidden_plus_scalar8":aug};report={"n":len(y),"added_features":["neutral_ic_mean","neutral_ic_std","odd_norm_eps02","odd_norm_eps05","odd_norm_eps10","even_norm_eps02","even_norm_eps05","even_norm_eps10"],"augmented_dim":9,"protocol":"random outer 5-fold; question probability is inner-OOF for meta training; scaler/meta fit training-only","entity_leakage_warning":True,"meta_coefficients":weights,"results":{k:{"overall":BASE.metrics(y,p,.5),"folds":[BASE.metrics(y[te],p[te],.5) for _,te in outer]} for k,p in pred.items()}}
    BASE.atomic_json(a.output_dir/"evaluation.json",report);BASE.atomic_json(a.output_dir/"config.json",{"seed":a.seed,"feature_whitelist":report["added_features"],"augmented_dim":9,"training_fold_only":True})
    with (a.output_dir/"predictions.jsonl").open("w") as f:
        for i,r in enumerate(rows):f.write(json.dumps({"key":r["key"],"known":int(y[i]),"probabilities":{k:float(p[i]) for k,p in pred.items()}})+"\n")
    (a.output_dir/"summary.md").write_text("# Hidden + 8 perturbation scalars\n\n"+"\n".join(f"- {k}: AUROC {v['overall']['auroc']:.4f}" for k,v in report["results"].items())+"\n");BASE.atomic_json(a.output_dir/"status.json",{"stage":"complete","completed":len(rows)});print(json.dumps(report,indent=2))
if __name__=="__main__":main()
