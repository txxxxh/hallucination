#!/usr/bin/env python3
"""Training-fold-only reduction of the 164 local-geometry features to <200 dims."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
PICS=importlib.import_module("163_pics_keen_known_unknown")
GEO=importlib.import_module("164_ic_local_geometry")
DEFAULT=Path(__file__).resolve().parent/"runs/165_reduced_local_geometry"
KS=(64,128,192)

def structural(x):
    """Fixed 156-D representation: 8 layers, epsilon-averaged odd/even, summaries."""
    base=x[:198].reshape(33,6)
    odd=x[198:792].reshape(3,33,6)
    even=x[792:1386].reshape(3,33,6)
    tail=x[1386:]
    ll=np.asarray(PICS.KEEN_LAYERS)
    return np.r_[base[ll].ravel(),odd[:,ll].mean(0).ravel(),even[:,ll].mean(0).ravel(),tail]

def weighted_fit_predict(X,y,tr,te,k,seed,return_idx=False):
    """Rank coordinates using standardized training-only LR weights, then refit top-k."""
    sc=StandardScaler().fit(X[tr]);xa=sc.transform(X[tr]);xb=sc.transform(X[te])
    ranker=LogisticRegression(C=.03,class_weight="balanced",max_iter=4000,solver="liblinear",random_state=seed).fit(xa,y[tr])
    idx=np.argsort(np.abs(ranker.coef_[0]))[-min(k,X.shape[1]):]
    model=LogisticRegression(C=.3,class_weight="balanced",max_iter=4000,random_state=seed).fit(xa[:,idx],y[tr])
    p=model.predict_proba(xb[:,idx])[:,1]
    return (p,idx) if return_idx else p

def choose_k(X,y,tr,seed):
    inner=StratifiedKFold(3,shuffle=True,random_state=seed)
    scores={k:[] for k in KS}
    for it,iv in inner.split(X[tr],y[tr]):
        for k in KS:
            p=weighted_fit_predict(X,y,tr[it],tr[iv],k,seed)
            scores[k].append(roc_auc_score(y[tr[iv]],p))
    means={k:float(np.mean(v)) for k,v in scores.items()}
    return max(KS,key=lambda k:(means[k],-k)),means

def simple_fit(X,y,tr,te,seed):
    sc=StandardScaler().fit(X[tr]);m=LogisticRegression(C=.3,class_weight="balanced",max_iter=4000,random_state=seed).fit(sc.transform(X[tr]),y[tr]);return m.predict_proba(sc.transform(X[te]))[:,1]

def qfit(Q,y,tr,te,seed): return PICS.fit_q(Q,y,tr,te,seed)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source-dir",type=Path,default=GEO.OUT);ap.add_argument("--order-dir",type=Path,default=GEO.SRC);ap.add_argument("--output-dir",type=Path,default=DEFAULT);ap.add_argument("--seed",type=int,default=PICS.SEED);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    order=BASE.read_jsonl(a.order_dir/"predictions.jsonl");base={r["key"]:r for r in BASE.load_rows()[0]};rows=[base[z["key"]] for z in order if (a.source_dir/"features"/(z["key"]+".npz")).exists()]
    y=np.asarray([r["known"] for r in rows]);X=np.stack([np.load(a.source_dir/"features"/(r["key"]+".npz"))["local_geometry"] for r in rows]);S=np.stack([structural(x) for x in X]);Q=np.stack([np.load(BASE.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][PICS.KEEN_LAYERS].astype(np.float32) for r in rows])
    outer=list(StratifiedKFold(5,shuffle=True,random_state=a.seed).split(X,y));pred={k:np.zeros(len(y)) for k in ("strong_question","structural_156","weighted_128","weighted_topk","nested_fusion_128","nested_fusion")};chosen=[];selected=[]
    for fold,(tr,te) in enumerate(outer):
        seed=a.seed+fold;k,cv=choose_k(X,y,tr,seed+100);chosen.append({"fold":fold,"k":k,"inner_auroc":cv})
        pred["strong_question"][te]=qfit(Q,y,tr,te,seed);pred["structural_156"][te]=simple_fit(S,y,tr,te,seed);pred["weighted_128"][te]=weighted_fit_predict(X,y,tr,te,128,seed);pred["weighted_topk"][te],idx=weighted_fit_predict(X,y,tr,te,k,seed,True);selected.append(idx.tolist())
        inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+1).split(X[tr],y[tr]));meta=np.zeros((len(tr),2));meta128=np.zeros((len(tr),2))
        for it,iv in inner:
            ik,_=choose_k(X,y,tr[it],seed+200)
            qp=qfit(Q,y,tr[it],tr[iv],seed);meta[iv,0]=qp;meta[iv,1]=weighted_fit_predict(X,y,tr[it],tr[iv],ik,seed);meta128[iv,0]=qp;meta128[iv,1]=weighted_fit_predict(X,y,tr[it],tr[iv],128,seed)
        test=np.c_[pred["strong_question"][te],pred["weighted_topk"][te]];sc=StandardScaler().fit(meta);m=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc.transform(meta),y[tr]);pred["nested_fusion"][te]=m.predict_proba(sc.transform(test))[:,1]
        test128=np.c_[pred["strong_question"][te],pred["weighted_128"][te]];sc128=StandardScaler().fit(meta128);m128=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed).fit(sc128.transform(meta128),y[tr]);pred["nested_fusion_128"][te]=m128.predict_proba(sc128.transform(test128))[:,1]
    report={"n":len(y),"raw_dim":int(X.shape[1]),"structural_dim":int(S.shape[1]),"max_selected_dim":max(KS),"selection":"absolute standardized LR weights; k chosen by inner 3-fold only","entity_leakage_warning":True,"chosen_k":chosen,"results":{name:{"overall":BASE.metrics(y,p,.5),"folds":[BASE.metrics(y[te],p[te],.5) for _,te in outer]} for name,p in pred.items()}}
    BASE.atomic_json(a.output_dir/"evaluation.json",report);BASE.atomic_json(a.output_dir/"config.json",{"seed":a.seed,"feature_whitelist":["local_geometry"],"candidate_k":KS,"structural_dim":int(S.shape[1]),"selection_training_fold_only":True})
    with (a.output_dir/"predictions.jsonl").open("w") as f:
        for i,r in enumerate(rows):f.write(json.dumps({"key":r["key"],"known":int(y[i]),"probabilities":{k:float(v[i]) for k,v in pred.items()}})+"\n")
    (a.output_dir/"summary.md").write_text("# Reduced local geometry\n\n"+"\n".join(f"- {k}: AUROC {v['overall']['auroc']:.4f}" for k,v in report["results"].items())+"\n");BASE.atomic_json(a.output_dir/"status.json",{"stage":"complete","completed":len(rows)});print(json.dumps(report,indent=2))
if __name__=="__main__":main()
