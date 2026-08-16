#!/usr/bin/env python3
"""Predict probe-derived knowledge labels from independent main-task hidden states only."""
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs"

def transform(x,tr,te,d,seed):
 sc=StandardScaler().fit(x[tr]);a=sc.transform(x[tr]);b=sc.transform(x[te]);d=min(d,a.shape[1],len(tr)-1)
 pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(a);return pc.transform(a),pc.transform(b)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cache",type=Path,default=RUNS/"141_scientist_all_trajectory_l8");a=ap.parse_args()
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())};man={x["key"]:x for x in map(json.loads,(RUNS/"76_closedbook_fact_probe_manifest.jsonl").open())};base=importlib.import_module("139_scientist_pairwise_swap_probes");rows=[]
 for fp in sorted(a.cache.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   k=str(z["key"].item());p=probes[k];known=int(p["n_discriminative_facts"]>=1 and p["binary_accuracy"]>.5 and p["pairwise_owner_accuracy"]>.5)
   rows.append({**man[k],"key":k,"known":known,"layers":z["layers"].copy(),"last":z["last"].astype(np.float32),"mean":z["mean"].astype(np.float32),"last_stats":z["last_stats"].astype(np.float32),"mean_stats":z["mean_stats"].astype(np.float32)})
 if len(rows)!=2894:raise RuntimeError(f"incomplete cache {len(rows)}/2894")
 y=np.array([r["known"]for r in rows]);g=base.components(rows);layers=rows[0]["layers"].tolist();views={n:np.stack([r[n]for r in rows])for n in("last","mean","last_stats","mean_stats")};configs=[]
 for view,x in views.items():
  for li in range(x.shape[1]):
   for d in((2,4,8,16,32,48)if x.shape[2]>11 else(2,4,8,11)):
    for C in(.003,.01,.03,.1,.3):configs.append((view,li,d,C))
 reports=[];allp=[]
 for seed in(42,43,44):
  pred=np.zeros(len(y));choices=[];outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te)in enumerate(outer.split(np.zeros(len(y)),y,g),1):
   inner=StratifiedGroupKFold(3,shuffle=True,random_state=seed+100+fold);best=None
   for cfg in configs:
    view,li,d,C=cfg;x=views[view][:,li];q=np.zeros(len(tr));yt=y[tr];gt=g[tr]
    for itr,iva in inner.split(np.zeros(len(tr)),yt,gt):
     xa,xv=transform(x[tr],itr,iva,d,seed+fold);m=LogisticRegression(C=C,max_iter=3000,class_weight="balanced",solver="liblinear").fit(xa,yt[itr]);q[iva]=m.predict_proba(xv)[:,list(m.classes_).index(1)]
    score=roc_auc_score(yt,q);z=(score,-d,-C,cfg)
    if best is None or z>best:best=z
   view,li,d,C=best[-1];xa,xv=transform(views[view][:,li],tr,te,d,seed+fold);m=LogisticRegression(C=C,max_iter=3000,class_weight="balanced",solver="liblinear").fit(xa,y[tr]);pred[te]=m.predict_proba(xv)[:,list(m.classes_).index(1)];choices.append({"fold":fold,"train_n":len(tr),"test_n":len(te),"view":view,"layer":int(layers[li]),"pca":d,"C":C,"inner_auroc":best[0]})
  allp.append(pred);reports.append({"seed":seed,"auroc":roc_auc_score(y,pred),"choices":choices})
 p=np.mean(allp,0);h=p>=.5;pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1],zero_division=0);report={"protocol":"probe labels only as y; X independent unperturbed main-task hidden; nested candidate-component grouped OOF","n":len(y),"components":len(set(g)),"layers":layers,"accuracy":accuracy_score(y,h),"balanced_accuracy":balanced_accuracy_score(y,h),"auroc":roc_auc_score(y,p),"macro_f1":f1_score(y,h,average="macro"),"confusion_rows_unknown_known":confusion_matrix(y,h,labels=[0,1]).tolist(),"precision_unknown_known":pr.tolist(),"recall_unknown_known":rc.tolist(),"f1_unknown_known":f.tolist(),"per_seed":reports}
 out=RUNS/"141_hidden_knowledge_probe_report.json";out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
