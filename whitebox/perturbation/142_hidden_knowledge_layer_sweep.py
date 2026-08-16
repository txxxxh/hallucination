#!/usr/bin/env python3
"""Leakage-free hidden-state layer sweep for probe-derived knowledge labels."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs";CACHE=RUNS/"141_scientist_all_trajectory_l8"

def score(y,p):
 h=p>=.5;pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1],zero_division=0)
 return {"auroc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,h)),"balanced_accuracy":float(balanced_accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),"confusion":confusion_matrix(y,h,labels=[0,1]).tolist(),"precision":pr.tolist(),"recall":rc.tolist(),"f1":f.tolist()}
def run(x,y,g,grouped,seed):
 cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)if grouped else StratifiedKFold(5,shuffle=True,random_state=seed);p=np.zeros(len(y));split=cv.split(x,y,g)if grouped else cv.split(x,y)
 for tr,te in split:
  d=min(16,len(tr)-1,x.shape[1]);m=make_pipeline(StandardScaler(),PCA(d,whiten=True,svd_solver="randomized",random_state=seed),LogisticRegression(C=.03,max_iter=3000,class_weight="balanced",solver="liblinear"));m.fit(x[tr],y[tr]);p[te]=m.predict_proba(x[te])[:,1]
 return p
def main():
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())};man={x["key"]:x for x in map(json.loads,(RUNS/"76_closedbook_fact_probe_manifest.jsonl").open())};rec={x["key"]:x for x in map(json.loads,(ROOT/"tool_gate_correctness_names_llama31_8b"/"records.jsonl").open())};base=importlib.import_module("139_scientist_pairwise_swap_probes");rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   k=str(z["key"].item())
   if not rec[k].get("parse_valid",True):continue
   p=probes[k];rows.append({**man[k],"key":k,"y":int(p["n_discriminative_facts"]>=1 and p["binary_accuracy"]>.5 and p["pairwise_owner_accuracy"]>.5),"layers":z["layers"].copy(),"last":z["last"].astype(np.float32),"mean":z["mean"].astype(np.float32),"last_stats":z["last_stats"].astype(np.float32),"mean_stats":z["mean_stats"].astype(np.float32)})
 assert len(rows)==2894,len(rows);y=np.array([r["y"]for r in rows]);g=base.components(rows);layers=rows[0]["layers"].tolist();out=[]
 for view in("last","mean","last_stats","mean_stats"):
  z=np.stack([r[view]for r in rows])
  for li,layer in enumerate(layers):
   item={"view":view,"layer":int(layer)}
   for protocol,grouped in(("stratified",False),("identity_grouped",True)):
    ps=[run(z[:,li],y,g,grouped,s)for s in(42,43,44)];item[protocol]=score(y,np.mean(ps,0));item[protocol]["per_seed_auroc"]=[float(roc_auc_score(y,p))for p in ps]
   out.append(item);print(view,layer,item["stratified"]["auroc"],item["identity_grouped"]["auroc"],flush=True)
 out.sort(key=lambda q:q["stratified"]["auroc"],reverse=True);report={"protocol":"probe data used only to define y; X is unperturbed main-task answer hidden state; fixed PCA16 + balanced LR C=.03; 3x5 OOF","n":len(y),"components":len(set(g)),"labels":{"0":"probe_unknown","1":"probe_known"},"warning":"layer ranking uses stratified OOF and is diagnostic, not identity-transfer evidence","results":out};path=RUNS/"142_hidden_knowledge_layer_sweep.json";path.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"out":str(path),"top":out[:5]},indent=2))
if __name__=="__main__":main()
