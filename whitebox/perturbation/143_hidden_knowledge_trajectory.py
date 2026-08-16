#!/usr/bin/env python3
"""Cross-layer hidden trajectory probes; probe outputs are labels only."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs";CACHE=RUNS/"141_scientist_all_trajectory_l8"
def blocks(rows,kind):
 last=np.stack([r["last"]for r in rows]);mean=np.stack([r["mean"]for r in rows]);ls=np.stack([r["last_stats"]for r in rows]);ms=np.stack([r["mean_stats"]for r in rows])
 if kind=="last_all":return[last[:,i]for i in range(8)]
 if kind=="last_mean_all":return[last[:,i]for i in range(8)]+[mean[:,i]for i in range(8)]
 if kind=="late_last_mean":return[last[:,i]for i in range(3,8)]+[mean[:,i]for i in range(3,8)]
 if kind=="deltas":return[last[:,i]-last[:,i-1]for i in range(1,8)]
 if kind=="stats_trajectory":return[np.c_[ls.reshape(len(rows),-1),ms.reshape(len(rows),-1)]]
 raise KeyError(kind)
def fold_features(bs,tr,te,seed):
 a=[];b=[]
 for x in bs:
  sc=StandardScaler().fit(x[tr]);u=sc.transform(x[tr]);v=sc.transform(x[te]);d=min(8,u.shape[1],len(tr)-1);pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(u);a.append(pc.transform(u));b.append(pc.transform(v))
 return np.concatenate(a,1),np.concatenate(b,1)
def met(y,p):
 h=p>=.5;return{"auroc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,h)),"balanced_accuracy":float(balanced_accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),"confusion":confusion_matrix(y,h,labels=[0,1]).tolist()}
def main():
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())};man={x["key"]:x for x in map(json.loads,(RUNS/"76_closedbook_fact_probe_manifest.jsonl").open())};rec={x["key"]:x for x in map(json.loads,(ROOT/"tool_gate_correctness_names_llama31_8b"/"records.jsonl").open())};base=importlib.import_module("139_scientist_pairwise_swap_probes");rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   k=str(z["key"].item())
   if not rec[k].get("parse_valid",True):continue
   q=probes[k];rows.append({**man[k],"y":int(q["n_discriminative_facts"]>=1 and q["binary_accuracy"]>.5 and q["pairwise_owner_accuracy"]>.5),**{n:z[n].astype(np.float32)for n in("last","mean","last_stats","mean_stats")}})
 y=np.array([r["y"]for r in rows]);g=base.components(rows);results=[]
 for kind in("last_all","last_mean_all","late_last_mean","deltas","stats_trajectory"):
  bs=blocks(rows,kind)
  for model in("linear","hist"):
   z={"features":kind,"model":model}
   for name,grouped in(("stratified",False),("identity_grouped",True)):
    ps=[]
    for seed in(42,43,44):
     p=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)if grouped else StratifiedKFold(5,shuffle=True,random_state=seed);spl=cv.split(np.zeros(len(y)),y,g)if grouped else cv.split(np.zeros(len(y)),y)
     for fold,(tr,te)in enumerate(spl):
      a,b=fold_features(bs,tr,te,seed+fold);m=(LogisticRegression(C=.03,max_iter=3000,class_weight="balanced",solver="liblinear")if model=="linear" else HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=5,learning_rate=.05,random_state=seed));m.fit(a,y[tr]);p[te]=m.predict_proba(b)[:,1]
     ps.append(p)
    z[name]=met(y,np.mean(ps,0));z[name]["per_seed_auroc"]=[float(roc_auc_score(y,p))for p in ps]
   results.append(z);print(kind,model,z["stratified"]["auroc"],z["identity_grouped"]["auroc"],flush=True)
 results.sort(key=lambda x:x["stratified"]["auroc"],reverse=True);report={"protocol":"probe only defines y; independent unperturbed hidden trajectory; fold-local scaler/PCA; 3x5 OOF","n":len(y),"components":len(set(g)),"results":results};out=RUNS/"143_hidden_knowledge_trajectory.json";out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"out":str(out),"top":results[:3]},indent=2))
if __name__=="__main__":main()
