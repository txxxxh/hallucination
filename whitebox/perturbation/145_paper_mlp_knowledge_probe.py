#!/usr/bin/env python3
"""LLMsKnow-style layer/token sweep for probe-derived knowledge labels."""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs";CACHE=RUNS/"144_paper_exact_mlp"
def met(y,p):
 h=p>=.5;return{"auroc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,h)),"balanced_accuracy":float(balanced_accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),"confusion":confusion_matrix(y,h,labels=[0,1]).tolist()}
def fit(x,y,tr,te,seed):
 m=LogisticRegression(random_state=seed,max_iter=1000).fit(x[tr],y[tr]);return m.predict_proba(x[te])[:,1]
def main():
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())};man={x["key"]:x for x in map(json.loads,(RUNS/"76_closedbook_fact_probe_manifest.jsonl").open())};base=importlib.import_module("139_scientist_pairwise_swap_probes");rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   k=str(z["key"].item());q=probes[k];rows.append({**man[k],"key":k,"y":int(q["n_discriminative_facts"]>=1 and q["binary_accuracy"]>.5 and q["pairwise_owner_accuracy"]>.5),"x":z["mlp"].astype(np.float32)})
 assert len(rows)==2894,len(rows);y=np.array([r["y"]for r in rows]);g=base.components(rows);X=np.stack([r["x"]for r in rows]);positions=["before_first","first","last","after_last"];results=[]
 for li in range(X.shape[1]):
  for pi,pos in enumerate(positions):
   x=X[:,li,pi];vals=[]
   for seed in(0,5,26,42,63):
    tr,te=train_test_split(np.arange(len(y)),test_size=min(1000,round(.2*len(y))),random_state=seed,stratify=y);p=fit(x,y,tr,te,seed);vals.append(met(y[te],p))
   z={"layer":li,"token":pos,"paper_split_mean_auroc":float(np.mean([v["auroc"]for v in vals])),"paper_split_std_auroc":float(np.std([v["auroc"]for v in vals])),"paper_split":vals};results.append(z);print(li,pos,z["paper_split_mean_auroc"],flush=True)
 results.sort(key=lambda z:z["paper_split_mean_auroc"],reverse=True)
 for z in results[:10]:
  x=X[:,z["layer"],positions.index(z["token"])];ps=[]
  for seed in(42,43,44):
   p=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
   for tr,te in cv.split(x,y,g):p[te]=fit(x,y,tr,te,seed)
   ps.append(p)
  z["identity_grouped_oof"]=met(y,np.mean(ps,0));z["identity_grouped_oof"]["per_seed_auroc"]=[float(roc_auc_score(y,p))for p in ps]
 out=RUNS/"145_paper_mlp_knowledge_probe.json";out.write_text(json.dumps({"protocol":"paper-faithful MLP output + exact-answer token + raw LogisticRegression; probe only defines y","n":len(y),"components":len(set(g)),"results":results},indent=2)+"\n");print(json.dumps({"out":str(out),"top":results[:10]},indent=2))
if __name__=="__main__":main()
