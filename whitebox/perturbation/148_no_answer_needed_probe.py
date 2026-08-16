#!/usr/bin/env python3
"""No Answer Needed-style layer sweep for probe-derived knowledge labels.

X is the residual-stream state at the final prompt token, before answer
generation, from a question-only prompt.  Existing probes are used only to
construct y; none of their scores or thresholds is included in X.
"""
import importlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support,roc_auc_score
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"; CACHE=RUNS/"147_question_only_hidden_v3"

def fit_predict(x,y,tr,te,seed):
 m=LogisticRegression(C=1.0,max_iter=2000,random_state=seed).fit(x[tr],y[tr])
 return m.predict_proba(x[te])[:,1]

def metrics(y,p,t=.5):
 h=p>=t; pr,rc,f,_=precision_recall_fscore_support(y,h,labels=[0,1],zero_division=0)
 return {"auroc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,h)),
  "balanced_accuracy":float(balanced_accuracy_score(y,h)),"macro_f1":float(f1_score(y,h,average="macro")),
  "confusion_rows_unknown_known":confusion_matrix(y,h,labels=[0,1]).tolist(),
  "precision_unknown_known":pr.tolist(),"recall_unknown_known":rc.tolist(),"f1_unknown_known":f.tolist()}

def oof(x,y,groups=None,seeds=(42,),folds=3):
 ps=[]
 for seed in seeds:
  p=np.zeros(len(y)); cv=(StratifiedKFold(folds,shuffle=True,random_state=seed) if groups is None else StratifiedGroupKFold(folds,shuffle=True,random_state=seed))
  split=cv.split(x,y) if groups is None else cv.split(x,y,groups)
  for tr,te in split:p[te]=fit_predict(x,y,tr,te,seed)
  ps.append(p)
 return np.mean(ps,axis=0),[float(roc_auc_score(y,p)) for p in ps]

def main():
 probes={x["key"]:x for x in map(json.loads,(RUNS/"77_closedbook_fact_probe_results.jsonl").open())}
 manifest={x["key"]:x for x in map(json.loads,(RUNS/"76_closedbook_fact_probe_manifest.jsonl").open())}
 base=importlib.import_module("139_scientist_pairwise_swap_probes")
 rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item()); q=probes[k]
   rows.append({**manifest[k],"key":k,"y":int(q["n_discriminative_facts"]>=1 and q["binary_accuracy"]>.5 and q["pairwise_owner_accuracy"]>.5),"x":z["hidden"].astype(np.float32)})
 assert len(rows)==2894,len(rows)
 keys=[r["key"] for r in rows]; y=np.asarray([r["y"] for r in rows]); X=np.stack([r["x"] for r in rows]); groups=base.components(rows)
 # The paper sweeps every other layer for <10B models. HF hidden state 0 is the
 # embedding output and state i+1 is the output of transformer block i.
 results=[]
 for hi in range(0,X.shape[1],2):
  p,seeds=oof(X[:,hi],y,seeds=(42,),folds=3); z={"hidden_state_index":hi,"transformer_block_output":None if hi==0 else hi-1,**metrics(y,p),"per_seed_auroc":seeds};results.append(z)
  print(hi,z["auroc"],flush=True)
 results.sort(key=lambda z:z["auroc"],reverse=True); best=results[0]; hi=best["hidden_state_index"]
 # Stable estimates at the selected layer. Random CV matches the paper; grouped
 # CV additionally prevents two questions from the same identity component
 # appearing on opposite sides of a fold.
 p,seed_auc=oof(X[:,hi],y,seeds=(0,5,26,42,63),folds=5)
 pg,gseed_auc=oof(X[:,hi],y,groups=groups,seeds=(0,5,26,42,63),folds=5)
 report={"method":"No Answer Needed: question-final residual activation before generation + linear LogisticRegression",
  "features":"question-only chat prompt; no answer, profile, candidate, or probe-derived feature",
  "label":"known iff n_discriminative_facts>=1 and binary_accuracy>0.5 and pairwise_owner_accuracy>0.5; label only",
  "n":len(y),"n_unknown":int((y==0).sum()),"n_known":int((y==1).sum()),"layer_sweep_protocol":"every 2 hidden states, stratified 3-fold OOF (paper protocol)",
  "layer_sweep":results,"selected_hidden_state_index":hi,"selected_transformer_block_output":best["transformer_block_output"],
  "selected_repeated_stratified_5fold":{**metrics(y,p),"per_seed_auroc":seed_auc},
  "selected_repeated_identity_grouped_5fold":{**metrics(y,pg),"per_seed_auroc":gseed_auc}}
 (RUNS/"148_no_answer_needed_probe_report.json").write_text(json.dumps(report,indent=2)+"\n")
 with (RUNS/"148_no_answer_needed_probe_oof.jsonl").open("w") as f:
  for k,z,a,b in zip(keys,y,p,pg):f.write(json.dumps({"key":k,"probe_known":int(z),"prob_known":float(a),"grouped_prob_known":float(b)})+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="layer_sweep"},indent=2))

if __name__=="__main__":main()
