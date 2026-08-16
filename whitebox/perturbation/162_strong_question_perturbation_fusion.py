#!/usr/bin/env python3
"""Fair same-fold comparison: script-150 layer ensemble + frozen perturbation fusion."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs"
BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
LAYERS=[8,10,12,14,16,18,20,22]

def bootstrap(y,a,b,seed,n=2000):
 from sklearn.metrics import roc_auc_score
 rng=np.random.default_rng(seed);neg=np.where(y==0)[0];pos=np.where(y==1)[0]
 ix=[np.r_[rng.choice(neg,len(neg),True),rng.choice(pos,len(pos),True)] for _ in range(n)]
 d=np.array([roc_auc_score(y[z],b[z])-roc_auc_score(y[z],a[z]) for z in ix])
 return {"delta_auroc":float(roc_auc_score(y,b)-roc_auc_score(y,a)),"paired_stratified_bootstrap_95ci":np.quantile(d,[.025,.975]).tolist(),"p_delta_le_0":float(np.mean(d<=0)),"replicates":n}

def main():
 p=argparse.ArgumentParser();p.add_argument("--suite-dir",type=Path,default=RUNS/"161_known_unknown_perturbation_suite_n500_confirm");p.add_argument("--exploratory-dir",type=Path,default=RUNS/"161_known_unknown_perturbation_suite");p.add_argument("--output-dir",type=Path,default=RUNS/"162_strong_question_perturbation_fusion_n500");p.add_argument("--seed",type=int,default=BASE.SEED);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 items=BASE.read_jsonl(a.suite_dir/"items.jsonl");old={x["key"] for x in BASE.read_jsonl(a.exploratory_dir/"items.jsonl")};keys=[x["key"] for x in items];y=np.array([x["known"] for x in items]);X=[]
 for k in keys:
  with np.load(BASE.QUESTION_CACHE/(k+".npz")) as z:X.append(z["hidden"][LAYERS].astype(np.float32))
 X=np.stack(X);splits=list(StratifiedKFold(5,shuffle=True,random_state=a.seed).split(X,y));layer_p=[]
 for li,layer in enumerate(LAYERS):
  pred=np.zeros(len(y))
  for tr,te in splits:
   m=LogisticRegression(C=.3,max_iter=3000,random_state=a.seed).fit(X[tr,li],y[tr]);pred[te]=m.predict_proba(X[te,li])[:,1]
  layer_p.append(pred);print(f"layer {layer}",flush=True)
 strong=np.mean(layer_p,axis=0);prior={r["key"]:r for r in BASE.read_jsonl(a.suite_dir/"predictions.jsonl")}
 basin=np.array([prior[k]["probabilities"]["basin_all"] for k in keys]);stability=np.array([prior[k]["probabilities"]["stability_hidden"] for k in keys])
 pred={"strong_question_layer_ensemble":strong,"strong_plus_basin":np.mean([strong,basin],axis=0),"strong_plus_basin_stability":np.mean([strong,basin,stability],axis=0)}
 result={name:{"overall":BASE.metrics(y,z,.5),"folds":[BASE.metrics(y[te],z[te],.5) for _,te in splits]} for name,z in pred.items()}
 fresh=np.array([i for i,k in enumerate(keys) if k not in old]);yf=y[fresh]
 fresh_result={name:BASE.metrics(yf,z[fresh],.5) for name,z in pred.items()}
 report={"n":len(y),"known":int(y.sum()),"protocol":"same fixed random stratified 5-fold; exact script-150 architecture: C=.3 independent LR at hidden indices 8,10,...,22 then mean probabilities; threshold .5","entity_leakage_warning":True,"layers":LAYERS,"results":result,"new_372":{"n":len(fresh),"known":int(yf.sum()),"results":fresh_result,"strong_vs_frozen_fusion":bootstrap(yf,strong[fresh],pred["strong_plus_basin_stability"][fresh],a.seed)}}
 BASE.atomic_json(a.output_dir/"evaluation.json",report)
 with (a.output_dir/"predictions.jsonl").open("w") as f:
  for i,k in enumerate(keys):f.write(json.dumps({"key":k,"known":int(y[i]),"is_new_372":bool(i in set(fresh)),"probabilities":{n:float(z[i]) for n,z in pred.items()}})+"\n")
 ranking=sorted((v["overall"]["auroc"],k) for k,v in result.items())[::-1];(a.output_dir/"summary.md").write_text("# Strong question + perturbation fusion\n\n"+"\n".join(f"- {k}: AUROC {v:.4f}" for v,k in ranking)+"\n")
 print(json.dumps({"overall":{k:v["overall"] for k,v in result.items()},"new_372":report["new_372"]},indent=2))
if __name__=="__main__":main()
