#!/usr/bin/env python3
"""Transfer the fixed Scientist paired-Aiersilan P+R detector to three datasets."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
old=importlib.import_module("293_multibench_p_aiersilan_fusion");RUNS=old.RUNS;ROOT=RUNS/"305_multibench_paired_aiersilan";OUT=ROOT/"evaluation";SEEDS=(42,43,44,45,46,47)
def transform(blocks,tr,te,dims,seed):
 a,b=[],[]
 for v,d in zip(blocks,dims):
  s=StandardScaler().fit(v[tr]);x,z=s.transform(v[tr]),s.transform(v[te])
  if d:p=PCA(min(d,len(tr)-1,x.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(x);x,z=p.transform(x),p.transform(z)
  a.append(x);b.append(z)
 return np.concatenate(a,1),np.concatenate(b,1)
def met(y,s):return{"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s))}
def evaluate(ds):
 keys,correct,pb,_=old.load_p(ds);y=1-correct;chosen_file=RUNS/"293_multibench_p_aiersilan_fusion/hidden_states"/f"llama3.1-8b__{ds}.pt";sv=torch.load(chosen_file,map_location="cpu");m={k:sv["hidden_states"][i,14].float().numpy()for i,k in enumerate(sv["keys"])};ch=np.stack([m[k]for k in keys]);other=np.stack([np.load(ROOT/"alternative"/ds/f"{k}.npy").astype(np.float32)for k in keys]);diff=other-ch;p_dims=[None,16,16,16,16,96];cfg={"P_only":(pb,p_dims),"R_difference_only":([diff],[96]),"P_plus_chosen":(pb+[ch],p_dims+[96]),"P_plus_difference":(pb+[diff],p_dims+[96])};per={k:[]for k in cfg};pred=[]
 for seed in SEEDS:
  tr,te=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed))
  for name,(blocks,dims)in cfg.items():
   x,z=transform(blocks,tr,te,dims,seed);score=LogisticRegression(C=.01,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(x,y[tr]).predict_proba(z)[:,1];mm=met(y[te],score);per[name].append({"seed":seed,**mm});pred += [{"dataset":ds,"seed":seed,"method":name,"key":keys[i],"error":int(y[i]),"score":float(s)}for i,s in zip(te,score)]
 summary={n:{k+"_mean":float(np.mean([r[k]for r in v]))for k in("auroc","auprc")}|{k+"_std":float(np.std([r[k]for r in v]))for k in("auroc","auprc")}for n,v in per.items()};return{"dataset":ds,"n":len(y),"errors":int(y.sum()),"protocol":"fixed Scientist winner: stratified outer 80/20 seeds42-47; P hidden PCA16x4 + layer PCA96; R layer14 alternative-minus-chosen PCA96; balanced LR C=.01; no target-dataset tuning","summary":summary,"per_seed":per},pred
def main():
 p=argparse.ArgumentParser();p.add_argument("datasets",nargs="+",choices=("trivia","gsm8k","drop"));a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 for ds in a.datasets:
  r,pred=evaluate(ds);(OUT/f"{ds}_report.json").write_text(json.dumps(r,indent=2)+"\n");(OUT/f"{ds}_predictions.jsonl").write_text("".join(json.dumps(x)+"\n"for x in pred));print(json.dumps(r,indent=2),flush=True)
if __name__=="__main__":main()
