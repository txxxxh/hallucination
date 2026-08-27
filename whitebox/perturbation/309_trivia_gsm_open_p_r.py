#!/usr/bin/env python3
"""Strict-open P+generated-R evaluation for TriviaQA and GSM8K."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pmod=importlib.import_module("297_multibench_pred_only");RUNS=pmod.RUNS;OUT=RUNS/"309_trivia_gsm_open_p_r";SEEDS=(42,43,44,45,46,47)
def fixed(x,n=6):x=np.asarray(x,np.float32);return np.pad(x[:n],(0,max(0,n-len(x))))
def ch(x):
 x=fixed(x);u=x[0]-x[1:];z=abs(float(x[0]))+1e-6;return np.r_[x[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(x):x=fixed(x);return np.r_[x[0],x[0]-x[1:]]
def wd(h,x):
 x=np.asarray(x,np.float32);u=x[0]-x[1:];d=np.asarray(h,np.float32)[1:]-np.asarray(h,np.float32)[0];n=min(len(u),len(d));return(d[:n]*u[:n,None]).sum(0)/(np.abs(u[:n]).sum()+1e-9)
def proj(v,tr,te,d,seed):
 s=StandardScaler().fit(v[tr]);a,b=s.transform(v[tr]),s.transform(v[te])
 if d is not None:p=PCA(min(d,len(tr)-1,a.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=p.transform(a),p.transform(b)
 return a,b
def design(blocks,dims,tr,te,seed):
 z=[proj(v,tr,te,d,seed)for v,d in zip(blocks,dims)];return np.concatenate([q[0]for q in z],1),np.concatenate([q[1]for q in z],1)
def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p))}
def evaluate(ds):
 cache=RUNS/"297_multibench_pred_only"/ds;rows=[]
 for fp in sorted(cache.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   p=z["stage1_pred"].astype(np.float32);q=z["stage2_pred"].astype(np.float32);h=z["pred_hidden"].astype(np.float32);rows.append((str(z["key"].item()),1-int(z["correct"]),np.r_[ch(p),ch2(q),p[0]-q[0]],h[0],wd(h,p),z["layer14"].astype(np.float32)))
 expected={"trivia":1000,"gsm8k":942,"drop":1000}[ds]
 if len(rows)!=expected:raise RuntimeError(f"{ds} pred-only cache {len(rows)}/{expected}")
 keys=[r[0]for r in rows];y=np.array([r[1]for r in rows]);pb=[np.stack([r[i]for r in rows])for i in range(2,6)];pd=[None,16,16,96]
 sv=torch.load(RUNS/"293_multibench_p_aiersilan_fusion"/"hidden_states"/f"llama3.1-8b__{ds}.pt",map_location="cpu");hm={k:sv["hidden_states"][i,14].float().numpy()for i,k in enumerate(sv["keys"])};rgen=np.stack([hm[k]for k in keys]);cfg={"P_pred_only":(pb,pd),"R_generated_only":([rgen],[96]),"P_pred_only_plus_R_generated":(pb+[rgen],pd+[96])};per={k:[]for k in cfg}
 for seed in SEEDS:
  tr,te=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed))
  for name,(blocks,dims)in cfg.items():
   a,b=design(blocks,dims,tr,te,seed);s=LogisticRegression(C=.01,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(a,y[tr]).predict_proba(b)[:,1];per[name].append({"seed":seed,**met(y[te],s)})
 summary={n:{m+"_mean":float(np.mean([x[m]for x in a]))for m in("auroc","auprc")}|{m+"_std":float(np.std([x[m]for x in a]))for m in("auroc","auprc")}for n,a in per.items()};report={"dataset":ds,"n":len(y),"protocol":"strict open pred-only P + generated-only layer14-last R; 80/20 seeds42-47; PCA16/96; LR C=.01","summary":summary,"per_seed":per};OUT.mkdir(parents=True,exist_ok=True);(OUT/f"{ds}_report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument("datasets",nargs="+",choices=("trivia","gsm8k","drop"));z=a.parse_args()
 for ds in z.datasets:evaluate(ds)
if __name__=="__main__":main()
