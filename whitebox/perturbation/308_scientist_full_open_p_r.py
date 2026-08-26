#!/usr/bin/env python3
"""Strict open Scientist-full: pred-only P plus generated-only R."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pmod=importlib.import_module("296_scientist_known_pred_only")
full=importlib.import_module("286_aiersilan_full_scientist")
RUNS=pmod.RUNS;CACHE=RUNS/"308_scientist_full_open_p_r"/"pred_only";OUT=RUNS/"308_scientist_full_open_p_r"

def jobs():
 return [(r["key"],r["group"],int(r["correct"]),r["raw"]["prompt"],r["pred"],r["pred"])for r in full.rows()]

def fixed(x,n=6):
 x=np.asarray(x,np.float32);return np.pad(x[:n],(0,max(0,n-len(x))))
def ch(x):
 x=fixed(x);u=x[0]-x[1:];z=abs(float(x[0]))+1e-6;return np.r_[x[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(x):x=fixed(x);return np.r_[x[0],x[0]-x[1:]]
def wd(h,x):
 x=fixed(x);u=x[0]-x[1:];d=np.asarray(h,np.float32)[1:]-np.asarray(h,np.float32)[0];return(d[:len(u)]*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def proj(v,tr,te,d,seed):
 s=StandardScaler().fit(v[tr]);a,b=s.transform(v[tr]),s.transform(v[te])
 if d is not None:p=PCA(min(d,len(tr)-1,a.shape[1]),whiten=True,svd_solver="randomized",random_state=seed).fit(a);a,b=p.transform(a),p.transform(b)
 return a,b
def design(blocks,dims,tr,te,seed):
 z=[proj(v,tr,te,d,seed)for v,d in zip(blocks,dims)];return np.concatenate([x[0]for x in z],1),np.concatenate([x[1]for x in z],1)
def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p))}

def collect(args):
 pmod.CACHE=CACHE;pmod.jobs=jobs
 pmod.score=importlib.import_module("297_resume_scientist_pred_only_fast").score
 pmod.collect(args)

def evaluate():
 rows=jobs();keys=[r[0]for r in rows];groups=np.array([r[1]for r in rows]);correct=np.array([r[2]for r in rows]);y=1-correct;data=[]
 for key in keys:
  with np.load(CACHE/f"{key}.npz",allow_pickle=True)as z:
   p=z["stage1_pred"].astype(np.float32);q=z["stage2_pred"].astype(np.float32);h=z["pred_hidden"].astype(np.float32);data.append((np.r_[ch(p),ch2(q),p[0]-q[0]],h[0],wd(h,p),z["layer14"].astype(np.float32)))
 pb=[np.stack([r[i]for r in data])for i in range(4)];pd=[None,16,16,96]
 saved=torch.load(RUNS/"286_aiersilan_full_scientist/hidden_states.pt",map_location="cpu");hm={k:saved["hidden_states"][i,14].float().numpy()for i,k in enumerate(saved["keys"])};rgen=np.stack([hm[k]for k in keys]);cfg={"P_pred_only":(pb,pd),"R_generated_only":([rgen],[96]),"P_pred_only_plus_R_generated":(pb+[rgen],pd+[96])};per={k:[]for k in cfg};pred=[]
 for seed in(42,43,44,45,46,47):
  tr,te=map(np.asarray,train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed))
  for name,(blocks,dims)in cfg.items():
   a,b=design(blocks,dims,tr,te,seed);s=LogisticRegression(C=.01,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(a,y[tr]).predict_proba(b)[:,1];per[name].append({"seed":seed,**met(y[te],s)});pred.extend({"seed":seed,"method":name,"key":keys[i],"error":int(y[i]),"score":float(v)}for i,v in zip(te,s))
  print("evaluate",seed,flush=True)
 summary={n:{m+"_mean":float(np.mean([x[m]for x in a]))for m in("auroc","auprc")}|{m+"_std":float(np.std([x[m]for x in a]))for m in("auroc","auprc")}for n,a in per.items()};report={"dataset":"Scientist full parse-valid","n":len(y),"errors":int(y.sum()),"protocol":"strict open: alternative never loaded by P or R; pred-only two-stage exact P; generated-answer layer14-last R; stratified 80/20 seeds42-47; fold-local PCA; balanced LR C=.01","summary":summary,"per_seed":per};OUT.mkdir(parents=True,exist_ok=True);(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n");(OUT/"predictions.jsonl").write_text("".join(json.dumps(x)+"\n"for x in pred));print(json.dumps(report,indent=2))

def main():
 a=argparse.ArgumentParser();a.add_argument("stage",choices=("collect","evaluate","all"));a.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");a.add_argument("--batch",type=int,default=32);a.add_argument("--resume",action="store_true");z=a.parse_args()
 if z.stage in("collect","all"):collect(z)
 if z.stage in("evaluate","all"):evaluate()
if __name__=="__main__":main()
