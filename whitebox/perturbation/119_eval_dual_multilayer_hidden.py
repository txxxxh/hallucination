#!/usr/bin/env python3
"""Evaluate compact dual-candidate hidden trajectories across six layers."""
from __future__ import annotations
import importlib, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"

def channel(score,u):
 scale=abs(float(score))+1e-6
 return np.r_[score,u,u/scale,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]

def wd(h,u):
 d=h[1:].astype(np.float32)-h[0].astype(np.float32)
 return (d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)

def cosine(a,b):
 return np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9)

def met(y,p):
 return {"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),
         "balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}

def main():
 mod=importlib.import_module("101_fuse_sota_trajectory")
 keys,groups,y,margin,old_hidden,_,_=mod.load_response("scientist")
 _,_,last,_=mod.trajectory("scientist",keys)
 margin=np.c_[margin[:,:5],margin[:,10:]]
 separate={}; multi={}
 for fp in (RUNS/"112_separate_candidate_top5").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item()); separate[k]=np.r_[channel(z["pred_scores"][0],z["pred_u"]),channel(z["other_scores"][0],z["other_u"])]
 for fp in (RUNS/"118_dual_candidate_multilayer_top5").glob("*.npz"):
  with np.load(fp,allow_pickle=True) as z:
   k=str(z["key"].item()); ph=z["pred_hidden"].astype(np.float32); oh=z["other_hidden"].astype(np.float32)
   pu=z["pred_u"].astype(np.float32); ou=z["other_u"].astype(np.float32)
   multi[k]=[(ph[0,j],wd(ph[:,j],pu),oh[0,j],wd(oh[:,j],ou)) for j in range(ph.shape[1])]
 missing=[k for k in keys if k not in separate or k not in multi]
 if missing: raise RuntimeError(f"missing {len(missing)} rows")
 separate=np.stack([separate[k] for k in keys]).astype(np.float32)
 blocks=[]; geom=[]
 for j in range(6):
  p0=np.stack([multi[k][j][0] for k in keys]); pd=np.stack([multi[k][j][1] for k in keys])
  o0=np.stack([multi[k][j][2] for k in keys]); od=np.stack([multi[k][j][3] for k in keys])
  blocks.append(np.c_[p0,pd,o0,od])
  geom.append(np.c_[cosine(p0,o0),cosine(pd,od),np.log((np.linalg.norm(pd,axis=1)+1e-6)/(np.linalg.norm(od,axis=1)+1e-6))])
 geometry=np.concatenate(geom,1).astype(np.float32)
 variants=["margin_old_full","separate_old_full","separate_multilayer48","separate_multilayer48_geometry",
           "separate_multilayer48_plus_layer14","separate_multilayer48_geometry_plus_layer14"]
 scores={v:[] for v in variants}
 for seed in (42,43,44):
  prediction={v:np.zeros(len(y)) for v in variants}
  cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(cv.split(margin,y,groups),1):
   def scale(x):
    s=StandardScaler().fit(x[tr]); return s.transform(x[tr]),s.transform(x[te])
   mt,mv=scale(margin); st,sv=scale(separate); gt,gv=scale(geometry)
   oldt,oldv=[],[]
   for x in old_hidden:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver="randomized",random_state=seed).fit(q)
    oldt.append(pc.transform(q)); oldv.append(pc.transform(s.transform(x[te])))
   x=last[:,3]; s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(48,whiten=True,svd_solver="randomized",random_state=seed).fit(q)
   ltr,lte=pc.transform(q),pc.transform(s.transform(x[te]))
   oldt=np.concatenate([*oldt,ltr],1); oldv=np.concatenate([*oldv,lte],1)
   mtr,mte=[],[]
   for x in blocks:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(8,whiten=True,svd_solver="randomized",random_state=seed).fit(q)
    mtr.append(pc.transform(q)); mte.append(pc.transform(s.transform(x[te])))
   mtr=np.concatenate(mtr,1); mte=np.concatenate(mte,1)
   sets={"margin_old_full":(np.c_[mt,oldt],np.c_[mv,oldv]),
         "separate_old_full":(np.c_[st,oldt],np.c_[sv,oldv]),
         "separate_multilayer48":(np.c_[st,mtr],np.c_[sv,mte]),
         "separate_multilayer48_geometry":(np.c_[st,mtr,gt],np.c_[sv,mte,gv]),
         "separate_multilayer48_plus_layer14":(np.c_[st,mtr,ltr],np.c_[sv,mte,lte]),
         "separate_multilayer48_geometry_plus_layer14":(np.c_[st,mtr,gt,ltr],np.c_[sv,mte,gv,lte])}
   for name,(xtr,xte) in sets.items():
    clf=LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(xtr,y[tr])
    prediction[name][te]=clf.predict_proba(xte)[:,1]
   print(f"seed={seed} fold={fold}/5",flush=True)
  for name in variants: scores[name].append(met(y,prediction[name]))
 out=[]
 for name,vals in scores.items(): out.append({"variant":name,**{f"mean_{k}":float(np.mean([v[k] for v in vals])) for k in vals[0]},"per_seed":vals})
 out.sort(key=lambda x:x["mean_auroc"],reverse=True)
 report={"protocol":"Scientist question-grouped 3x5-fold OOF","layers":[10,14,18,22,26,30],"dimensions":{"separate":32,"multilayer":48,"geometry":18},"results":out}
 path=RUNS/"119_dual_multilayer_hidden.json"; path.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
