#!/usr/bin/env python3
"""Grouped-CV ablation for compact multi-alpha response-curve features."""
from __future__ import annotations
import argparse, glob, json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent/'runs'; ALPHA=np.asarray([0,.25,.5,.75,1.],np.float32)

def wmean(x,u,pos):
 m=u>0 if pos else u<0; w=u if pos else -u
 return (x[m]*w[m,None]).sum(0)/(np.abs(w[m]).sum()+1e-9) if m.any() else np.zeros(x.shape[-1],np.float32)

def load():
 src={x['key']:x for x in map(json.loads,(ROOT/'88_known_gt05_n1084.jsonl').open())}
 oracle={x['key']:x for x in map(json.loads,(ROOT/'88_oracle_top11_known_gt05.jsonl').open())}; rows=[]
 for cp in sorted(glob.glob(str(ROOT/'93_response_curve_top5'/'*.npz'))):
  with np.load(cp,allow_pickle=True) as c:
   key=str(c['key'].item()); old=np.load(ROOT/'88_hidden_delta_top11_known_gt05'/f'{key}.npz')
   u=np.asarray(c['top_u'],np.float32); ua=np.asarray(oracle[key]['u'],np.float32); s0=float(oracle[key]['S0'])
   h=np.asarray(old['answer_last'],np.float32)[0]; h0=h[0]; hend=h[1:6]
   curve=np.empty((5,5,4096),np.float32); curve[:,0]=h0; curve[:,1:4]=np.asarray(c['answer_last_mid'],np.float32); curve[:,4]=hend
   margins=np.empty((5,5),np.float32); margins[:,0]=s0; margins[:,1:4]=c['margin_mid']; margins[:,4]=s0-u
   margin=np.r_[u,np.abs(u),u/(abs(s0)+1e-6),u.max(initial=0),u.min(initial=0),np.abs(u).mean(),np.abs(u).sum()/(np.abs(ua).sum()+1e-9),np.mean(ua>0),np.std(ua)]
   delta=hend-h0; pos=wmean(delta,u,True); neg=wmean(delta,u,False)
   # Hidden nonlinear residuals relative to the endpoint chord, aggregated by sign.
   residual=curve[:,1:4]-h0-ALPHA[1:4,None]*delta[:,None,:]
   rblocks=[]
   for ai in range(3): rblocks += [wmean(residual[:,ai],u,True),wmean(residual[:,ai],u,False)]
   # Directional geometry (24 dims): chord projection error and orthogonal deviation.
   mid=curve[:,1:4]-h0; den=np.sum(delta**2,1)[:,None]+1e-8
   proj=np.einsum('kad,kd->ka',mid,delta)/den-ALPHA[1:4]
   orth=np.linalg.norm(mid-(proj+ALPHA[1:4])[...,None]*delta[:,None],axis=2)/(np.linalg.norm(delta,axis=1)[:,None]+1e-8)
   geom=np.r_[proj.mean(0),proj.std(0),np.max(np.abs(proj),0),np.average(proj,axis=0,weights=np.abs(u)+1e-8),orth.mean(0),orth.std(0),orth.max(0),np.average(orth,axis=0,weights=np.abs(u)+1e-8)]
   gains=s0-margins; norm=gains/(np.abs(u)[:,None]+.1); nonlinear=norm-ALPHA
   scalar=np.r_[nonlinear[:,1:4].mean(0),nonlinear[:,1:4].std(0),np.max(np.abs(nonlinear[:,1:4]),0),np.average(nonlinear[:,1:4],axis=0,weights=np.abs(u)+1e-8)]
   # DCT only on normalized ordered alpha trajectories; retain two non-DC modes' summaries.
   cf=dct(norm,type=2,norm='ortho',axis=1)[:,1:3]
   fft=np.r_[cf.mean(0),cf.std(0),np.max(np.abs(cf),0),np.average(cf,axis=0,weights=np.abs(u)+1e-8)]
   rows.append((key,src[key]['group'],int(src[key]['correct']),margin,h0,pos,neg,np.stack(rblocks),np.r_[geom,scalar],fft))
 assert len(rows)==1084,len(rows); return rows

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dims',type=int,nargs='+',default=[1,2,3,4]); p.add_argument('--Cs',type=float,nargs='+',default=[.01,.03,.05,.075,.1,.15]); p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46]); p.add_argument('--out',type=Path,default=ROOT/'94_response_curve_report.json'); a=p.parse_args()
 r=load(); y=np.array([x[2] for x in r]); groups=np.array([x[1] for x in r]); M=np.stack([x[3] for x in r]); H=[np.stack([x[i] for x in r]) for i in (4,5,6)]; R=np.stack([x[7] for x in r]); G=np.stack([x[8] for x in r]); F=np.stack([x[9] for x in r])
 variants=['baseline','curve_scalar','curve_hidden','curve_both','dct_only','curve_all']; configs=list(product(variants,a.dims,a.Cs)); scores={c:[] for c in configs}
 for seed in a.seeds:
  pred={c:np.zeros(len(y),np.float32) for c in configs}; cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for fold,(tr,te) in enumerate(cv.split(M,y,groups),1):
   def scale(x): s=StandardScaler().fit(x[tr]); return s.transform(x[tr]),s.transform(x[te])
   mt,mv=scale(M); bt,bv=[mt],[mv]
   for x in H:
    s=StandardScaler().fit(x[tr]); z=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(z); bt.append(pc.transform(z)); bv.append(pc.transform(s.transform(x[te])))
   base=(np.concatenate(bt,1),np.concatenate(bv,1)); gt,gv=scale(G); ft,fv=scale(F)
   # Six residual-vector blocks, each strongly bottlenecked to 1--4 PCs.
   rp=[]
   for bi in range(6):
    s=StandardScaler().fit(R[tr,bi]); z=s.transform(R[tr,bi]); pc=PCA(max(a.dims),whiten=True,svd_solver='randomized',random_state=seed).fit(z); rp.append((pc.transform(z),pc.transform(s.transform(R[te,bi]))))
   for dim in a.dims:
    rt=np.concatenate([q[0][:,:dim] for q in rp],1); rv=np.concatenate([q[1][:,:dim] for q in rp],1)
    parts={'baseline':base,'curve_scalar':(np.c_[base[0],gt],np.c_[base[1],gv]),'curve_hidden':(np.c_[base[0],rt],np.c_[base[1],rv]),'curve_both':(np.c_[base[0],gt,rt],np.c_[base[1],gv,rv]),'dct_only':(np.c_[base[0],ft],np.c_[base[1],fv]),'curve_all':(np.c_[base[0],gt,rt,ft],np.c_[base[1],gv,rv,fv])}
    for variant,C in product(variants,a.Cs):
     xt,xv=parts[variant]; clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear',random_state=seed).fit(xt,y[tr]); pred[(variant,dim,C)][te]=clf.predict_proba(xv)[:,1]
   print(f'seed={seed} fold={fold}/5',flush=True)
  for c,q in pred.items(): scores[c].append({'auroc':float(roc_auc_score(y,q)),'auprc':float(average_precision_score(y,q)),'balanced_accuracy':float(balanced_accuracy_score(y,q>=.5))})
 out=[]
 for c,v in scores.items():
  z=dict(zip(['variant','residual_pca','C'],c))
  for m in ['auroc','auprc','balanced_accuracy']:
   q=np.array([x[m] for x in v]); z['mean_'+m]=float(q.mean()); z['std_'+m]=float(q.std(ddof=1))
  z['per_seed']=v; out.append(z)
 out.sort(key=lambda x:(x['mean_auroc'],x['mean_auprc']),reverse=True); report={'warning':'same-data feature selection','n':len(y),'variants':variants,'results':out}; a.out.write_text(json.dumps(report,indent=2)); print(json.dumps({'out':str(a.out),'top_15':out[:15]},indent=2))
if __name__=='__main__': main()
