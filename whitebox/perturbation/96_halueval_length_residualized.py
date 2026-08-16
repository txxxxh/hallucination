#!/usr/bin/env python3
"""Remove answer-length/style nuisance signals fold-wise from HaluEval features."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent/'runs'
def wm(x,u,pos):
 m=u>0 if pos else u<0; w=u if pos else -u
 return (x[m]*w[m,None]).sum(0)/(np.abs(w[m]).sum()+1e-9) if m.any() else np.zeros(x.shape[-1],np.float32)
def residualize(xtr,xte,ztr,zte):
 # Fold-fitted quadratic nuisance model; no labels or test statistics are used.
 def design(z): return np.c_[np.ones(len(z)),z,z**2]
 a,b=design(ztr),design(zte); coef=np.linalg.lstsq(a,xtr,rcond=1e-5)[0]
 return xtr-a@coef,xte-b@coef
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path('/home/tong56/other_bench/qa_data (2).json')); p.add_argument('--cache',type=Path,default=ROOT/'95_halueval_q128_gradient_curves'); p.add_argument('--out',type=Path,default=ROOT/'96_halueval_length_residualized_report.json'); a=p.parse_args(); src=[json.loads(x) for x in a.data.open() if x.strip()][:128]; rows=[]
 for fp in sorted(a.cache.glob('*.npz')):
  with np.load(fp,allow_pickle=True) as z:
   key=str(z['key'].item()); qi=int(re.search(r'hq(\d+)_',key).group(1)); ans=src[qi]['right_answer'] if key.endswith('_right') else src[qi]['hallucinated_answer']; u=z['top_u'].astype(np.float32); ua=z['all_u'].astype(np.float32); s=float(z['S0']); c=z['curve'].astype(np.float32); h0=c[0,0]; d=c[:,-1]-h0
   m=np.r_[u,np.abs(u),u/(abs(s)+1e-6),u.max(initial=0),u.min(initial=0),np.abs(u).mean(),np.abs(u).sum()/(np.abs(ua).sum()+1e-9),np.mean(ua>0),np.std(ua)]
   res=c[:,1:4]-h0-np.asarray([.25,.5,.75])[None,:,None]*d[:,None]; rb=[]
   for j in range(3): rb += [wm(res[:,j],u,True),wm(res[:,j],u,False)]
   words=ans.split(); nuisance=np.asarray([np.log1p(len(ans)),np.log1p(len(words)),ans.count('.'),ans.count(','),len(words)>6],float)
   rows.append((str(z['group'].item()),int(z['correct']),nuisance,m,h0,wm(d,u,True),wm(d,u,False),np.stack(rb)))
 y=np.array([x[1] for x in rows]); groups=np.array([x[0] for x in rows]); Z=np.stack([x[2] for x in rows]); M=np.stack([x[3] for x in rows]); H=[np.stack([x[i] for x in rows]) for i in (4,5,6)]; R=np.stack([x[7] for x in rows]); names=['style_only','raw_baseline','resid_margin','resid_baseline','resid_curve']; vals={n:[] for n in names}
 for seed in [42,43,44,45,46]:
  pred={n:np.zeros(len(y)) for n in names}; cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(M,y,groups):
   zs=StandardScaler().fit(Z[tr]); zt,zv=zs.transform(Z[tr]),zs.transform(Z[te]); mr,mv=residualize(M[tr],M[te],zt,zv); ms=StandardScaler().fit(mr); mt,me=ms.transform(mr),ms.transform(mv)
   rawms=StandardScaler().fit(M[tr]); rawparts=[rawms.transform(M[tr])]; rawtest=[rawms.transform(M[te])]; parts=[mt]; test=[me]
   for x in H:
    xr,xv=residualize(x[tr],x[te],zt,zv); sc=StandardScaler().fit(xr); q=sc.transform(xr); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); parts.append(pc.transform(q)); test.append(pc.transform(sc.transform(xv)))
    sc0=StandardScaler().fit(x[tr]); q0=sc0.transform(x[tr]); pc0=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q0); rawparts.append(pc0.transform(q0)); rawtest.append(pc0.transform(sc0.transform(x[te])))
   base=np.concatenate(parts,1),np.concatenate(test,1); raw=np.concatenate(rawparts,1),np.concatenate(rawtest,1); rr=[];rv=[]
   for j in range(6):
    q,v=residualize(R[tr,j],R[te,j],zt,zv); sc=StandardScaler().fit(q); q=sc.transform(q); pc=PCA(4,whiten=True,svd_solver='randomized',random_state=seed).fit(q); rr.append(pc.transform(q)); rv.append(pc.transform(sc.transform(v)))
   sets={'style_only':(zt,zv),'raw_baseline':raw,'resid_margin':(mt,me),'resid_baseline':base,'resid_curve':(np.concatenate([base[0]]+rr,1),np.concatenate([base[1]]+rv,1))}
   for n,(xtr,xte) in sets.items():
    C=.03 if n=='resid_curve' else .075; clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear').fit(xtr,y[tr]); pred[n][te]=clf.predict_proba(xte)[:,1]
  for n,q in pred.items(): vals[n].append({'auroc':float(roc_auc_score(y,q)),'auprc':float(average_precision_score(y,q)),'balanced_accuracy':float(balanced_accuracy_score(y,q>=.5))})
 report={'protocol':'answer nuisance residualization fit inside each question-grouped fold; 5 CV seeds','n':len(y),'nuisance':['log_chars','log_words','periods','commas','words_gt6'],'results':{n:{m:float(np.mean([x[m] for x in v])) for m in ['auroc','auprc','balanced_accuracy']} for n,v in vals.items()},'per_seed':vals}; a.out.write_text(json.dumps(report,indent=2)); print(json.dumps(report['results'],indent=2))
if __name__=='__main__': main()
