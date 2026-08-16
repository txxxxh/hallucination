#!/usr/bin/env python3
"""HaluEval QA paired-candidate pilot for perturbation response curves."""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(HERE),str(HERE.parent)]
from spanattr.core import Item,SpanAttributor,set_seed

def hidden_last(att,prep,alphas,layer):
 import torch
 ans=prep.pred_variant_ids[0]; out=[]
 for i in range(0,len(alphas),att.max_rows):
  a=alphas[i:i+att.max_rows]; pe=att._embeds(prep,a); ae=att.emb_layer(ans).detach().unsqueeze(0).expand(len(a),-1,-1); seq=torch.cat([pe,ae.to(pe.dtype)],1); mask=torch.ones(seq.shape[:2],dtype=torch.long,device=att.device)
  with torch.inference_mode(): z=att.model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
  out.append(z.hidden_states[layer][:,pe.shape[1]+len(ans)-1].float().cpu()); del z,seq,pe
 return torch.cat(out).numpy()

def collect(a):
 import torch
 rows=[json.loads(x) for x in a.data.open() if x.strip()][:a.questions]; a.cache.mkdir(parents=True,exist_ok=True)
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,a.dtype,a.device); att=SpanAttributor(model,tok,device=a.device,baseline='mean',length_norm=True,max_rows=a.batch)
 jobs=[]
 for qi,r in enumerate(rows):
  jobs += [(f'hq{qi:05d}_right',qi,r['right_answer'],r['hallucinated_answer'],1,r),(f'hq{qi:05d}_hall',qi,r['hallucinated_answer'],r['right_answer'],0,r)]
 for n,(key,qi,pred,other,label,r) in enumerate(jobs,1):
  target=a.cache/f'{key}.npz'
  if target.exists() and a.resume: continue
  item=Item(key,r['knowledge'],r['question'],other,pred); prep=att.prepare(item); spans=att.build_word_spans(prep,widths=(2,3),stride=1)
  s0=att.S0(prep)
  if a.selector=='oracle':
   proposal,_=att.u_of_sets(prep,[[i] for i in range(len(spans))],S0=s0)
  else:
   proposal=att.u_hat_first_order(prep,spans,g=att.grad_alpha(prep))
  ids=np.argsort(-np.abs(proposal))[:a.topk]; gates=[att.alpha_from_spans(prep,[int(i)]) for i in ids]
  alpha=torch.stack([torch.zeros_like(gates[0]),*[g*x for g in gates for x in (.25,.5,.75,1.)]])
  h=hidden_last(att,prep,alpha,a.layer); margins=att.S_batched(prep,alpha).numpy(); curve=np.empty((a.topk,5,h.shape[1]),np.float32); curve[:,0]=h[0]; curve[:,1:]=h[1:].reshape(a.topk,4,-1); mc=np.empty((a.topk,5),np.float32); mc[:,0]=margins[0]; mc[:,1:]=margins[1:].reshape(a.topk,4)
  top_u=(s0-mc[:,-1]).astype(np.float32)
  np.savez_compressed(target,key=np.asarray(key),group=np.asarray(f'hq{qi:05d}'),correct=np.asarray(label),S0=np.asarray(s0,np.float32),top_u=top_u,all_u=proposal.astype(np.float32),selector=np.asarray(a.selector),curve=curve.astype(np.float16),margin_curve=mc)
  print(f'[{n}/{len(jobs)}] {key} y={label} spans={len(spans)}',flush=True)

def wmean(x,u,pos):
 m=u>0 if pos else u<0; w=u if pos else -u
 return (x[m]*w[m,None]).sum(0)/(np.abs(w[m]).sum()+1e-9) if m.any() else np.zeros(x.shape[-1],np.float32)

def train(a):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
 from sklearn.model_selection import StratifiedGroupKFold
 from sklearn.preprocessing import StandardScaler
 files=sorted(a.cache.glob('*.npz')); rows=[]
 for p in files:
  with np.load(p,allow_pickle=True) as z:
   u=z['top_u'].astype(np.float32); ua=z['all_u'].astype(np.float32); s=float(z['S0']); c=z['curve'].astype(np.float32); h0=c[0,0]; d=c[:,-1]-h0
   m=np.r_[u,np.abs(u),u/(abs(s)+1e-6),u.max(initial=0),u.min(initial=0),np.abs(u).mean(),np.abs(u).sum()/(np.abs(ua).sum()+1e-9),np.mean(ua>0),np.std(ua)]
   res=c[:,1:4]-h0-np.asarray([.25,.5,.75])[None,:,None]*d[:,None]; rb=[]
   for j in range(3): rb += [wmean(res[:,j],u,True),wmean(res[:,j],u,False)]
   rows.append((str(z['group'].item()),int(z['correct']),m,h0,wmean(d,u,True),wmean(d,u,False),np.stack(rb)))
 y=np.array([x[1] for x in rows]); groups=np.array([x[0] for x in rows]); M=np.stack([x[2] for x in rows]); H=[np.stack([x[i] for x in rows]) for i in (3,4,5)]; R=np.stack([x[6] for x in rows]); cv=StratifiedGroupKFold(5,shuffle=True,random_state=42); pred={v:np.zeros(len(y)) for v in ['margin','baseline','curve']}
 for tr,te in cv.split(M,y,groups):
  ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); bt,bv=[mt],[mv]
  for x in H:
   s=StandardScaler().fit(x[tr]); z=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=42).fit(z); bt.append(pc.transform(z)); bv.append(pc.transform(s.transform(x[te])))
  base=np.concatenate(bt,1),np.concatenate(bv,1); rt,rv=[],[]
  for j in range(6):
   s=StandardScaler().fit(R[tr,j]); z=s.transform(R[tr,j]); pc=PCA(4,whiten=True,svd_solver='randomized',random_state=42).fit(z); rt.append(pc.transform(z)); rv.append(pc.transform(s.transform(R[te,j])))
  sets={'margin':(mt,mv),'baseline':base,'curve':(np.concatenate([base[0]]+rt,axis=1),np.concatenate([base[1]]+rv,axis=1))}
  for name,(xt,xv) in sets.items():
   C=.03 if name=='curve' else .075; clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear').fit(xt,y[tr]); pred[name][te]=clf.predict_proba(xv)[:,1]
 report={'protocol':'128 HaluEval questions, paired right/hallucinated candidates, question-grouped 5-fold OOF','n':len(y),'groups':len(set(groups)),'results':{n:{'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))} for n,p in pred.items()}}; a.report.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

def main():
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['collect','train','all']); p.add_argument('--data',type=Path,default=Path('/home/tong56/other_bench/qa_data (2).json')); p.add_argument('--cache',type=Path,default=HERE/'runs/95_halueval_q128_curves'); p.add_argument('--report',type=Path,default=HERE/'runs/95_halueval_q128_report.json'); p.add_argument('--questions',type=int,default=128); p.add_argument('--model',default='/tmp/Meta-Llama-3.1-8B-Instruct'); p.add_argument('--dtype',default='bfloat16'); p.add_argument('--device',default='cuda'); p.add_argument('--batch',type=int,default=8); p.add_argument('--layer',type=int,default=16); p.add_argument('--topk',type=int,default=5); p.add_argument('--selector',choices=['oracle','gradient'],default='gradient'); p.add_argument('--resume',action='store_true'); a=p.parse_args(); set_seed(42)
 if a.stage in ['collect','all']: collect(a)
 if a.stage in ['train','all']: train(a)
if __name__=='__main__': main()
