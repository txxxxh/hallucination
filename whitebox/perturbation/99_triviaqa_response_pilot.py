#!/usr/bin/env python3
"""Collect and evaluate gradient-top5 response features on balanced TriviaQA generations."""
from __future__ import annotations
import argparse,importlib,json,string,re,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(HERE),str(HERE.parent)]
from spanattr.core import Item,SpanAttributor,set_seed
def norm(s):
 s=s.lower(); s=''.join(c if c not in string.punctuation else ' ' for c in s); s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())
def hidden(att,prep,A,layer):
 import torch
 ans=prep.pred_variant_ids[0]; out=[]
 for i in range(0,len(A),att.max_rows):
  a=A[i:i+att.max_rows]; pe=att._embeds(prep,a); ae=att.emb_layer(ans).detach().unsqueeze(0).expand(len(a),-1,-1); seq=torch.cat([pe,ae.to(pe.dtype)],1); mask=torch.ones(seq.shape[:2],dtype=torch.long,device=att.device)
  with torch.inference_mode(): z=att.model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
  out.append(z.hidden_states[layer][:,pe.shape[1]+len(ans)-1].float().cpu()); del z,seq,pe
 return torch.cat(out).numpy()
def selected(path):
 r=[json.loads(x) for x in path.open() if x.strip()]; r=[x for x in r if not (x['correct'] and norm(x['other_answer']) in [norm(y) for y in x['aliases']])]; g=[x for x in r if x['correct']]; b=[x for x in r if not x['correct']]; n=min(len(g),len(b)); return g[:n]+b[:n]
def collect(a):
 import torch
 rows=selected(a.manifest); a.cache.mkdir(parents=True,exist_ok=True); model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda'); att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 for i,r in enumerate(rows,1):
  fp=a.cache/f"{r['key']}.npz"
  if fp.exists() and a.resume: continue
  item=Item(r['key'],r['context'],r['question'],r['other_answer'],r['generation']); prep=att.prepare(item); spans=att.build_word_spans(prep,widths=(2,3),stride=1); s0=att.S0(prep); proposal=att.u_hat_first_order(prep,spans,g=att.grad_alpha(prep)); ids=np.argsort(-np.abs(proposal))[:5]; gates=[att.alpha_from_spans(prep,[int(j)]) for j in ids]; A=torch.stack([torch.zeros_like(gates[0]),*[g*x for g in gates for x in (.25,.5,.75,1.)]]); h=hidden(att,prep,A,16); sm=att.S_batched(prep,A).numpy(); c=np.empty((5,5,4096),np.float32); c[:,0]=h[0]; c[:,1:]=h[1:].reshape(5,4,-1); mc=np.empty((5,5),np.float32); mc[:,0]=sm[0]; mc[:,1:]=sm[1:].reshape(5,4); u=s0-mc[:,-1]
  np.savez_compressed(fp,key=np.asarray(r['key']),correct=np.asarray(int(r['correct'])),words=np.asarray(r['generation_words']),S0=np.asarray(s0),top_u=u,all_u=proposal,curve=c.astype(np.float16),margin_curve=mc); print(f'[{i}/{len(rows)}] {r["key"]} y={int(r["correct"])}',flush=True)
def wm(x,u,pos):
 m=u>0 if pos else u<0; w=u if pos else -u
 return (x[m]*w[m,None]).sum(0)/(np.abs(w[m]).sum()+1e-9) if m.any() else np.zeros(x.shape[-1],np.float32)
def train(a):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
 from sklearn.model_selection import StratifiedKFold
 from sklearn.preprocessing import StandardScaler
 rows=[]
 for fp in sorted(a.cache.glob('*.npz')):
  with np.load(fp,allow_pickle=True) as z:
   u=z['top_u'].astype(np.float32); ua=z['all_u'].astype(np.float32); s=float(z['S0']); c=z['curve'].astype(np.float32); h0=c[0,0]; d=c[:,-1]-h0; m=np.r_[u,np.abs(u),u/(abs(s)+1e-6),u.max(initial=0),u.min(initial=0),np.abs(u).mean(),np.abs(u).sum()/(np.abs(ua).sum()+1e-9),np.mean(ua>0),np.std(ua)]; res=c[:,1:4]-h0-np.array([.25,.5,.75])[None,:,None]*d[:,None]; rb=[]
   for j in range(3): rb += [wm(res[:,j],u,True),wm(res[:,j],u,False)]
   rows.append((int(z['correct']),float(z['words']),m,h0,wm(d,u,True),wm(d,u,False),np.stack(rb)))
 y=np.array([x[0] for x in rows]); W=np.array([[np.log1p(x[1])] for x in rows]); M=np.stack([x[2] for x in rows]); H=[np.stack([x[i] for x in rows]) for i in (3,4,5)]; R=np.stack([x[6] for x in rows]); names=['length','margin','baseline','curve']; vals={n:[] for n in names}
 for seed in [42,43,44,45,46]:
  p={n:np.zeros(len(y)) for n in names}; cv=StratifiedKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(M,y):
   ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); bt,bv=[mt],[mv]
   for x in H:
    sc=StandardScaler().fit(x[tr]); q=sc.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); bt.append(pc.transform(q)); bv.append(pc.transform(sc.transform(x[te])))
   base=np.concatenate(bt,1),np.concatenate(bv,1); rt,rv=[],[]
   for j in range(6):
    sc=StandardScaler().fit(R[tr,j]); q=sc.transform(R[tr,j]); pc=PCA(4,whiten=True,svd_solver='randomized',random_state=seed).fit(q); rt.append(pc.transform(q)); rv.append(pc.transform(sc.transform(R[te,j])))
   ws=StandardScaler().fit(W[tr]); sets={'length':(ws.transform(W[tr]),ws.transform(W[te])),'margin':(mt,mv),'baseline':base,'curve':(np.concatenate([base[0]]+rt,1),np.concatenate([base[1]]+rv,1))}
   for n,(xt,xv) in sets.items():
    C=.03 if n=='curve' else .075; clf=LogisticRegression(C=C,max_iter=5000,class_weight='balanced',solver='liblinear').fit(xt,y[tr]); p[n][te]=clf.predict_proba(xv)[:,1]
  for n,q in p.items(): vals[n].append({'auroc':float(roc_auc_score(y,q)),'auprc':float(average_precision_score(y,q)),'balanced_accuracy':float(balanced_accuracy_score(y,q>=.5))})
 report={'protocol':'236 balanced free generations; aliases labels; 5x repeated stratified 5-fold OOF','n':len(y),'results':{n:{m:float(np.mean([x[m] for x in v])) for m in ['auroc','auprc','balanced_accuracy']} for n,v in vals.items()},'per_seed':vals}; a.report.write_text(json.dumps(report,indent=2)); print(json.dumps(report['results'],indent=2))
def main():
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['collect','train','all']); p.add_argument('--manifest',type=Path,default=Path('runs/98_triviaqa_balanced_n238.jsonl')); p.add_argument('--cache',type=Path,default=Path('runs/99_triviaqa_response_n236')); p.add_argument('--report',type=Path,default=Path('runs/99_triviaqa_response_report.json')); p.add_argument('--model',default='/tmp/Meta-Llama-3.1-8B-Instruct'); p.add_argument('--batch',type=int,default=32); p.add_argument('--resume',action='store_true'); a=p.parse_args(); set_seed(42)
 if a.stage in ['collect','all']: collect(a)
 if a.stage in ['train','all']: train(a)
if __name__=='__main__': main()
