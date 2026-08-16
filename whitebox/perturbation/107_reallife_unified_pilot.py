#!/usr/bin/env python3
"""Evaluate the frozen unified detector on a 200-question RealLifeQA pilot."""
import argparse,importlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'

def wm(x,u,pos):
 m=u>0 if pos else u<0; w=u if pos else -u
 return (x[m]*w[m,None]).sum(0)/(np.abs(w[m]).sum()+1e-9) if m.any() else np.zeros(x.shape[-1],np.float32)

def collect(a):
 import torch
 from spanattr.core import Item,SpanAttributor
 stats=importlib.import_module('100_collect_multilayer_trajectory')._stats; rows=json.load(open(a.data))[:a.questions]; a.cache.mkdir(parents=True,exist_ok=True); model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda'); att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=16); layers=np.unique(np.rint(np.linspace(1,model.config.num_hidden_layers,8)).astype(int)).tolist()
 jobs=[]
 for qi,r in enumerate(rows):
  right=r['answer']-1
  for oi in [0,1]: jobs.append((f'rl{qi:05d}_{oi+1}',f'rl{qi:05d}',int(oi==right),r['options'][oi],r['options'][1-oi],r))
 for n,(key,group,label,pred,other,r) in enumerate(jobs,1):
  fp=a.cache/f'{key}.npz'
  if fp.exists() and a.resume: continue
  q=f"Option1: {r['options'][0]}\nOption2: {r['options'][1]}\nWhich option is correct?"; item=Item(key,r['question'],q,other,pred); prep=att.prepare(item); spans=att.build_word_spans(prep,widths=(2,3),stride=1); s0=att.S0(prep); proposal=att.u_hat_first_order(prep,spans,g=att.grad_alpha(prep)); ids=np.argsort(-np.abs(proposal))[:5]; gates=[att.alpha_from_spans(prep,[int(i)]) for i in ids]; A=torch.stack([torch.zeros_like(gates[0]),*gates]); ans=prep.pred_variant_ids[0]
  pe=att._embeds(prep,A); ae=att.emb_layer(ans).detach().unsqueeze(0).expand(len(A),-1,-1); seq=torch.cat([pe,ae.to(pe.dtype)],1); mask=torch.ones(seq.shape[:2],dtype=torch.long,device='cuda')
  with torch.inference_mode(): out=model(inputs_embeds=seq,attention_mask=mask,output_hidden_states=True,use_cache=False)
  plen,alen=pe.shape[1],len(ans); h16=out.hidden_states[16][:,plen+alen-1].float(); h0=h16[0].cpu().numpy(); delta=h16[1:].cpu().numpy()-h0; margins=att.S_batched(prep,A).numpy(); u=(s0-margins[1:]).astype(np.float32); allh=torch.stack([out.hidden_states[l][0,plen+alen-1] for l in layers]); allm=torch.stack([out.hidden_states[l][0,plen:plen+alen].mean(0) for l in layers]); ls=stats(allh).cpu().numpy(); ms=stats(allm).cpu().numpy(); from scipy.fft import dct; T=np.r_[ls.ravel(),ms.ravel(),np.diff(ls,axis=0).ravel(),np.diff(ms,axis=0).ravel(),dct(ls,axis=0,norm='ortho')[1:4].ravel(),dct(ms,axis=0,norm='ortho')[1:4].ravel()].astype(np.float32)
  logits=out.logits[0,plen-1:plen+alen-1].float(); lp=logits.log_softmax(-1); tlp=lp.gather(1,ans[:,None]).squeeze(1); p=lp.exp(); ent=-(p*lp).sum(-1); top2=logits.topk(2,-1).values; L=torch.stack([tlp.mean(),tlp.amin(),tlp.std(unbiased=False),ent.mean(),ent.amax(),(top2[:,0]-top2[:,1]).mean(),torch.tensor(float(alen),device='cuda')]).cpu().numpy().astype(np.float32)
  M=np.r_[u,np.abs(u),u/(abs(s0)+1e-6),u.max(initial=0),u.min(initial=0),np.abs(u).mean(),np.abs(u).sum()/(np.abs(proposal).sum()+1e-9),np.mean(proposal>0),np.std(proposal)].astype(np.float32)
  np.savez_compressed(fp,key=np.asarray(key),group=np.asarray(group),correct=np.asarray(label),M=M,H0=h0.astype(np.float16),HP=wm(delta,u,True).astype(np.float16),HN=wm(delta,u,False).astype(np.float16),T=T,L=L,last14=allh[3].float().cpu().numpy().astype(np.float16)); print(f'[{n}/{len(jobs)}] {key}',flush=True)

def train(a):
 from sklearn.decomposition import PCA
 from sklearn.ensemble import HistGradientBoostingClassifier
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
 from sklearn.model_selection import StratifiedGroupKFold
 from sklearn.preprocessing import StandardScaler
 rows=[np.load(p) for p in sorted(a.cache.glob('*.npz'))]; y=np.array([int(x['correct']) for x in rows]); g=np.array([str(x['group']) for x in rows]); M=np.stack([x['M'] for x in rows]); H=[np.stack([x[k].astype(np.float32) for x in rows]) for k in ['H0','HP','HN']]; T=np.stack([x['T'] for x in rows]); L=np.stack([x['L'] for x in rows]); X14=np.stack([x['last14'].astype(np.float32) for x in rows]); vals=[]
 for seed in [42,43,44]:
  pl=np.zeros(len(y)); pt=np.zeros(len(y)); cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(M,y,g):
   ms=StandardScaler().fit(M[tr]); mt,mv=ms.transform(M[tr]),ms.transform(M[te]); parts=[]
   for x in H:
    s=StandardScaler().fit(x[tr]); q=s.transform(x[tr]); pc=PCA(12,whiten=True,svd_solver='randomized',random_state=seed).fit(q); parts.append((pc.transform(q),pc.transform(s.transform(x[te]))))
   base=np.concatenate([mt]+[x[0] for x in parts],1),np.concatenate([mv]+[x[1] for x in parts],1); s=StandardScaler().fit(X14[tr]); q=s.transform(X14[tr]); pc=PCA(48,whiten=True,svd_solver='randomized',random_state=seed).fit(q); a14,b14=pc.transform(q),pc.transform(s.transform(X14[te])); lr=LogisticRegression(C=.03,max_iter=5000,class_weight='balanced',solver='liblinear').fit(np.c_[base[0],a14],y[tr]); pl[te]=lr.predict_proba(np.c_[base[1],b14])[:,1]
   hi=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=3,learning_rate=.05,random_state=seed).fit(np.c_[M[tr],T[tr],L[tr]],y[tr]); pt[te]=hi.predict_proba(np.c_[M[te],T[te],L[te]])[:,1]
  p=.8*pl+.2*pt; vals.append({'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))})
 report={'protocol':f'frozen unified detector; {len(set(g))} questions/{len(y)} paired candidates; question-grouped 3x5-fold','config':'0.8*(base+layer14-last PCA48 LR C=.03)+0.2*HistGB(base margin+trajectory stats+logits)','n':len(y),'groups':len(set(g)),'mean':{m:float(np.mean([x[m] for x in vals])) for m in vals[0]},'per_seed':vals}; a.report.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

def main():
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['collect','train','all']); p.add_argument('--data',type=Path,default=Path('/home/tong56/whitebox/question_and_result.json')); p.add_argument('--questions',type=int,default=200); p.add_argument('--cache',type=Path,default=RUNS/'107_reallife_unified_n200'); p.add_argument('--report',type=Path,default=RUNS/'107_reallife_unified_n200_report.json'); p.add_argument('--model',default='/tmp/Meta-Llama-3.1-8B-Instruct'); p.add_argument('--resume',action='store_true'); a=p.parse_args();
 if a.stage in ['collect','all']:collect(a)
 if a.stage in ['train','all']:train(a)
if __name__=='__main__':main()
