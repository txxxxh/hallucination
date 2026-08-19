#!/usr/bin/env python3
"""Exact-gradient, random-projection, and entity-interpolation known/unknown pilot."""
from __future__ import annotations
import argparse,importlib,json,math,os,tempfile,time,zlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs";BASE=importlib.import_module("160_symmetric_evidence_known_unknown");OUT=RUNS/"173_known_unknown_margin_geometry_n100";ALPHAS=np.linspace(0,1,11,dtype=np.float32)
def atomic_npz(path,**v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent);os.close(fd)
 try:np.savez_compressed(tmp,**v);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def prompt_for(r):
 a,b=r["candidate_pool"];return "Choose which candidate best matches the description.\nCandidate A: "+a+"\nCandidate B: "+b+"\n\nDescription:\n"+r["question"]+"\n\nAnswer with the candidate name only.",a,b
def span(off,a,b):
 z=[i for i,(x,y)in enumerate(off)if x<b and y>a]
 if not z:raise RuntimeError("token span not found")
 return z
def conc(v):
 v=np.maximum(np.asarray(v,float),0);s=v.sum()+1e-12;p=v/s;o=np.sort(v);n=len(v);ent=-np.sum(p*np.log(p+1e-12))/max(math.log(max(n,2)),1e-12);g=2*np.sum((np.arange(n)+1)*o)/(n*s)-(n+1)/n;return np.asarray([v.mean(),v.std(),v.max(),np.median(v),ent,g,o[-min(3,n):].sum()/s,o[-min(5,n):].sum()/s],np.float32)
def curvefeat(C):
 arms=[]
 for c in C:
  d=c-c[0];x=np.sign(c[:-1])!=np.sign(c[1:]);cross=float(ALPHAS[np.argmax(x)+1])if np.any(x)else 1.25;arms.append([cross,np.max(abs(d)),np.mean(abs(d)),np.trapezoid(abs(d),ALPHAS),np.max(abs(np.diff(c))),c[-1]-c[0]])
 arms=np.asarray(sorted(arms,key=lambda x:tuple(x)),np.float32).ravel();mir=abs((C[0]-C[0,0])+(C[1]-C[1,0]));return np.r_[abs(C[0,0]),arms,mir.mean(),mir.max()].astype(np.float32)
def score(model,emb,E,ans):
 import torch
 ae=emb(ans).detach()[None];seq=torch.cat([E,ae.to(E.dtype)],1);out=model(inputs_embeds=seq,attention_mask=torch.ones(seq.shape[:2],dtype=torch.long,device=seq.device),logits_to_keep=ans.numel()+1,use_cache=False);lg=out.logits[:,:ans.numel()].float();return torch.log_softmax(lg,-1).gather(-1,ans.view(1,-1,1)).squeeze(-1).mean(-1)
def margin(model,emb,E,a,b):return score(model,emb,E,a)-score(model,emb,E,b)
def one(model,tok,r,draws):
 import torch
 prompt,a,b=prompt_for(r);text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt",return_offsets_mapping=True,add_special_tokens=False);ids=enc.input_ids.cuda();off=enc.offset_mapping[0].tolist();q0=text.find(r["question"]);qi=span(off,q0,q0+len(r["question"]));a0=text.find(a);b0=text.find(b);ai=span(off,a0,a0+len(a));bi=span(off,b0,b0+len(b));aid=tok(a,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();bid=tok(b,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();emb=model.get_input_embeddings();E=emb(ids).detach().requires_grad_(True);m=margin(model,emb,E,aid,bid).sum();G,=torch.autograd.grad(m,E);g=G[0].float();e=E[0].detach().float();qg=g[qi];mean=emb.weight.detach().float().mean(0);exact=np.r_[abs(float(m.detach())),conc(qg.norm(dim=-1).cpu()),conc(abs((qg*e[qi]).sum(-1)).cpu()),conc(abs((qg*(mean-e[qi])).sum(-1)).cpu()),conc(g[ai].norm(dim=-1).cpu()),conc(g[bi].norm(dim=-1).cpu())].astype(np.float32);gen=torch.Generator(device="cuda");gen.manual_seed(BASE.SEED+zlib.crc32(r["key"].encode()));rp=[]
 for _ in range(draws):
  n=torch.randn(qg.shape,generator=gen,device=qg.device);n=n/(n.norm(dim=-1,keepdim=True)+1e-8)*e[qi].norm(dim=-1,keepdim=True);rp.append(float((qg*n).sum()))
 rp=np.asarray(rp,np.float32);rnd=np.asarray([abs(float(m.detach())),np.mean(abs(rp)),np.std(rp),np.max(abs(rp)),np.mean(rp),np.std(abs(rp)),np.mean(rp>0)],np.float32);C=[]
 with torch.inference_mode():
  for ix,target in((ai,e[bi].mean(0)),(bi,e[ai].mean(0))):
   vals=[]
   for alpha in ALPHAS:
    ep=e.clone()[None];ep[:,ix]=(1-float(alpha))*ep[:,ix]+float(alpha)*target;vals.append(float(margin(model,emb,ep,aid,bid)[0]))
   C.append(vals)
 C=np.asarray(C,np.float32);return exact,rnd,curvefeat(C),C
def collect(a,rows):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 (a.output_dir/"features").mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.pad_token=tok.eos_token;model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).cuda().eval()
 for i,r in enumerate(rows,1):
  fp=a.output_dir/"features"/(r["key"]+".npz")
  if a.resume and fp.exists():continue
  try:x,n,c,C=one(model,tok,r,a.random_draws);atomic_npz(fp,exact_gradient=x,random_projection=n,entity_interpolation=c,curves=C);print(f"[{i}/{len(rows)}] {r['key']}",flush=True)
  except Exception as e:BASE.append_error(a.output_dir/"errors.jsonl",{"key":r["key"],"error":repr(e)});print("ERROR",r["key"],repr(e),flush=True)
def met(y,p):
 from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
 return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}
def pred(X,y,g,grouped,seed):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,g)if grouped else StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y);p=np.zeros(len(y))
 for tr,te in cv:
  s=[StandardScaler()]
  if X.shape[1]>32:s.append(PCA(min(16,len(tr)-2),whiten=True,random_state=seed))
  s.append(LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=seed));m=make_pipeline(*s).fit(X[tr],y[tr]);p[te]=m.predict_proba(X[te])[:,1]
 return p
def evaluate(a,rows):
 rows=[r for r in rows if(a.output_dir/"features"/(r["key"]+".npz")).exists()];y=np.asarray([r["known"]for r in rows]);g=np.asarray([r["group"]for r in rows]);F={n:np.stack([np.load(a.output_dir/"features"/(r["key"]+".npz"))[n]for r in rows])for n in("exact_gradient","random_projection","entity_interpolation")};q=np.stack([np.load(BASE.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][[8,10,12,14,16,18,20,22]].astype(np.float32)for r in rows]).reshape(len(rows),-1);report={"n":len(y),"known":int(y.sum()),"unknown":int((1-y).sum()),"groups":len(set(g)),"results":{}}
 for grouped in(False,True):
  kind="entity_grouped"if grouped else"stratified_random";report["results"][kind]={};A={**F,"question_hidden":q,"question_plus_exact":np.c_[q,F["exact_gradient"]],"question_plus_entity":np.c_[q,F["entity_interpolation"]]}
  for name,X in A.items():
   ps=[pred(X,y,g,grouped,s)for s in(42,43,44)];report["results"][kind][name]={"mean":met(y,np.mean(ps,0)),"per_seed":[met(y,z)for z in ps]}
 BASE.atomic_json(a.output_dir/"evaluation.json",report);print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("collect","evaluate","all"));p.add_argument("--limit",type=int,default=100);p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--random-draws",type=int,default=16);p.add_argument("--resume",action="store_true");a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);rows,_=BASE.audit(a.output_dir);rows=BASE.select_balanced(rows,a.limit,BASE.SEED);BASE.atomic_json(a.output_dir/"config.json",{"model":a.model,"n":len(rows),"alphas":ALPHAS.tolist(),"random_draws":a.random_draws,"label_used_during_collection":False,"created":time.time()})
 if a.stage in("collect","all"):collect(a,rows)
 if a.stage in("evaluate","all"):evaluate(a,rows)
if __name__=="__main__":main()
