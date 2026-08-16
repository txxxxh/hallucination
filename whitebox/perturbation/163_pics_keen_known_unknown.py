#!/usr/bin/env python3
"""Perturbed Internal Confidence Surfaces + symmetric KEEN entity representations."""
from __future__ import annotations
import argparse, importlib, json, os, re, tempfile, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs";OUT=RUNS/"163_pics_keen_pilot"
BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
SEED=BASE.SEED;K=6;MAX_CLAUSES=6;KEEN_LAYERS=[8,10,12,14,16,18,20,22]
SEMANTIC=[
 "{q}",
 "Read independently and carefully. {q}",
 "Focus on the identifying content rather than its style. {q}",
 "Verify whether your internal knowledge is sufficient. {q}",
]
SELF=("Respond only with Yes or No: are you capable of answering the following query accurately?\n\n{q}\n\nAnswer Yes or No:",
      "Respond only with Yes or No: are you capable of answering the following query accurately?\n\n{q}\n\nAnswer No or Yes:")
CAND=("Respond only with Yes or No: are you capable of accurately determining whether the candidate {c} matches the description below?\n\n{q}\n\nAnswer {order}:")
FEATURES=["ic_base","semantic_consistency","clause_occlusion","candidate_ic","keen_pair"]
IO_EXCLUSIONS={"question_2628","question_2251"}  # deterministic JuiceFS EIO; one item per class

def atomic_npz(path,**kw):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent);os.close(fd)
 try:np.savez(tmp,**kw);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def clauses(q):
 z=[x.strip() for x in re.split(r"(?<=[.!?])\s+|;\s*",q) if x.strip()]
 if len(z)<2:z=[x.strip() for x in q.split(",") if x.strip()]
 return z[:MAX_CLAUSES]
def remove_clause(q,cs,i):
 return re.sub(r"\s+"," ",q.replace(cs[i],"This detail is unavailable.",1)).strip()
def prompts(q,x,y):
 sem=[[t.format(q=s.format(q=q)) for t in SELF] for s in SEMANTIC]
 cs=clauses(q);occ=[[t.format(q=remove_clause(q,cs,i)) for t in SELF] for i in range(len(cs))]
 cand=[CAND.format(c=c,q=q,order=o) for c in (x,y) for o in ("Yes or No","No or Yes")]
 return [p for pair in sem for p in pair],[p for pair in occ for p in pair],cand,cs
def audit_templates():
 s,o,c,cs=prompts("First clue. Second clue?","Alice","Bob")
 return {"semantic":len(s)==8,"occlusion":len(o)==4,"candidate":len(c)==4,
         "candidate_symmetric":c[0].replace("Alice","<C>")==c[2].replace("Bob","<C>"),
         "order_symmetric":c[0].rsplit("Answer ",1)[0]==c[1].rsplit("Answer ",1)[0]}

def token_id(tok,text):
 ids=tok.encode(text,add_special_tokens=False)
 if len(ids)!=1:
  ids=tok.encode(" "+text,add_special_tokens=False)
 if len(ids)!=1:raise RuntimeError(f"label token not singleton: {text} {ids}")
 return ids[0]
def ic_maps(model,tok,prompts):
 import torch
 yes,no=token_id(tok,"Yes"),token_id(tok,"No");maps=[]
 for p in prompts:
  text=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True);ids=tok(text,return_tensors="pt").input_ids.to("cuda:0")
  with torch.inference_mode():z=model(ids,output_hidden_states=True,use_cache=False)
  hs=torch.stack([h[0,-min(K,h.shape[1]):] for h in z.hidden_states])
  if hs.shape[1]<K:hs=torch.cat([hs[:,:1].expand(-1,K-hs.shape[1],-1),hs],1)
  with torch.inference_mode():lg=torch.nn.functional.linear(hs,model.lm_head.weight[[yes,no]]).float();maps.append(torch.softmax(lg,-1)[...,0].cpu().numpy())
 return np.stack(maps).astype(np.float32)
def keen(model,tok,cands):
 import torch
 vals=[]
 for c in cands:
  p=f"Tell me about {c}.";text=tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=False);ids=tok(text,return_tensors="pt").input_ids.to("cuda:0");cid=tok(c,add_special_tokens=False).input_ids;seq=ids[0].tolist();starts=[i for i in range(len(seq)-len(cid)+1) if seq[i:i+len(cid)]==cid]
  if not starts:
   cid=tok(" "+c,add_special_tokens=False).input_ids;starts=[i for i in range(len(seq)-len(cid)+1) if seq[i:i+len(cid)]==cid]
  if not starts:raise RuntimeError("entity span not found")
  pos=starts[-1]+len(cid)-1
  with torch.inference_mode():z=model(ids,output_hidden_states=True,use_cache=False)
  vals.append(torch.stack([z.hidden_states[i][0,pos] for i in KEEN_LAYERS]).float().cpu().numpy())
 return np.stack(vals)
def fixed_features(sem,occ,cand,kh,nclauses):
 # sem [4 verbalizations,2 label orders,L,K]
 sem=sem.reshape(len(SEMANTIC),2,*sem.shape[1:]);base=sem[0]
 ic_base=np.r_[base.reshape(-1),base.mean((1,2)),base.std((1,2)),np.abs(base[0]-base[1]).mean()]
 d=sem[1:]-base;semantic_consistency=np.r_[d.reshape(-1),d.mean((0,1,2)),d.std((0,1,2)),np.abs(d).mean((0,1)).reshape(-1)]
 if nclauses:
  oc=occ.reshape(nclauses,2,*occ.shape[1:])-base[None];a=np.abs(oc);stats=np.stack([oc.mean(0),oc.std(0),a.max(0),np.partition(a,-min(2,nclauses),axis=0)[-min(2,nclauses)]])
 else:stats=np.zeros((4,*base.shape),np.float32)
 clause_occlusion=np.r_[stats.reshape(-1),float(nclauses)]
 ca=cand.reshape(2,2,*cand.shape[1:]);candidate_ic=np.r_[ca.min(0).reshape(-1),ca.max(0).reshape(-1),np.abs(ca[0]-ca[1]).reshape(-1),np.abs(ca[:,0]-ca[:,1]).reshape(-1)]
 keen_pair=np.r_[(kh[0]+kh[1]).reshape(-1),np.abs(kh[0]-kh[1]).reshape(-1)]
 return {k:np.asarray(v,np.float32) for k,v in locals().items() if k in FEATURES}

def collect(a,rows):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 out=a.output_dir;(out/"features").mkdir(parents=True,exist_ok=True);(out/"audit_items").mkdir(exist_ok=True);(out/"errors.jsonl").touch(exist_ok=True)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.pad_token=tok.eos_token;model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).to("cuda:0").eval()
 for i,r in enumerate(rows,1):
  if r["key"] in IO_EXCLUSIONS:continue
  fp=out/"features"/(r["key"]+".npz");ip=out/"audit_items"/(r["key"]+".json")
  if a.resume and fp.exists() and ip.exists():continue
  try:
   x,y=r["candidate_pool"];sp,op,cp,cs=prompts(r["question"],x,y);sm=ic_maps(model,tok,sp);om=ic_maps(model,tok,op) if op else np.zeros((0,*sm.shape[1:]),np.float32);cm=ic_maps(model,tok,cp);kh=keen(model,tok,[x,y]);atomic_npz(fp,**fixed_features(sm,om,cm,kh,len(cs)))
   BASE.atomic_json(ip,{"key":r["key"],"known":r["known"],"group":r["group"],"candidate_pool":[x,y],"clauses":cs});BASE.rebuild_items(out);BASE.atomic_json(out/"status.json",{"stage":"collect","completed":len(list((out/"audit_items").glob("*.json"))),"expected":len(rows),"last_key":r["key"],"updated":time.time()});print(f"[{i}/{len(rows)}] {r['key']}",flush=True)
  except Exception as e:BASE.append_error(out/"errors.jsonl",{"key":r["key"],"error":repr(e)});print("ERROR",r["key"],repr(e),flush=True)

def pipe(x,n,seed):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 steps=[StandardScaler()]
 if x.shape[1]>128:steps.append(PCA(n_components=min(24,n-2,x.shape[1]),svd_solver="randomized",random_state=seed))
 steps.append(LogisticRegression(C=.3,max_iter=3000,class_weight="balanced",random_state=seed));return make_pipeline(*steps)
def fit_q(X,y,tr,te,seed):
 from sklearn.linear_model import LogisticRegression
 ps=[]
 for li in range(X.shape[1]):
  m=LogisticRegression(C=.3,max_iter=3000,random_state=seed).fit(X[tr,li],y[tr]);ps.append(m.predict_proba(X[te,li])[:,1])
 return np.mean(ps,0)
def fit_head(name,X,y,tr,te,seed):
 if name=="strong_question":return fit_q(X,y,tr,te,seed)
 m=pipe(X,len(tr),seed).fit(X[tr],y[tr]);return m.predict_proba(X[te])[:,1]
def evaluate(a,rows):
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import StratifiedKFold
 from sklearn.preprocessing import StandardScaler
 out=a.output_dir;have={x["key"] for x in BASE.read_jsonl(out/"items.jsonl")};rows=[r for r in rows if r["key"] in have];y=np.array([r["known"] for r in rows]);keys=[r["key"] for r in rows];B={k:[] for k in FEATURES}
 for r in rows:
  with np.load(out/"features"/(r["key"]+".npz")) as z:
   for k in FEATURES:B[k].append(z[k].astype(np.float32))
 B={k:np.stack(v) for k,v in B.items()};q=[]
 for r in rows:
  with np.load(BASE.QUESTION_CACHE/(r["key"]+".npz")) as z:q.append(z["hidden"][KEEN_LAYERS].astype(np.float32))
 B["strong_question"]=np.stack(q);B["pics"]=np.c_[B["ic_base"],B["semantic_consistency"],B["clause_occlusion"],B["candidate_ic"]]
 heads=["strong_question","ic_base","semantic_consistency","clause_occlusion","candidate_ic","keen_pair","pics"];outer=list(StratifiedKFold(5,shuffle=True,random_state=a.seed).split(np.zeros(len(y)),y));pred={h:np.zeros(len(y)) for h in heads};pred["nested_stack"]=np.zeros(len(y));pred["nested_ic_gate"]=np.zeros(len(y));weights=[];gate_thresholds=[]
 for fold,(tr,te) in enumerate(outer):
  inner=list(StratifiedKFold(3,shuffle=True,random_state=a.seed+fold+1).split(np.zeros(len(tr)),y[tr]));meta=np.zeros((len(tr),4));test=np.zeros((len(te),4))
  for j,h in enumerate(("strong_question","pics","keen_pair","ic_base")):
   for itr,iva in inner:meta[iva,j]=fit_head(h,B[h],y,tr[itr],tr[iva],a.seed+fold)
   test[:,j]=fit_head(h,B[h],y,tr,te,a.seed+fold);pred[h][te]=test[:,j]
  for h in heads:
   if h not in ("strong_question","pics","keen_pair","ic_base"):pred[h][te]=fit_head(h,B[h],y,tr,te,a.seed+fold)
  sc=StandardScaler().fit(meta[:,:3]);m=LogisticRegression(C=.3,class_weight="balanced",max_iter=2000,random_state=a.seed).fit(sc.transform(meta[:,:3]),y[tr]);pred["nested_stack"][te]=m.predict_proba(sc.transform(test[:,:3]))[:,1];weights.append(m.coef_[0].tolist())
  from sklearn.metrics import roc_auc_score
  grid=(.025,.05,.075,.1,.15);gd=max(grid,key=lambda d:roc_auc_score(y[tr],np.where(np.abs(meta[:,0]-.5)<d,meta[:,3],meta[:,0])));gate_thresholds.append(gd);pred["nested_ic_gate"][te]=np.where(np.abs(test[:,0]-.5)<gd,test[:,3],test[:,0])
 result={h:{"overall":BASE.metrics(y,p,.5),"folds":[BASE.metrics(y[te],p[te],.5) for _,te in outer]} for h,p in pred.items()};report={"n":len(y),"known":int(y.sum()),"protocol":"USER-authorized random outer 5-fold; nested 3-fold stacking/gating; all scaler/PCA/base/meta/gate selection training-only; threshold .5","entity_leakage_warning":True,"stack_members":["strong_question","pics","keen_pair"],"meta_coefficients":weights,"gate_grid":[.025,.05,.075,.1,.15],"selected_gate_thresholds":gate_thresholds,"results":result};BASE.atomic_json(out/"evaluation.json",report)
 with (out/"predictions.jsonl").open("w") as f:
  for i,k in enumerate(keys):f.write(json.dumps({"key":k,"known":int(y[i]),"probabilities":{h:float(p[i]) for h,p in pred.items()}})+"\n")
 ranking=sorted((v["overall"]["auroc"],k) for k,v in result.items())[::-1];(out/"summary.md").write_text("# PICS + KEEN\n\n"+"\n".join(f"- {k}: AUROC {v:.4f}" for v,k in ranking)+"\n");BASE.atomic_json(out/"status.json",{"stage":"complete","completed":len(rows),"updated":time.time()});print(json.dumps(ranking,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["selftest","collect","evaluate","all"]);p.add_argument("--limit",type=int,default=128);p.add_argument("--resume",action="store_true");p.add_argument("--batch",type=int,default=1);p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--seed",type=int,default=SEED);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);rows,_=BASE.audit(a.output_dir);rows=BASE.select_balanced(rows,a.limit,a.seed);checks=audit_templates();BASE.atomic_json(a.output_dir/"template_audit.json",checks);BASE.atomic_json(a.output_dir/"config.json",{"model":a.model,"seed":a.seed,"selected_n":len(rows),"io_exclusions":sorted(IO_EXCLUSIONS),"K_last_tokens":K,"max_clauses":MAX_CLAUSES,"keen_layers":KEEN_LAYERS,"feature_whitelist":FEATURES,"nested_stacking":True,"templates":{"semantic":SEMANTIC,"self":SELF,"candidate":CAND}})
 if a.stage=="selftest":assert all(checks.values());assert len(clauses("A. B? C!"))==3;print("2 CPU tests passed");return
 if a.stage in ("collect","all"):collect(a,rows)
 if a.stage in ("evaluate","all"):evaluate(a,rows)
if __name__=="__main__":main()
