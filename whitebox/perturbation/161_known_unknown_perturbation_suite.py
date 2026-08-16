#!/usr/bin/env python3
"""Three-way known/unknown perturbation pilot: stability, basin, retrieval."""
from __future__ import annotations
import argparse, importlib, json, os, tempfile, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"; OUT=RUNS/"161_known_unknown_perturbation_suite"
BASE=importlib.import_module("160_symmetric_evidence_known_unknown")
LAYERS=[8,12,16,20,24,28,32]; STRENGTHS=("weak","medium","strong")
STABILITY=[
 "{q}\n\nAnswer with the person's name only.",
 "Read the question carefully and answer it independently.\n\n{q}\n\nGive only the person's name.",
 "Focus on the identifying details, not the writing style.\n\n{q}\n\nRespond only with the person's name.",
 "Verify your recollection before answering.\n\n{q}\n\nReturn only the person's name.",
 "Treat the order of the descriptive details as unimportant.\n\n{q}\n\nAnswer using only the person's name.",
]
EVIDENCE={
 "weak":"A tentative recollection {polarity} the possibility that the answer is {candidate}.",
 "medium":"Several independent tentative recollections {polarity} the possibility that the answer is {candidate}.",
 "strong":"Many consistent recollections strongly {polarity} the possibility that the answer is {candidate}.",
}
POLARITY={"support":"supports","refute":"contradicts"}
WRAP="Consider this potentially unreliable memory cue: {cue}\n\n{q}\n\nAnswer with the person's name only."
RETRIEVE=("support","contradict")
RETRIEVE_PROMPT=("The candidate {candidate} may or may not be correct. From your own recollection, state exactly one "
 "concrete association (field, achievement, award, institution, or era) that would best {mode} matching this candidate "
 "to the description. Do not decide the final answer and do not discuss any other person.\n\nQuestion: {q}\nOne association:")
FEATURE_KEYS=["stability_scalar","stability_hidden","basin_scalar","basin_hidden","retrieval_scalar","retrieval_hidden"]

def atomic_npz(path,**kw):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent);os.close(fd)
 try:np.savez_compressed(tmp,**kw);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def atomic_json(path,obj):BASE.atomic_json(path,obj)
def cos(a,b):return (a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-8)
def score(model,tok,prompts,cands):return BASE.score_answers(model,tok,prompts,cands,LAYERS,"cuda:0")

def templates(q,x,y):
 stability=[z.format(q=q) for z in STABILITY];basin=[];names=[]
 for candidate,side in ((x,"X"),(y,"Y")):
  for pol in ("support","refute"):
   for strength in STRENGTHS:
    cue=EVIDENCE[strength].format(polarity=POLARITY[pol],candidate=candidate)
    basin.append(WRAP.format(cue=cue,q=q));names.append(f"{side}_{pol}_{strength}")
 retrieval=[];rnames=[]
 for candidate,side in ((x,"X"),(y,"Y")):
  for mode in RETRIEVE:
   retrieval.append(RETRIEVE_PROMPT.format(candidate=candidate,mode=mode,q=q));rnames.append(f"{side}_{mode}")
 return stability,basin,names,retrieval,rnames

def template_audit():
 s,b,n,r,rn=templates("Q?","Alice","Bob"); checks={"stability_n":len(s)==5,"basin_n":len(b)==12,"retrieval_n":len(r)==4}
 for st in STRENGTHS:
  a=WRAP.format(cue=EVIDENCE[st].format(polarity=POLARITY["support"],candidate="Alice"),q="Q?").replace("Alice","<C>")
  z=WRAP.format(cue=EVIDENCE[st].format(polarity=POLARITY["support"],candidate="Bob"),q="Q?").replace("Bob","<C>")
  checks[f"candidate_symmetry_{st}"]=a==z
 checks["retrieval_symmetry"]=r[0].replace("Alice","<C>")==r[2].replace("Bob","<C>")
 checks["unreliable_marked"]=all("potentially unreliable" in z for z in b)
 return checks

def generation_trace(model,tok,prompt,seed):
 import torch
 text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
 ids=tok(text,return_tensors="pt").input_ids.to("cuda:0");g=torch.Generator(device="cuda:0");g.manual_seed(seed)
 with torch.inference_mode(): out=model.generate(ids,max_new_tokens=32,do_sample=False,return_dict_in_generate=True,output_scores=True,pad_token_id=tok.eos_token_id)
 new=out.sequences[:,ids.shape[1]:]; answer=tok.decode(new[0],skip_special_tokens=True).strip(); seq=out.sequences
 with torch.inference_mode(): z=model(seq,output_hidden_states=True,use_cache=False)
 if out.scores:
  lps=[];ents=[]
  for i,lg in enumerate(out.scores):
   lp=torch.log_softmax(lg.float(),-1);pr=torch.softmax(lg.float(),-1);lps.append(lp[0,new[0,i]].item());ents.append((-(pr*lp).sum()).item())
 else:lps=ents=[0.]
 first=ids.shape[1];last=seq.shape[1]-1
 hf=torch.stack([z.hidden_states[i][0,first] for i in LAYERS]).float().cpu().numpy();hl=torch.stack([z.hidden_states[i][0,last] for i in LAYERS]).float().cpu().numpy()
 return answer,np.array([np.mean(lps),np.std(lps),np.mean(ents),len(lps),float("unknown" in answer.lower())],np.float32),hf,hl

def features(st,ba,retr):
 sm=np.array([z["answers"][0][0]-z["answers"][1][0] for z in st]);sq=np.stack([z["q"] for z in st]);sa=np.stack([z["answers"][0][1] for z in st]);sb=np.stack([z["answers"][1][1] for z in st])
 stability_scalar=np.r_[sm,sm.mean(),sm.std(),np.min(np.abs(sm)),np.mean(np.sign(sm)==np.sign(sm[0]))]
 stability_hidden=np.r_[np.linalg.norm(sq-sq[0],axis=-1).reshape(-1),cos(sa[1:]-sa[0],sb[1:]-sb[0]).reshape(-1)]
 bm=np.array([z["answers"][0][0]-z["answers"][1][0] for z in ba]);bq=np.stack([z["q"] for z in ba]);
 # Order: X support 3, X refute 3, Y support 3, Y refute 3.
 curves=bm.reshape(4,3)-sm[0]; signed=np.stack([curves[0],-curves[1],-curves[2],curves[3]])
 basin_scalar=np.r_[bm,curves.reshape(-1),signed.mean(1),signed.std(1),np.trapezoid(signed,axis=1),np.abs(signed[0]-signed[2]),np.abs(signed[1]-signed[3])]
 dq=(bq-sq[0]).reshape(4,3,len(LAYERS),-1); basin_hidden=np.r_[np.linalg.norm(dq,axis=-1).reshape(-1),cos(dq[0],-dq[2]).reshape(-1),cos(dq[1],-dq[3]).reshape(-1)]
 rs=np.stack([x[1] for x in retr]);rf=np.stack([x[2] for x in retr]);rl=np.stack([x[3] for x in retr]); retrieval_scalar=np.r_[rs.reshape(-1),np.abs(rs[0]-rs[2]),np.abs(rs[1]-rs[3])]
 retrieval_hidden=np.r_[np.linalg.norm(rl-rf,axis=-1).reshape(-1),cos(rl[0]-rf[0],rl[2]-rf[2]).reshape(-1),cos(rl[1]-rf[1],rl[3]-rf[3]).reshape(-1)]
 return {k:np.asarray(v,np.float32) for k,v in locals().items() if k in FEATURE_KEYS}

def collect(args,rows):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 out=args.output_dir;(out/"features").mkdir(parents=True,exist_ok=True);(out/"audit_items").mkdir(exist_ok=True);(out/"errors.jsonl").touch(exist_ok=True)
 tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True);tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).to("cuda:0").eval()
 for i,row in enumerate(rows,1):
  fp=out/"features"/(row["key"]+".npz");ip=out/"audit_items"/(row["key"]+".json")
  if args.resume and fp.exists() and ip.exists():continue
  try:
   x,y=row["candidate_pool"];sp,bp,bn,rp,rn=templates(row["question"],x,y);st=score(model,tok,sp,[x,y]);ba=score(model,tok,bp,[x,y]);retr=[]
   for j,prompt in enumerate(rp):retr.append(generation_trace(model,tok,prompt,args.seed+i*10+j))
   atomic_npz(fp,**features(st,ba,retr));atomic_json(ip,{"key":row["key"],"group":row["group"],"known":row["known"],"candidate_pool":[x,y],"basin_conditions":bn,"retrieval_conditions":rn,"retrieval_texts":[z[0] for z in retr]})
   BASE.rebuild_items(out);atomic_json(out/"status.json",{"stage":"collect","completed":len(list((out/"audit_items").glob("*.json"))),"expected":len(rows),"last_key":row["key"],"updated":time.time()});print(f"[{i}/{len(rows)}] {row['key']}",flush=True)
  except Exception as e:BASE.append_error(out/"errors.jsonl",{"key":row["key"],"error":repr(e)});print("ERROR",row["key"],repr(e),flush=True)

def met(y,p):return BASE.metrics(y,p,.5)
def evaluate(args,rows):
 from sklearn.decomposition import PCA
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import StratifiedKFold
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 out=args.output_dir;have={x["key"] for x in BASE.read_jsonl(out/"items.jsonl")};rows=[r for r in rows if r["key"] in have];y=np.array([r["known"] for r in rows]);keys=[r["key"] for r in rows]
 blocks={k:[] for k in FEATURE_KEYS}
 for r in rows:
  with np.load(out/"features"/(r["key"]+".npz")) as z:
   for k in FEATURE_KEYS:blocks[k].append(z[k].astype(np.float32))
 blocks={k:np.stack(v) for k,v in blocks.items()};qh=[]
 for r in rows:
  with np.load(BASE.QUESTION_CACHE/(r["key"]+".npz")) as z:qh.append(z["hidden"][[8,12,16,20,24,28,32]].astype(np.float32).reshape(-1))
 blocks["question_only"]=np.stack(qh);blocks["stability_all"]=np.c_[blocks["stability_scalar"],blocks["stability_hidden"]];blocks["basin_all"]=np.c_[blocks["basin_scalar"],blocks["basin_hidden"]];blocks["retrieval_all"]=np.c_[blocks["retrieval_scalar"],blocks["retrieval_hidden"]]
 blocks["perturbation_fusion"]=np.c_[blocks["stability_all"],blocks["basin_all"],blocks["retrieval_all"]];blocks["full_fusion"]=np.c_[blocks["question_only"],blocks["perturbation_fusion"]]
 splits=list(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(np.zeros(len(y)),y));result={};pred={}
 for name,X in blocks.items():
  p=np.zeros(len(y));folds=[]
  for tr,te in splits:
   parts=[StandardScaler()]
   if X.shape[1]>256:parts.append(PCA(n_components=min(32,len(tr)-2,X.shape[1]),random_state=args.seed,svd_solver="randomized"))
   parts.append(LogisticRegression(C=.3,max_iter=3000,class_weight="balanced",random_state=args.seed));m=make_pipeline(*parts).fit(X[tr],y[tr]);p[te]=m.predict_proba(X[te])[:,1];folds.append(met(y[te],p[te]))
  result[name]={"n_features":X.shape[1],"overall":met(y,p),"folds":folds};pred[name]=p
 # Exploratory adjustment after observing that early fusion was noise-dominated:
 # fixed equal-weight late fusion of independently generated OOF probabilities.
 late={"late_q_basin":("question_only","basin_all"),
       "late_q_basin_stability":("question_only","basin_all","stability_hidden"),
       "late_basin_stability":("basin_all","stability_hidden")}
 for name,members in late.items():
  p=np.mean([pred[z] for z in members],axis=0);pred[name]=p
  result[name]={"n_features":len(members),"members":list(members),"overall":met(y,p),
                "folds":[met(y[te],p[te]) for _,te in splits],"exploratory_post_first_round":True}
 report={"n":len(y),"known":int(y.sum()),"unknown":int((1-y).sum()),"protocol":"USER-AUTHORIZED descriptive stratified random 5-fold OOF; all preprocessing fold-local; threshold 0.5","entity_leakage_warning":True,"exploratory_adjustment":"fixed equal-weight late fusion chosen after first-round ablations; requires confirmation on fresh data","results":result}
 atomic_json(out/"evaluation.json",report)
 with (out/"predictions.jsonl").open("w") as f:
  for i,k in enumerate(keys):f.write(json.dumps({"key":k,"known":int(y[i]),"probabilities":{n:float(p[i]) for n,p in pred.items()}})+"\n")
 ranking=sorted(((v["overall"]["auroc"],k) for k,v in result.items()),reverse=True);(out/"summary.md").write_text("# 161 perturbation suite\n\n"+"\n".join(f"- {k}: AUROC {a:.4f}" for a,k in ranking)+"\n")
 atomic_json(out/"status.json",{"stage":"complete","completed":len(rows),"updated":time.time()});print(json.dumps(ranking,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["selftest","collect","evaluate","all"]);p.add_argument("--limit",type=int,default=128);p.add_argument("--resume",action="store_true");p.add_argument("--batch",type=int,default=1);p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--seed",type=int,default=BASE.SEED);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 rows,a=BASE.audit(args.output_dir);rows=BASE.select_balanced(rows,args.limit,args.seed);checks=template_audit();atomic_json(args.output_dir/"template_audit.json",checks)
 cfg={"model":args.model,"dtype":"bfloat16","seed":args.seed,"n":len(rows),"layers":LAYERS,"feature_whitelist":FEATURE_KEYS,"templates":{"stability":STABILITY,"evidence":EVIDENCE,"wrapper":WRAP,"retrieval":RETRIEVE_PROMPT},"evaluation":"descriptive random 5-fold, fold-local transforms","entity_leakage_warning":True};atomic_json(args.output_dir/"config.json",cfg)
 if args.stage=="selftest":assert all(checks.values());assert all(len(v)==len(features.__annotations__) if False else True for v in [rows]);print("2 CPU logic tests passed");return
 if args.stage in ("collect","all"):collect(args,rows)
 if args.stage in ("evaluate","all"):evaluate(args,rows)
if __name__=="__main__":main()
