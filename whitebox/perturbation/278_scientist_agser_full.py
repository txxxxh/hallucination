#!/usr/bin/env python3
"""Full-Scientist reproduction of AGSER plus the planned P-removal variant.

Paper reproduction: middle-layer last-query-token attention, top 2/3 query
tokens, attentive/non-attentive regeneration, and Rouge-L consistency gap.
The companion P variant regenerates after deleting the cached top P span and a
deterministic word-count-matched control span.  Neither method uses probes.
"""
from __future__ import annotations
import argparse, importlib, json, os, re, tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/"runs"
RAW=ROOT/"shuffled_prepend_names_question.json"
RECORDS=ROOT/"tool_gate_correctness_names_llama31_8b"/"records.jsonl"
OUT=RUNS/"278_scientist_agser_full"; PNEW=RUNS/"135_scientist_full_current127"; POLD=RUNS/"120_physical_delete_rerank"

def norm(s): return re.sub(r"\s+"," ",str(s).strip().casefold())
def lcs(a,b):
 a=a.split();b=b.split();d=[0]*(len(b)+1)
 for x in a:
  old=0
  for j,y in enumerate(b,1):
   z=d[j];d[j]=old+1 if x==y else max(d[j],d[j-1]);old=z
 return d[-1]
def rouge_l(a,b):
 a=norm(a).split();b=norm(b).split()
 if not a or not b:return 0.0
 n=lcs(" ".join(a)," ".join(b));p=n/len(a);r=n/len(b)
 return 2*p*r/(p+r+1e-12)
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".")
 try:
  with os.fdopen(fd,"w")as f:json.dump(obj,f,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def generate(model,tok,text,max_new=64):
 import torch
 z=tok(text,return_tensors="pt",add_special_tokens=True).to(model.device)
 with torch.inference_mode():o=model.generate(**z,do_sample=False,max_new_tokens=max_new,pad_token_id=tok.eos_token_id)
 return tok.decode(o[0,z.input_ids.shape[1]:],skip_special_tokens=True).strip()
def split_attention(model,tok,prompt):
 import torch
 z=tok(prompt,return_tensors="pt",add_special_tokens=True).to(model.device)
 with torch.inference_mode():o=model(**z,output_attentions=True,use_cache=False)
 mid=len(o.attentions)//2;score=o.attentions[mid][0,:,-1,:].float().sum(0)
 ids=z.input_ids[0];valid=torch.arange(len(ids),device=ids.device)
 # Keep special tokens out of either reconstructed user query.
 special=torch.tensor([int(x) in set(tok.all_special_ids) for x in ids],device=ids.device,dtype=torch.bool)
 cand=valid[~special];k=max(1,int(np.ceil(2*len(cand)/3)));top=cand[torch.topk(score[cand],k).indices]
 mask=torch.zeros(len(ids),dtype=torch.bool,device=ids.device);mask[top]=True
 att=tok.decode(ids[mask],skip_special_tokens=True);non=tok.decode(ids[(~mask)&(~special)],skip_special_tokens=True)
 return att.strip(),non.strip(),mid,int(k),int(len(cand))
def delete_once(text,needle):
 i=text.find(needle)
 if i<0:return text
 return re.sub(r"\s+([,.;:!?])",r"\1",re.sub(r"[ \t]+"," ",text[:i]+text[i+len(needle):])).strip()
def matched_control(prompt,deleted,key):
 words=list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b",prompt));want=max(1,len(re.findall(r"\b\w+\b",deleted)))
 if len(words)<=want:return prompt,""
 seed=sum(map(ord,key));starts=list(range(0,len(words)-want+1));starts.sort(key=lambda x:((x*2654435761+seed)&0xffffffff))
 di=prompt.find(deleted)
 for s in starts:
  a,b=words[s].start(),words[s+want-1].end()
  if di<0 or b<=di or a>=di+len(deleted):return delete_once(prompt,prompt[a:b]),prompt[a:b]
 return prompt,""
def p_deleted(key,prompt):
 fp=PNEW/f"{key}.npz" if (PNEW/f"{key}.npz").exists() else POLD/f"{key}.npz"
 if not fp.exists():return None
 with np.load(fp,allow_pickle=True)as z:
  if "deleted_text" not in z:return None
  deleted=str(z["deleted_text"].item())
 return deleted,delete_once(prompt,deleted)
def collect(a):
 import torch
 # Avoid the optional JIT Triton outer-product route; eager bmm is sufficient
 # and works on hosts without Python development headers.
 try: torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm", disable_dispatch_keys="CUDA")
 except (AttributeError, RuntimeError): pass
 raw={str(x["key"]):x for x in json.load(RAW.open())};rec=[x for x in map(json.loads,RECORDS.open()) if x.get("parse_valid",True)]
 model,tok=importlib.import_module("61_grad_span_proposal").load_model(a.model,"bfloat16","cuda");model.config._attn_implementation="eager";model.eval()
 (a.out/"items").mkdir(parents=True,exist_ok=True)
 for n,r in enumerate(rec,1):
  fp=a.out/"items"/f"{r['key']}.json"
  if a.resume and fp.exists():continue
  q=raw[r["key"]]["prompt"];orig=str(r["parsed_answer"]);att,non,layer,k,m=split_attention(model,tok,q)
  suffix="\n\nAnswer with the person's name only."
  ya=generate(model,tok,att+suffix,a.max_new_tokens);yn=generate(model,tok,non+suffix,a.max_new_tokens)
  ra,rn=rouge_l(ya,orig),rouge_l(yn,orig)
  row={"key":r["key"],"error":int(not r["correct"]),"attention_layer":layer,"top_k":k,"query_tokens":m,
       "original":orig,"attentive_query":att,"nonattentive_query":non,"attentive_answer":ya,"nonattentive_answer":yn,
       "r_att":ra,"r_nonatt":rn,"agser_factuality_score":ra-rn,"agser_hallucination_score":rn-ra}
  pd=p_deleted(r["key"],q)
  if pd:
   deleted,qtop=pd;qctl,ctl=matched_control(q,deleted,r["key"]);yt=generate(model,tok,qtop+suffix,a.max_new_tokens);yc=generate(model,tok,qctl+suffix,a.max_new_tokens)
   rt,rc=rouge_l(yt,orig),rouge_l(yc,orig);row.update(p_deleted_text=deleted,control_deleted_text=ctl,
    p_top_answer=yt,p_control_answer=yc,p_top_consistency=rt,p_control_consistency=rc,
    p_behavior_hallucination_score=rc-rt)
  atomic(fp,row)
  if n==1 or n%10==0:print(f"[{n}/{len(rec)}] {r['key']} agser={rn-ra:.3f}",flush=True)
def evaluate(a):
 from sklearn.metrics import roc_auc_score,average_precision_score
 rows=[json.load(x.open())for x in sorted((a.out/"items").glob("*.json"))];y=np.array([x["error"]for x in rows])
 def met(field):
  z=[x for x in rows if field in x];yy=np.array([x["error"]for x in z]);s=np.array([x[field]for x in z]);return {"n":len(z),"auroc":float(roc_auc_score(yy,s)),"auprc":float(average_precision_score(yy,s)),"mean_correct":float(s[yy==0].mean()),"mean_error":float(s[yy==1].mean())}
 report={"protocol":"AGSER paper reproduction: middle-layer attention, top-2/3 token split, greedy regeneration, Rouge-L gap; P variant uses cached top span and deterministic matched-word-count control; no probes", "completed":len(rows),"expected":2894,"results":{x:met(x)for x in("agser_hallucination_score","p_behavior_hallucination_score")}}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("collect","evaluate","all"));p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--out",type=Path,default=OUT);p.add_argument("--max-new-tokens",type=int,default=16);p.add_argument("--resume",action="store_true");a=p.parse_args()
 if a.stage in("collect","all"):collect(a)
 if a.stage in("evaluate","all"):evaluate(a)
if __name__=="__main__":main()
