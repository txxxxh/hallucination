#!/usr/bin/env python3
"""LaaB logical self-judgment with Scientist perturbation (P) features.

The loss follows ICTMCG/LaaB model/loss_funcs.py.  Both response and judgment
branches use the same gold-free 63-dimensional current127 P representation.
Only the response branch is used for OOF inference.
"""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs"
RAW=ROOT/"shuffled_prepend_names_question.json";RECORDS=ROOT/"tool_gate_correctness_names_llama31_8b"/"records.jsonl"
MANIFEST=RUNS/"76_closedbook_fact_probe_manifest.jsonl";OUT=RUNS/"279_scientist_laab_p";RNEW=RUNS/"135_scientist_full_current127";ROLD=RUNS/"120_physical_delete_rerank"

def read(path):return [json.loads(x)for x in Path(path).open()if x.strip()]
def fixed(x,n=6):x=np.asarray(x,np.float32);return np.pad(x[:n],(0,max(0,n-len(x))))
def ch(x):x=fixed(x);u=x[0]-x[1:];s=abs(float(x[0]))+1e-6;return np.r_[x[0],u,u/s,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(x):x=fixed(x);return np.r_[x[0],x[0]-x[1:]]
def pfeat(a,b,c,d):return np.r_[ch(a),ch(b),ch2(c),ch2(d),a[0]-c[0],b[0]-d[0],(a[0]-b[0])-(c[0]-d[0])].astype(np.float32)
def response_p(key):
 fp=RNEW/f"{key}.npz" if (RNEW/f"{key}.npz").exists()else ROLD/f"{key}.npz"
 with np.load(fp,allow_pickle=True)as z:
  names=("stage1_pred","stage1_other","stage2_pred","stage2_other")if"stage1_pred"in z else("stage1_pred_scores","stage1_other_scores","stage2_pred_scores","stage2_other_scores")
  return pfeat(*(z[x]for x in names))
def judgment_prompt(q,answer):return q+"\n\nProposed answer:\n"+answer+"\n\nIs the proposed answer correct? Answer Yes or No only."
def collect(a):
 import torch
 try:torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm",disable_dispatch_keys="CUDA")
 except(AttributeError,RuntimeError):pass
 from spanattr.core import Item,SpanAttributor,set_seed
 mod=importlib.import_module("125_collect_current_three_benchmarks");raw={str(x["key"]):x for x in json.load(RAW.open())};rows=[x for x in read(RECORDS)if x.get("parse_valid",True)]
 if a.limit and a.limit<len(rows):
  # Spread the pilot across the full Scientist set to cover candidate groups.
  take=np.linspace(0,len(rows)-1,a.limit,dtype=int);rows=[rows[i]for i in take]
 set_seed(42);model,tok=importlib.import_module("61_grad_span_proposal").load_model(a.model,"bfloat16","cuda");att=SpanAttributor(model,tok,device="cuda",baseline="mean",length_norm=True,max_rows=a.batch);a.out.mkdir(parents=True,exist_ok=True)
 for n,r in enumerate(rows,1):
  fp=a.out/f"{r['key']}.npz"
  if a.resume and fp.exists():continue
  text=judgment_prompt(raw[r["key"]]["prompt"],str(r["parsed_answer"]));yes=tok("Yes",add_special_tokens=False,return_tensors="pt").input_ids.to("cuda");no=tok("No",add_special_tokens=False,return_tensors="pt").input_ids.to("cuda");ids=tok(text,return_tensors="pt").input_ids.to("cuda")
  with torch.inference_mode():lp=model(ids,use_cache=False).logits[:,-1].float().log_softmax(-1)
  aff=bool(lp[0,yes[0,0]]>=lp[0,no[0,0]]);pred,other=("Yes","No")if aff else("No","Yes")
  item=Item(r["key"],text,"",other,pred);prep=att.prepare(item);ss,cc=mod.spans(att,prep);p,o=mod.scan(att,prep,ss);u=(p[0]-p[1:])-(o[0]-o[1:]);top=int(np.argmax(np.abs(u)));ids_top=np.argsort(-np.abs(u))[:min(5,len(u))]
  ca,cb=cc[top];deleted=re.sub(r"[ \t]+"," ",text[:ca]+text[cb:]);deleted=re.sub(r"\s+([,.;:!?])",r"\1",deleted).strip();z=att.prepare(Item(r["key"]+"_d",deleted,"",other,pred));ss2,_=mod.spans(att,z);q,s=mod.scan(att,z,ss2);ids2=np.argsort(-np.abs((q[0]-q[1:])-(s[0]-s[1:])))[:min(5,len(q)-1)]
  np.savez_compressed(fp,key=np.asarray(r["key"]),label_R=np.asarray(int(r["correct"])),judgment_affirms=np.asarray(int(aff)),label_E=np.asarray(int(r["correct"])if aff else int(not r["correct"])),judgment=np.asarray(pred),stage1_pred=np.r_[p[0],p[1:][ids_top]],stage1_other=np.r_[o[0],o[1:][ids_top]],stage2_pred=np.r_[q[0],q[1:][ids2]],stage2_other=np.r_[s[0],s[1:][ids2]])
  if n==1 or n%10==0:print(f"[{n}/{len(rows)}] {r['key']} judgment={pred}",flush=True)

def groups(keys):
 # Match the established full-Scientist protocol used by scripts 272/273:
 # questions about the same true person stay in the same fold.
 man={str(x["key"]):x for x in read(MANIFEST)}
 return np.array([man[k]["right_qid"]for k in keys])
def met(y,p):
 from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
 return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}
def evaluate(a):
 import torch;from torch import nn;import torch.nn.functional as F
 from sklearn.model_selection import StratifiedGroupKFold
 class M(nn.Module):
  def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,2))
  def forward(self,x):return self.net(x)
 rows=[]
 for fp in sorted(a.out.glob("question_*.npz")):
  with np.load(fp,allow_pickle=True)as z:rows.append((str(z["key"].item()),int(z["label_R"]),int(z["label_E"]),response_p(str(z["key"].item())),pfeat(z["stage1_pred"],z["stage1_other"],z["stage2_pred"],z["stage2_other"])))
 keys=[x[0]for x in rows];yr=np.array([x[1]for x in rows]);ye=np.array([x[2]for x in rows]);xr=np.stack([x[3]for x in rows]);xe=np.stack([x[4]for x in rows]);g=groups(keys);allp=[];basep=[]
 for seed in(42,43,44):
  torch.manual_seed(seed);po=np.zeros(len(rows));pb=np.zeros(len(rows));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(xr,yr,g):
   mu=xr[tr].mean(0);sd=xr[tr].std(0)+1e-6;ar=torch.tensor((xr[tr]-mu)/sd).float();ae=torch.tensor((xe[tr]-xe[tr].mean(0))/(xe[tr].std(0)+1e-6)).float();br=torch.tensor((xr[te]-mu)/sd).float()
   R,E=M(xr.shape[1]),M(xe.shape[1]);opt=torch.optim.AdamW(list(R.parameters())+list(E.parameters()),lr=5e-4,weight_decay=1e-5);ytr=torch.tensor(yr[tr]);etr=torch.tensor(ye[tr])
   for epoch in range(100):
    R.train();E.train();opt.zero_grad();lr,le=R(ar),E(ae);pr,pe=lr.softmax(1)[:,1],le.softmax(1)[:,1];agree=(ytr==etr).float();logic=agree*F.huber_loss(pr,pe,delta=.5,reduction="none")+(1-agree)*F.huber_loss(pr+pe,torch.ones_like(pr),delta=.5,reduction="none");loss=F.cross_entropy(lr,ytr)+F.cross_entropy(le,etr)+logic.mean();loss.backward();opt.step()
   R.eval();po[te]=R(br).softmax(1)[:,1].detach().numpy()
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   b=make_pipeline(StandardScaler(),LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear")).fit(xr[tr],yr[tr]);pb[te]=b.predict_proba(xr[te])[:,1]
  allp.append(po);basep.append(pb);print(seed,met(yr,pb),met(yr,po),flush=True)
 p=np.mean(allp,0);b=np.mean(basep,0);report={"protocol":"LaaB official Huber logical constraint adapted to response-P/judgment-P; right-person-QID grouped 3x5 OOF; inference uses response branch only","n":len(rows),"P_baseline":met(yr,b),"LaaB_P":met(yr,p),"per_seed":{"baseline":[met(yr,x)for x in basep],"laab":[met(yr,x)for x in allp]}}
 (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("collect","evaluate","all"));p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--batch",type=int,default=32);p.add_argument("--limit",type=int,default=0,help="evenly-spaced collection pilot; 0 means full set");p.add_argument("--out",type=Path,default=OUT);p.add_argument("--resume",action="store_true");a=p.parse_args()
 if a.stage in("collect","all"):collect(a)
 if a.stage in("evaluate","all"):evaluate(a)
if __name__=="__main__":main()
