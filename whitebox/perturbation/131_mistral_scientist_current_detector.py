#!/usr/bin/env python3
"""Build and evaluate the current compact detector on Mistral-known ScientistQA."""
from __future__ import annotations
import argparse, importlib, json, re, unicodedata
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from spanattr.core import Item, SpanAttributor, set_seed

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/"runs"
DATA=ROOT/"shuffled_prepend_names_question.json"
RAW_RECORDS=ROOT/"tool_gate_correctness_names_mistral7b_v03"/"records.jsonl"
PROBES=RUNS/"130_mistral_closedbook_fact_probe_results.jsonl"
RECORDS=RUNS/"131_mistral_reparsed_records.jsonl"
SOURCE=RUNS/"131_mistral_known_gt05.jsonl"
CACHE=RUNS/"131_mistral_known_gt05_current127"
REPORT=RUNS/"131_mistral_known_gt05_current127_report.json"

def canon(s):
 return " ".join(re.sub(r"[^\w\s]"," ",unicodedata.normalize("NFKC",str(s)).casefold()).split())
def first_choice(text,right,wrong):
 t=canon(text); names=[canon(right),canon(wrong)]; hits=[]
 for i,n in enumerate(names):
  p=t.find(n)
  if p>=0:hits.append((p,i))
 if not hits:
  for i,n in enumerate(names):
   s=n.split()[-1];m=re.search(rf"(?<!\w){re.escape(s)}(?!\w)",t)
   if m:hits.append((m.start(),i))
 return min(hits)[1] if hits else None

def prepare():
 raw=[json.loads(x) for x in RAW_RECORDS.open() if x.strip()]; probes={x["key"]:x for x in map(json.loads,PROBES.open())}; data={str(x["key"]):x for x in json.load(DATA.open())}
 repaired=[]
 for r in raw:
  choice=first_choice(r["generation"],r["right_answer"],r["wrong_answer"]); q=dict(r)
  q["original_parse_valid"]=bool(r["parse_valid"]);q["choice_rule"]="first explicit candidate identity in generation"
  q["parse_valid"]=choice is not None;q["parsed_answer"]=(r["right_answer"] if choice==0 else r["wrong_answer"] if choice==1 else None);q["correct"]=choice==0
  repaired.append(q)
 with RECORDS.open("w") as f:
  for x in repaired:f.write(json.dumps(x,ensure_ascii=False)+"\n")
 by={x["key"]:x for x in repaired}; selected=[]; invalid_known=0
 for key,p in probes.items():
  if not(p["n_discriminative_facts"]>=1 and p["binary_accuracy"]>.5 and p["pairwise_owner_accuracy"]>.5):continue
  r=by[key]
  if not r["parse_valid"]:invalid_known+=1;continue
  selected.append({"key":key,"group":p["right_qid"],"correct":bool(r["correct"]),"pred":r["parsed_answer"],"knowledge_binary_accuracy":p["binary_accuracy"],"knowledge_pairwise_owner_accuracy":p["pairwise_owner_accuracy"]})
 with SOURCE.open("w") as f:
  for x in selected:f.write(json.dumps(x,ensure_ascii=False)+"\n")
 summary={"full_n":len(repaired),"full_parse_valid":sum(x["parse_valid"] for x in repaired),"full_correct":sum(x["correct"] for x in repaired),"knowledge_gt05_before_parse_filter":len(selected)+invalid_known,"excluded_unparsed":invalid_known,"detector_n":len(selected),"groups":len({x["group"] for x in selected}),"correct":sum(x["correct"] for x in selected),"incorrect":sum(not x["correct"] for x in selected)}
 print(json.dumps(summary,indent=2))

def collect(args):
 mod=importlib.import_module("125_collect_current_three_benchmarks");rows=[json.loads(x) for x in SOURCE.open() if x.strip()];data={str(x["key"]):x for x in json.load(DATA.open())};set_seed(42);CACHE.mkdir(parents=True,exist_ok=True)
 model,tok=importlib.import_module("61_grad_span_proposal").load_model(args.model,"bfloat16","cuda");att=SpanAttributor(model,tok,device="cuda",baseline="mean",length_norm=True,max_rows=args.batch)
 for num,r in enumerate(rows,1):
  fp=CACHE/f"{r['key']}.npz"
  if fp.exists() and args.resume:continue
  raw=data[r["key"]];pred=str(r["pred"]);right=str(raw["rgt_ans"]);wrong=str(raw["wrg_ans"]);other=wrong if pred==right else right
  item=Item.from_dict(dict(raw,pred=pred,gold=other));item.pred,item.gold=pred,other;prep=att.prepare(item);ss,cc=mod.spans(att,prep);p1,o1=mod.scan(att,prep,ss);u=(p1[0]-p1[1:])-(o1[0]-o1[1:]);top=int(np.argmax(np.abs(u)));ids=np.argsort(-np.abs(u))[:min(5,len(u))];ph,oh,h14=mod.selected_hidden(att,prep,ids)
  ca,cb=cc[top];deleted=re.sub(r"[ \t]+"," ",item.context[:ca]+item.context[cb:]);deleted=re.sub(r"\s+([,.;:!?])",r"\1",deleted).strip();raw2=dict(raw);raw2["prompt"]=deleted;item2=Item.from_dict(dict(raw2,pred=pred,gold=other));item2.pred,item2.gold=pred,other;prep2=att.prepare(item2);ss2,_=mod.spans(att,prep2);p2,o2=mod.scan(att,prep2,ss2);u2=(p2[0]-p2[1:])-(o2[0]-o2[1:]);ids2=np.argsort(-np.abs(u2))[:min(5,len(u2))]
  np.savez_compressed(fp,key=np.asarray(r["key"]),group=np.asarray(r["group"]),correct=np.asarray(int(r["correct"])),stage1_pred=np.r_[p1[0],p1[1:][ids]],stage1_other=np.r_[o1[0],o1[1:][ids]],stage2_pred=np.r_[p2[0],p2[1:][ids2]],stage2_other=np.r_[o2[0],o2[1:][ids2]],pred_hidden=ph.astype(np.float16),other_hidden=oh.astype(np.float16),layer14=h14.astype(np.float16));print(f"[{num}/{len(rows)}] {r['key']}",flush=True)

def ch(s):
 u=s[0]-s[1:];z=abs(float(s[0]))+1e-6;return np.r_[s[0],u,u/z,u.max(initial=0),u.min(initial=0),np.abs(u).mean(),u.std(),np.mean(u>0)]
def ch2(s):return np.r_[s[0],s[0]-s[1:]]
def wd(h,u):
 d=h[1:].astype(np.float32)-h[0].astype(np.float32);return(d*u[:,None]).sum(0)/(np.abs(u).sum()+1e-9)
def met(y,p):return{"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5))}
def evaluate():
 rows=[]
 for fp in sorted(CACHE.glob("*.npz")):
  with np.load(fp,allow_pickle=True)as z:
   p=z["stage1_pred"].astype(np.float32);o=z["stage1_other"].astype(np.float32);q=z["stage2_pred"].astype(np.float32);r=z["stage2_other"].astype(np.float32);ph=z["pred_hidden"].astype(np.float32);oh=z["other_hidden"].astype(np.float32);scalar=np.r_[ch(p),ch(o),ch2(q),ch2(r),p[0]-q[0],o[0]-r[0],(p[0]-o[0])-(q[0]-r[0])];rows.append((str(z["key"].item()),str(z["group"].item()),int(z["correct"]),scalar,(ph[0],wd(ph,p[0]-p[1:]),oh[0],wd(oh,o[0]-o[1:])),z["layer14"].astype(np.float32)))
 expected=sum(1 for x in SOURCE.open() if x.strip());assert len(rows)==expected,(len(rows),expected)
 keys=np.array([x[0]for x in rows]);g=np.array([x[1]for x in rows]);y=np.array([x[2]for x in rows]);S=np.stack([x[3]for x in rows]);H=[np.stack([x[4][j]for x in rows])for j in range(4)];L=np.stack([x[5]for x in rows]);vals=[]
 for seed in(42,43,44):
  pred=np.zeros(len(y));cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(S,y,g):
   sc=StandardScaler().fit(S[tr]);parts_t=[sc.transform(S[tr])];parts_v=[sc.transform(S[te])]
   for x,d in[*[(x,8)for x in H],(L,48)]:
    sc=StandardScaler().fit(x[tr]);z=sc.transform(x[tr]);pc=PCA(d,whiten=True,svd_solver="randomized",random_state=seed).fit(z);parts_t.append(pc.transform(z));parts_v.append(pc.transform(sc.transform(x[te])))
   clf=LogisticRegression(C=.03,max_iter=5000,class_weight="balanced",solver="liblinear",random_state=seed).fit(np.concatenate(parts_t,1),y[tr]);pred[te]=clf.predict_proba(np.concatenate(parts_v,1))[:,1]
  vals.append(met(y,pred))
 report={"model":"mistralai/Mistral-7B-Instruct-v0.3","selection":"closed-book n_facts>=1, binary_accuracy>0.5, pairwise_owner_accuracy>0.5; parse-valid candidate responses","protocol":"current127 detector; Scientist question-grouped 3x5-fold OOF","n":len(y),"groups":len(set(g)),"correct":int(y.sum()),"incorrect":int(len(y)-y.sum()),"mean":{k:float(np.mean([v[k]for v in vals]))for k in vals[0]},"per_seed":vals}
 REPORT.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","collect","evaluate","all"]);p.add_argument("--model",default="/tmp/Mistral-7B-Instruct-v0.3");p.add_argument("--batch",type=int,default=24);p.add_argument("--resume",action="store_true");a=p.parse_args()
 if a.stage in("prepare","all"):prepare()
 if a.stage in("collect","all"):collect(a)
 if a.stage in("evaluate","all"):evaluate()
if __name__=="__main__":main()
