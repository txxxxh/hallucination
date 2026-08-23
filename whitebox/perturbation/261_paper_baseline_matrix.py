#!/usr/bin/env python3
"""Paper baselines on the frozen four-dataset Llama matrix.

Representation: Aiersilan (2026), arXiv:2606.02628v1, SAPLMA-style
last-answer-token per-layer linear probe.  Uncertainty: K=6 stochastic
greedy-answer disagreement (exact/canonicalized self-consistency).
"""
from __future__ import annotations
import argparse, importlib, json, re, unicodedata
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"

def read(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def rows(ds):
    if ds=="scientist":
        return importlib.import_module("100_collect_multilayer_trajectory")._scientist_rows("known")
    if ds=="trivia":
        return [dict(key=x["key"],group=x["key"],correct=int(x["correct"]),context=x["context"],question=x["question"],pred=x["generation"],gold=x["other_answer"],raw=x) for x in read(RUNS/"127_triviaqa_balanced_n1000.jsonl")]
    if ds=="gsm8k":
        return [dict(key=x["key"],group=x["group"],correct=int(x["correct"]),context=x["question"],question=x["question"],pred=x["generation"],gold=x["reference_solution"],raw=x) for x in read(RUNS/"140_gsm8k_natural/natural_balanced_n942.jsonl")]
    return [dict(key=x["key"],group=x["group"],correct=int(x["correct"]),context=x["context"],question=x["question"],pred=x["generation"],gold=x["other_answer"],raw=x) for x in read(RUNS/"166_drop1000/drop_balanced_n1000.jsonl")]

def user_text(ds,r):
    if ds=="scientist": return r["raw"]["prompt"]
    if ds=="trivia": return f"Answer using the context. Output only the short answer.\n\nContext:\n{r['context']}\n\nQuestion: {r['question']}"
    if ds=="drop": return f"Read the passage and answer the question. Return only the shortest direct answer, with no explanation.\n\nPassage:\n{r['context']}\n\nQuestion: {r['question']}"
    return "Solve the following grade-school math problem. Show your reasoning step by step. End your response with the final numeric answer in exactly this format: #### <number>\n\nProblem:\n"+r["question"]

def canon_text(x):
    x=unicodedata.normalize("NFKC",str(x)).casefold(); return " ".join(re.sub(r"[^\w\s]"," ",x).split())
def canon_num(x):
    z=re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?",str(x)); return z[-1].replace(",","") if z else "<invalid>"
def scientist_choice(text,r):
    names=[str(r["raw"]["rgt_ans"]),str(r["raw"]["wrg_ans"])]; v=canon_text(text)
    hit=[i for i,n in enumerate(names) if canon_text(n) in v]
    if len(hit)==1:return str(hit[0])
    last=[canon_text(n).split()[-1] for n in names]; hit=[i for i,n in enumerate(last) if re.search(rf"(?<!\w){re.escape(n)}(?!\w)",v)]
    return str(hit[0]) if len(hit)==1 and last[0]!=last[1] else "<invalid>"
def canon(ds,x,r):
    if ds=="scientist": return scientist_choice(x,r)
    if ds=="gsm8k": return canon_num(x)
    return canon_text(x)

def collect(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    rs=rows(args.dataset); out=args.out/args.dataset; (out/"hidden").mkdir(parents=True,exist_ok=True)
    sample_path=out/"samples.jsonl"; done={x["key"]:x for x in read(sample_path)} if args.resume and sample_path.exists() else {}
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={"":0},low_cpu_mem_usage=True,attn_implementation="sdpa",local_files_only=True).eval()
    mode="a" if args.resume and sample_path.exists() else "w"
    with sample_path.open(mode) as fh:
      for st in range(0,len(rs),args.batch):
        part=rs[st:st+args.batch]; rendered=[tok.apply_chat_template([{"role":"user","content":user_text(args.dataset,r)}],tokenize=False,add_generation_prompt=True) for r in part]
        # Paper representation: supplied primary answer, last answer token, every layer.
        pending=[]
        for r,p in zip(part,rendered):
          hp=out/"hidden"/(r["key"]+".npz")
          if not (args.resume and hp.exists()): pending.append((r,p,hp))
        for r,p,hp in pending:
          pi=tok.encode(p,add_special_tokens=False); ai=tok.encode(" "+str(r["pred"]),add_special_tokens=False); ids=torch.tensor([pi+ai],device=model.device)
          with torch.inference_mode(): z=model(ids,output_hidden_states=True,use_cache=False)
          h=torch.stack([q[0,-1] for q in z.hidden_states]).float().cpu().numpy().astype(np.float16)
          np.savez_compressed(hp,key=r["key"],group=r["group"],correct=r["correct"],hidden=h)
        todo=[(r,p) for r,p in zip(part,rendered) if r["key"] not in done]
        if todo:
          z=tok([p for _,p in todo],return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
          torch.manual_seed(args.seed+st)
          with torch.inference_mode(): g=model.generate(**z,do_sample=True,temperature=.7,top_p=.95,num_return_sequences=6,max_new_tokens=192 if args.dataset=="gsm8k" else 32,pad_token_id=tok.pad_token_id)
          texts=tok.batch_decode(g[:,z.input_ids.shape[1]:],skip_special_tokens=True)
          for i,(r,_) in enumerate(todo):
            vals=texts[i*6:(i+1)*6]; target=canon(args.dataset,r["pred"],r); cs=[canon(args.dataset,x,r) for x in vals]
            rec={"key":r["key"],"correct":r["correct"],"greedy":target,"samples":vals,"canonical":cs,"score":1-sum(x==target for x in cs)/6}
            fh.write(json.dumps(rec,ensure_ascii=False)+"\n");fh.flush();done[r["key"]]=rec
        print(args.dataset,min(st+len(part),len(rs)),"/",len(rs),flush=True)

def evaluate(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score,average_precision_score
    from sklearn.model_selection import train_test_split
    out=args.out/args.dataset; sam={x["key"]:x for x in read(out/"samples.jsonl")}; fs=sorted((out/"hidden").glob("*.npz")); hs=[];ys=[];keys=[]
    for f in fs:
      z=np.load(f);keys.append(str(z["key"]));ys.append(1-int(z["correct"]));hs.append(z["hidden"].astype(np.float32))
    H=np.stack(hs);y=np.array(ys); curves=[]
    for seed in (42,43,44):
      pool,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
      tr,_=train_test_split(pool,test_size=.125,stratify=y[pool],random_state=seed)
      vals=[]
      for l in range(1,H.shape[1]):
        clf=LogisticRegression(C=1,max_iter=3000,class_weight="balanced",solver="liblinear",random_state=seed).fit(H[tr,l],y[tr]);p=clf.predict_proba(H[te,l])[:,1];vals.append(float(roc_auc_score(y[te],p)))
      curves.append(vals)
    mean=np.mean(curves,0); best=int(np.argmax(mean))+1; u=np.array([sam[k]["score"] for k in keys])
    report={"dataset":args.dataset,"n":len(y),"errors":int(y.sum()),"representation":{"method":"Aiersilan 2026 SAPLMA-style linear probe; last answer token; 70/10/20 stratified split; 3 seeds","best_layer":best,"mean_auroc":float(mean[best-1]),"per_seed_auroc":[x[best-1] for x in curves],"layer_curve_mean":mean.tolist()},"uncertainty":{"method":"K=6 greedy-answer disagreement; temperature=.7 top_p=.95","auroc":float(roc_auc_score(y,u)),"auprc":float(average_precision_score(y,u))}}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","evaluate"]);p.add_argument("dataset",choices=["scientist","trivia","gsm8k","drop"]);p.add_argument("--batch",type=int,default=8);p.add_argument("--seed",type=int,default=20260822);p.add_argument("--resume",action="store_true");p.add_argument("--out",type=Path,default=RUNS/"261_paper_baseline_matrix");a=p.parse_args();(collect if a.stage=="collect" else evaluate)(a)
if __name__=="__main__":main()
