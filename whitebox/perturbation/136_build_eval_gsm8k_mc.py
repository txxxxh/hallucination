#!/usr/bin/env python3
"""Build deterministic two-choice GSM8K and evaluate Llama choice probabilities."""
from __future__ import annotations
import argparse, hashlib, json, re
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT.parent/"other_bench/GSM8K"; OUT=ROOT/"perturbation/runs/136_gsm8k_mc"

def final_number(answer):
 s=answer.rsplit("####",1)[-1].strip().replace(",","")
 m=re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",s)
 if not m: raise ValueError(answer)
 return Decimal(m.group())

def fmt(x):
 s=format(x,"f"); return s.rstrip("0").rstrip(".") if "." in s else s

def distractor(gold, key):
 mag=abs(gold); step=Decimal(1) if mag<20 else max(Decimal(1),(mag*Decimal("0.1")).quantize(Decimal(1)))
 sign=Decimal(1) if int(hashlib.sha1(key.encode()).hexdigest(),16)%2 else Decimal(-1)
 x=gold+sign*step
 if x==gold or (gold>=0 and x<0): x=gold+step
 return x

def build(split,n,seed):
 import pandas as pd
 fn="train-00000-of-00001.parquet" if split=="train" else "test-00000-of-00001 (2).parquet"
 df=pd.read_parquet(SRC/fn)
 if n and n<len(df):
  rng=np.random.default_rng(seed); idx=np.sort(rng.choice(len(df),n,replace=False)); df=df.iloc[idx]
 rows=[]
 for j,(orig,r) in enumerate(df.iterrows()):
  key=f"gsm8k_{split}_{orig:05d}"; gold=final_number(r.answer); wrong=distractor(gold,key)
  flip=int(hashlib.sha1((key+str(seed)).encode()).hexdigest(),16)%2
  order=[gold,wrong] if flip==0 else [wrong,gold]
  q=str(r.question).strip(); prompt=("Choose one of the following two options as the answer to the math problem below. "
      "Output only 1 or 2.\n1. "+fmt(order[0])+"\n2. "+fmt(order[1])+"\nQuestion:\n"+q)
  rows.append({"id":key,"source_index":int(orig),"split":split,"question":q,"gold_solution":r.answer,
   "correct_answer":fmt(gold),"wrong_answer":fmt(wrong),"candidate_order":[fmt(x) for x in order],
   "correct_position":1+order.index(gold),"prepend_names_prompt":prompt})
 return rows

def evaluate(rows,model_name,batch):
 import torch
 try: torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
 except (AttributeError,ImportError): pass
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model_name,use_fast=True,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side="left"
 one=tok.encode("1",add_special_tokens=False);two=tok.encode("2",add_special_tokens=False)
 if len(one)!=1 or len(two)!=1:raise RuntimeError((one,two))
 model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map={"":0},low_cpu_mem_usage=True,attn_implementation="sdpa",local_files_only=True).eval()
 out=[]
 for st in range(0,len(rows),batch):
  b=rows[st:st+batch]; texts=[tok.apply_chat_template([{"role":"user","content":x["prepend_names_prompt"]}],tokenize=False,add_generation_prompt=True) for x in b]
  enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode(): logits=model(**enc,use_cache=False).logits[:,-1,[one[0],two[0]]].float()
  probs=torch.softmax(logits,-1).cpu().numpy()
  for x,p in zip(b,probs):
   choice=1+int(p[1]>p[0]);correct=choice==x["correct_position"]; pred=x["candidate_order"][choice-1]
   out.append({"id":x["id"],"choice":choice,"p_choice1":float(p[0]),"p_choice2":float(p[1]),"generation":pred,
    "name_correct":correct,"name_outcome":"correct" if correct else "wrong","probe_state":"not_applicable"})
  print(f"{st+len(b)}/{len(rows)}",flush=True)
 return out

def dump(path,rows):
 with path.open("w") as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")

def main():
 p=argparse.ArgumentParser();p.add_argument("--train-n",type=int,default=1500);p.add_argument("--seed",type=int,default=42);p.add_argument("--batch",type=int,default=64);p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 for split,n in (("train",a.train_n),("test",0)):
  rows=build(split,n,a.seed);res=evaluate(rows,a.model,a.batch);dump(OUT/f"{split}.jsonl",rows);dump(OUT/f"{split}_llama.jsonl",res)
  print(split,{"n":len(rows),"correct":sum(x["name_correct"] for x in res),"accuracy":np.mean([x["name_correct"] for x in res]),"answer_pos1":sum(x["correct_position"]==1 for x in rows)})
if __name__=="__main__":main()
