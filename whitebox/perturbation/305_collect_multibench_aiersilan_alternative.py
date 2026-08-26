#!/usr/bin/env python3
"""Collect alternative-candidate layer-14 Aiersilan states on three benchmarks."""
from __future__ import annotations
import argparse,importlib
from pathlib import Path
import numpy as np,torch
src=importlib.import_module("282_aiersilan_exact_original_benchmarks")
RUNS=src.RUNS;OUT=RUNS/"305_multibench_paired_aiersilan/alternative"
def alternative(ds,r):
 if ds in ("trivia","drop"):return r["raw"]["other_answer"]
 return r["raw"]["reference_solution"]
def main():
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 p=argparse.ArgumentParser();p.add_argument("datasets",nargs="+",choices=("trivia","gsm8k","drop"));p.add_argument("--batch",type=int,default=8);a=p.parse_args()
 bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16);tok=AutoTokenizer.from_pretrained(src.MODEL_SNAPSHOT,use_fast=True,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side="right";model=AutoModelForCausalLM.from_pretrained(src.MODEL_SNAPSHOT,quantization_config=bnb,device_map="auto",dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).eval()
 for ds in a.datasets:
  out=OUT/ds;out.mkdir(parents=True,exist_ok=True);rows=src.rows(ds);todo=[r for r in rows if not(out/f"{r['key']}.npy").exists()]
  for st in range(0,len(todo),a.batch):
   part=todo[st:st+a.batch];texts=[src.user_text(ds,r)+" "+str(alternative(ds,r))for r in part];enc=tok(texts,truncation=True,max_length=512,padding=True,return_tensors="pt").to(model.device)
   with torch.inference_mode():h=model(**enc,output_hidden_states=True,use_cache=False).hidden_states[14]
   pos=enc["attention_mask"].sum(1)-1;x=h[torch.arange(len(part),device=h.device),pos].float().cpu().numpy()
   for r,v in zip(part,x):np.save(out/f"{r['key']}.npy",v.astype(np.float16))
   if(st//a.batch)%10==0 or st+a.batch>=len(todo):print(ds,min(st+a.batch,len(todo)),"/",len(todo),flush=True)
if __name__=="__main__":main()
