#!/usr/bin/env python3
"""Collect four pre-confirmed symmetric candidate-gradient concentration scalars."""
import argparse,importlib,os,tempfile
from pathlib import Path
import numpy as np
B=importlib.import_module("160_symmetric_evidence_known_unknown");G=importlib.import_module("173_known_unknown_margin_geometry");RUNS=Path(__file__).resolve().parent/"runs";OUT=RUNS/"182_sparse_candidate_gradients"
def save(path,**v):
 fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent);os.close(fd)
 try:np.savez_compressed(tmp,**v);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def one(model,tok,r):
 import torch
 prompt,a,b=G.prompt_for(r);text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt",return_offsets_mapping=True,add_special_tokens=False);ids=enc.input_ids.cuda();off=enc.offset_mapping[0].tolist();a0=text.find(a);b0=text.find(b);ai=G.span(off,a0,a0+len(a));bi=G.span(off,b0,b0+len(b));aid=tok(a,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();bid=tok(b,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();emb=model.get_input_embeddings();E=emb(ids).detach().requires_grad_(True);m=G.margin(model,emb,E,aid,bid).sum();grad,=torch.autograd.grad(m,E);ca=G.conc(grad[0,ai].float().norm(dim=-1).cpu());cb=G.conc(grad[0,bi].float().norm(dim=-1).cpu());return np.asarray([max(ca[4],cb[4]),min(ca[1],cb[1]),max(ca[6],cb[6]),min(ca[4],cb[4])],np.float32)
def main():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 p=argparse.ArgumentParser();p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--limit",type=int,default=0);p.add_argument("--resume",action="store_true");a=p.parse_args();(a.output_dir/"features").mkdir(parents=True,exist_ok=True);rows,*_=B.load_rows();rows=rows[:a.limit or None];tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).cuda().eval()
 for i,r in enumerate(rows,1):
  fp=a.output_dir/"features"/(r["key"]+".npz")
  if a.resume and fp.exists():continue
  try:save(fp,key=np.asarray(r["key"]),sparse_gradient=one(model,tok,r));print(f"[{i}/{len(rows)}] {r['key']}",flush=True)
  except Exception as e:B.append_error(a.output_dir/"errors.jsonl",{"key":r["key"],"error":repr(e)});print("ERROR",r["key"],repr(e),flush=True)
if __name__=="__main__":main()
