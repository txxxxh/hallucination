#!/usr/bin/env python3
"""Collect No Answer Needed-style question-final activations before generation."""
import argparse,importlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");ap.add_argument("--out",type=Path,default=RUNS/"147_question_only_hidden_v3");ap.add_argument("--resume",action="store_true");a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 import torch
 load=importlib.import_module("61_grad_span_proposal").load_model;model,tok=load(a.model,"bfloat16","cuda");model.config._attn_implementation="eager";raw={str(x["key"]):x for x in json.load(open(ROOT/"shuffled_prepend_names_question.json"))};rec={x["key"]:x for x in map(json.loads,(ROOT/"tool_gate_correctness_names_llama31_8b"/"records.jsonl").open())};manifest={x["key"]:x for x in map(json.loads,(RUNS/"76_closedbook_fact_probe_manifest.jsonl").open())};rows=[(k,raw[k])for k,r in rec.items()if r.get("parse_valid",True)]
 for num,(k,z)in enumerate(rows,1):
  fp=a.out/(k+".npz")
  if fp.exists()and a.resume:continue
  # Closed-book natural-language question only: no profiles, candidates, answers, or probe text.
  q=str(manifest[k]["question"]).strip();text=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True);ids=tok(text,return_tensors="pt").input_ids.to("cuda")
  qids=tok(q,add_special_tokens=False).input_ids; seq=ids[0].tolist(); starts=[i for i in range(len(seq)-len(qids)+1) if seq[i:i+len(qids)]==qids]; assert starts,(k,len(seq),len(qids)); qlast=starts[-1]+len(qids)-1
  with torch.inference_mode():out=model(ids,output_hidden_states=True,use_cache=False)
  h=torch.stack([x[0,qlast]for x in out.hidden_states]).float().cpu().numpy().astype(np.float16)
  np.savez_compressed(fp,key=np.asarray(k),hidden=h)
  if num==1 or num%100==0:print(f"[{num}/{len(rows)}]",flush=True)
if __name__=="__main__":main()
