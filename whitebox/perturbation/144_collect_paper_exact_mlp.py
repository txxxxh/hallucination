#!/usr/bin/env python3
"""Collect LLMsKnow-style MLP activations at exact-answer token locations."""
import argparse,importlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs"
def rows():
 raw={str(x["key"]):x for x in json.load(open(ROOT/"shuffled_prepend_names_question.json"))};rec={x["key"]:x for x in map(json.loads,(ROOT/"tool_gate_correctness_names_llama31_8b"/"records.jsonl").open())};man={x["key"]:x for x in map(json.loads,(RUNS/"76_closedbook_fact_probe_manifest.jsonl").open())};out=[]
 for k,r in rec.items():
  if not r.get("parse_valid",True):continue
  z=raw[k];pred=str(r["parsed_answer"]);right=str(z["rgt_ans"]);wrong=str(z["wrg_ans"]);out.append({**man[k],"key":k,"raw":z,"pred":pred,"other":wrong if pred==right else right})
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");ap.add_argument("--out",type=Path,default=RUNS/"144_paper_exact_mlp");ap.add_argument("--resume",action="store_true");a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 import torch
 from spanattr.core import Item,SpanAttributor
 model,tok=importlib.import_module("61_grad_span_proposal").load_model(a.model,"bfloat16","cuda");att=SpanAttributor(model,tok,device="cuda",baseline="mean",length_norm=True,max_rows=1);layers=list(range(model.config.num_hidden_layers));acts={};hooks=[]
 for li in layers:
  def hook(_m,_i,o,li=li):acts[li]=o.detach()
  hooks.append(model.model.layers[li].mlp.register_forward_hook(hook))
 try:
  rr=rows()
  for num,r in enumerate(rr,1):
   fp=a.out/(r["key"]+".npz")
   if fp.exists()and a.resume:continue
   item=Item.from_dict(dict(r["raw"],pred=r["pred"],gold=r["other"]));item.pred,item.gold=r["pred"],r["other"];prep=att.prepare(item);ans=prep.pred_variant_ids[0];pe=prep.E.unsqueeze(0);ae=att.emb_layer(ans).detach().unsqueeze(0).to(pe.dtype);seq=torch.cat([pe,ae],1);mask=torch.ones(seq.shape[:2],dtype=torch.long,device="cuda");acts.clear()
   with torch.inference_mode():model(inputs_embeds=seq,attention_mask=mask,use_cache=False)
   first=pe.shape[1];last=first+len(ans)-1;pos=[max(0,first-1),first,last,min(seq.shape[1]-1,last+1)];x=torch.stack([acts[li][0,pos]for li in layers]).float().cpu().numpy().astype(np.float16)
   np.savez_compressed(fp,key=np.asarray(r["key"]),layers=np.asarray(layers),positions=np.asarray(["before_first","first","last","after_last"]),mlp=x)
   if num==1 or num%100==0:print(f"[{num}/{len(rr)}]",flush=True)
 finally:
  for h in hooks:h.remove()
if __name__=="__main__":main()
