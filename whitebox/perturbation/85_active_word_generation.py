#!/usr/bin/env python3
"""Stage 85: put Stage-84 recovered words into the prompt and regenerate."""
from __future__ import annotations
import argparse,importlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from spanattr.core import Item,SpanAttributor,set_seed

def generate(att,ids,n,temp,max_new,seed):
 import torch
 out=[]
 for k in range(n):
  torch.manual_seed(seed+k)
  with torch.inference_mode():
   g=att.model.generate(input_ids=ids.unsqueeze(0),attention_mask=torch.ones_like(ids).unsqueeze(0),
      max_new_tokens=max_new,do_sample=temp>0,temperature=max(temp,1e-5),top_p=.95,
      pad_token_id=getattr(att.tok,"pad_token_id",0) or 0)
  out.append(att.tok.decode(g[0,ids.shape[0]:].tolist()).strip())
 return out

def flags(att,gens,targets): return [bool(att.match_rate([g],targets)>0) for g in gens]

def main():
 p=argparse.ArgumentParser(); p.add_argument("--in84",default="runs/84_active_vocab_decode.jsonl")
 p.add_argument("--items",required=True); p.add_argument("--out",default="runs/85_active_word_generation.jsonl")
 p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct"); p.add_argument("--dtype",default="bfloat16")
 p.add_argument("--device",default="cuda"); p.add_argument("--top_edits",type=int,default=1)
 p.add_argument("--n_gen",type=int,default=10); p.add_argument("--temperature",type=float,default=.8)
 p.add_argument("--max_new_tokens",type=int,default=24); p.add_argument("--seed",type=int,default=42); a=p.parse_args(); set_seed(a.seed)
 items={x.item_id:x for x in [Item.from_dict(d) for d in json.load(open(a.items))]}; rows=[json.loads(x) for x in open(a.in84) if x.strip()]
 loader=importlib.import_module("61_grad_span_proposal").load_model; model,tok=loader(a.model,a.dtype,a.device)
 att=SpanAttributor(model,tok,device=a.device,baseline="mean",length_norm=True,max_rows=16)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,"w") as fh:
  for ni,row in enumerate(rows):
   item=items[row["item_id"]]; prep=att.prepare(item); seed=a.seed+10000*ni
   gold=[item.gold]+item.gold_variants; pred=[item.pred]+item.pred_variants
   base=generate(att,prep.prompt_ids,a.n_gen,a.temperature,a.max_new_tokens,seed)
   bg,bp=flags(att,base,gold),flags(att,base,pred)
   out={"item_id":item.item_id,"n_gen":a.n_gen,"temperature":a.temperature,
        "baseline":{"generations":base,"gold_match":bg,"pred_match":bp,
                    "p_gold":float(np.mean(bg)),"p_pred":float(np.mean(bp))},"strategies":{}}
   for name,s in row["strategies"].items():
    edits=[]
    for rank,edit in enumerate(s["joint_top"][:a.top_edits]):
     ids=prep.prompt_ids.clone()
     for sub in edit["substitutions"]: ids[int(sub["pos"])]=int(sub["id"])
     gens=generate(att,ids,a.n_gen,a.temperature,a.max_new_tokens,seed)
     gm,pm=flags(att,gens,gold),flags(att,gens,pred)
     edits.append({"rank":rank+1,"source_u_realized":edit["u_realized"],
       "substitutions":edit["substitutions"],"generations":gens,"gold_match":gm,"pred_match":pm,
       "p_gold":float(np.mean(gm)),"p_pred":float(np.mean(pm)),
       "rise_p_gold":float(np.mean(gm)-np.mean(bg)),"drop_p_pred":float(np.mean(bp)-np.mean(pm)),
       "paired_pred_to_gold":float(np.mean([bp[k] and gm[k] for k in range(a.n_gen)])),
       "any_gold":any(gm)})
    out["strategies"][name]={"span_text":s["span_text"],"edits":edits}
   fh.write(json.dumps(out,ensure_ascii=False)+"\n"); fh.flush()
   print(item.item_id,{k:({"p_gold":v["edits"][0]["p_gold"],"p_pred":v["edits"][0]["p_pred"]} if v["edits"] else None) for k,v in out["strategies"].items()},flush=True)

if __name__=="__main__": main()
