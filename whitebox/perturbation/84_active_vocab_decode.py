#!/usr/bin/env python3
"""Stage 84: recover discrete words from mean/active spans and directions.

Four crossed strategies separate span selection from direction construction:
mean/active selected spans x gradient/active-ZO correction directions.
One token edit is proposed per disjoint span, then the small Cartesian product
is scored with real vocabulary substitutions and teacher-forced margin.
"""
from __future__ import annotations
import argparse,itertools,json,math,importlib,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from spanattr.core import Item,SpanAttributor,set_seed
zoutil=importlib.import_module("81_zo_span_keywords")

def optimize_correction(att,prep,span,basis,s0,steps,directions,mu,lr,seed):
 import torch
 width=span.end-span.start
 budget=float((prep.Ebar[span.start:span.end]-prep.E[span.start:span.end]).float().norm())
 radius=budget/math.sqrt(max(width,1)); r=basis.shape[1]
 gen=torch.Generator(device="cpu").manual_seed(seed); z=torch.zeros(r,device=prep.E.device)
 best_z=z.clone(); best_s=float(s0)
 # Random initialization is part of the optimizer, not an uncounted oracle.
 rz=zoutil.project_ball(torch.randn(2*steps*directions,r,generator=gen)).to(prep.E.device)
 rs=zoutil.score_embeds(att,prep,zoutil.embeds_for_z(prep,span,basis,rz,radius))
 j=int(rs.argmin())
 if float(rs[j])<best_s: best_s=float(rs[j]); best_z=rz[j].clone()
 queries=len(rz)
 for _ in range(steps):
  u=torch.randn(directions,r,generator=gen); u=(u/u.norm(dim=1,keepdim=True).clamp_min(1e-12)).to(prep.E.device)
  zp=zoutil.project_ball(z[None]+mu*u); zm=zoutil.project_ball(z[None]-mu*u); cand=torch.cat([zp,zm])
  scores=zoutil.score_embeds(att,prep,zoutil.embeds_for_z(prep,span,basis,cand,radius)); queries+=len(cand)
  j=int(scores.argmin())
  if float(scores[j])<best_s: best_s=float(scores[j]); best_z=cand[j].clone()
  sp,sm=scores[:directions].to(prep.E.device),scores[directions:].to(prep.E.device)
  gh=(((sp-sm)/(2*mu))[:,None]*u).mean(0); z=zoutil.project_ball(z-lr*gh/gh.norm().clamp_min(1e-12))
  sv=float(zoutil.score_embeds(att,prep,zoutil.embeds_for_z(prep,span,basis,z[None],radius))[0]); queries+=1
  if sv<best_s: best_s=sv; best_z=z.clone()
 return {"z":best_z.detach(),"delta":radius*(best_z.to(basis.dtype)@basis.T),
         "continuous_u":float(s0-best_s),"fro_budget":budget,"queries":queries}

def vocab_candidates(att,prep,span,directions,topn,chunk):
 """Top normalized vocabulary directions over all token positions in a span."""
 import torch
 W=att.emb_layer.weight.detach(); V=W.shape[0]; special=set(getattr(att.tok,"all_special_ids",[]) or [])
 pool=[]
 for local,t in enumerate(range(span.start,span.end)):
  e=prep.E[t].float(); d=directions[local].float(); best=[]
  for lo in range(0,V,chunk):
   hi=min(V,lo+chunk); D=W[lo:hi].float()-e; score=(D@d)/(D.norm(dim=1)+1e-8)
   k=min(topn,len(score)); val,idx=torch.topk(score,k)
   best.extend((float(v),lo+int(i)) for v,i in zip(val.cpu(),idx.cpu()))
  for pred,v in sorted(best,reverse=True)[:topn*2]:
   if v==int(prep.prompt_ids[t]) or v in special: continue
   text=att.tok.decode([v])
   if not text.strip(): continue
   pool.append({"pos":t,"id":v,"tok":text,"orig":att.tok.decode([int(prep.prompt_ids[t])]),
                "direction_score":pred})
 return sorted(pool,key=lambda x:-x["direction_score"])[:topn]

def measured_span_edits(att,prep,s0,candidates,keep):
 import torch
 if not candidates:return []
 ids=prep.prompt_ids.unsqueeze(0).repeat(len(candidates),1)
 for j,c in enumerate(candidates): ids[j,c["pos"]]=c["id"]
 scores=att.score_ids_batched(prep,ids).numpy(); order=np.argsort(-(s0-scores))[:keep]
 return [dict(candidates[int(j)],u_realized=float(s0-scores[j]),score=float(scores[j])) for j in order]

def joint_search(att,prep,s0,per_span,top,cap):
 if not per_span or any(not x for x in per_span): return []
 combos=list(itertools.product(*per_span))
 if len(combos)>cap: combos=combos[:cap]
 ids=prep.prompt_ids.unsqueeze(0).repeat(len(combos),1)
 for j,combo in enumerate(combos):
  for e in combo: ids[j,e["pos"]]=e["id"]
 scores=att.score_ids_batched(prep,ids).numpy(); order=np.argsort(-(s0-scores))[:top]
 return [{"u_realized":float(s0-scores[j]),"score":float(scores[j]),
          "substitutions":[{k:e[k] for k in ("pos","id","tok","orig","u_realized")} for e in combos[int(j)]]}
         for j in order]

def main():
 import torch
 p=argparse.ArgumentParser(); p.add_argument("--in82",default="runs/82_zo_active_keywords.jsonl")
 p.add_argument("--items",required=True); p.add_argument("--basis",required=True)
 p.add_argument("--out",default="runs/84_active_vocab_decode.jsonl")
 p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct"); p.add_argument("--dtype",default="bfloat16")
 p.add_argument("--device",default="cuda"); p.add_argument("--steps",type=int,default=2); p.add_argument("--directions",type=int,default=4)
 p.add_argument("--mu",type=float,default=.25); p.add_argument("--lr",type=float,default=.35)
 p.add_argument("--topn",type=int,default=6); p.add_argument("--keep_per_span",type=int,default=3)
 p.add_argument("--joint_top",type=int,default=5); p.add_argument("--joint_cap",type=int,default=5000)
 p.add_argument("--vocab_chunk",type=int,default=4096); p.add_argument("--seed",type=int,default=42); a=p.parse_args(); set_seed(a.seed)
 source={x.item_id:x for x in [Item.from_dict(d) for d in json.load(open(a.items))]}
 rows=[json.loads(x) for x in open(a.in82) if x.strip()]
 loader=importlib.import_module("61_grad_span_proposal").load_model; model,tok=loader(a.model,a.dtype,a.device)
 att=SpanAttributor(model,tok,device=a.device,baseline="mean",length_norm=True,max_rows=16)
 saved=torch.load(a.basis,map_location="cpu",weights_only=True); rank=rows[0]["rank"]
 B=saved["basis"][:,:rank].to(a.device,dtype=att.emb_layer.weight.dtype)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,"w") as fh:
  for ni,row in enumerate(rows):
   item=source[row["item_id"]]; prep=att.prepare(item); spans=att.build_word_spans(prep,widths=(2,3),stride=1); prep.spans=spans
   if [s.text for s in spans]!=row["span_text"]: raise ValueError(f"{item.item_id}: span reconstruction drift")
   s0=att.S0(prep); G=att.grad_embed(prep); union=sorted(set(row["selection"]["mean"])|set(row["selection"]["active"]))
   active_dir={i:optimize_correction(att,prep,spans[i],B,s0,a.steps,a.directions,a.mu,a.lr,a.seed+100003*ni+i) for i in union}
   strategies={}
   for span_source in ("mean","active"):
    chosen=row["selection"][span_source]
    for direction_name in ("gradient","active"):
     per=[]
     for i in chosen:
      sp=spans[i]
      if direction_name=="gradient": dirs=[-G[t] for t in range(sp.start,sp.end)]
      else: dirs=[active_dir[i]["delta"].float().cpu().numpy()]*(sp.end-sp.start)
      cand=vocab_candidates(att,prep,sp,dirs,a.topn,a.vocab_chunk)
      per.append(measured_span_edits(att,prep,s0,cand,a.keep_per_span))
     key=f"{span_source}_span__{direction_name}_direction"
     strategies[key]={"span_ids":chosen,"span_text":[spans[i].text for i in chosen],
       "per_span_edits":per,"joint_top":joint_search(att,prep,s0,per,a.joint_top,a.joint_cap)}
   out={"item_id":item.item_id,"S0":s0,"rank":rank,"strategies":strategies,
        "active_continuous":{str(i):{k:v for k,v in active_dir[i].items() if k not in ("z","delta")} for i in union},
        "config":vars(a)}
   fh.write(json.dumps(out,ensure_ascii=False)+"\n"); fh.flush(); print(item.item_id,{k:(v["joint_top"][0]["u_realized"] if v["joint_top"] else None) for k,v in strategies.items()},flush=True)

if __name__=="__main__": main()
