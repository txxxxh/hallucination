#!/usr/bin/env python3
"""Stage 82: held-out forward-only ZO in random/vocabulary/active subspaces."""
from __future__ import annotations
import argparse, importlib, json, math, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from spanattr.core import Item,SpanAttributor,build_toy,nms_disjoint,set_seed,spearman
zo=importlib.import_module("81_zo_span_keywords")

SMOKE=[Item("test","red green blue cyan magenta yellow black white",
            "which colour","cyan","yellow")]

def basis_random(d,r,seed):
    import torch
    gen=torch.Generator(device="cpu").manual_seed(seed)
    return torch.linalg.qr(torch.randn(d,r,generator=gen),mode="reduced")[0]

def basis_vocab(weight,r,seed):
    import torch
    gen=torch.Generator(device="cpu").manual_seed(seed)
    ids=torch.randint(weight.shape[0],(2*r,),generator=gen)
    w=weight.detach().float().cpu()
    return torch.linalg.qr((w[ids[:r]]-w[ids[r:]]).T,mode="reduced")[0]

def optimize_abs(att,prep,span,basis,s0,steps,directions,mu,lr,seed):
    """Maximize |S-S0|; only score_embeds forward queries are used."""
    import torch
    width=span.end-span.start
    budget=float((prep.Ebar[span.start:span.end]-prep.E[span.start:span.end]).float().norm())
    radius=budget/math.sqrt(max(width,1)); r=basis.shape[1]
    gen=torch.Generator(device="cpu").manual_seed(seed)
    z=torch.zeros(r,device=prep.E.device); best_s=float(s0); best_abs=0.; trace=[0.]
    nq=2*steps*directions
    rz=zo.project_ball(torch.randn(nq,r,generator=gen)).to(prep.E.device)
    rs=zo.score_embeds(att,prep,zo.embeds_for_z(prep,span,basis,rz,radius))
    random_abs=float((rs-s0).abs().max()); queries=0
    for _ in range(steps):
        u=torch.randn(directions,r,generator=gen)
        u=(u/u.norm(dim=1,keepdim=True).clamp_min(1e-12)).to(prep.E.device)
        zp=zo.project_ball(z[None]+mu*u); zm=zo.project_ball(z[None]-mu*u)
        cand=torch.cat([zp,zm]); scores=zo.score_embeds(
            att,prep,zo.embeds_for_z(prep,span,basis,cand,radius))
        utility=(scores-s0).abs(); queries+=len(cand)
        j=int(utility.argmax())
        if float(utility[j])>best_abs: best_abs,best_s=float(utility[j]),float(scores[j])
        loss=-utility.to(prep.E.device); lp,lm=loss[:directions],loss[directions:]
        gh=(((lp-lm)/(2*mu))[:,None]*u).mean(0)
        z=zo.project_ball(z-lr*gh/gh.norm().clamp_min(1e-12))
        sv=float(zo.score_embeds(att,prep,zo.embeds_for_z(prep,span,basis,z[None],radius))[0]); queries+=1
        if abs(sv-s0)>best_abs: best_abs,best_s=abs(sv-s0),sv
        trace.append(best_abs)
    return {"abs_u":best_abs,"signed_u":float(s0-best_s),"random_best_abs_u":random_abs,
            "fro_budget":budget,"queries":queries,"trace":trace}

def main():
    import torch
    p=argparse.ArgumentParser(); p.add_argument("--items"); p.add_argument("--item_id",nargs="+")
    p.add_argument("--basis",default="runs/81_active_basis.pt"); p.add_argument("--rank",type=int,default=16)
    p.add_argument("--out",default="runs/82_zo_active_keywords.jsonl")
    p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device",default=None)
    p.add_argument("--steps",type=int,default=4); p.add_argument("--directions",type=int,default=8)
    p.add_argument("--mu",type=float,default=.25); p.add_argument("--lr",type=float,default=.35)
    p.add_argument("--topk",type=int,default=5); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--smoke",action="store_true"); a=p.parse_args(); set_seed(a.seed)
    a.device=a.device or ("cuda" if torch.cuda.is_available() else "cpu"); extra={}
    if a.smoke:
        model,tok=build_toy(); items=SMOKE; a.device="cpu"; a.basis="runs/81_active_smoke.pt"
        a.out="runs/82_active_smoke.jsonl"; a.rank=4; a.steps=2; a.directions=3
        extra=dict(prefix="ctx: ",middle=" q: {question} a: ")
    else:
        if not a.items: raise SystemExit("--items is required")
        loader=importlib.import_module("61_grad_span_proposal").load_model
        model,tok=loader(a.model,a.dtype,a.device)
        items=[Item.from_dict(x) for x in json.load(open(a.items))]
        if a.item_id: items=[x for x in items if x.item_id in set(a.item_id)]
    att=SpanAttributor(model,tok,device=a.device,baseline="mean",length_norm=True,max_rows=16,**extra)
    saved=torch.load(a.basis,map_location="cpu",weights_only=True)
    overlap=set(saved["calibration_item_ids"]) & {x.item_id for x in items}
    if overlap: raise ValueError(f"calibration/test leakage: {sorted(overlap)}")
    active=saved["basis"][:,:a.rank]; r=active.shape[1]
    bases={"random":basis_random(att.d,r,a.seed),"vocab":basis_vocab(att.emb_layer.weight,r,a.seed),"active":active}
    bases={k:v.to(a.device,dtype=att.emb_layer.weight.dtype) for k,v in bases.items()}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w") as fh:
      for ni,item in enumerate(items):
        prep=att.prepare(item); spans=att.build_word_spans(prep,widths=(2,3),stride=1); prep.spans=spans
        s0=att.S0(prep); mean,_=att.u_of_sets(prep,[[i] for i in range(len(spans))],S0=s0)
        methods={name:[] for name in bases}
        for name,B in bases.items():
          for i,sp in enumerate(spans): methods[name].append(optimize_abs(
              att,prep,sp,B,s0,a.steps,a.directions,a.mu,a.lr,a.seed+100003*ni+i))
        selection={"mean":nms_disjoint(np.abs(mean),spans,a.topk)}
        for name in bases: selection[name]=nms_disjoint(np.array([x["abs_u"] for x in methods[name]]),spans,a.topk)
        row={"item_id":item.item_id,"S0":s0,"span_text":[s.text for s in spans],"mean_u":mean.tolist(),
             "methods":methods,"selection":selection,
             "keywords":{k:[spans[i].text for i in ids] for k,ids in selection.items()},
             "rho_vs_mean":{k:spearman(np.abs(mean),[x["abs_u"] for x in methods[k]]) for k in bases},
             "rank":r,"config":vars(a),"calibration_item_ids":saved["calibration_item_ids"]}
        fh.write(json.dumps(row,ensure_ascii=False)+"\n"); fh.flush()
        print(item.item_id,json.dumps(row["keywords"],ensure_ascii=False),flush=True)

if __name__=="__main__": main()
