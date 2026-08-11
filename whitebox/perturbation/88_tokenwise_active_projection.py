#!/usr/bin/env python3
"""Token-wise active-subspace perturbation and discrete projection.

Unlike Stage 87's shared span direction, each token owns an independent
rank-dimensional active coefficient.  Their concatenation is constrained by
one span-level L2 ball, preserving the mean-baseline Frobenius budget.
"""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from spanattr.core import Item,SpanAttributor,set_seed

zo=importlib.import_module("81_zo_span_keywords")
p87=importlib.import_module("87_projection_aware_decode")


def embeds_tokenwise(prep,span,basis,z,budget):
    """z: [N,width*rank], globally unit-bounded; output [N,P,d]."""
    width=span.end-span.start; r=basis.shape[1]
    delta=budget*(z.reshape(len(z),width,r).to(basis.dtype)@basis.T)
    E=prep.E.unsqueeze(0).repeat(len(z),1,1)
    E[:,span.start:span.end]=E[:,span.start:span.end]+delta
    return E,delta


def optimize_tokenwise(att,prep,span,basis,s0,budget,steps,directions,mu,lr,seed):
    import torch
    width=span.end-span.start; dim=width*basis.shape[1]
    gen=torch.Generator(device="cpu").manual_seed(seed)
    z=torch.zeros(dim,device=prep.E.device); best_z=z.clone(); best_s=float(s0)
    rz=zo.project_ball(torch.randn(2*steps*directions,dim,generator=gen)).to(prep.E.device)
    emb,_=embeds_tokenwise(prep,span,basis,rz,budget)
    scores=zo.score_embeds(att,prep,emb); j=int(scores.argmin())
    if float(scores[j])<best_s: best_s=float(scores[j]); best_z=rz[j].clone()
    queries=len(rz)
    for _ in range(steps):
        u=torch.randn(directions,dim,generator=gen)
        u=(u/u.norm(dim=1,keepdim=True).clamp_min(1e-12)).to(prep.E.device)
        zp=zo.project_ball(z[None]+mu*u); zm=zo.project_ball(z[None]-mu*u)
        cand=torch.cat([zp,zm]); emb,_=embeds_tokenwise(prep,span,basis,cand,budget)
        scores=zo.score_embeds(att,prep,emb); queries+=len(cand)
        j=int(scores.argmin())
        if float(scores[j])<best_s: best_s=float(scores[j]); best_z=cand[j].clone()
        sp,sm=scores[:directions].to(z.device),scores[directions:].to(z.device)
        gh=(((sp-sm)/(2*mu))[:,None]*u).mean(0)
        z=zo.project_ball(z-lr*gh/gh.norm().clamp_min(1e-12))
        emb,_=embeds_tokenwise(prep,span,basis,z[None],budget)
        sv=float(zo.score_embeds(att,prep,emb)[0]); queries+=1
        if sv<best_s: best_s=sv; best_z=z.clone()
    _,delta=embeds_tokenwise(prep,span,basis,best_z[None],budget)
    return {"delta":delta[0].float(),"continuous_u":float(s0-best_s),"queries":queries,
            "fro_budget":float(budget)}


def quantized_tokenwise(att,prep,span,basis,s0,budget,n_dirs,rounds,scales,seed,chunk):
    import torch
    width=span.end-span.start; dim=width*basis.shape[1]
    gen=torch.Generator(device="cpu").manual_seed(seed); center=torch.zeros(dim); out=[]
    for scale in scales:
      center.zero_()
      for rd in range(rounds):
        z=torch.randn(n_dirs,dim,generator=gen)
        if rd: z=z+center[None]
        z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)
        _,delta=embeds_tokenwise(prep,span,basis,z.to(basis.device),budget*scale)
        # Stage-87 projector accepts [N,d] shared deltas; implement token-wise target flattening here.
        W=att.emb_layer.weight.detach(); V=W.shape[0]
        targets=(prep.E[span.start:span.end].float()[None]+delta.float()).reshape(-1,W.shape[1])
        best_d=torch.full((len(targets),),float("inf"),device=targets.device)
        best_i=torch.zeros(len(targets),dtype=torch.long,device=targets.device)
        special=set(getattr(att.tok,"all_special_ids",[]) or [])
        for lo in range(0,V,chunk):
            hi=min(V,lo+chunk); X=W[lo:hi].float()
            dist=targets.square().sum(1)[:,None]+X.square().sum(1)[None]-2*targets@X.T
            bad=[v-lo for v in special if lo<=v<hi]
            if bad: dist[:,bad]=float("inf")
            val,idx=dist.min(1); take=val<best_d
            best_d[take]=val[take]; best_i[take]=idx[take]+lo
        vids=best_i.reshape(n_dirs,width); gaps=best_d.clamp_min(0).sqrt().reshape(n_dirs,width)
        ids=prep.prompt_ids.unsqueeze(0).repeat(n_dirs,1)
        for j in range(n_dirs): ids[j,span.start:span.end]=vids[j]
        scores=att.score_ids_batched(prep,ids).numpy(); order=np.argsort(scores)
        center=z[int(order[0])].cpu()
        for q in order[:min(5,len(order))]:
            subs=[]
            for k,t in enumerate(range(span.start,span.end)):
                vid=int(vids[int(q),k]); old=int(prep.prompt_ids[t])
                subs.append({"pos":t,"id":vid,"tok":att.tok.decode([vid]),"orig":att.tok.decode([old]),
                             "target_distance":float(gaps[int(q),k])})
            out.append({"score":float(scores[q]),"u_realized":float(s0-scores[q]),
                        "scale":float(scale),"round":rd,"substitutions":subs})
    return sorted(out,key=lambda x:x["score"])


def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument("--in82",required=True); p.add_argument("--items",required=True); p.add_argument("--basis",required=True)
    p.add_argument("--out",default="runs/88_tokenwise_active_projection.jsonl")
    p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device",default="cuda")
    p.add_argument("--steps",type=int,default=2); p.add_argument("--directions",type=int,default=4)
    p.add_argument("--mu",type=float,default=.25); p.add_argument("--lr",type=float,default=.35)
    p.add_argument("--pool",type=int,default=2); p.add_argument("--top_spans",type=int,default=3)
    p.add_argument("--quant_dirs",type=int,default=16); p.add_argument("--quant_rounds",type=int,default=2)
    p.add_argument("--quant_scales",type=float,nargs="+",default=[1.,2.,3.,4.])
    p.add_argument("--vocab_chunk",type=int,default=4096); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); set_seed(a.seed)
    items={x.item_id:x for x in [Item.from_dict(d) for d in json.load(open(a.items))]}
    rows=[json.loads(x) for x in open(a.in82) if x.strip()]
    loader=importlib.import_module("61_grad_span_proposal").load_model
    model,tok=loader(a.model,a.dtype,a.device)
    att=SpanAttributor(model,tok,device=a.device,baseline="mean",length_norm=True,max_rows=1)
    saved=torch.load(a.basis,map_location="cpu",weights_only=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w") as fh:
      for ni,row in enumerate(rows):
        item=items[row["item_id"]]; prep=att.prepare(item)
        spans=att.build_word_spans(prep,widths=(2,3),stride=1); prep.spans=spans
        rank=int(row["rank"]); B=saved["basis"][:,:rank].to(a.device,dtype=att.emb_layer.weight.dtype)
        s0=att.S0(prep); results=[]
        for sid in row["selection"]["active"][:a.top_spans]:
            sp=spans[sid]
            budget=float((prep.Ebar[sp.start:sp.end]-prep.E[sp.start:sp.end]).float().norm())
            cont=optimize_tokenwise(att,prep,sp,B,s0,budget,a.steps,a.directions,a.mu,a.lr,
                                    a.seed+100003*ni+sid)
            tables=p87.candidate_tables(att,prep,sp,cont["delta"],a.pool,a.vocab_chunk)
            direction=p87.score_combos(att,prep,s0,[x["direction"] for x in tables])
            nearest=p87.score_combos(att,prep,s0,[x["nearest"] for x in tables])
            union=[]
            for x in tables:
                seen={}
                for e in x["direction"]+x["nearest"]: seen[e["id"]]=e
                union.append(list(seen.values()))
            margin=p87.score_combos(att,prep,s0,union)
            quant=quantized_tokenwise(att,prep,sp,B,s0,budget,a.quant_dirs,a.quant_rounds,
                                      a.quant_scales,a.seed+700001*ni+sid,a.vocab_chunk)
            results.append({"span_id":sid,"span_text":sp.text,
                "continuous":{"u_realized":cont["continuous_u"],"crossed":bool(s0-cont["continuous_u"]<0),
                              "fro_budget":budget},
                "direction":direction,"nearest":nearest,"margin_oracle":margin,
                "quantization_aware":quant})
        out={"item_id":item.item_id,"S0":s0,"rank":rank,"results":results,"config":vars(a)}
        fh.write(json.dumps(out,ensure_ascii=False)+"\n"); fh.flush()
        print(item.item_id,[(x["span_text"],x["continuous"]["u_realized"],
          {k:(x[k][0]["u_realized"] if x[k] else None) for k in
           ("direction","nearest","margin_oracle","quantization_aware")}) for x in results],flush=True)

if __name__=="__main__": main()
