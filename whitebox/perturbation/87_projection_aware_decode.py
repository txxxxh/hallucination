#!/usr/bin/env python3
"""Compare discrete projections and projection-aware active-subspace search.

The first experiment projects the same continuous active target by (1) cosine
direction, (2) nearest target embedding, and (3) true-margin reranking of their
candidate union.  The second experiment searches only realizable vocabulary
displacements, using active-subspace alignment for proposal and true margin for
beam selection.  Thus its final perturbation has zero token-projection gap.
"""
from __future__ import annotations
import argparse, importlib, itertools, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, SpanAttributor, set_seed

stage84 = importlib.import_module("84_active_vocab_decode")


def _valid_token(tok, old_id, new_id):
    if new_id == old_id or new_id in set(getattr(tok, "all_special_ids", []) or []):
        return False
    text = tok.decode([new_id])
    return bool(text.strip())


def candidate_tables(att, prep, span, delta, pool, chunk):
    """Return per-position cosine and nearest-target candidate tables."""
    import torch
    W = att.emb_layer.weight.detach()
    V = W.shape[0]
    ans = []
    for local,t in enumerate(range(span.start, span.end)):
        e = prep.E[t].float()
        raw = delta[local] if getattr(delta,"ndim",1)==2 else delta
        d = torch.as_tensor(raw, device=e.device, dtype=torch.float32)
        target = e + d
        cos_best, near_best = [], []
        for lo in range(0, V, chunk):
            hi = min(V, lo + chunk)
            X = W[lo:hi].float()
            D = X - e
            cos = (D @ d) / (D.norm(dim=1) * d.norm() + 1e-8)
            dist = (X - target).square().sum(1)
            k = min(pool, hi-lo)
            cv, ci = torch.topk(cos, k)
            nv, ni = torch.topk(-dist, k)
            cos_best.extend((float(v), lo+int(i)) for v, i in zip(cv.cpu(), ci.cpu()))
            near_best.extend((-float(v), lo+int(i)) for v, i in zip(nv.cpu(), ni.cpu()))
        old = int(prep.prompt_ids[t])
        def clean(rows, reverse=False):
            rows = sorted(rows, reverse=reverse)
            out, seen = [], set()
            for metric, vid in rows:
                if vid in seen or not _valid_token(att.tok, old, vid): continue
                seen.add(vid)
                vec = W[vid].float()
                disp = vec-e
                out.append({"pos":t,"id":vid,"tok":att.tok.decode([vid]),
                            "orig":att.tok.decode([old]),"cosine":float((disp@d)/(disp.norm()*d.norm()+1e-8)),
                            "target_distance":float((vec-target).norm())})
                if len(out) >= pool: break
            return out
        ans.append({"direction":clean(cos_best, True), "nearest":clean(near_best, False)})
    return ans


def score_combos(att, prep, s0, choices, top=5, cap=20000):
    """True-margin score a Cartesian product of per-position token choices."""
    if not choices or any(not x for x in choices): return []
    combos = list(itertools.product(*choices))
    if len(combos) > cap: combos = combos[:cap]
    ids = prep.prompt_ids.unsqueeze(0).repeat(len(combos), 1)
    for j, combo in enumerate(combos):
        for e in combo: ids[j, e["pos"]] = e["id"]
    scores = att.score_ids_batched(prep, ids).numpy()
    order = np.argsort(scores)[:top]
    return [{"score":float(scores[j]), "u_realized":float(s0-scores[j]),
             "substitutions":list(combos[int(j)])} for j in order]


def active_feasible_pool(att, prep, span, basis, pool, chunk):
    """Tokens with the nearest reachable Voronoi boundary in active space."""
    import torch
    W = att.emb_layer.weight.detach(); V = W.shape[0]
    out = []
    for t in range(span.start, span.end):
        e = prep.E[t].float(); best=[]
        for lo in range(0,V,chunk):
            hi=min(V,lo+chunk); D=W[lo:hi].float()-e
            proj=(D@basis.float()).norm(dim=1)
            boundary=D.square().sum(1)/(2*proj+1e-8)
            k=min(pool,len(boundary)); val,idx=torch.topk(-boundary,k)
            best.extend((-float(v),lo+int(i)) for v,i in zip(val.cpu(),idx.cpu()))
        old=int(prep.prompt_ids[t]); rows=[]; seen=set()
        for boundary,vid in sorted(best):
            if vid in seen or not _valid_token(att.tok,old,vid): continue
            seen.add(vid); rows.append({"pos":t,"id":vid,"tok":att.tok.decode([vid]),
                "orig":att.tok.decode([old]),"active_boundary_radius":boundary})
            if len(rows)>=pool: break
        out.append(rows)
    return out


def project_targets_nearest(att, prep, span, deltas, chunk):
    """Project a batch of shared span deltas to exact nearest vocabulary rows."""
    import torch
    W=att.emb_layer.weight.detach(); V=W.shape[0]; width=span.end-span.start
    targets=(prep.E[span.start:span.end].float()[None]+deltas[:,None].float()).reshape(-1,W.shape[1])
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
    return best_i.reshape(len(deltas),width),best_d.clamp_min(0).sqrt().reshape(len(deltas),width)


def quantization_aware_search(att,prep,span,basis,s0,radius,n_dirs,rounds,scales,seed,chunk):
    """Optimize projected-token margin, putting vocabulary sparsity inside search."""
    import torch
    gen=torch.Generator(device="cpu").manual_seed(seed); r=basis.shape[1]
    center=torch.zeros(r); all_rows=[]
    for scale in scales:
      center.zero_()
      for rd in range(rounds):
        z=torch.randn(n_dirs,r,generator=gen)
        if rd: z=z+center[None]
        z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)
        delta=(radius*scale)*(z.to(basis.device,dtype=basis.dtype)@basis.T)
        vids,gaps=project_targets_nearest(att,prep,span,delta,chunk)
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
            all_rows.append({"score":float(scores[q]),"u_realized":float(s0-scores[q]),
                             "scale":float(scale),"round":rd,"substitutions":subs})
    # Keep all retained candidates so scale-specific failures are observable;
    # callers may still use element zero as the global best.
    return sorted(all_rows,key=lambda x:x["score"])


def boundary_targeted_search(att,prep,span,basis,s0,radius,proposals,scales,chunk):
    """Aim active directions at nearby vocabulary Voronoi boundaries."""
    import torch
    W=att.emb_layer.weight.detach(); zrows=[]; meta=[]
    for rows in proposals:
        for e in rows:
            d=W[e["id"]].float()-prep.E[e["pos"]].float()
            z=d@basis.float()
            if float(z.norm())>1e-8:
                zrows.append(z/z.norm()); meta.append(e)
    if not zrows: return []
    Z=torch.stack(zrows); out=[]
    for scale in scales:
        delta=(radius*scale)*(Z.to(basis.dtype)@basis.T)
        vids,gaps=project_targets_nearest(att,prep,span,delta,chunk)
        ids=prep.prompt_ids.unsqueeze(0).repeat(len(Z),1)
        for j in range(len(Z)): ids[j,span.start:span.end]=vids[j]
        scores=att.score_ids_batched(prep,ids).numpy()
        for j in range(len(Z)):
            subs=[]
            for k,t in enumerate(range(span.start,span.end)):
                vid=int(vids[j,k]); old=int(prep.prompt_ids[t])
                subs.append({"pos":t,"id":vid,"tok":att.tok.decode([vid]),"orig":att.tok.decode([old]),
                             "target_distance":float(gaps[j,k])})
            out.append({"score":float(scores[j]),"u_realized":float(s0-scores[j]),"scale":float(scale),
                        "boundary_target":meta[j],"substitutions":subs})
    return sorted(out,key=lambda x:x["score"])


def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument("--in82",required=True); p.add_argument("--items",required=True); p.add_argument("--basis",required=True)
    p.add_argument("--out",default="runs/87_projection_aware.jsonl")
    p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device",default="cuda")
    p.add_argument("--steps",type=int,default=2); p.add_argument("--directions",type=int,default=4)
    p.add_argument("--mu",type=float,default=.25); p.add_argument("--lr",type=float,default=.35)
    p.add_argument("--pool",type=int,default=12); p.add_argument("--top_spans",type=int,default=5)
    p.add_argument("--quant_dirs",type=int,default=16); p.add_argument("--quant_rounds",type=int,default=2)
    p.add_argument("--quant_scales",type=float,nargs="+",default=[1.,2.,3.,4.])
    p.add_argument("--vocab_chunk",type=int,default=4096); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); set_seed(a.seed)
    items={x.item_id:x for x in [Item.from_dict(d) for d in json.load(open(a.items))]}
    rows=[json.loads(x) for x in open(a.in82) if x.strip()]
    loader=importlib.import_module("61_grad_span_proposal").load_model
    model,tok=loader(a.model,a.dtype,a.device)
    # Keep discrete scores batch-invariant under bfloat16 kernels.
    att=SpanAttributor(model,tok,device=a.device,baseline="mean",length_norm=True,max_rows=1)
    saved=torch.load(a.basis,map_location="cpu",weights_only=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w") as fh:
      for ni,row in enumerate(rows):
        item=items[row["item_id"]]; prep=att.prepare(item)
        spans=att.build_word_spans(prep,widths=(2,3),stride=1); prep.spans=spans
        if [s.text for s in spans] != row["span_text"]: raise ValueError("span reconstruction drift")
        rank=int(row["rank"]); B=saved["basis"][:,:rank].to(a.device,dtype=att.emb_layer.weight.dtype)
        s0=att.S0(prep); chosen=row["selection"]["active"][:a.top_spans]; results=[]
        for sid in chosen:
          sp=spans[sid]
          cont=stage84.optimize_correction(att,prep,sp,B,s0,a.steps,a.directions,a.mu,a.lr,a.seed+100003*ni+sid)
          delta=cont["delta"].float()
          tables=candidate_tables(att,prep,sp,delta,a.pool,a.vocab_chunk)
          direction=score_combos(att,prep,s0,[x["direction"][:a.pool] for x in tables])
          nearest=score_combos(att,prep,s0,[x["nearest"][:a.pool] for x in tables])
          union=[]
          for x in tables:
            seen={};
            for e in x["direction"]+x["nearest"]: seen[e["id"]]=e
            union.append(list(seen.values()))
          margin=score_combos(att,prep,s0,union)
          feasible=active_feasible_pool(att,prep,sp,B,max(a.pool,16),a.vocab_chunk)
          projection_aware=score_combos(att,prep,s0,[x[:a.pool] for x in feasible])
          width=sp.end-sp.start
          budget=float((prep.Ebar[sp.start:sp.end]-prep.E[sp.start:sp.end]).float().norm())
          quantized=quantization_aware_search(att,prep,sp,B,s0,budget/math.sqrt(max(width,1)),
              a.quant_dirs,a.quant_rounds,a.quant_scales,a.seed+700001*ni+sid,a.vocab_chunk)
          boundary=boundary_targeted_search(att,prep,sp,B,s0,budget/math.sqrt(max(width,1)),
              feasible,a.quant_scales,a.vocab_chunk)
          results.append({"span_id":sid,"span_text":sp.text,
            "continuous":{"u_realized":cont["continuous_u"],"crossed":bool(s0-cont["continuous_u"]<0)},
            "direction":direction,"nearest":nearest,"margin_oracle":margin,
            "projection_aware":projection_aware,"quantization_aware":quantized,
            "boundary_targeted":boundary})
        out={"item_id":item.item_id,"S0":s0,"rank":rank,"results":results,"config":vars(a)}
        fh.write(json.dumps(out,ensure_ascii=False)+"\n"); fh.flush()
        print(item.item_id,[(x["span_text"],{k:(x[k][0]["u_realized"] if x[k] else None) for k in
          ("direction","nearest","margin_oracle","projection_aware","quantization_aware","boundary_targeted")}) for x in results],flush=True)

if __name__=="__main__": main()
