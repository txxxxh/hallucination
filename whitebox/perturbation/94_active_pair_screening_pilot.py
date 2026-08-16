#!/usr/bin/env python3
"""Exact-within-budget token-wise active pair screening pilot.

For each span and span pair, optimize token-wise coefficients in a frozen
active basis.  Each span keeps its own mean-neutralization Frobenius budget;
the pair objective jointly re-optimizes both coefficient blocks.  The resulting
joint-active values are used as truth for testing singleton-sum screening.
"""
from __future__ import annotations
import argparse, importlib, itertools, json, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from spanattr.core import Item,SpanAttributor,set_seed
zo=importlib.import_module('81_zo_span_keywords')


def project_blocks(z,dims):
    """Project every span's coefficient block onto its own unit L2 ball."""
    import torch
    out=z.clone(); lo=0
    for dim in dims:
        block=out[...,lo:lo+dim]
        norm=block.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        block.mul_(torch.clamp(1/norm,max=1.0)); lo+=dim
    return out


def embeds_blocks(prep,spans,basis,z,budgets):
    width_rank=[(s.end-s.start)*basis.shape[1] for s in spans]
    E=prep.E.unsqueeze(0).repeat(len(z),1,1); lo=0
    for span,dim,budget in zip(spans,width_rank,budgets):
        width=span.end-span.start
        delta=budget*(z[:,lo:lo+dim].reshape(len(z),width,basis.shape[1]).to(basis.dtype)@basis.T)
        E[:,span.start:span.end]+=delta; lo+=dim
    return E


def optimize(att,prep,spans,basis,s0,steps,directions,mu,lr,seed,repeats=1,inits=None):
    import torch
    dims=[(s.end-s.start)*basis.shape[1] for s in spans]
    budgets=[float((prep.Ebar[s.start:s.end]-prep.E[s.start:s.end]).float().norm()) for s in spans]
    dim=sum(dims); best_s=float(s0); best_z=torch.zeros(dim,device=att.device); queries=0
    inits=list(inits or [])
    for rep in range(repeats):
        gen=torch.Generator(device='cpu').manual_seed(seed+1000003*rep)
        z=torch.zeros(dim,device=att.device)
        pool=project_blocks(torch.randn(2*steps*directions,dim,generator=gen),dims).to(att.device)
        if inits:
            pool=torch.cat([pool,*[project_blocks(q.detach().cpu()[None],dims).to(att.device) for q in inits]])
        scores=zo.score_embeds(att,prep,embeds_blocks(prep,spans,basis,pool,budgets)); queries+=len(pool)
        j=int(scores.argmin())
        if float(scores[j])<best_s: best_s=float(scores[j]); best_z=pool[j].clone()
        z=pool[j].clone()
        for _ in range(steps):
            u=torch.randn(directions,dim,generator=gen)
            u=(u/u.norm(dim=1,keepdim=True).clamp_min(1e-12)).to(att.device)
            cand=torch.cat([project_blocks(z[None]+mu*u,dims),project_blocks(z[None]-mu*u,dims)])
            scores=zo.score_embeds(att,prep,embeds_blocks(prep,spans,basis,cand,budgets)); queries+=len(cand)
            j=int(scores.argmin())
            if float(scores[j])<best_s: best_s=float(scores[j]); best_z=cand[j].clone()
            sp,sm=scores[:directions].to(att.device),scores[directions:].to(att.device)
            gh=(((sp-sm)/(2*mu))[:,None]*u).mean(0)
            z=project_blocks(z-lr*gh/gh.norm().clamp_min(1e-12),dims)
            sv=float(zo.score_embeds(att,prep,embeds_blocks(prep,spans,basis,z[None],budgets))[0]); queries+=1
            if sv<best_s: best_s=sv; best_z=z.clone()
    return {'u':float(s0-best_s),'score':best_s,'z':best_z,'queries':queries,'dims':dims,'budgets':budgets}


def evaluate(order,pairs,joint,budgets):
    true=int(np.argmax(joint)); out={}
    for B in budgets:
        ids=order[:min(B,len(order))]; chosen=max(ids,key=lambda k:joint[k])
        out[str(B)]={'recall':bool(true in ids),'regret':float(joint[true]-joint[chosen]),
                     'selected_pair':list(pairs[chosen]),'true_pair':list(pairs[true])}
    return out


def main():
    import torch
    p=argparse.ArgumentParser(); p.add_argument('--in82',default='runs/82_active_n30_r32_q4.jsonl')
    p.add_argument('--items',default='data/items_n128_generation_flip.json')
    p.add_argument('--basis',default='runs/81_q0000_active_basis.pt')
    p.add_argument('--out',default='runs/94_active_pair_screening_n3_m8.json')
    p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--dtype',default='bfloat16'); p.add_argument('--device',default='cuda')
    p.add_argument('--samples',type=int,default=3); p.add_argument('--m',type=int,default=8)
    p.add_argument('--rank',type=int,default=32); p.add_argument('--steps',type=int,default=2)
    p.add_argument('--directions',type=int,default=4); p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--mu',type=float,default=.25); p.add_argument('--lr',type=float,default=.35)
    p.add_argument('--budgets',type=int,nargs='+',default=[5,10,20])
    p.add_argument('--max_rows',type=int,default=16); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); set_seed(a.seed)
    rows=[json.loads(x) for x in open(a.in82) if x.strip()][:a.samples]
    items={x.item_id:x for x in (Item.from_dict(d) for d in json.load(open(a.items)))}
    saved=torch.load(a.basis,map_location='cpu',weights_only=True)
    overlap=set(saved['calibration_item_ids'])&{x['item_id'] for x in rows}
    if overlap: raise ValueError(f'calibration leakage: {sorted(overlap)}')
    loader=importlib.import_module('61_grad_span_proposal').load_model
    model,tok=loader(a.model,a.dtype,a.device)
    att=SpanAttributor(model,tok,device=a.device,baseline='mean',length_norm=True,max_rows=a.max_rows)
    basis=saved['basis'][:,:a.rank].to(a.device,dtype=att.emb_layer.weight.dtype)
    outputs=[]; begin=time.time()
    for ni,row in enumerate(rows):
        prep=att.prepare(items[row['item_id']]); all_spans=att.build_word_spans(prep,widths=(2,3),stride=1)
        if [s.text for s in all_spans]!=row['span_text']: raise ValueError('span reconstruction drift')
        ids=row['selection']['active'][:a.m]; spans=[all_spans[i] for i in ids]; s0=att.S0(prep)
        singles=[]
        for k,span in enumerate(spans):
            singles.append(optimize(att,prep,[span],basis,s0,a.steps,a.directions,a.mu,a.lr,
                                    a.seed+100003*ni+1009*k,repeats=a.repeats))
        pairs=list(itertools.combinations(range(len(spans)),2)); joint=[]; t0=time.time()
        for pk,(i,j) in enumerate(pairs):
            # Warm starts guarantee joint search can reproduce either optimized singleton.
            zi,zj=singles[i]['z'],singles[j]['z']
            init=[torch.cat([zi,zj]),torch.cat([zi,torch.zeros_like(zj)]),torch.cat([torch.zeros_like(zi),zj])]
            joint.append(optimize(att,prep,[spans[i],spans[j]],basis,s0,a.steps,a.directions,a.mu,a.lr,
                                  a.seed+700001*ni+7919*pk,repeats=a.repeats,inits=init))
            print(f"  {row['item_id']} pair {pk+1}/{len(pairs)}",flush=True)
        su=np.asarray([x['u'] for x in singles]); ju=np.asarray([x['u'] for x in joint])
        additive=np.asarray([su[i]+su[j] for i,j in pairs]); interaction=ju-additive
        order_add=np.argsort(-additive); order_maxsingle=np.argsort(-np.asarray([max(su[i],su[j]) for i,j in pairs]))
        out={'item_id':row['item_id'],'m':len(spans),'n_pairs':len(pairs),'span_ids':ids,
             'span_text':[s.text for s in spans],'single_u':su.tolist(),'joint_u':ju.tolist(),
             'interaction':interaction.tolist(),'pair_seconds':time.time()-t0,
             'methods':{'active_singleton_sum':evaluate(order_add,pairs,ju,a.budgets),
                        'max_singleton':evaluate(order_maxsingle,pairs,ju,a.budgets)}}
        outputs.append(out); print(f"[{ni+1}/{len(rows)}] {row['item_id']} best={pairs[int(ju.argmax())]} u={ju.max():.3f}",flush=True)
    summary={}
    for method in outputs[0]['methods']:
        summary[method]={}
        for B in map(str,a.budgets):
            z=[x['methods'][method][B] for x in outputs]
            summary[method][B]={'recall':float(np.mean([q['recall'] for q in z])),
                                'regret':float(np.mean([q['regret'] for q in z]))}
    report={'config':vars(a),'elapsed_seconds':time.time()-begin,'items':outputs,'summary':summary}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2)); print('wrote',a.out)
if __name__=='__main__': main()
