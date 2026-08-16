#!/usr/bin/env python3
"""Run Stage 94 with the Stage-82 absolute active-effect objective."""
from __future__ import annotations
import importlib

p94=importlib.import_module('94_active_pair_screening_pilot')
zo=importlib.import_module('81_zo_span_keywords')


def optimize_abs(att,prep,spans,basis,s0,steps,directions,mu,lr,seed,repeats=1,inits=None):
    import torch
    dims=[(s.end-s.start)*basis.shape[1] for s in spans]
    budgets=[float((prep.Ebar[s.start:s.end]-prep.E[s.start:s.end]).float().norm()) for s in spans]
    dim=sum(dims); best_u=0.; best_s=float(s0); best_z=torch.zeros(dim,device=att.device); queries=0
    inits=list(inits or [])
    for rep in range(repeats):
        gen=torch.Generator(device='cpu').manual_seed(seed+1000003*rep)
        pool=p94.project_blocks(torch.randn(2*steps*directions,dim,generator=gen),dims).to(att.device)
        if inits:
            pool=torch.cat([pool,*[p94.project_blocks(q.detach().cpu()[None],dims).to(att.device) for q in inits]])
        scores=zo.score_embeds(att,prep,p94.embeds_blocks(prep,spans,basis,pool,budgets)); queries+=len(pool)
        utility=(scores-s0).abs(); j=int(utility.argmax()); z=pool[j].clone()
        if float(utility[j])>best_u: best_u=float(utility[j]); best_s=float(scores[j]); best_z=z.clone()
        for _ in range(steps):
            u=torch.randn(directions,dim,generator=gen)
            u=(u/u.norm(dim=1,keepdim=True).clamp_min(1e-12)).to(att.device)
            cand=torch.cat([p94.project_blocks(z[None]+mu*u,dims),p94.project_blocks(z[None]-mu*u,dims)])
            scores=zo.score_embeds(att,prep,p94.embeds_blocks(prep,spans,basis,cand,budgets)); queries+=len(cand)
            utility=(scores-s0).abs(); j=int(utility.argmax())
            if float(utility[j])>best_u: best_u=float(utility[j]); best_s=float(scores[j]); best_z=cand[j].clone()
            lp,lm=-utility[:directions].to(att.device),-utility[directions:].to(att.device)
            gh=(((lp-lm)/(2*mu))[:,None]*u).mean(0)
            z=p94.project_blocks(z-lr*gh/gh.norm().clamp_min(1e-12),dims)
            sv=float(zo.score_embeds(att,prep,p94.embeds_blocks(prep,spans,basis,z[None],budgets))[0]); queries+=1
            if abs(sv-s0)>best_u: best_u=abs(sv-s0); best_s=sv; best_z=z.clone()
    return {'u':best_u,'signed_u':float(s0-best_s),'score':best_s,'z':best_z,
            'queries':queries,'dims':dims,'budgets':budgets}


if __name__=='__main__':
    p94.optimize=optimize_abs
    p94.main()
