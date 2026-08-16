#!/usr/bin/env python3
"""Direct top-2 optimization versus exhaustive pair enumeration.

Uses three white-box proposal signals: a one-backward additive score, a
randomized finite-difference HVP curvature model, and projected continuous
top-2 optimization.  Proposals are evaluated against exact pair truth, but the
reported low-query selection only sees the requested number of verified pairs.
"""
from __future__ import annotations
import argparse, importlib, itertools, json, sys, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from spanattr.core import Item,Span,SpanAttributor,nms_disjoint,set_seed


def capped_simplex(x,total=2.0):
    """Euclidean projection onto {0<=x<=1, sum(x)=total}."""
    x=np.asarray(x,float); lo=float(x.min()-1); hi=float(x.max())
    for _ in range(60):
        mid=(lo+hi)/2; z=np.clip(x-mid,0,1)
        if z.sum()>total: lo=mid
        else: hi=mid
    return np.clip(x-hi,0,1)


def candidate_spans(row,m):
    spans=[Span(i,int(s['start']),int(s['end']),s['text']) for i,s in enumerate(row['spans'])]
    ids=nms_disjoint(np.abs([float(s['u']) for s in row['spans']]),spans,m)
    return ids,[spans[i] for i in ids]


def gate_matrix(prep,spans,device):
    import torch
    M=torch.zeros(len(spans),len(prep.prompt_ids),device=device)
    for i,s in enumerate(spans): M[i,s.start:s.end]=1
    return M


def score_grad(att,prep,M,z):
    import torch
    z=z.detach().clone().requires_grad_(True)
    score=att.S(prep,(z@M).unsqueeze(0),grad=True)[0]
    grad,=torch.autograd.grad(score,z)
    return float(score.detach()),grad.detach()


def curvature_proposal(att,prep,M,s0,rank,eps,seed):
    import torch
    m=len(M); z0=torch.zeros(m,device=att.device)
    _,g=score_grad(att,prep,M,z0)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    V=torch.linalg.qr(torch.randn(m,min(rank,m),generator=gen),mode='reduced')[0].to(att.device)
    cols=[]
    for k in range(V.shape[1]):
        _,gp=score_grad(att,prep,M,eps*V[:,k])
        _,gm=score_grad(att,prep,M,-eps*V[:,k])
        cols.append((gp-gm)/(2*eps))
    Y=torch.stack(cols,dim=1).float()
    Vf=V.float(); core=Vf.T@Y
    H=Y@torch.linalg.pinv(core)@Y.T
    H=((H+H.T)/2).cpu().numpy(); g=g.float().cpu().numpy()
    pairs=list(itertools.combinations(range(m),2))
    additive=[]; quadratic=[]
    for i,j in pairs:
        additive.append(-(g[i]+g[j]))
        quadratic.append(-(g[i]+g[j])-.5*(H[i,i]+2*H[i,j]+H[j,j]))
    return pairs,np.asarray(additive),np.asarray(quadratic),g,H


def continuous_candidates(att,prep,M,steps,restarts,lr,binary_penalty,seed):
    import torch
    m=len(M); gen=torch.Generator(device='cpu').manual_seed(seed); found=[]; traces=[]
    # Include a neutral start, random starts, and starts biased by the local gradient.
    _,g0=score_grad(att,prep,M,torch.zeros(m,device=att.device))
    starts=[capped_simplex((-g0).float().cpu().numpy())]
    for _ in range(max(0,restarts-1)):
        starts.append(capped_simplex(torch.rand(m,generator=gen).numpy()))
    for init in starts:
        z=torch.tensor(init,device=att.device,dtype=torch.float32,requires_grad=True)
        opt=torch.optim.Adam([z],lr=lr); best=(float('inf'),None)
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            score=att.S(prep,(z@M).unsqueeze(0),grad=True)[0]
            penalty=(z*(1-z)).sum()
            loss=score+binary_penalty*penalty
            loss.backward(); opt.step()
            with torch.no_grad():
                z.copy_(torch.tensor(capped_simplex(z.detach().cpu().numpy()),device=z.device,dtype=z.dtype))
                pair=tuple(sorted(torch.topk(z,2).indices.cpu().tolist()))
                if float(score.detach())<best[0]: best=(float(score.detach()),pair)
                found.append(pair)
        traces.append({'best_continuous_score':best[0],'pair':list(best[1])})
    # Frequency first, then best continuous score among traces proposing that pair.
    counts={p:found.count(p) for p in set(found)}
    ranked=sorted(counts,key=lambda p:(-counts[p],min((x['best_continuous_score'] for x in traces if tuple(x['pair'])==p),default=1e9)))
    return ranked,traces


def rank_fusion(rankings,boost=None,k0=20):
    score={}
    for ranking in rankings:
        for r,p in enumerate(ranking): score[p]=score.get(p,0)+1/(k0+r+1)
    for p in boost or []: score[p]=score.get(p,0)+2/k0
    return sorted(score,key=score.get,reverse=True)


def evaluate_order(order,pairs,pair_gain,interaction,singles,budgets):
    pair_to_k={p:k for k,p in enumerate(pairs)}
    utility=np.asarray([singles[i]+singles[j]+interaction[k] for k,(i,j) in enumerate(pairs)])
    best_u=int(np.argmax(utility)); best_i=int(np.argmax(np.abs(interaction)))
    out={}
    for budget in budgets:
        visible=[pair_to_k[p] for p in order[:min(budget,len(order))]]
        # Verification chooses the best exact pair among proposed pairs.
        chosen=max(visible,key=lambda k:pair_gain[k])
        out[str(budget)]={
            'best_effect_recall':bool(best_u in visible),
            'best_interaction_recall':bool(best_i in visible),
            'selected_pair':list(pairs[chosen]),
            'true_best_pair':list(pairs[best_u]),
            'effect_regret':float(pair_gain[best_u]-pair_gain[chosen]),
        }
    return out


def main():
    import torch
    p=argparse.ArgumentParser(); p.add_argument('--in61',default='runs/61.jsonl')
    p.add_argument('--out',default='runs/93_top2_optimization_pilot.json')
    p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--dtype',default='bfloat16'); p.add_argument('--device',default='cuda')
    p.add_argument('--samples',type=int,default=5); p.add_argument('--m',type=int,default=24)
    p.add_argument('--hvp_rank',type=int,default=8); p.add_argument('--hvp_eps',type=float,default=.2)
    p.add_argument('--steps',type=int,default=20); p.add_argument('--restarts',type=int,default=6)
    p.add_argument('--lr',type=float,default=.15); p.add_argument('--binary_penalty',type=float,default=.1)
    p.add_argument('--budgets',type=int,nargs='+',default=[10,20,40])
    p.add_argument('--max_rows',type=int,default=16); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); set_seed(a.seed)
    rows=[json.loads(x) for x in open(a.in61) if x.strip()][:a.samples]
    loader=importlib.import_module('61_grad_span_proposal').load_model
    model,tok=loader(a.model,a.dtype,a.device)
    att=SpanAttributor(model,tok,device=a.device,baseline='mean',length_norm=True,max_rows=a.max_rows)
    outputs=[]; t_all=time.time()
    for ni,row in enumerate(rows):
        item=Item(row['item_id'],row['context'],row['question'],row['gold'],row['pred'],
                  context_prefix=row.get('context_prefix',''),gold_variants=row.get('gold_variants',[]),pred_variants=row.get('pred_variants',[]))
        prep=att.prepare(item); source_ids,spans=candidate_spans(row,a.m); prep.spans=spans
        singles=np.asarray([float(row['spans'][i]['u']) for i in source_ids]); s0=att.S0(prep)
        pairs=list(itertools.combinations(range(len(spans)),2)); t0=time.time()
        gains,_=att.u_of_sets(prep,[list(p) for p in pairs],S0=s0)
        interaction=np.asarray([gains[k]-singles[i]-singles[j] for k,(i,j) in enumerate(pairs)])
        exact_sec=time.time()-t0; M=gate_matrix(prep,spans,a.device)
        hpairs,add,quad,g,H=curvature_proposal(att,prep,M,s0,a.hvp_rank,a.hvp_eps,a.seed+1009*ni)
        cont,traces=continuous_candidates(att,prep,M,a.steps,a.restarts,a.lr,a.binary_penalty,a.seed+100003*ni)
        add_order=[hpairs[k] for k in np.argsort(-add)]
        quad_order=[hpairs[k] for k in np.argsort(-quad)]
        # Main-effect oracle is already paid for by Stage 1 and is a strong baseline.
        main_order=sorted(pairs,key=lambda p:singles[p[0]]+singles[p[1]],reverse=True)
        hybrid=rank_fusion([quad_order,add_order,main_order],boost=cont)
        methods={
            'singleton_sum':evaluate_order(main_order,pairs,gains,interaction,singles,a.budgets),
            'gradient':evaluate_order(add_order,pairs,gains,interaction,singles,a.budgets),
            'hvp_quadratic':evaluate_order(quad_order,pairs,gains,interaction,singles,a.budgets),
            'hybrid':evaluate_order(hybrid,pairs,gains,interaction,singles,a.budgets),
        }
        out={'item_id':row['item_id'],'m':len(spans),'n_pairs':len(pairs),'exact_seconds':exact_sec,
             'source_ids':source_ids,'span_text':[s.text for s in spans],
             'continuous_pairs':[list(x) for x in cont],'continuous_traces':traces,'methods':methods}
        outputs.append(out)
        print(f"[{ni+1}/{len(rows)}] {row['item_id']} m={len(spans)} pairs={len(pairs)} exact={exact_sec:.1f}s cont={cont[:3]}",flush=True)
    summary={}
    for method in outputs[0]['methods']:
        summary[method]={}
        for budget in map(str,a.budgets):
            z=[x['methods'][method][budget] for x in outputs]
            summary[method][budget]={'best_effect_recall':float(np.mean([q['best_effect_recall'] for q in z])),
                                     'best_interaction_recall':float(np.mean([q['best_interaction_recall'] for q in z])),
                                     'effect_regret':float(np.mean([q['effect_regret'] for q in z]))}
    report={'config':vars(a),'elapsed_seconds':time.time()-t_all,'items':outputs,'summary':summary}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2)); print('wrote',a.out)
if __name__=='__main__': main()
