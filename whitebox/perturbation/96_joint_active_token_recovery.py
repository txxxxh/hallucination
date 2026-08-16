#!/usr/bin/env python3
"""Recover real tokens from the best correction-oriented joint-active pair."""
from __future__ import annotations
import argparse, importlib, itertools, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from spanattr.core import Item,SpanAttributor,set_seed
p87=importlib.import_module('87_projection_aware_decode')
p90=importlib.import_module('90_active_tokenwise_generation')
p94=importlib.import_module('94_active_pair_screening_pilot')


def deltas_from_z(prep,spans,basis,z):
    out=[]; lo=0
    for span in spans:
        width=span.end-span.start; dim=width*basis.shape[1]
        budget=float((prep.Ebar[span.start:span.end]-prep.E[span.start:span.end]).float().norm())
        delta=budget*(z[lo:lo+dim].reshape(width,basis.shape[1]).to(basis.dtype)@basis.T)
        out.append(delta.float()); lo+=dim
    return out


def token_choices(att,prep,spans,deltas,basis,pool,chunk):
    tables={}; delta_norm={}
    for span,delta in zip(spans,deltas):
        projected=p87.candidate_tables(att,prep,span,delta,pool,chunk)
        feasible=p87.active_feasible_pool(att,prep,span,basis,pool,chunk)
        for local,pos in enumerate(range(span.start,span.end)):
            old=int(prep.prompt_ids[pos]); seen={old:{'pos':pos,'id':old,'tok':att.tok.decode([old]),
                                                       'orig':att.tok.decode([old]),'source':'original'}}
            for source,rows in [('direction',projected[local]['direction']),
                                ('nearest',projected[local]['nearest']),('feasible',feasible[local])]:
                for e in rows:
                    q=dict(e); q['source']=source; seen[int(e['id'])]=q
            tables[pos]=list(seen.values()); delta_norm[pos]=float(delta[local].norm())
    return tables,delta_norm


def beam_search(att,prep,s0,tables,order,beam_width):
    import torch
    beam=[(prep.prompt_ids.clone(),float(s0),[])]
    for pos in order:
        expanded=[]
        for ids,_,subs in beam:
            for e in tables[pos]:
                q=ids.clone(); q[pos]=int(e['id'])
                ns=subs if int(e['id'])==int(prep.prompt_ids[pos]) else subs+[e]
                expanded.append((q,ns))
        batch=torch.stack([x[0] for x in expanded]); scores=att.score_ids_batched(prep,batch).numpy()
        keep=np.argsort(scores)[:min(beam_width,len(scores))]
        beam=[(expanded[int(k)][0],float(scores[k]),expanded[int(k)][1]) for k in keep]
    return beam


def main():
    import torch
    p=argparse.ArgumentParser(); p.add_argument('--in82',default='runs/82_active_n30_r32_q4.jsonl')
    p.add_argument('--pair_report',default='runs/94_active_pair_screening_n3_m8.json')
    p.add_argument('--items',default='data/items_n128_generation_flip.json')
    p.add_argument('--basis',default='runs/81_q0000_active_basis.pt')
    p.add_argument('--out',default='runs/96_joint_active_token_recovery_n3.jsonl')
    p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--dtype',default='bfloat16'); p.add_argument('--device',default='cuda')
    p.add_argument('--rank',type=int,default=32); p.add_argument('--steps',type=int,default=4)
    p.add_argument('--directions',type=int,default=8); p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--mu',type=float,default=.25); p.add_argument('--lr',type=float,default=.35)
    p.add_argument('--pool',type=int,default=4); p.add_argument('--beam',type=int,default=32)
    p.add_argument('--vocab_chunk',type=int,default=4096); p.add_argument('--n_gen',type=int,default=3)
    p.add_argument('--temperature',type=float,default=1.0); p.add_argument('--max_new_tokens',type=int,default=24)
    p.add_argument('--seed',type=int,default=42); a=p.parse_args(); set_seed(a.seed)
    rows82={x['item_id']:x for x in map(json.loads,open(a.in82))}
    report=json.load(open(a.pair_report)); pair_rows=report['items']
    raw=json.load(open(a.items)); raw_by={str(x.get('item_id',x.get('key'))):x for x in raw}
    items={x.item_id:x for x in (Item.from_dict(d) for d in raw)}
    loader=importlib.import_module('61_grad_span_proposal').load_model
    model,tok=loader(a.model,a.dtype,a.device)
    att=SpanAttributor(model,tok,device=a.device,baseline='mean',length_norm=True,max_rows=16)
    saved=torch.load(a.basis,map_location='cpu',weights_only=True)
    basis=saved['basis'][:,:a.rank].to(a.device,dtype=att.emb_layer.weight.dtype)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,'w') as fh:
      for ni,pr in enumerate(pair_rows):
        item=items[pr['item_id']]; row=rows82[item.item_id]; prep=att.prepare(item)
        all_spans=att.build_word_spans(prep,widths=(2,3),stride=1)
        ids=pr['span_ids']; local_pairs=list(itertools.combinations(range(len(ids)),2))
        best_local=local_pairs[int(np.argmax(pr['joint_u']))]; span_ids=[ids[k] for k in best_local]
        spans=[all_spans[k] for k in span_ids]; s0=att.S0(prep)
        cont=p94.optimize(att,prep,spans,basis,s0,a.steps,a.directions,a.mu,a.lr,
                          a.seed+700001*ni,repeats=a.repeats)
        deltas=deltas_from_z(prep,spans,basis,cont['z'])
        tables,dnorm=token_choices(att,prep,spans,deltas,basis,a.pool,a.vocab_chunk)
        natural=sorted(tables); reverse=list(reversed(natural)); strongest=sorted(tables,key=dnorm.get,reverse=True)
        candidates=[]
        for order_name,order in [('natural',natural),('reverse',reverse),('delta_norm',strongest)]:
            for ids_edit,score,subs in beam_search(att,prep,s0,tables,order,a.beam):
                candidates.append((score,ids_edit,subs,order_name))
        score,edit_ids,subs,order_name=min(candidates,key=lambda x:x[0])
        seed=a.seed+10000*ni
        baseline=p90.generate(att,prep.prompt_ids,a.n_gen,a.temperature,a.max_new_tokens,seed)
        edited=p90.generate(att,edit_ids,a.n_gen,a.temperature,a.max_new_tokens,seed)
        rr=raw_by[item.item_id]; gold=[str(rr.get('eval_gold',rr.get('rgt_ans',item.gold)))]+list(rr.get('eval_gold_variants',[]))
        pred=[item.pred]+item.pred_variants
        bg,bp=p90.flags(att,baseline,gold),p90.flags(att,baseline,pred)
        eg,ep=p90.flags(att,edited,gold),p90.flags(att,edited,pred)
        out={'item_id':item.item_id,'span_ids':span_ids,'span_text':[s.text for s in spans],
             'S0':s0,'continuous_score':cont['score'],'continuous_u':cont['u'],
             'continuous_crossed':bool(cont['score']<0),'discrete_score':score,
             'discrete_u':float(s0-score),'discrete_crossed':bool(score<0),
             'beam_order':order_name,'substitutions':subs,
             'baseline':{'generations':baseline,'gold_match':bg,'pred_match':bp},
             'edit':{'generations':edited,'gold_match':eg,'pred_match':ep,
                     'p_gold':float(np.mean(eg)),'p_pred':float(np.mean(ep)),
                     'rise_p_gold':float(np.mean(eg)-np.mean(bg)),
                     'drop_p_pred':float(np.mean(bp)-np.mean(ep)),
                     'correction_rate_paired':float(np.mean([(not bg[k]) and eg[k] for k in range(a.n_gen)]))}}
        fh.write(json.dumps(out,ensure_ascii=False)+'\n'); fh.flush()
        print(f"[{ni+1}/{len(pair_rows)}] {item.item_id} continuous={cont['score']:+.3f} discrete={score:+.3f} p_gold={out['edit']['p_gold']:.2f}",flush=True)
if __name__=='__main__': main()
