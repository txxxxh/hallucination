#!/usr/bin/env python3
"""Stage 96 restricted to common, ordinary English-like vocabulary tokens."""
from __future__ import annotations
import importlib,re

p87=importlib.import_module('87_projection_aware_decode')
p96=importlib.import_module('96_joint_active_token_recovery')

WORD=re.compile(r"^ ?[A-Za-z]+(?:['.-][A-Za-z]+)*$")
PUNCT=re.compile(r"^ ?[.,;:!?()'-]+$")


def acceptable(e,old_id):
    text=str(e['tok']); vid=int(e['id'])
    return vid==old_id or (vid<50000 and text.isascii() and (WORD.fullmatch(text) or PUNCT.fullmatch(text)))


def token_choices_common(att,prep,spans,deltas,basis,pool,chunk):
    tables={}; delta_norm={}
    for span,delta in zip(spans,deltas):
        wide=max(pool*5,20)
        projected=p87.candidate_tables(att,prep,span,delta,wide,chunk)
        feasible=p87.active_feasible_pool(att,prep,span,basis,wide,chunk)
        for local,pos in enumerate(range(span.start,span.end)):
            old=int(prep.prompt_ids[pos]); old_text=att.tok.decode([old])
            seen={old:{'pos':pos,'id':old,'tok':old_text,'orig':old_text,'source':'original'}}
            for source,rows in [('direction',projected[local]['direction']),
                                ('nearest',projected[local]['nearest']),('feasible',feasible[local])]:
                kept=0
                for e in rows:
                    if not acceptable(e,old): continue
                    q=dict(e); q['source']=source; seen[int(e['id'])]=q; kept+=1
                    if kept>=pool: break
            tables[pos]=list(seen.values()); delta_norm[pos]=float(delta[local].norm())
    return tables,delta_norm


if __name__=='__main__':
    p96.token_choices=token_choices_common
    p96.main()
