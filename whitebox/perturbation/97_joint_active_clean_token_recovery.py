#!/usr/bin/env python3
"""Stage 96 with strict natural-text vocabulary filtering."""
from __future__ import annotations
import importlib

p87=importlib.import_module('87_projection_aware_decode')
p96=importlib.import_module('96_joint_active_token_recovery')


def clean_text(s):
    return bool(s.strip()) and s.isascii() and all(c.isprintable() for c in s) and '<|' not in s and '\ufffd' not in s


def token_choices_clean(att,prep,spans,deltas,basis,pool,chunk):
    tables={}; delta_norm={}
    for span,delta in zip(spans,deltas):
        projected=p87.candidate_tables(att,prep,span,delta,max(pool*3,pool),chunk)
        feasible=p87.active_feasible_pool(att,prep,span,basis,max(pool*3,pool),chunk)
        for local,pos in enumerate(range(span.start,span.end)):
            old=int(prep.prompt_ids[pos]); old_text=att.tok.decode([old])
            seen={old:{'pos':pos,'id':old,'tok':old_text,'orig':old_text,'source':'original'}}
            for source,rows in [('direction',projected[local]['direction']),
                                ('nearest',projected[local]['nearest']),('feasible',feasible[local])]:
                kept=0
                for e in rows:
                    if not clean_text(str(e['tok'])): continue
                    q=dict(e); q['source']=source; seen[int(e['id'])]=q; kept+=1
                    if kept>=pool: break
            tables[pos]=list(seen.values()); delta_norm[pos]=float(delta[local].norm())
    return tables,delta_norm


if __name__=='__main__':
    p96.token_choices=token_choices_clean
    p96.main()
