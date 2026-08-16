#!/usr/bin/env python3
"""Joint-active span recovery using contextual LM-top-k token candidates."""
from __future__ import annotations
import importlib,re

p96=importlib.import_module('96_joint_active_token_recovery')

WORD=re.compile(r"^ ?[A-Za-z]+(?:['.-][A-Za-z]+)*$")
PUNCT=re.compile(r"^ ?[.,;:!?()'-]+$")


def token_choices_lm(att,prep,spans,deltas,basis,pool,chunk):
    import torch
    with torch.inference_mode():
        logits=att.model(input_ids=prep.prompt_ids.unsqueeze(0),use_cache=False).logits[0].float()
    tables={}; delta_norm={}; special=set(getattr(att.tok,'all_special_ids',[]) or [])
    for span,delta in zip(spans,deltas):
        for local,pos in enumerate(range(span.start,span.end)):
            old=int(prep.prompt_ids[pos]); old_text=att.tok.decode([old])
            rows=[{'pos':pos,'id':old,'tok':old_text,'orig':old_text,'source':'original','lm_logprob':0.0}]
            lp=torch.log_softmax(logits[pos-1],dim=-1); vals,ids=torch.topk(lp,min(512,len(lp)))
            seen={old}; old_leading=old_text.startswith(' ')
            for val,vid0 in zip(vals.cpu(),ids.cpu()):
                vid=int(vid0); text=att.tok.decode([vid])
                if vid in seen or vid in special or vid>=50000: continue
                if not (WORD.fullmatch(text) or PUNCT.fullmatch(text)): continue
                # Preserve the BPE word-boundary role whenever the original has one.
                if old_leading != text.startswith(' '): continue
                seen.add(vid); rows.append({'pos':pos,'id':vid,'tok':text,'orig':old_text,
                                            'source':'causal_lm_topk','lm_logprob':float(val)})
                if len(rows)>=pool+1: break
            tables[pos]=rows; delta_norm[pos]=float(delta[local].norm())
    return tables,delta_norm


if __name__=='__main__':
    p96.token_choices=token_choices_lm
    p96.main()
