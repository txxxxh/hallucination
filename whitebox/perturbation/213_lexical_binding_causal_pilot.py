#!/usr/bin/env python3
"""Lexical-realization causal pilot for the fixed fact 'university teacher'."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'
FORMS=[
 'university teacher',
 'university professor',
 'university educator',
 'academic faculty member',
 'higher-education instructor',
 'faculty member at a university',
]

def score(model,tok,prompts,answers,batch):
 return importlib.import_module('212_within_question_binding_competition').candidate_logprob(model,tok,prompts,answers,batch)

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=32);p.add_argument('--out',type=Path,default=RUNS/'213_lexical_binding_causal_pilot');a=p.parse_args()
 import torch
 try:
  from torch._native.registry import deregister_op_overrides
  deregister_op_overrides(disable_op_symbols='bmm')
 except (ImportError,RuntimeError):pass
 a.out.mkdir(parents=True,exist_ok=True)
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');tok.padding_side='left'
 jobs=[x for x in importlib.import_module('152_scientist_attention_pruned_current127').jobs() if 'university teacher' in x[3].casefold()]
 rows=[]
 for key,group,gcorrect,prompt,pred,other in jobs:
  right,wrong=(pred,other) if gcorrect else (other,pred)
  rows.append({'key':key,'group':group,'right':right,'wrong':wrong,'generation_correct':bool(gcorrect),'prompt':prompt})
 # Closed-book cue association, calibrated by a cue-free name prior.
 ps=[];ans=[];idx=[]
 for i,r in enumerate(rows):
  for form in FORMS:
   cue=f"Based only on general background knowledge, complete with the most associated person's name.\nPhrase: {form}\nPerson:"
   for owner,name in [('wrong',r['wrong']),('right',r['right'])]:ps.append(cue);ans.append(' '+name);idx.append((i,form,'cue',owner))
  null='Complete the following with a person\'s name.\nPerson:'
  for owner,name in [('wrong',r['wrong']),('right',r['right'])]:ps.append(null);ans.append(' '+name);idx.append((i,'', 'null',owner))
 vals=score(model,tok,ps,ans,a.batch);cells={k:float(v) for k,v in zip(idx,vals)}
 for i,r in enumerate(rows):
  null=cells[i,'','null','wrong']-cells[i,'','null','right'];r['binding']={f:(cells[i,f,'cue','wrong']-cells[i,f,'cue','right'])-null for f in FORMS}
 # Score original plus every semantic-preserving lexical realization in the original task.
 ps=[];ans=[];idx=[]
 for i,r in enumerate(rows):
  for form in FORMS:
   q=r['prompt'].replace('university teacher',form).replace('University teacher',form.capitalize())
   for owner,name in [('wrong',r['wrong']),('right',r['right'])]:ps.append(q);ans.append(' '+name);idx.append((i,form,owner))
 vals=score(model,tok,ps,ans,a.batch);cells={k:float(v) for k,v in zip(idx,vals)}
 for i,r in enumerate(rows):
  margins={f:cells[i,f,'wrong']-cells[i,f,'right'] for f in FORMS};orig=FORMS[0];lo=min(FORMS,key=lambda f:r['binding'][f]);hi=max(FORMS,key=lambda f:r['binding'][f]);r.update(margins=margins,original_margin=margins[orig],likelihood_error=margins[orig]>0,low_form=lo,high_form=hi,delta_low=margins[lo]-margins[orig],delta_high=margins[hi]-margins[orig],binding_low=r['binding'][lo],binding_original=r['binding'][orig],binding_high=r['binding'][hi],necessity_repair=bool(margins[orig]>0 and margins[lo]<0),sufficiency_induce=bool(margins[orig]<0 and margins[hi]>0))
 with (a.out/'items.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps({k:v for k,v in r.items() if k!='prompt'},ensure_ascii=False)+'\n')
 err=[r for r in rows if r['likelihood_error']];ok=[r for r in rows if not r['likelihood_error']]
 def boot(q,field):
  x=np.array([r[field] for r in q]);rng=np.random.default_rng(42);b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(10000)]);return {'n':len(x),'mean_delta':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'fraction_desired_direction':float(np.mean(x<0 if field=='delta_low' else x>0))}
 report={'cue':'university teacher','forms':FORMS,'total':len(rows),'necessity':{**boot(err,'delta_low'),'repair_n':sum(r['necessity_repair'] for r in err),'repair_rate':float(np.mean([r['necessity_repair'] for r in err]))},'sufficiency':{**boot(ok,'delta_high'),'induce_n':sum(r['sufficiency_induce'] for r in ok),'induce_rate':float(np.mean([r['sufficiency_induce'] for r in ok]))}}
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
