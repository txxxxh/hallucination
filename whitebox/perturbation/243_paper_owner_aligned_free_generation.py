#!/usr/bin/env python3
"""Paper §3.2: owner-aligned free-generation frequency differences."""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs'
SRC=RUNS/'241_paper_keyword_reliance';OUT=RUNS/'243_paper_owner_aligned_free_generation'
FIELDS=('award_received','education','field','occupation','position_held')
PROMPTS={
 'occupation':["Describe {person}'s occupation in one short sentence.","What was {person}'s occupation? Answer briefly.","Write one brief biographical sentence about {person}'s occupation.","State {person}'s professional occupation in a short phrase."],
 'position_held':["Name one formal position held by {person}.","What official position did {person} hold? Answer briefly.","Write one short sentence about a position held by {person}.","State one institutional role held by {person}."],
 'education':["Name one institution where {person} studied.","Where was {person} educated? Answer briefly.","Write one short sentence about {person}'s education.","State one educational institution attended by {person}."],
 'award_received':["Name one award or honor received by {person}.","What honor did {person} receive? Answer briefly.","Write one short sentence about an award received by {person}.","State one prize or honor associated with {person}."],
 'field':["State {person}'s principal academic field.","What field did {person} work in? Answer briefly.","Write one short sentence about {person}'s academic field.","Name the main scholarly field associated with {person}."],
}
STOP=set('the a an of in at for and or to as was is one member received award prize honor field worked work occupation position held formal official academic principal associated university society college institute served'.split())
EQUIV={'teacher':'teachrole','professor':'teachrole','lecturer':'teachrole','faculty':'teachrole','ceo':'executive','chairman':'chair','chairwoman':'chair','chemist':'chemistry','physicist':'physics','biologist':'biology','mathematician':'mathematics'}
def norm(s):return ' '.join(re.findall(r'[a-z0-9]+',s.casefold()))
def tokens(s):return [EQUIV.get(x,x) for x in re.findall(r'[a-z0-9]+',s.casefold()) if x not in STOP]
def hits(k,o):
 exact=norm(k) in norm(o);kt=set(tokens(k));ot=set(tokens(o));n=len(kt&ot)
 loose=exact or (bool(kt) and (kt<=ot or (n>=2 and n/len(kt)>=.6)))
 return exact,loose
def cohort():
 out=[]
 for p in sorted(SRC.glob('question_*.json')):
  x=json.loads(p.read_text());seen=set()
  for t in x['targets']:
   if t['field'] not in FIELDS:continue
   ident=(t['span_start'],t['span_end'],t['field'])
   if ident in seen:continue
   seen.add(ident);owner=t['owner_name'];other=x['wrong'] if owner==x['right'] else x['right']
   out.append({'item_id':f"{x['key']}::{t['span_start']}::{t['span_end']}::{t['field']}",'key':x['key'],'group':x['group'],'correct':x['correct'],'field':t['field'],'keyword':t['span_text'],'owner_person':owner,'other_person':other})
 return out
def collect(a):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 OUT.mkdir(parents=True,exist_ok=True);rows=cohort();raw=OUT/'generations.jsonl';done={}
 if a.resume and raw.exists():done={x['unit_id']:x for x in map(json.loads,raw.open())}
 units=[]
 for r in rows:
  for side,person in (('owner',r['owner_person']),('other',r['other_person'])):
   for ti,p in enumerate(PROMPTS[r['field']]):
    uid=f"{r['item_id']}::{side}::t{ti}"
    if uid not in done:units.append((uid,r,side,p.format(person=person)))
 if units:
  tok=AutoTokenizer.from_pretrained(a.model,use_fast=True,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
  model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True).eval();torch.manual_seed(a.seed)
  with raw.open('a' if a.resume else 'w') as f:
   for st in range(0,len(units),a.batch):
    q=units[st:st+a.batch];tx=[tok.apply_chat_template([{'role':'user','content':z[3]}],tokenize=False,add_generation_prompt=True) for z in q];z=tok(tx,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
    with torch.inference_mode():y=model.generate(**z,do_sample=True,temperature=.8,top_p=.95,num_return_sequences=a.samples,max_new_tokens=a.max_new_tokens,pad_token_id=tok.eos_token_id)
    dec=tok.batch_decode(y[:,z['input_ids'].shape[1]:],skip_special_tokens=True)
    for j,(uid,r,side,prompt) in enumerate(q):
     rec={'unit_id':uid,'item_id':r['item_id'],'key':r['key'],'group':r['group'],'correct':r['correct'],'field':r['field'],'keyword':r['keyword'],'side':side,'prompt':prompt,'outputs':dec[j*a.samples:(j+1)*a.samples]};f.write(json.dumps(rec,ensure_ascii=False)+'\n');done[uid]=rec
    f.flush();print(f'[{min(st+a.batch,len(units))}/{len(units)}]',flush=True)
 score(rows,done,a.seed)
def group_summary(rows,metric,seed):
 by_item=defaultdict(list)
 for r in rows:by_item[r['key']].append(r[metric])
 item={k:np.mean(v) for k,v in by_item.items()};kg={r['key']:r['group'] for r in rows};g=defaultdict(list)
 for k,v in item.items():g[kg[k]].append(v)
 vals=np.array([np.mean(v) for v in g.values()]);rng=np.random.default_rng(seed);boot=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(10000)])
 return {'n_items':len(item),'n_keywords':len(rows),'owner_frequency':float(np.mean([r[metric.replace('_difference','_owner')] for r in rows])),'other_frequency':float(np.mean([r[metric.replace('_difference','_other')] for r in rows])),'frequency_difference':float(vals.mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
def score(rows,done,seed=42):
 scored=[]
 for r in rows:
  cnt={}
  for side in ('owner','other'):
   outs=sum((done[f"{r['item_id']}::{side}::t{i}"]['outputs'] for i in range(4)),[]);h=[hits(r['keyword'],x) for x in outs];cnt[side]={'n':len(outs),'exact':np.mean([x[0] for x in h]),'loose':np.mean([x[1] for x in h])}
  scored.append({**r,'exact_owner':float(cnt['owner']['exact']),'exact_other':float(cnt['other']['exact']),'exact_difference':float(cnt['owner']['exact']-cnt['other']['exact']),'loose_owner':float(cnt['owner']['loose']),'loose_other':float(cnt['other']['loose']),'loose_difference':float(cnt['owner']['loose']-cnt['other']['loose'])})
 with (OUT/'items.jsonl').open('w') as f:
  for r in scored:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 report={'protocol':'241 frozen discriminative keywords; 4 prompts x 5 samples per person; owner-aligned recall difference; item aggregation and person-group bootstrap','n_keywords':len(scored),'cells':{},'by_field':{}}
 for c in (False,True):
  name='correct' if c else 'error';z=[r for r in scored if r['correct']==c];report['cells'][name]={m:group_summary(z,f'{m}_difference',seed+int(c)) for m in ('exact','loose')}
 for field in FIELDS:
  report['by_field'][field]={}
  for c in (False,True):
   z=[r for r in scored if r['correct']==c and r['field']==field]
   report['by_field'][field]['correct' if c else 'error']={m:group_summary(z,f'{m}_difference',seed+10+int(c)) for m in ('exact','loose')}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','score']);p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--batch',type=int,default=64);p.add_argument('--samples',type=int,default=5);p.add_argument('--max-new-tokens',type=int,default=32);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.stage=='collect':collect(a)
 else:
  done={x['unit_id']:x for x in map(json.loads,(OUT/'generations.jsonl').open())};score(cohort(),done,a.seed)
if __name__=='__main__':main()
