#!/usr/bin/env python3
"""Entity-binding override pilot grounded in signed exact perturbations."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RUNS=HERE/'runs'
STOP=set('the a an this that who person individual their his her is was did and or of in to for with'.split())
LOGIC=set('never not without despite however except neither nor no'.split())

def toks(x): return set(re.findall(r"[a-z0-9]+",x.casefold()))-STOP
def loadj(path): return [json.loads(x) for x in path.open() if x.strip()]
def facts_by_item():
 out={}
 for r in loadj(RUNS/'76_closedbook_fact_probe_manifest.jsonl'):
  fs={}
  for p in r['probes']:
   fid=p['probe_id'].split('::')[1]
   fs.setdefault(fid,{'field':p['field'],'value':p['value']})
  out[r['key']]=list(fs.values())
 return out

def match_fact(span,facts):
 s=toks(span)
 scored=[]
 for f in facts:
  v=toks(f['value']); overlap=len(s&v)/max(1,len(s|v))
  if s&v: scored.append((overlap,len(s&v),f))
 return max(scored,key=lambda x:(x[0],x[1]))[2] if scored else None

def score_ab(model,tok,prompts,batch):
 import torch
 aid=tok.encode('A',add_special_tokens=False);bid=tok.encode('B',add_special_tokens=False)
 if len(aid)!=1 or len(bid)!=1: raise RuntimeError((aid,bid))
 vals=[]
 for st in range(0,len(prompts),batch):
  text=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in prompts[st:st+batch]]
  z=tok(text,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode():
   q=model(**z,use_cache=False).logits[:,-1,[aid[0],bid[0]]].float()
   vals.extend(torch.log_softmax(q,-1).cpu().numpy().tolist())
 return np.asarray(vals)

def binding_prompts(row,keyword,nonce):
 right,wrong=row['right'],row['wrong']; out=[]
 # Correct-answer log margin is scored after counterbalancing candidate order.
 for cue in (keyword,nonce):
  for owner_name in (right,wrong):
   other=wrong if owner_name==right else right
   for swap in (0,1):
    a,b=(owner_name,other) if not swap else (other,owner_name)
    text=("This is a fictional card-assignment game. Ignore real-world facts.\n"
          f"For this round only, the card '{cue}' is assigned to {owner_name}.\n"
          f"The card 'Blue Circle' is assigned to {other}.\n"
          f"According only to these assignments, who has the '{cue}' card?\n"
          f"A. {a}\nB. {b}\nAnswer exactly A or B.")
    gold=0 if not swap else 1
    out.append({'cue':'real' if cue==keyword else 'nonce','owner':'right' if owner_name==right else 'wrong','swap':swap,'gold':gold,'prompt':text})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--limit',type=int,default=32);p.add_argument('--batch',type=int,default=32);p.add_argument('--out',type=Path,default=RUNS/'204_binding_override_pilot');p.add_argument('--resume',action='store_true');a=p.parse_args()
 import torch
 from spanattr.core import Item,SpanAttributor,set_seed
 set_seed(42);a.out.mkdir(parents=True,exist_ok=True)
 loader=importlib.import_module('61_grad_span_proposal'); model,tok=loader.load_model(a.model,'bfloat16','cuda');tok.padding_side='left'
 att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 spanmod=importlib.import_module('125_collect_current_three_benchmarks'); facts=facts_by_item(); jobs=importlib.import_module('152_scientist_attention_pruned_current127').jobs()
 selected=[]
 for key,group,label,prompt,pred,other in jobs:
  if label or len(selected)>=a.limit: continue
  fp=a.out/f'{key}.json'
  if fp.exists() and a.resume: selected.append(json.loads(fp.read_text()));continue
  prep=att.prepare(Item.from_dict({'key':key,'prompt':prompt,'pred':pred,'gold':other}));ss,_=spanmod.spans(att,prep);pr,ot=spanmod.scan(att,prep,ss);base=float(pr[0]-ot[0]);u=(pr[0]-pr[1:])-(ot[0]-ot[1:])
  ranked=[]
  for i in np.argsort(-u):
   f=match_fact(ss[int(i)].text,facts.get(key,[])); words=toks(ss[int(i)].text)
   ranked.append({'rank_positive':len(ranked)+1,'text':ss[int(i)].text,'u':float(u[i]),'logic':bool(words&LOGIC),'fact':f})
  entity=next((x for x in ranked if x['u']>0 and x['fact'] is not None),None)
  rec={'key':key,'group':group,'right':other,'wrong':pred,'base_margin_wrong_minus_right':base,'top_signed':ranked[:10],'entity':entity}
  fp.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');selected.append(rec);print(f"attribution {len(selected)}/{a.limit} {key} entity={bool(entity)}",flush=True)
  torch.cuda.empty_cache()
 # Only actual positive perturbation entities enter the binding experiment.
 probes=[];meta=[]
 for n,r in enumerate(selected):
  if not r['entity']:continue
  keyword=r['entity']['fact']['value'];nonce=f'ZORP-{100+n}'
  for x in binding_prompts(r,keyword,nonce):meta.append((r,x));probes.append(x['prompt'])
 lp=score_ab(model,tok,probes,a.batch) if probes else np.zeros((0,2));by={}
 for (r,x),v in zip(meta,lp):
  correct=float(v[x['gold']]-v[1-x['gold']]);by.setdefault(r['key'],[]).append({**x,'correct_margin':correct})
 rows=[]
 for r in selected:
  z=by.get(r['key']);
  if not z:continue
  means={(cue,owner):np.mean([q['correct_margin'] for q in z if q['cue']==cue and q['owner']==owner]) for cue in ('real','nonce') for owner in ('right','wrong')}
  # wrong assignment is aligned with the hypothesized erroneous binding.
  real=means['real','wrong']-means['real','right']; null=means['nonce','wrong']-means['nonce','right'];
  rows.append({'key':r['key'],'keyword':r['entity']['fact']['value'],'field':r['entity']['fact']['field'],'perturb_u':r['entity']['u'],'real_override_asymmetry':float(real),'nonce_asymmetry':float(null),'binding_effect':float(real-null),'conditions':z})
 with (a.out/'binding_items.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 if rows:
  from scipy.stats import spearmanr,wilcoxon
  d=np.array([r['binding_effect'] for r in rows]);u=np.array([r['perturb_u'] for r in rows]);sp=spearmanr(d,u) if len(rows)>2 else None
  try:w=wilcoxon(d,alternative='greater')
  except ValueError:w=None
  report={'requested_wrong_items':a.limit,'attributed_items':len(selected),'entity_items':len(rows),'mean_binding_effect':float(d.mean()),'fraction_positive':float(np.mean(d>0)),'wilcoxon_greater_p':None if w is None else float(w.pvalue),'binding_vs_perturb_spearman':None if sp is None else {'rho':float(sp.statistic),'p':float(sp.pvalue)}}
 else:report={'requested_wrong_items':a.limit,'attributed_items':len(selected),'entity_items':0}
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
