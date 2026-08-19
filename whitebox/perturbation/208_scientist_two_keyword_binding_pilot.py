#!/usr/bin/env python3
"""Two-stage, two-keyword contextual-override binding pilot on Scientist errors."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RUNS=HERE/'runs'

def score_ab(model,tok,prompts,batch):
 import torch
 ai=tok.encode('A',add_special_tokens=False);bi=tok.encode('B',add_special_tokens=False)
 out=[]
 for st in range(0,len(prompts),batch):
  text=[tok.apply_chat_template([{'role':'user','content':x}],tokenize=False,add_generation_prompt=True) for x in prompts[st:st+batch]]
  z=tok(text,return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  with torch.inference_mode():q=model(**z,use_cache=False).logits[:,-1,[ai[0],bi[0]]].float()
  out.extend(torch.log_softmax(q,-1).cpu().numpy().tolist())
 return np.asarray(out)

def override_prompts(right,wrong,cues,label):
 out=[]
 shown=' and '.join(f"'{x}'" for x in cues)
 controls=['Blue Circle','Green Triangle'][:len(cues)]
 for owner_name in (right,wrong):
  other=wrong if owner_name==right else right
  for swap in (0,1):
   a,b=(owner_name,other) if not swap else(other,owner_name);gold=0 if not swap else 1
   assign='\n'.join([f"- The card '{x}' is assigned to {owner_name}." for x in cues]+[f"- The card '{x}' is assigned to {other}." for x in controls])
   prompt=("This is a fictional card-assignment game. Ignore real-world facts.\nFor this round only:\n"+assign+f"\nAccording only to these assignments, who has all of the cards {shown}?\nA. {a}\nB. {b}\nAnswer exactly A or B.")
   out.append({'set':label,'owner':'right' if owner_name==right else'wrong','swap':swap,'gold':gold,'prompt':prompt})
 return out

def scan_stage(att,spanmod,strict,attributes,prep):
 ss=att.build_word_spans(prep,widths=(2,),stride=1);pr,ot=spanmod.scan(att,prep,ss);base=float(pr[0]-ot[0]);u=(pr[0]-pr[1:])-(ot[0]-ot[1:]);rank=[]
 for i in np.argsort(-u):
  fact=strict.strict_match(ss[int(i)].text,attributes)
  rank.append({'text':ss[int(i)].text,'u':float(u[i]),'fact':fact,'rank':len(rank)+1})
 entity=next((x for x in rank[:10]if x['u']>0 and x['fact']),None)
 return base,rank,entity

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');p.add_argument('--limit',type=int,default=24);p.add_argument('--batch',type=int,default=48);p.add_argument('--out',type=Path,default=RUNS/'208_two_keyword_binding_pilot');a=p.parse_args()
 import torch
 from spanattr.core import Item,SpanAttributor,set_seed
 set_seed(42);a.out.mkdir(parents=True,exist_ok=True);loader=importlib.import_module('61_grad_span_proposal');model,tok=loader.load_model(a.model,'bfloat16','cuda');tok.padding_side='left';att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=a.batch)
 spanmod=importlib.import_module('125_collect_current_three_benchmarks');strict=importlib.import_module('207_scientist_strict_attribute_binding_pilot');attrs=importlib.import_module('206_scientist_attribute_binding_pilot').full_profile_attributes();jobs=importlib.import_module('152_scientist_attention_pruned_current127').jobs();items=[]
 for key,group,label,prompt,pred,other in jobs:
  if label or pred in ('','None','null')or other in ('','None','null')or len(items)>=a.limit:continue
  prep=att.prepare(Item.from_dict({'key':key,'prompt':prompt,'pred':pred,'gold':other}));b1,r1,e1=scan_stage(att,spanmod,strict,attrs[key],prep)
  if not e1:items.append({'key':key,'right':other,'wrong':pred,'stage1':None,'stage2':None});continue
  # Physical deletion mirrors current127. Delete only one exact occurrence.
  start=prompt.casefold().find(e1['text'].casefold());deleted=prompt if start<0 else(prompt[:start]+prompt[start+len(e1['text']):]);deleted=re.sub(r'\s+([,.;:!?])',r'\1',re.sub(r'[ \t]+',' ',deleted))
  prep2=att.prepare(Item.from_dict({'key':key+'_d','prompt':deleted,'pred':pred,'gold':other}));b2,r2,e2=scan_stage(att,spanmod,strict,attrs[key],prep2)
  rec={'key':key,'right':other,'wrong':pred,'stage1':e1,'stage2':e2,'stage1_top':r1[:10],'stage2_top':r2[:10],'base1':b1,'base2':b2};items.append(rec);print(f"[{len(items)}/{a.limit}] {key} k1={e1['text']} k2={None if not e2 else e2['text']}",flush=True);torch.cuda.empty_cache()
 prompts=[];meta=[]
 for n,x in enumerate(items):
  if not x['stage1']or not x['stage2']:continue
  k1=x['stage1']['fact']['value'];k2=x['stage2']['fact']['value'];sets=[([k1],'k1'),([k2],'k2'),([k1,k2],'pair'),([f'ZORP-{n}'],'n1'),([f'KETA-{n}'],'n2'),([f'ZORP-{n}',f'KETA-{n}'],'npair')]
  for cues,label in sets:
   for q in override_prompts(x['right'],x['wrong'],cues,label):prompts.append(q['prompt']);meta.append((x,q))
 lp=score_ab(model,tok,prompts,a.batch);by={}
 for (x,q),v in zip(meta,lp):q={**q,'correct_margin':float(v[q['gold']]-v[1-q['gold']])};by.setdefault(x['key'],[]).append(q)
 rows=[]
 for x in items:
  q=by.get(x['key']);
  if not q:continue
  def asym(label):
   z={o:np.mean([r['correct_margin']for r in q if r['set']==label and r['owner']==o])for o in('right','wrong')};return float(z['wrong']-z['right'])
  d1=asym('k1')-asym('n1');d2=asym('k2')-asym('n2');dp=asym('pair')-asym('npair')
  rows.append({'key':x['key'],'k1':x['stage1'],'k2':x['stage2'],'d1':d1,'d2':d2,'d_pair':dp,'pair_gain_over_max':dp-max(d1,d2),'pair_interaction':dp-d1-d2,'conditions':q})
 with(a.out/'items.jsonl').open('w')as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 from scipy.stats import wilcoxon
 report={'requested':a.limit,'two_keyword_items':len(rows)}
 for name in('d1','d2','d_pair','pair_gain_over_max','pair_interaction'):
  z=np.array([x[name]for x in rows]);
  try:pv=float(wilcoxon(z,alternative='greater').pvalue)
  except ValueError:pv=None
  report[name]={'mean':float(z.mean()),'fraction_positive':float(np.mean(z>0)),'wilcoxon_greater_p':pv}
 # Prompt-engineering view: which first-keyword property types show binding?
 for field in sorted(set(x['k1']['fact']['field']for x in rows)):
  z=[x['d_pair']for x in rows if x['k1']['fact']['field']==field];report.setdefault('by_k1_field',{})[field]={'n':len(z),'mean_pair_binding':float(np.mean(z))}
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
