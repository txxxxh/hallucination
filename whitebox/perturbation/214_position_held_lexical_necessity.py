#!/usr/bin/env python3
"""Strict lexical-necessity case series for position-held top-5 cues."""
import importlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/214_position_held_lexical_necessity'
CASES={
'question_0032':('Astronomer Royal',['Astronomer Royal','the British royal astronomer','holder of the Astronomer Royal office','the monarch’s senior astronomical adviser']),
'question_0600':('president of the French Academy of Sciences',['president of the French Academy of Sciences','head of the French Academy of Sciences','leader of the French Academy of Sciences','presiding officer of the French Academy of Sciences']),
'question_0770':('Director General of CERN',['Director General of CERN','CERN director-general','chief executive of CERN','head of CERN']),
'question_0994':('Chief Economist of the World Bank',['Chief Economist of the World Bank','World Bank chief economist','leading economist at the World Bank','head of economic research at the World Bank']),
'question_1125':('President of the Royal Society',['President of the Royal Society','Royal Society president','head of the Royal Society','presiding officer of the Royal Society']),
'question_1356':('Director General of CERN',['Director General of CERN','CERN director-general','chief executive of CERN','head of CERN']),
'question_1391':('Chief Economist of the World Bank',['Chief Economist of the World Bank','World Bank chief economist','leading economist at the World Bank','head of economic research at the World Bank']),
'question_1644':('deputy of the Supreme Soviet of the Soviet Union',['deputy of the Supreme Soviet of the Soviet Union','Supreme Soviet deputy','member of the Soviet Union’s Supreme Soviet','legislator in the Supreme Soviet of the Soviet Union']),
'question_2687':('Director-General of UNESCO',['Director-General of UNESCO','UNESCO director-general','chief executive of UNESCO','head of UNESCO']),
'question_2689':('President of the Nordic Council',['President of the Nordic Council','Nordic Council president','head of the Nordic Council','presiding officer of the Nordic Council']),
'question_2884':('Soviet deputy',['Soviet deputy','deputy in the Soviet legislature','member of a Soviet legislative body','Soviet legislative representative']),
}
def main():
 import torch
 try:
  from torch._native.registry import deregister_op_overrides;deregister_op_overrides(disable_op_symbols='bmm')
 except Exception:pass
 from spanattr.core import Item,SpanAttributor
 model,tok=importlib.import_module('61_grad_span_proposal').load_model('NousResearch/Meta-Llama-3.1-8B-Instruct','bfloat16','cuda');att=SpanAttributor(model,tok,device='cuda',baseline='mean',length_norm=True,max_rows=16)
 jobs={x[0]:x for x in importlib.import_module('152_scientist_attention_pruned_current127').jobs()}; score=importlib.import_module('212_within_question_binding_competition').candidate_logprob
 rows=[]
 for key,(cue,forms) in CASES.items():
  _,group,gcorrect,prompt,pred,other=jobs[key];right,wrong=(pred,other) if gcorrect else(other,pred)
  # closed-book calibrated binding
  ps=[];ans=[]
  for f in forms:
   q=f"Based only on general background knowledge, complete with the most associated person's name.\nPhrase: {f}\nPerson:"
   ps += [q,q];ans += [' '+wrong,' '+right]
  null="Complete the following with a person's name.\nPerson:";ps += [null,null];ans += [' '+wrong,' '+right]
  z=score(model,tok,ps,ans,16);prior=z[-2]-z[-1];B={f:float(z[2*i]-z[2*i+1]-prior) for i,f in enumerate(forms)};low=min(forms,key=B.get)
  margins={}
  for f in forms:
   q=prompt.replace(cue,f);prep=att.prepare(Item.from_dict({'key':key,'prompt':q,'pred':wrong,'gold':right}));a=torch.zeros((1,len(prep.prompt_ids)),device='cuda');w,r=att.class_scores(prep,a);margins[f]=float(w[0]-r[0])
  rows.append({'key':key,'right':right,'wrong':wrong,'cue':cue,'forms':forms,'binding':B,'low_form':low,'original_margin':margins[cue],'low_margin':margins[low],'delta_low':margins[low]-margins[cue],'repair':margins[cue]>0 and margins[low]<0})
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'items.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 # All keys were selected as likelihood errors by the identical 212 scorer.
 d=np.array([r['delta_low'] for r in rows]);rng=np.random.default_rng(42);b=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(20000)]);report={'n':len(rows),'mean_delta':float(d.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'median_delta':float(np.median(d)),'desired_direction_n':int(np.sum(d<0)),'desired_direction_rate':float(np.mean(d<0)),'repair_n':sum(r['repair'] for r in rows),'repair_rate':float(np.mean([r['repair'] for r in rows]))}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
