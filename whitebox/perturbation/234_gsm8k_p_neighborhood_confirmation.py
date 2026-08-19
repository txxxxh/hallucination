#!/usr/bin/env python3
"""GSM8K P validation in a held-out instruction neighbourhood with matched deletion."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;RUNS=HERE/'runs';ops=importlib.import_module('228_scientist_p_e_confirmation');scoremod=importlib.import_module('229_trivia_e_confirmation')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=RUNS/'140_gsm8k_natural/natural_balanced_n942.jsonl');ap.add_argument('--model',default='NousResearch/Meta-Llama-3.1-8B-Instruct');ap.add_argument('--batch',type=int,default=24);ap.add_argument('--out-dir',type=Path,default=RUNS/'234_gsm8k_p_neighborhood_confirmation');a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);rows={x['key']:x for x in map(json.loads,a.manifest.open())};cache=RUNS/'141_gsm8k_natural_current127';items=[];req=[]
 for key,x in rows.items():
  if x['correct']:continue
  with np.load(cache/f'{key}.npz',allow_pickle=True)as z:
   p,o=z['stage1_pred'],z['stage1_other'];u=float((p[0]-p[1])-(o[0]-o[1]));target=str(z['deleted_text'])
  if u>=0:continue
  held='Work out the following equivalent math problem carefully. Output only the final number.\nQuestion:\n'+x['question'];deleted=ops.delete_target(held,target);placebo,pt=ops.delete_matched(held,target,key)
  if deleted is None or placebo is None:continue
  items.append({'key':key,'p_score':-u,'target':target,'placebo_text':pt});right=x['gold_final'];wrong=x['predicted_final'];req.extend([{'prompt':held,'right':right,'wrong':wrong},{'prompt':deleted,'right':right,'wrong':wrong},{'prompt':placebo,'right':right,'wrong':wrong}])
 model,tok=importlib.import_module('61_grad_span_proposal').load_model(a.model,'bfloat16','cuda');tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';m=scoremod.score(model,tok,req,a.batch).reshape(-1,3)
 for x,z in zip(items,m):x.update(base_margin=float(z[0]),target_margin=float(z[1]),placebo_margin=float(z[2]),target_gain=float(z[1]-z[0]),placebo_gain=float(z[2]-z[0]),specific_gain=float(z[1]-z[2]))
 with(a.out_dir/'items.jsonl').open('w')as f:
  for x in items:f.write(json.dumps(x)+'\n')
 s=np.array([x['p_score']for x in items]);q=np.quantile(s,[.3,.7]);lo=[x['specific_gain']for x in items if x['p_score']<=q[0]];hi=[x['specific_gain']for x in items if x['p_score']>=q[1]];wrong=[x for x in items if x['base_margin']<0];report={'protocol':'original mean-neutralization selects signed wrong-support span; held-out instruction framing; physical target deletion vs same-length random deletion','n':len(items),'mean_target_gain':float(np.mean([x['target_gain']for x in items])),'mean_placebo_gain':float(np.mean([x['placebo_gain']for x in items])),'mean_specific_gain':float(np.mean([x['specific_gain']for x in items])),'rho':float(spearmanr(s,[x['specific_gain']for x in items]).statistic),'low_n':len(lo),'low_gain':float(np.mean(lo)),'high_n':len(hi),'high_gain':float(np.mean(hi)),'high_minus_low':float(np.mean(hi)-np.mean(lo)),'base_wrong_n':len(wrong),'target_repair_rate':float(np.mean([x['target_margin']>0 for x in wrong])),'placebo_repair_rate':float(np.mean([x['placebo_margin']>0 for x in wrong]))};(a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
