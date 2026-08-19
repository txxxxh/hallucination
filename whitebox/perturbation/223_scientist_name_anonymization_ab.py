#!/usr/bin/env python3
"""Name anonymization scored through counterbalanced A/B labels, not name tokens."""
from __future__ import annotations
import argparse, importlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent;RUNS=HERE/"runs"
old=importlib.import_module("222_scientist_name_anonymization_binding")
card=importlib.import_module("204_scientist_binding_override_pilot")

def choice_prompt(prompt,right,wrong,swap):
 a,b=(wrong,right) if not swap else (right,wrong)
 return (prompt+"\nChoose the answer from the two candidates below based on the preceding profiles and question.\n"
         f"A. {a}\nB. {b}\nAnswer exactly A or B.")

def boot(x,rng):
 x=np.asarray(x);z=np.mean(rng.choice(x,(10000,len(x)),replace=True),1)
 return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def summ(rows,rng):
 d=np.array([r["anon_minus_original"] for r in rows]);o=np.array([r["original_margin"] for r in rows]);a=np.array([r["anonymous_margin"] for r in rows])
 return {"n":len(rows),"original_mean":float(o.mean()),"anonymous_mean":float(a.mean()),"change":float(d.mean()),
  "change_ci95":boot(d,rng),"fraction_repair_direction":float(np.mean(d<0)),"original_error_rate":float(np.mean(o>0)),
  "anonymous_error_rate":float(np.mean(a>0)),"wrong_to_right_flips":int(np.sum((o>0)&(a<0))),"original_wrong_n":int(np.sum(o>0))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");ap.add_argument("--batch",type=int,default=32)
 ap.add_argument("--out",type=Path,default=RUNS/"223_scientist_name_anonymization_ab");a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 jobs=importlib.import_module("152_scientist_attention_pruned_current127").jobs();prompts=[];meta=[];base=[]
 for key,group,label,prompt,pred,other in jobs:
  right,wrong=(pred,other) if label else (other,pred);base.append({"key":key,"group":group,"generation_correct":bool(label),"right":right,"wrong":wrong})
  for swap in (0,1):prompts.append(choice_prompt(prompt,right,wrong,swap));meta.append((key,"original",swap))
  for pi,(x,y) in enumerate(old.ALIASES):
   for nameswap in (0,1):
    ar,aw=(x,y) if not nameswap else (y,x);p=old.replace_names(prompt,right,wrong,ar,aw)
    for swap in (0,1):prompts.append(choice_prompt(p,ar,aw,swap));meta.append((key,f"p{pi}s{nameswap}",swap))
 loader=importlib.import_module("61_grad_span_proposal");model,tok=loader.load_model(a.model,"bfloat16","cuda");tok.padding_side="left"
 lp=card.score_ab(model,tok,prompts,a.batch);vals=defaultdict(list)
 for (key,cond,swap),v in zip(meta,lp):vals[key,cond].append(float(v[0]-v[1]) if not swap else float(v[1]-v[0]))
 rows=[]
 bind=defaultdict(list)
 for line in (RUNS/"221_free_generation_person_attribute_binding/items.jsonl").open():
  z=json.loads(line);bind[z["key"]].append(z)
 for r in base:
  o=float(np.mean(vals[r["key"],"original"]));am=[]
  for pi in range(len(old.ALIASES)):
   for ns in (0,1):am.append(float(np.mean(vals[r["key"],f"p{pi}s{ns}"])))
  bz=bind.get(r["key"],[])
  rows.append({**r,"original_margin":o,"anonymous_margins":am,"anonymous_margin":float(np.mean(am)),"anon_minus_original":float(np.mean(am)-o),
   "free_binding_exact":None if not bz else float(np.mean([x["exact_binding"] for x in bz])),
   "free_binding_loose":None if not bz else float(np.mean([x["loose_binding"] for x in bz]))})
 with (a.out/"items.jsonl").open("w") as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 rng=np.random.default_rng(42);ge=[r for r in rows if not r["generation_correct"]];gc=[r for r in rows if r["generation_correct"]];le=[r for r in rows if r["original_margin"]>0];both=[r for r in ge if r["original_margin"]>0];br=[r for r in rows if r["free_binding_exact"] is not None]
 def corr(m):
  s=spearmanr([r[m] for r in br],[-r["anon_minus_original"] for r in br]);return {"n":len(br),"rho":float(s.statistic),"p":float(s.pvalue)}
 report={"design":"real vs crossed nonce names; A/B label scoring; candidate order counterbalanced","all":summ(rows,rng),"generation_error":summ(ge,rng),"generation_correct":summ(gc,rng),"likelihood_error":summ(le,rng),"generation_and_likelihood_error":summ(both,rng),"binding_predicts_repair":{"exact":corr("free_binding_exact"),"loose":corr("free_binding_loose")}}
 (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__":main()
