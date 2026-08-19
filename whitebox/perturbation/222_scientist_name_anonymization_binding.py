#!/usr/bin/env python3
"""Natural-data intervention: replace real scientist names with crossed nonce aliases."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent; RUNS=HERE/"runs"
ALIASES=[("Liora Venn","Marek Sol"),("Neris Vale","Tovan Rell"),("Sela Morn","Korin Dast"),
         ("Avela Kest","Daren Noll"),("Ilyra Saren","Pavel Trell")]

def replace_names(text,right,wrong,ar,aw):
 # Placeholders prevent collisions when one candidate name is a substring of another.
 pairs=sorted([(right,"ZXRIGHTZX"),(wrong,"ZXWRONGZX")],key=lambda x:len(x[0]),reverse=True)
 for old,new in pairs:text=re.sub(re.escape(old),new,text,flags=re.I)
 return text.replace("ZXRIGHTZX",ar).replace("ZXWRONGZX",aw)

def bootstrap(x,rng):
 x=np.asarray(x,float);b=np.mean(rng.choice(x,(10000,len(x)),replace=True),axis=1)
 return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]

def summary(rows,rng):
 d=np.array([r["anon_minus_original"] for r in rows]);orig=np.array([r["original_margin"] for r in rows]);anon=np.array([r["anonymous_margin"] for r in rows])
 return {"n":len(rows),"original_margin_mean":float(orig.mean()),"anonymous_margin_mean":float(anon.mean()),
  "anon_minus_original_mean":float(d.mean()),"ci95":bootstrap(d,rng),"fraction_repaired_direction":float(np.mean(d<0)),
  "original_likelihood_error_rate":float(np.mean(orig>0)),"anonymous_likelihood_error_rate":float(np.mean(anon>0)),
  "flip_wrong_to_right":int(np.sum((orig>0)&(anon<0))),"original_wrong_n":int(np.sum(orig>0))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
 ap.add_argument("--batch",type=int,default=16);ap.add_argument("--out",type=Path,default=RUNS/"222_scientist_name_anonymization_binding");a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 jobs=importlib.import_module("152_scientist_attention_pruned_current127").jobs()
 prompts=[];answers=[];meta=[];base_rows=[]
 for key,group,label,prompt,pred,other in jobs:
  right,wrong=(pred,other) if label else (other,pred)
  base_rows.append({"key":key,"group":group,"generation_correct":bool(label),"right":right,"wrong":wrong})
  prompts.extend([prompt,prompt]);answers.extend([" "+wrong," "+right]);meta.extend([(key,"original","wrong"),(key,"original","right")])
  for pi,(x,y) in enumerate(ALIASES):
   for swap in (0,1):
    ar,aw=(x,y) if not swap else (y,x)
    p=replace_names(prompt,right,wrong,ar,aw)
    prompts.extend([p,p]);answers.extend([" "+aw," "+ar]);meta.extend([(key,f"p{pi}s{swap}","wrong"),(key,f"p{pi}s{swap}","right")])
 loader=importlib.import_module("61_grad_span_proposal");model,tok=loader.load_model(a.model,"bfloat16","cuda");tok.padding_side="left"
 score=importlib.import_module("212_within_question_binding_competition").candidate_logprob
 z=score(model,tok,prompts,answers,a.batch);vals={}
 for m,v in zip(meta,z):vals[m]=float(v)
 rows=[]
 for r in base_rows:
  key=r["key"];orig=vals[key,"original","wrong"]-vals[key,"original","right"]
  am=[]
  for pi in range(len(ALIASES)):
   for swap in (0,1):am.append(vals[key,f"p{pi}s{swap}","wrong"]-vals[key,f"p{pi}s{swap}","right"])
  rows.append({**r,"original_margin":float(orig),"anonymous_margins":am,"anonymous_margin":float(np.mean(am)),
               "anon_minus_original":float(np.mean(am)-orig)})
 bind=defaultdict(list)
 for line in (RUNS/"221_free_generation_person_attribute_binding/items.jsonl").open():
  x=json.loads(line);bind[x["key"]].append(x)
 for r in rows:
  z0=bind.get(r["key"],[]);r["free_binding_exact"]=None if not z0 else float(np.mean([x["exact_binding"] for x in z0]));r["free_binding_loose"]=None if not z0 else float(np.mean([x["loose_binding"] for x in z0]))
 with (a.out/"items.jsonl").open("w") as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 rng=np.random.default_rng(42);generr=[r for r in rows if not r["generation_correct"]];gencor=[r for r in rows if r["generation_correct"]];likerr=[r for r in rows if r["original_margin"]>0];both=[r for r in generr if r["original_margin"]>0]
 br=[r for r in rows if r["free_binding_exact"] is not None]
 def corr(metric):
  # Positive binding predicts repair magnitude original-anonymous = -delta.
  s=spearmanr([r[metric] for r in br],[-r["anon_minus_original"] for r in br])
  return {"n":len(br),"rho":float(s.statistic),"p":float(s.pvalue)}
 report={"design":"5 nonce-name pairs crossed both directions; facts/order unchanged; length-normalized full-name likelihood",
  "all":summary(rows,rng),"generation_error":summary(generr,rng),"generation_correct":summary(gencor,rng),
  "original_likelihood_error":summary(likerr,rng),"generation_and_likelihood_error":summary(both,rng),
  "binding_predicts_repair":{"exact":corr("free_binding_exact"),"loose":corr("free_binding_loose")}}
 (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__":
 from collections import defaultdict
 main()
