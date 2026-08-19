#!/usr/bin/env python3
"""Within-question comparison of selected vs non-selected profile attributes."""
from __future__ import annotations
import importlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon,spearmanr

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RUNS=HERE/"runs"
gen=importlib.import_module("221_free_generation_person_attribute_binding")
selmod=importlib.import_module("220_top3_matched_attribute_binding")
builder=importlib.import_module("76_build_closedbook_fact_probes")

def wt(s):return set(re.findall(r"[a-z0-9]+",s.casefold()))
def matched(value,keyword,field):
 v,k=wt(value),wt(keyword)
 if builder.norm(value)==builder.norm(keyword):return True
 if field in {"occupation","field","position_held"}:return bool(v) and v<=k
 generic={"award","prize","medal","order","university","society","college","institute","member"}
 return len((v&k)-generic)>=2 or (len(v&k)>=2 and len((v&k)-generic)>=1)

def ci(x,rng):
 x=np.asarray(x);z=np.mean(rng.choice(x,(20000,len(x)),replace=True),1);return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]

def main():
 selected=selmod.candidates();selby=defaultdict(list)
 for x in selected:selby[x["key"],x["field"]].append(x)
 raw=[json.loads(x) for x in (RUNS/"221_free_generation_person_attribute_binding/generations.jsonl").open()]
 # Select one common set of 20 generations per question-field-owner, so every fact is judged on identical outputs.
 units=defaultdict(list)
 for x in raw:units[x["key"],x["field"],x["owner"]].append(x)
 outputs={}
 for k,z in units.items():
  first=min(x["item_id"] for x in z);q=[x for x in z if x["item_id"]==first]
  outputs[k]=sum((x["outputs"] for x in sorted(q,key=lambda y:y["unit_id"])),[])
 data={str(x["key"]):x for x in json.load((ROOT/"shuffled_prepend_profiles_question.json").open())}
 facts=[]
 for (key,field),ss in selby.items():
  profiles,question=builder.parse_item(data[key]);qnorm=builder.norm(question)
  right=ss[0]["right"];wrong=ss[0]["wrong"]
  pmap={p["name"]:p for p in profiles};rp,wp=pmap[right],pmap[wrong]
  rv={builder.norm(v):v for v in builder.values(rp,field)};wv={builder.norm(v):v for v in builder.values(wp,field)}
  for nv in sorted(set(rv)|set(wv)):
   value=rv.get(nv,wv.get(nv));
   if nv not in qnorm:continue
   owner="both" if nv in rv and nv in wv else ("right" if nv in rv else "wrong")
   is_sel=any(matched(value,s["keyword"],field) for s in ss)
   wo=outputs.get((key,field,"wrong"),[]);ro=outputs.get((key,field,"right"),[])
   if not wo or not ro:continue
   wh=[gen.hits(value,o) for o in wo];rh=[gen.hits(value,o) for o in ro]
   facts.append({"key":key,"field":field,"value":value,"owner":owner,"selected":is_sel,
    "n_wrong":len(wo),"n_right":len(ro),"wrong_exact":sum(x[0] for x in wh)/len(wo),"right_exact":sum(x[0] for x in rh)/len(ro),
    "wrong_loose":sum(x[1] for x in wh)/len(wo),"right_loose":sum(x[1] for x in rh)/len(ro),
    "exact_binding":sum(x[0] for x in wh)/len(wo)-sum(x[0] for x in rh)/len(ro),
    "loose_binding":sum(x[1] for x in wh)/len(wo)-sum(x[1] for x in rh)/len(ro)})
 out=RUNS/"224_selected_vs_nonselected_free_binding";out.mkdir(parents=True,exist_ok=True)
 with (out/"facts.jsonl").open("w") as f:
  for x in facts:f.write(json.dumps(x,ensure_ascii=False)+"\n")
 rng=np.random.default_rng(42);report={"n_facts":len(facts),"selected_n":sum(x["selected"] for x in facts),"owner_counts":{}}
 for own in ("wrong","right","both"):
  report["owner_counts"][own]={"selected":sum(x["selected"] and x["owner"]==own for x in facts),"nonselected":sum((not x["selected"]) and x["owner"]==own for x in facts)}
 for metric in ("exact_binding","loose_binding"):
  groups=defaultdict(lambda:{True:[],False:[]})
  for x in facts:groups[x["key"],x["field"],x["owner"]][x["selected"]].append(x[metric])
  diffs=[];meta=[]
  for k,z in groups.items():
   if z[True] and z[False]:diffs.append(float(np.mean(z[True])-np.mean(z[False])));meta.append(k)
  try:p=float(wilcoxon(diffs,alternative="greater").pvalue)
  except ValueError:p=None
  report[metric]={"matched_question_field_owner_groups":len(diffs),"selected_minus_nonselected_mean":float(np.mean(diffs)) if diffs else None,
   "ci95":ci(diffs,rng) if diffs else None,"fraction_positive":float(np.mean(np.array(diffs)>0)) if diffs else None,"wilcoxon_greater_p":p}
  for owner in ("wrong","right","both"):
   d=[v for v,k in zip(diffs,meta) if k[2]==owner]
   report[metric][owner]={"n":len(d),"mean":float(np.mean(d)) if d else None,"ci95":ci(d,rng) if d else None,"fraction_positive":float(np.mean(np.array(d)>0)) if d else None}
 (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__":main()
