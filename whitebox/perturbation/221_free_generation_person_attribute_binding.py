#!/usr/bin/env python3
"""Spontaneous recall of perturbation-selected attributes from person names."""
from __future__ import annotations
import argparse, importlib, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/221_free_generation_person_attribute_binding"
select = importlib.import_module("220_top3_matched_attribute_binding")

PROMPTS = {
 "occupation": ["Describe {person}'s occupation in one short sentence.", "What was {person}'s occupation? Answer briefly.",
                "Write one brief biographical sentence about {person}'s occupation.", "State {person}'s professional occupation in a short phrase."],
 "position_held": ["Name one formal position held by {person}.", "What official position did {person} hold? Answer briefly.",
                   "Write one short sentence about a position held by {person}.", "State one institutional role held by {person}."],
 "education": ["Name one institution where {person} studied.", "Where was {person} educated? Answer briefly.",
               "Write one short sentence about {person}'s education.", "State one educational institution attended by {person}."],
 "award_received": ["Name one award or honor received by {person}.", "What honor did {person} receive? Answer briefly.",
                    "Write one short sentence about an award received by {person}.", "State one prize or honor associated with {person}."],
 "field": ["State {person}'s principal academic field.", "What field did {person} work in? Answer briefly.",
           "Write one short sentence about {person}'s academic field.", "Name the main scholarly field associated with {person}."],
 "notable_work": ["Name one notable work associated with {person}.", "What is one notable work by {person}? Answer briefly.",
                  "Write one short sentence naming a notable work of {person}.", "State one work for which {person} is known."],
 "place_of_birth": ["State where {person} was born.", "Where was {person} born? Answer briefly.",
                    "Write one short sentence about {person}'s birthplace.", "Name {person}'s place of birth."],
}
STOP=set("the a an of in at for and or to as was is one member received award prize honor field worked work occupation position held formal official academic principal associated university society college institute served".split())
EQUIV={"teacher":"teachrole","professor":"teachrole","lecturer":"teachrole","faculty":"teachrole",
       "ceo":"executive","chairman":"chair","chairwoman":"chair","chemist":"chemistry",
       "physicist":"physics","biologist":"biology","mathematician":"mathematics"}

def norm(s): return " ".join(re.findall(r"[a-z0-9]+",s.casefold()))
def tokens(s): return [EQUIV.get(x,x) for x in re.findall(r"[a-z0-9]+",s.casefold()) if x not in STOP]
def hits(keyword, output):
 exact = norm(keyword) in norm(output)
 kt=set(tokens(keyword)); ot=set(tokens(output)); overlap=len(kt&ot)
 # Predeclared lexical-semantic recall: all informative tokens, or >=60% with >=2 tokens.
 loose = exact or (bool(kt) and (kt <= ot or (overlap >= 2 and overlap/len(kt) >= .6)))
 return exact, loose

def units(rows, done):
 out=[]
 for r in rows:
  for owner in ("wrong","right"):
   for ti,t in enumerate(PROMPTS[r["field"]]):
    uid=f"{r['item_id']}::{owner}::t{ti}"
    if uid not in done:
     out.append((uid,r,owner,t.format(person=r[owner])))
 return out

def cluster_summary(rows, metric, rng):
 byq=defaultdict(list)
 for r in rows: byq[r["key"]].append(r[metric])
 q=np.array([np.mean(v) for v in byq.values()]); raw=np.array([r[metric] for r in rows])
 boot=np.mean(rng.choice(q,(10000,len(q)),replace=True),axis=1)
 return {"n_keywords":len(rows),"n_questions":len(q),"keyword_mean":float(raw.mean()),
         "keyword_fraction_positive":float(np.mean(raw>0)),"question_cluster_mean":float(q.mean()),
         "question_cluster_fraction_positive":float(np.mean(q>0)),
         "cluster_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
 ap.add_argument("--batch",type=int,default=16);ap.add_argument("--samples",type=int,default=5)
 ap.add_argument("--max-new-tokens",type=int,default=32);ap.add_argument("--seed",type=int,default=42)
 ap.add_argument("--out",type=Path,default=OUT);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rawpath=a.out/"generations.jsonl";done={}
 if rawpath.exists():
  for line in rawpath.open():
   x=json.loads(line);done[x["unit_id"]]=x
 rows=select.candidates(); todo=units(rows,done)
 if todo:
  import torch
  loader=importlib.import_module("61_grad_span_proposal");model,tok=loader.load_model(a.model,"bfloat16","cuda")
  tok.padding_side="left";model.eval();torch.manual_seed(a.seed)
  with rawpath.open("a") as f:
   for st in range(0,len(todo),a.batch):
    chunk=todo[st:st+a.batch]
    texts=[tok.apply_chat_template([{"role":"user","content":x[3]}],tokenize=False,add_generation_prompt=True) for x in chunk]
    z=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
    with torch.inference_mode():
     y=model.generate(**z,do_sample=True,temperature=.8,top_p=.95,num_return_sequences=a.samples,
                      max_new_tokens=a.max_new_tokens,pad_token_id=tok.eos_token_id)
    dec=tok.batch_decode(y[:,z["input_ids"].shape[1]:],skip_special_tokens=True)
    for j,(uid,r,owner,prompt) in enumerate(chunk):
     rec={"unit_id":uid,"item_id":r["item_id"],"key":r["key"],"field":r["field"],
          "keyword":r["keyword"],"owner":owner,"person":r[owner],"prompt":prompt,
          "outputs":dec[j*a.samples:(j+1)*a.samples]}
     f.write(json.dumps(rec,ensure_ascii=False)+"\n");done[uid]=rec
    f.flush();print(f"generated {min(st+a.batch,len(todo))}/{len(todo)} prompt units",flush=True)
 scored=[]
 for r in rows:
  counts={}
  for owner in ("wrong","right"):
   outs=[]
   for ti in range(4): outs += done[f"{r['item_id']}::{owner}::t{ti}"]["outputs"]
   hh=[hits(r["keyword"],x) for x in outs]
   counts[owner]={"n":len(outs),"exact":sum(x[0] for x in hh),"loose":sum(x[1] for x in hh)}
  scored.append({**r,"counts":counts,
   "exact_binding":counts["wrong"]["exact"]/counts["wrong"]["n"]-counts["right"]["exact"]/counts["right"]["n"],
   "loose_binding":counts["wrong"]["loose"]/counts["wrong"]["n"]-counts["right"]["loose"]/counts["right"]["n"]})
 with (a.out/"items.jsonl").open("w") as f:
  for r in scored:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 rng=np.random.default_rng(a.seed);fields=sorted(set(r["field"] for r in scored))
 report={"model":a.model,"samples_per_person_keyword":a.samples*4,"n_keywords":len(scored),
  "exact_all":cluster_summary(scored,"exact_binding",rng),"loose_all":cluster_summary(scored,"loose_binding",rng),
  "by_field":{fld:{"exact":cluster_summary([r for r in scored if r["field"]==fld],"exact_binding",rng),
                         "loose":cluster_summary([r for r in scored if r["field"]==fld],"loose_binding",rng)} for fld in fields}}
 (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__":main()
