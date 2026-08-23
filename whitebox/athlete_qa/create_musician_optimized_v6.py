#!/usr/bin/env python3
"""Create a fixed-500 musician set focused on harder, model-independent relation families."""
import json, shutil
from pathlib import Path
H=Path(__file__).parent
SOURCES=[H/x for x in ("multidomain_v6_famous","multidomain_v6_famous_supplement","multidomain_v6_famous_supplement3","multidomain_v6_musician_targeted")]
BASE=H/"multidomain_v6_fixed500"; OUT=H/"multidomain_v6_fixed500_musician_opt"
QUOTAS={"record_label":86,"occupation":119,"birth_place":146,"instrument":74}
def read(p):return [json.loads(x) for x in p.open() if x.strip()]
def write(p,xs):p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in xs))
def key(x):return(tuple(sorted((x["correct_answer_qid"],x["wrong_answer_qid"]))),x["decisive_relation"]["field"],x["decisive_relation"]["value"])
def fame(x):return min(p.get("pageviews_60d",0) for p in x["profiles"].values())
def main():
 if OUT.exists():shutil.rmtree(OUT)
 shutil.copytree(BASE,OUT,ignore=shutil.ignore_patterns("gpt52_eval"))
 pool={}
 for s in SOURCES:
  states={x["id"]:x for x in read(s/"gpt52_probe_eval/results.jsonl") if x["domain"]=="musician"}
  for x in read(s/"musician/primary_questions.jsonl"):
   x["probe_state"]=states[x["id"]]["probe_state"];pool.setdefault(key(x),x)
 known=[]
 for field,n in QUOTAS.items():
  z=sorted((x for x in pool.values() if x["probe_state"]=="knows_both" and x["decisive_relation"]["field"]==field),key=lambda x:(-fame(x),key(x)))
  known+=z[:n]
 used={key(x) for x in known};other=sorted((x for x in pool.values() if key(x) not in used and x["probe_state"]!="knows_both" and x["decisive_relation"]["field"] in QUOTAS),key=lambda x:(-fame(x),key(x)))[:75]
 xs=sorted(known+other,key=key)
 assert len(xs)==500 and len(known)==425
 for i,x in enumerate(xs):x["original_id"]=x["id"];x["id"]=f"musician_opt_v6_qa_{i:04d}";x["calibration"]={"model":"gpt-5.2-2025-12-11","probe_state":x.pop("probe_state")}
 d=OUT/"musician";write(d/"primary_questions.jsonl",xs)
 write(d/"prepend_names.jsonl",[{"id":x["id"],"prompt":x["prepend_names_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]}for x in xs])
 write(d/"prepend_profiles.jsonl",[{"id":x["id"],"prompt":x["prepend_profiles_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]}for x in xs])
 write(d/"probes.jsonl",[{"id":f"{x['id']}_probe_{i}","parent_id":x["id"],**p}for x in xs for i,p in enumerate(x["probes"])])
 write(d/"profiles.jsonl",sorted({p["qid"]:p for x in xs for p in x["profiles"].values()}.values(),key=lambda x:x["qid"]))
 report=json.loads((OUT/"report.json").read_text());report["dataset"]="multidomain_v6_fixed500_musician_opt";report["musician_relation_quotas_known_both"]=QUOTAS;(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"items":len(xs),"knows_both":len(known),"quotas":QUOTAS},indent=2))
if __name__=="__main__":main()
