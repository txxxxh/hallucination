#!/usr/bin/env python3
"""Offline alias-aware regrading and aggregation of saved GPT-5.2 outputs."""
import json
from collections import defaultdict
from pathlib import Path
from eval_multidomain_gpt52 import match,stats
ROOT=Path(__file__).parent/"multidomain_v5/gpt52_eval"
rows=[]
for x in map(json.loads,(ROOT/"results.jsonl").open()):
 x["outcome"]=match(x["generation"],x["correct_answer"],x["wrong_answer"]);x["correct"]=x["outcome"]=="correct";rows.append(x)
with(ROOT/"results_regraded.jsonl").open("w")as f:
 for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
by_domain={};by_field={}
for d in("athlete","musician","building"):
 by_domain[d]={c:stats([x for x in rows if x["domain"]==d and x["condition"]==c])for c in("names","profiles")};by_domain[d]["profiles_gain_points"]=100*(by_domain[d]["profiles"]["accuracy"]-by_domain[d]["names"]["accuracy"])
for d in("athlete","musician","building"):
 for field in sorted({x["field"]for x in rows if x["domain"]==d}):
  z={c:stats([x for x in rows if x["domain"]==d and x["field"]==field and x["condition"]==c])for c in("names","profiles")};z["profiles_gain_points"]=100*(z["profiles"]["accuracy"]-z["names"]["accuracy"]);by_field[d+"/"+field]=z
overall={c:stats([x for x in rows if x["condition"]==c])for c in("names","profiles")};overall["profiles_gain_points"]=100*(overall["profiles"]["accuracy"]-overall["names"]["accuracy"])
usage={k:sum(x.get("usage",{}).get(k,0)or 0 for x in rows)for k in("input_tokens","output_tokens","total_tokens")}
report={"model":"gpt-5.2-2025-12-11","calls":len(rows),"grading":"exact candidate match, then unique Wikipedia-title alias after removing parenthetical/location suffix","overall":overall,"by_domain":by_domain,"by_domain_and_field":by_field,"usage":usage}
(ROOT/"summary_regraded.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n");print(json.dumps(report,indent=2,ensure_ascii=False))
