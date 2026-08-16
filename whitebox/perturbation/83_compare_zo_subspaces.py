#!/usr/bin/env python3
"""Stage 83: summarize random/vocabulary/active ZO keyword searches."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument("--in82",default="runs/82_zo_active_keywords.jsonl")
 p.add_argument("--out",default="runs/83_zo_subspace_comparison.json"); a=p.parse_args()
 rows=[json.loads(x) for x in open(a.in82) if x.strip()]; methods=["random","vocab","active"]
 report={"n_items":len(rows),"rank":rows[0]["rank"],"methods":{}}
 for name in methods:
  gains=np.concatenate([[x["abs_u"] for x in r["methods"][name]] for r in rows])
  rnd=np.concatenate([[x["random_best_abs_u"] for x in r["methods"][name]] for r in rows])
  j=[]
  for r in rows:
   m,z=set(r["selection"]["mean"]),set(r["selection"][name]); u=m|z
   j.append(len(m&z)/len(u) if u else 1.)
  report["methods"][name]={"mean_abs_gain":float(gains.mean()),
    "median_abs_gain":float(np.median(gains)),"mean_equal_query_random_best":float(rnd.mean()),
    "fraction_optimizer_beats_random_best":float(np.mean(gains>rnd)),
    "mean_rank_rho_vs_mean":float(np.nanmean([r["rho_vs_mean"][name] for r in rows])),
    "mean_topk_jaccard_vs_mean":float(np.mean(j))}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))

if __name__=="__main__": main()
