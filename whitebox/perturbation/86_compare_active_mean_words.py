#!/usr/bin/env python3
"""Stage 86: summarize the crossed mean/active discrete flip experiment."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument("--in85",default="runs/85_active_word_generation.jsonl")
 p.add_argument("--out",default="runs/86_active_mean_word_comparison.json"); a=p.parse_args()
 rows=[json.loads(x) for x in open(a.in85) if x.strip()]; names=sorted(rows[0]["strategies"])
 report={"n_items":len(rows),"strategies":{}}
 for name in names:
  e=[r["strategies"][name]["edits"][0] for r in rows if r["strategies"][name]["edits"]]
  report["strategies"][name]={k:float(np.mean([x[k] for x in e])) for k in
    ("source_u_realized","p_gold","p_pred","rise_p_gold","drop_p_pred","paired_pred_to_gold")}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__":main()
