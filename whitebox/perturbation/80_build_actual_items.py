#!/usr/bin/env python3
"""Build Item-compatible JSON from dataset rows and actual model generations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--records",required=True)
 p.add_argument("--item_id",nargs="+",required=True); p.add_argument("--out",required=True); a=p.parse_args()
 data={str(x.get("key",x.get("item_id"))):x for x in json.load(open(a.data))}
 rec={str(x["key"]):x for x in map(json.loads,open(a.records))}; out=[]
 for key in a.item_id:
  raw,r=data[key],rec[key]; pred=str(r["parsed_answer"]); right=str(raw["rgt_ans"]); wrong=str(raw["wrg_ans"])
  gold=wrong if pred==right else right
  out.append({"item_id":key,"prompt":raw["prompt"],"gold":gold,"pred":pred,
              "eval_gold":right,"actual_correct":bool(r["correct"]),
              "record_generation":r.get("generation")})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps([{"item_id":x["item_id"],"pred":x["pred"],"gold":x["gold"],"actual_correct":x["actual_correct"]} for x in out],indent=2))
if __name__=="__main__":main()
