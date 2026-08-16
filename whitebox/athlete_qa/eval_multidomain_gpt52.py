#!/usr/bin/env python3
"""Evaluate 1500 multidomain items in names-only and profiles conditions."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,random,re,threading,time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen

API="https://api.openai.com/v1/responses"
def norm(s):return " ".join(re.sub(r"[^\w\s]"," ",s.casefold()).split())
def aliases(s):
 z={norm(s)}
 z.add(norm(re.sub(r"\s*\([^)]*\)\s*$","",s)))
 z.add(norm(s.split(",",1)[0]))
 return z
def match(text,c,w):
 t=norm(text);cn,wn=norm(c),norm(w);hc,hw=cn in t,wn in t
 if t==cn:return "correct"
 if t==wn:return "wrong"
 ca,wa=aliases(c),aliases(w)
 if t in ca-wa:return "correct"
 if t in wa-ca:return "wrong"
 return "correct" if hc and not hw else "wrong" if hw and not hc else "unmatched"
def response_text(b):
 if b.get("output_text"):return b["output_text"]
 return "".join(c.get("text","")for x in b.get("output",[])for c in x.get("content",[])if c.get("type")=="output_text")
def request_one(task,model,retries,timeout):
 row,domain,condition=task;prompt=row["prepend_names_prompt" if condition=="names" else "prepend_profiles_prompt"]+"\nOutput only one candidate name."
 payload=json.dumps({"model":model,"reasoning":{"effort":"none"},"max_output_tokens":64,"store":False,"input":[{"role":"user","content":prompt}]}).encode()
 for attempt in range(retries+1):
  req=Request(API,data=payload,headers={"Authorization":"Bearer "+os.environ["OPENAI_API_KEY"],"Content-Type":"application/json"},method="POST")
  try:
   with urlopen(req,timeout=timeout)as res:b=json.load(res)
   text=response_text(b).strip();out=match(text,row["correct_answer"],row["wrong_answer"])
   return {"domain":domain,"id":row["id"],"condition":condition,"field":row["decisive_relation"]["field"],"correct_answer":row["correct_answer"],"wrong_answer":row["wrong_answer"],"generation":text,"outcome":out,"correct":out=="correct","response_id":b.get("id"),"usage":b.get("usage",{})}
  except HTTPError as e:
   detail=e.read().decode(errors="replace")
   if e.code not in(408,409,429,500,502,503,504)or attempt==retries:raise RuntimeError(f"HTTP {e.code}: {detail[:500]}")from e
  except(URLError,TimeoutError):
   if attempt==retries:raise
  time.sleep(min(60,2**attempt+random.random()))
def stats(rows):
 n=len(rows);c=sum(x["correct"]for x in rows);w=sum(x["outcome"]=="wrong"for x in rows);u=n-c-w
 return {"n":n,"correct":c,"accuracy":c/n if n else None,"wrong":w,"unmatched":u}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path(__file__).parent/"multidomain_v5");ap.add_argument("--out",type=Path,default=Path(__file__).parent/"multidomain_v5/gpt52_eval");ap.add_argument("--model",default="gpt-5.2-2025-12-11");ap.add_argument("--key-file",type=Path,default=Path("/home/tong56/.openai_api_key"));ap.add_argument("--workers",type=int,default=16);ap.add_argument("--retries",type=int,default=8);ap.add_argument("--timeout",type=int,default=120);ap.add_argument("--resume",action="store_true");ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
 if not os.environ.get("OPENAI_API_KEY") and a.key_file.exists():os.environ["OPENAI_API_KEY"]=a.key_file.read_text().strip()
 if not os.environ.get("OPENAI_API_KEY"):raise RuntimeError("OPENAI_API_KEY is not set")
 tasks=[]
 for d in("athlete","musician","building"):
  rows=[json.loads(x)for x in open(a.root/d/"primary_questions.jsonl")][:a.limit or None]
  tasks += [(x,d,c)for x in rows for c in("names","profiles")]
 a.out.mkdir(parents=True,exist_ok=True);path=a.out/"results.jsonl";done={}
 if a.resume and path.exists():
  for x in map(json.loads,path.open()):done[(x["domain"],x["id"],x["condition"])]=x
 pending=[x for x in tasks if(x[1],x[0]["id"],x[2])not in done];lock=threading.Lock();mode="a"if a.resume and path.exists()else"w"
 with path.open(mode)as f,concurrent.futures.ThreadPoolExecutor(a.workers)as pool:
  fs={pool.submit(request_one,x,a.model,a.retries,a.timeout):x for x in pending}
  for i,z in enumerate(concurrent.futures.as_completed(fs),1):
   r=z.result();key=(r["domain"],r["id"],r["condition"])
   with lock:done[key]=r;f.write(json.dumps(r,ensure_ascii=False)+"\n");f.flush()
   if i%25==0 or i==len(pending):print(f"[{i}/{len(pending)}] stored={len(done)}/{len(tasks)}",flush=True)
 rows=[done[(d,x["id"],c)]for x,d,c in tasks];by_domain={};by_field={}
 for d in("athlete","musician","building"):
  by_domain[d]={c:stats([x for x in rows if x["domain"]==d and x["condition"]==c])for c in("names","profiles")};by_domain[d]["profiles_gain_points"]=100*(by_domain[d]["profiles"]["accuracy"]-by_domain[d]["names"]["accuracy"])
 for d in("athlete","musician","building"):
  for field in sorted({x["field"]for x in rows if x["domain"]==d}):by_field[d+"/"+field]={c:stats([x for x in rows if x["domain"]==d and x["field"]==field and x["condition"]==c])for c in("names","profiles")}
 usage={k:sum(x.get("usage",{}).get(k,0)or 0 for x in rows)for k in("input_tokens","output_tokens","total_tokens")}
 report={"model":a.model,"calls":len(rows),"overall":{c:stats([x for x in rows if x["condition"]==c])for c in("names","profiles")},"by_domain":by_domain,"by_domain_and_field":by_field,"usage":usage}
 report["overall"]["profiles_gain_points"]=100*(report["overall"]["profiles"]["accuracy"]-report["overall"]["names"]["accuracy"])
 (a.out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n");print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
