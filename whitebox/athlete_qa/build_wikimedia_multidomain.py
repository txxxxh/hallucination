#!/usr/bin/env python3
"""Wikimedia-API frontend for multidomain ScientistQA-style construction."""
import argparse,json,re,time
from pathlib import Path
import requests
import build_multidomain_trapqa as core
from build_tennis_pilot import Wiki,wikidata_entities,labels_for,claim_qids

API="https://en.wikipedia.org/w/api.php";UA="MultidomainTRAPQA/0.1 research dataset"
CATEGORIES={
 "athlete":["Association football players","Basketball players","Olympic athletes","Swimmers","Boxers","Tennis players","Track and field athletes","Baseball players"],
 "musician":["Singers","Guitarists","Pianists","Composers","Rappers","Drummers","Violinists","Jazz musicians"],
 "building":["Skyscrapers","Stadiums","Museums","Palaces","Castles","Church buildings","Railway stations","Concert halls"],
}
PIDS={
 "athlete":{"sport":["P641"],"team":["P54"],"position":["P413"],"birth_place":["P19"],"nationality":["P27"],"college":["P69"]},
 "musician":{"genre":["P136"],"instrument":["P1303"],"record_label":["P264"],"associated_act":["P463","P361"],"birth_place":["P19"],"occupation":["P106"]},
 "building":{"architect":["P84"],"architectural_style":["P149"],"location":["P131","P17"],"owner":["P127"],"tenant":["P466"],"material":["P186"]},
}
def get(s,cache,key,params):
 p=cache/(key+".json")
 if p.exists():return json.loads(p.read_text())
 for k in range(8):
  r=s.get(API,params={"action":"query","format":"json",**params},timeout=90)
  if r.status_code==200:break
  time.sleep(min(60,2**k))
 r.raise_for_status();z=r.json();p.write_text(json.dumps(z,ensure_ascii=False));time.sleep(.3);return z
def pages(domain,s,cache,max_pages):
 out={};queue=[(x,0,x)for x in CATEGORIES[domain]];seen=set();call=0
 while queue and len(out)<max_pages:
  cat,depth,root=queue.pop(0)
  if cat in seen:continue
  seen.add(cat);cont=None
  while True:
   params={"generator":"categorymembers","gcmtitle":"Category:"+cat,"gcmtype":"page|subcat","gcmlimit":"max","prop":"pageprops"}
   if cont:params["gcmcontinue"]=cont
   z=get(s,cache,f"{domain}_bfs_{call}",params);call+=1
   for p in z.get("query",{}).get("pages",{}).values():
    if p.get("ns")==14 and depth<2:queue.append((p["title"].removeprefix("Category:"),depth+1,root));continue
    q=p.get("pageprops",{}).get("wikibase_item")
    if q and p.get("ns")==0:out[q]={"name":p["title"],"source_title":p["title"],"qid":q,"pageid":p["pageid"],"category":root,"wikipedia_url":"https://en.wikipedia.org/wiki/"+p["title"].replace(" ","_")}
   cont=z.get("continue",{}).get("gcmcontinue")
   if not cont:break
 return sorted(out.values(),key=lambda x:x["pageid"])
def profiles(domain,rows,s,cache):
 ents=wikidata_entities(sorted({x["qid"]for x in rows}),s,cache);refs=[]
 for e in ents.values():
  for ps in PIDS[domain].values():
   for p in ps:refs+=claim_qids(e,p)
 labs=labels_for(refs,s,cache);out=[]
 for x in rows:
  e=ents.get(x["qid"],{});attrs={}
  for f,ps in PIDS[domain].items():
   vals=[]
   for p in ps:vals += [labs.get(q,q)for q in claim_qids(e,p)]
   vals=sorted(set(v for v in vals if v and v!=x["name"] and not re.fullmatch(r"Q\d+",v)))[:12]
   if vals:attrs[f]=vals
  x["attributes"]=attrs;x["tag_count"]=len(attrs);x["source_category"]=x.pop("category")
  if len(attrs)>=2:out.append(x)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",default="multidomain_v3");ap.add_argument("--n",type=int,default=500);ap.add_argument("--max-pages",type=int,default=1500);ap.add_argument("--seed",type=int,default=42);ap.add_argument("--domains",nargs="*",default=list(CATEGORIES));a=ap.parse_args();root=Path(a.out);cache=root/"cache";cache.mkdir(parents=True,exist_ok=True);s=requests.Session();s.headers["User-Agent"]=UA;summary={}
 for di,d in enumerate(a.domains):
  raw=pages(d,s,cache,a.max_pages);ps=profiles(d,raw,s,cache);items,fields,nc=core.mine(d,ps,a.n,a.seed+di);out=root/d;out.mkdir(exist_ok=True)
  core.dump(out/"profiles.jsonl",ps);core.dump(out/"primary_questions.jsonl",items);core.dump(out/"prepend_names.jsonl",[{"id":x["id"],"prompt":x["prepend_names_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]}for x in items]);core.dump(out/"prepend_profiles.jsonl",[{"id":x["id"],"prompt":x["prepend_profiles_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]}for x in items]);core.dump(out/"probes.jsonl",[{"id":f"{x['id']}_probe_{i}","parent_id":x["id"],**p}for x in items for i,p in enumerate(x["probes"])]);summary[d]={"historical_pages":len(raw),"profiles":len(ps),"generated":len(items),"candidate_relations":nc,"field_counts":dict(fields)};print(d,summary[d],flush=True)
 (root/"report.json").write_text(json.dumps({"fact_cutoff":core.CUTOFF,"source":"English Wikipedia revisions at/before cutoff + current stable Wikidata claims","summary":summary},indent=2,ensure_ascii=False)+"\n")
if __name__=="__main__":main()
