#!/usr/bin/env python3
"""Build 500-item Athlete/Musician/Building QA sets in ScientistQA style.

Candidates come from early-created English Wikipedia pages exposed by DBpedia.
Only stable, pre-2020-compatible relations are retained. Pair mining follows the
ScientistQA recipe: name-free profile similarity with a sparsity penalty, then a
single complementary decisive relation and two closed-book probes.
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import unquote
import numpy as np, requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ENDPOINT="https://dbpedia.org/sparql"; CUTOFF="2019-12-31"; UA="MultidomainTRAPQA/0.1 research dataset"
D="http://dbpedia.org/ontology/"; P="http://dbpedia.org/property/"
DOMAINS={
 "athlete":{"class":"Athlete","noun":"athlete","person":True,"props":{
  "sport":[D+"sport",P+"sport"],"team":[D+"team",P+"team"],"position":[D+"position",P+"position"],"birth_place":[D+"birthPlace"],"nationality":[D+"nationality"],"college":[D+"college",P+"college"]}},
 "musician":{"class":"MusicalArtist","noun":"musician","person":True,"props":{
  "genre":[D+"genre",P+"genre"],"instrument":[D+"instrument",P+"instrument"],"record_label":[D+"recordLabel",P+"label"],"associated_act":[D+"associatedBand",D+"associatedMusicalArtist",P+"associatedActs"],"birth_place":[D+"birthPlace"],"occupation":[D+"occupation"]}},
 "building":{"class":"Building","noun":"building","person":False,"props":{
  "architect":[D+"architect",P+"architect"],"architectural_style":[D+"architecturalStyle",P+"architecturalStyle"],"location":[D+"location",D+"city",D+"country"],"owner":[D+"owner",P+"owner"],"tenant":[D+"tenant",P+"tenant"],"material":[D+"buildingMaterial",P+"material"]}},
}
LABELS={"sport":"sport","team":"team","position":"playing position","birth_place":"birthplace","nationality":"nationality","college":"college",
 "genre":"musical genre","instrument":"instrument","record_label":"record label","associated_act":"associated act","occupation":"occupation",
 "architect":"architect","architectural_style":"architectural style","location":"location","owner":"owner","tenant":"tenant","material":"construction material"}

def sparql(query,cache,key):
 path=cache/(key+".json")
 if path.exists():return json.loads(path.read_text())
 s=requests.Session();s.headers["User-Agent"]=UA
 for k in range(8):
  r=s.get(ENDPOINT,params={"query":query,"format":"application/sparql-results+json"},timeout=180)
  if r.status_code==200:break
  time.sleep(min(120,2**k))
 else:raise RuntimeError((r.status_code,r.text[:500]))
 z=r.json();path.write_text(json.dumps(z,ensure_ascii=False));time.sleep(1);return z

def val(x):return x.get("value") if x else None
def label_uri(u):
 s=unquote(u.rsplit("/",1)[-1]).replace("_"," ");return re.sub(r"\s*\([^)]*\)$","",s).strip()
def qid(row):
 for x in row.get("same",[]):
  m=re.fullmatch(r"http://www\.wikidata\.org/entity/(Q\d+)",x)
  if m:return m.group(1)
 return None
def clean_literal(x):
 x=re.sub(r"\s+"," ",x).strip();return x if 1<len(x)<100 and not x.startswith("http") else None

def candidates(domain,cache,limit):
 c=DOMAINS[domain]["class"]
 q=f'''SELECT DISTINCT ?s ?name ?pageid ?abstract WHERE {{ ?s a <{D+c}>; <{D}wikiPageID> ?pageid; <{D}abstract> ?abstract; <http://www.w3.org/2000/01/rdf-schema#label> ?name. FILTER(lang(?abstract)='en' && lang(?name)='en') }} ORDER BY ASC(?pageid) LIMIT {limit}'''
 rows=[]
 for b in sparql(q,cache,f"{domain}_candidates_{limit}")["results"]["bindings"]:
  rows.append({"uri":val(b["s"]),"name":val(b["name"]),"pageid":int(float(val(b["pageid"]))),"abstract":val(b["abstract"])})
 return rows

def enrich(domain,rows,cache):
 props=DOMAINS[domain]["props"];by={x["uri"]:x for x in rows}
 for x in rows:x.update({k:[] for k in props});x["same"]=[]
 for bi in range(0,len(rows),80):
  batch=rows[bi:bi+80];values=" ".join("<"+x["uri"]+">" for x in batch);unions=[]
  for field,uris in props.items():
   for u in uris:unions.append(f'{{ ?s <{u}> ?o. BIND("{field}" AS ?field) }}')
  unions.append('{ ?s <http://www.w3.org/2002/07/owl#sameAs> ?o. BIND("same" AS ?field) }')
  q=f'SELECT ?s ?field ?o WHERE {{ VALUES ?s {{ {values} }} {{ '+" UNION ".join(unions)+' }} }'
  for b in sparql(q,cache,f"{domain}_props_{bi}_{len(batch)}")["results"]["bindings"]:
   s=val(b["s"]);f=val(b["field"]);o=val(b["o"])
   if s not in by:continue
   if f=="same":by[s][f].append(o);continue
   z=label_uri(o) if b["o"].get("type")=="uri" else clean_literal(o)
   if z and z.lower()!=by[s]["name"].lower() and z not in by[s][f]:by[s][f].append(z)
 out=[]
 for x in rows:
  x["qid"]=qid(x);x["wikipedia_url"]="https://en.wikipedia.org/wiki/"+x["uri"].rsplit("/",1)[-1]
  x["attributes"]={k:sorted(v)[:12] for k,v in ((k,x[k])for k in props) if v};x["tag_count"]=len(x["attributes"])
  if x["qid"] and x["tag_count"]>=2:out.append(x)
 return out

def linear(p):return "; ".join(f"{LABELS[k]}: {', '.join(v)}" for k,v in sorted(p["attributes"].items()))
def mine(domain,profiles,n,seed,candidate_offset=0,allowed_fields=None):
 texts=[linear(x) for x in profiles];X=TfidfVectorizer(ngram_range=(1,2),min_df=1).fit_transform(texts);S=cosine_similarity(X);lam=float(np.median([x["tag_count"]for x in profiles]));cs=[]
 # Nearest 30 neighbours are enough and avoid quadratic candidate materialization.
 for i,a in enumerate(profiles):
  for j in np.argpartition(S[i],-31)[-31:]:
   if j<=i:continue
   b=profiles[j];common=set(a["attributes"])&set(b["attributes"])
   for f in common:
    av=set(a["attributes"][f]);bv=set(b["attributes"][f])
    for target in sorted(av^bv):
     holder,non=(a,b) if target in av else (b,a);dense=min(a["tag_count"],b["tag_count"]);score=float(S[i,j])*dense/(dense+lam)
     cs.append((score,f,target,holder,non))
 cs.sort(key=lambda z:z[0],reverse=True);rng=np.random.default_rng(seed);picked=[];pairs=set();cnt=Counter();fields=Counter()
 for score,f,target,holder,non in cs[candidate_offset:]:
  if allowed_fields and f not in allowed_fields:continue
  pair=tuple(sorted((holder["qid"],non["qid"])))
  if pair in pairs or cnt[holder["qid"]]>=4 or cnt[non["qid"]]>=4:continue
  # Keep relation families reasonably balanced.
  if fields[f]>n//(len(allowed_fields) if allowed_fields else len(DOMAINS[domain]["props"]))+20:continue
  picked.append((score,f,target,holder,non));pairs.add(pair);cnt[holder["qid"]]+=1;cnt[non["qid"]]+=1;fields[f]+=1
  if len(picked)==n:break
 items=[];noun=DOMAINS[domain]["noun"]
 for ix,(score,f,target,wrong,correct) in enumerate(picked):
  shared=[]
  for k in set(wrong["attributes"])&set(correct["attributes"]):shared += [(k,v)for v in sorted(set(wrong["attributes"][k])&set(correct["attributes"][k]))]
  lead=f"This {noun} has a documented profile established before the end of 2019."
  if shared:
   k,v=shared[0];lead+=f" Like the other candidate, this {noun} is associated with the {LABELS[k]} {v}."
  question=lead+f" However, this {noun} was not associated with the {LABELS[f]} {target} by the end of 2019. What {'is the name of this person' if domain!='building' else 'building is this'}?"
  order=[correct,wrong] if rng.random()<.5 else [wrong,correct]
  probe=lambda p:f"By the end of 2019, was {p['name']} associated with the {LABELS[f]} {target}?"
  item={"id":f"{domain}_qa_{ix:04d}","domain":domain,"question":question,"correct_answer":correct["name"],"correct_answer_qid":correct["qid"],"wrong_answer":wrong["name"],"wrong_answer_qid":wrong["qid"],"candidate_order":[x["name"]for x in order],"decisive_relation":{"field":f,"label":LABELS[f],"value":target,"fact_cutoff":CUTOFF,"correct_candidate_has":False,"wrong_candidate_has":True},"pair_similarity":score,"probes":[{"question":probe(wrong),"correct_answer":1},{"question":probe(correct),"correct_answer":0}],"profiles":{correct["qid"]:correct,wrong["qid"]:wrong}}
  fmt=lambda p:"name: "+p["name"]+"\n"+linear(p)
  item["prepend_names_prompt"]="Choose one of the following two options as the answer to the question below:\n1. "+order[0]["name"]+"\n2. "+order[1]["name"]+"\nQuestion:\n"+question
  item["prepend_profiles_prompt"]="Given two profiles:\n"+fmt(order[0])+"\n"+fmt(order[1])+"\nChoose exactly one and output its name as the answer:\n"+question;items.append(item)
 return items,fields,len(cs)

def dump(path,rows):path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n"for x in rows))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",default="multidomain_v1");ap.add_argument("--n",type=int,default=500);ap.add_argument("--candidate-limit",type=int,default=3000);ap.add_argument("--seed",type=int,default=42);ap.add_argument("--domains",nargs="*",default=list(DOMAINS));a=ap.parse_args();root=Path(a.out);cache=root/"cache";cache.mkdir(parents=True,exist_ok=True);summary={}
 for di,d in enumerate(a.domains):
  out=root/d;out.mkdir(parents=True,exist_ok=True);ps=enrich(d,candidates(d,cache,a.candidate_limit),cache);items,fields,nc=mine(d,ps,a.n,a.seed+di)
  dump(out/"profiles.jsonl",ps);dump(out/"primary_questions.jsonl",items);dump(out/"prepend_names.jsonl",[{"id":x["id"],"prompt":x["prepend_names_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]}for x in items]);dump(out/"prepend_profiles.jsonl",[{"id":x["id"],"prompt":x["prepend_profiles_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]}for x in items]);dump(out/"probes.jsonl",[{"id":f"{x['id']}_probe_{i}","parent_id":x["id"],**p}for x in items for i,p in enumerate(x["probes"])]);summary[d]={"requested":a.n,"generated":len(items),"profiles":len(ps),"candidate_relations":nc,"field_counts":fields}
 (root/"report.json").write_text(json.dumps({"created_at":date.today().isoformat(),"fact_cutoff":CUTOFF,"source":"DBpedia/Wikipedia/Wikidata","selection":"early English Wikipedia page IDs; stable relations only","summary":summary},indent=2,ensure_ascii=False,default=dict)+"\n");print(json.dumps(summary,indent=2,default=dict))
if __name__=="__main__":main()
