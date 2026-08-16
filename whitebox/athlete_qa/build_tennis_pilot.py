#!/usr/bin/env python3
"""Build a provenance-rich TennisQA pilot from Wikipedia historical revisions.

The factual cutoff is 2019-12-31.  Grand Slam winner sets are parsed from the
last English-Wikipedia revision at or before the cutoff.  Current lists are
also parsed only to reject negatives that became winners after the cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CUTOFF = "2019-12-31T23:59:59Z"
API = "https://en.wikipedia.org/w/api.php"
TOURNAMENTS = {
    "australian_open": {
        "label": "Australian Open", "men": "List of Australian Open men's singles champions",
        "women": "List of Australian Open women's singles champions"},
    "french_open": {
        "label": "French Open", "men": "List of French Open men's singles champions",
        "women": "List of French Open women's singles champions"},
    "wimbledon": {
        "label": "Wimbledon", "men": "List of Wimbledon gentlemen's singles champions",
        "women": "List of Wimbledon ladies' singles champions"},
    "us_open": {
        "label": "US Open", "men": "List of US Open men's singles champions",
        "women": "List of US Open women's singles champions"},
}
UA = "TennisQAPilot/0.1 (research dataset; contact: local workspace owner)"


class Wiki:
    def __init__(self, cache: Path, delay: float = .5):
        self.cache = cache; self.cache.mkdir(parents=True, exist_ok=True)
        self.delay = delay; self.s = requests.Session(); self.s.headers["User-Agent"] = UA

    def get(self, params, key):
        path = self.cache / f"{key}.json"
        if path.exists(): return json.loads(path.read_text())
        for attempt in range(8):
            r = self.s.get(API, params=params, timeout=60)
            if r.status_code != 429:
                r.raise_for_status(); break
            wait = float(r.headers.get("Retry-After", min(60, 2 ** attempt)))
            time.sleep(max(wait, 2 ** attempt))
        else:
            raise RuntimeError(f"Wikipedia rate limit persisted for {key}")
        out = r.json()
        path.write_text(json.dumps(out, ensure_ascii=False)); time.sleep(self.delay); return out

    def revision(self, title, cutoff=CUTOFF):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        j = self.get({"action":"query","format":"json","prop":"revisions|pageprops",
                      "titles":title,"rvprop":"ids|timestamp","rvstart":cutoff,
                      "rvdir":"older","rvlimit":1}, f"rev_{safe}_{cutoff[:10]}")
        page = next(iter(j["query"]["pages"].values()))
        if "revisions" not in page: return None
        rev = page["revisions"][0]
        return {"title":page["title"], "pageid":page["pageid"],
                "qid":page.get("pageprops",{}).get("wikibase_item"), **rev}

    def parsed(self, oldid, key):
        j = self.get({"action":"parse","format":"json","oldid":oldid,
                      "prop":"text"}, f"parse_{key}_{oldid}")
        return BeautifulSoup(j["parse"]["text"]["*"], "html.parser")

    def current_parsed(self, title, key):
        j = self.get({"action":"parse","format":"json","page":title,
                      "prop":"text|revid"}, f"parse_current_{key}")
        return BeautifulSoup(j["parse"]["text"]["*"], "html.parser"), j["parse"]["revid"]

    def qids(self, titles):
        out={}
        for bi in range(0,len(titles),40):
            batch=titles[bi:bi+40]; key=f"qids_{bi}_"+str(abs(hash(tuple(batch))))
            j=self.get({"action":"query","format":"json","prop":"pageprops",
                        "titles":"|".join(batch)},key)
            for p in j["query"]["pages"].values():
                if p.get("pageprops",{}).get("wikibase_item"): out[p["title"]]=p["pageprops"]["wikibase_item"]
        return out


    def revisions_bulk(self, titles, cutoff=CUTOFF):
        out = {}
        for bi in range(0, len(titles), 40):
            batch = titles[bi:bi+40]
            j = self.get({"action":"query","format":"json","prop":"revisions|pageprops",
                "titles":"|".join(batch),"rvprop":"ids|timestamp","rvstart":cutoff,
                "rvdir":"older","rvlimit":1}, f"bulk_revs_{cutoff[:10]}_{bi}")
            for page in j["query"]["pages"].values():
                if "revisions" not in page: continue
                rev=page["revisions"][0]
                out[page["title"]]={"title":page["title"],"pageid":page["pageid"],
                    "qid":page.get("pageprops",{}).get("wikibase_item"),**rev}
        return out


def clean_title(a):
    if not a: return None
    href=a.get("href","")
    if href.startswith("./"): raw=href[2:]
    elif href.startswith("/wiki/"): raw=href[6:]
    else: return None
    title=unquote(raw).replace("_"," ").split("#")[0]
    if title.startswith(("File:","Flag of ","List of ")): return None
    return title


def champion_rows(soup):
    """Extract (year, linked champion title) from tables with a Champion header."""
    rows=[]
    for table in soup.select("table.wikitable"):
        trs=table.select("tr"); header=None
        for tr in trs[:5]:
            cells=[x.get_text(" ",strip=True).lower() for x in tr.find_all(["th","td"],recursive=False)]
            if any(re.fullmatch(r"champion(?:\s*\[[^]]+\])?",x) for x in cells): header=cells; break
        if not header: continue
        ci=next(i for i,x in enumerate(header) if x.startswith("champion"))
        for tr in trs:
            cells=tr.find_all(["th","td"],recursive=False)
            if len(cells)<=ci: continue
            ym=re.search(r"\b(18|19|20)\d{2}\b",cells[0].get_text(" ",strip=True))
            if not ym: continue
            links=[clean_title(a) for a in cells[ci].select("a[href]")]
            links=[x for x in links if x]
            if links: rows.append((int(ym.group()),links[0]))
    return rows


def wikidata_entities(qids, session, cache):
    out={}
    for bi in range(0,len(qids),40):
        batch=qids[bi:bi+40]; digest=hashlib.sha1("|".join(batch).encode()).hexdigest()[:16]; path=cache/f"entities_{digest}.json"
        if path.exists(): j=json.loads(path.read_text())
        else:
            for attempt in range(8):
                r=session.get("https://www.wikidata.org/w/api.php",params={"action":"wbgetentities",
                    "format":"json","ids":"|".join(batch),"props":"labels|claims","languages":"en"},timeout=60)
                if r.status_code != 429:
                    r.raise_for_status(); break
                time.sleep(max(float(r.headers.get("Retry-After",0) or 0), 2 ** attempt))
            else: raise RuntimeError("Wikidata rate limit persisted")
            j=r.json(); path.write_text(json.dumps(j,ensure_ascii=False)); time.sleep(.5)
        out.update(j["entities"])
    return out


def claim_qids(e,p):
    z=[]
    for c in e.get("claims",{}).get(p,[]):
        try: z.append(c["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError,TypeError): pass
    return z


def claim_time(e,p):
    try:return e["claims"][p][0]["mainsnak"]["datavalue"]["value"]["time"][1:11]
    except (KeyError,IndexError,TypeError):return None


def labels_for(qids, session, cache):
    ents=wikidata_entities(sorted(set(qids)),session,cache)
    return {q:(e.get("labels",{}).get("en",{}).get("value") or q) for q,e in ents.items()}


def profile_text(p):
    won=", ".join(p["slam_labels"])
    return f"tennis player; gender {p['sex']}; country {p.get('country','unknown')}; " \
           f"born {p.get('birth_year','unknown')}; Grand Slam singles titles {won}"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="pilot_v1")
    ap.add_argument("--n",type=int,default=100); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); root=Path(a.out); cache=root/"cache"; root.mkdir(parents=True,exist_ok=True)
    wiki=Wiki(cache); cutoff_sets=defaultdict(set); current_sets=defaultdict(set); provenance={}
    title_sex={}
    for tid,meta in TOURNAMENTS.items():
        for sex in ("men","women"):
            key=f"{tid}_{sex}"; title=meta[sex]; rev=wiki.revision(title)
            if not rev: raise RuntimeError(f"missing historical revision: {title}")
            oldrows=champion_rows(wiki.parsed(rev["revid"],key)); cur_soup,cur_rev=wiki.current_parsed(title,key)
            currows=champion_rows(cur_soup)
            cutoff_sets[key]={name for year,name in oldrows if year<=2019}
            current_sets[key]={name for year,name in currows}
            for name in cutoff_sets[key]: title_sex.setdefault(name,sex)
            provenance[key]={"source_title":title,"source_url":"https://en.wikipedia.org/wiki/"+title.replace(" ","_"),
                "cutoff_revision_id":rev["revid"],"cutoff_revision_timestamp":rev["timestamp"],
                "current_revision_id":cur_rev,"n_cutoff_champions":len(cutoff_sets[key]),
                "n_current_champions":len(current_sets[key])}
    titles=sorted(title_sex); qmap=wiki.qids(titles); titles=[x for x in titles if x in qmap]
    sess=requests.Session(); sess.headers["User-Agent"]=UA
    ents=wikidata_entities(sorted(set(qmap.values())),sess,cache)
    ref=[]
    for e in ents.values(): ref += claim_qids(e,"P27")+claim_qids(e,"P21")
    labs=labels_for(ref,sess,cache)
    profiles=[]
    for title in titles:
        qid=qmap[title]; e=ents[qid]; sex=title_sex[title]
        countries=[labs.get(x,x) for x in claim_qids(e,"P27")]
        born=claim_time(e,"P569"); slams=[]
        for tid,m in TOURNAMENTS.items():
            if title in cutoff_sets[f"{tid}_{sex}"]: slams.append(tid)
        if not slams: continue
        profiles.append({"name":e.get("labels",{}).get("en",{}).get("value",title),"source_title":title,"qid":qid,"sex":sex,"date_of_birth":born,
            "birth_year":int(born[:4]) if born else None,"country":countries[0] if countries else None,
            "grand_slam_singles_titles_through_2019":slams,
            "slam_labels":[TOURNAMENTS[x]["label"] for x in slams],
            "wikipedia_url":"https://en.wikipedia.org/wiki/"+title.replace(" ","_"),
            "cutoff_evidence":"historical Grand Slam champion-list revision"})
    texts=[profile_text(p) for p in profiles]
    X=TfidfVectorizer(ngram_range=(1,2)).fit_transform(texts); sim=cosine_similarity(X)
    candidates=[]
    for i,p in enumerate(profiles):
      for j in range(i+1,len(profiles)):
        q=profiles[j]
        if p["sex"]!=q["sex"]: continue
        py,qy=p.get("birth_year"),q.get("birth_year")
        if py and qy and abs(py-qy)>18: continue
        ps=set(p["grand_slam_singles_titles_through_2019"]); qs=set(q["grand_slam_singles_titles_through_2019"])
        for target in sorted(ps^qs):
            winner,nonwinner=(p,q) if target in ps else (q,p)
            # Reject a negative that later became a champion of this event.
            if nonwinner["source_title"] in current_sets[f"{target}_{p['sex']}"]: continue
            shared=sorted(ps&qs); country_same=bool(p.get("country") and p.get("country")==q.get("country"))
            score=float(sim[i,j])+0.25*bool(shared)+0.12*country_same
            if py and qy: score+=0.15*max(0,1-abs(py-qy)/18)
            candidates.append({"score":score,"target":target,"winner":winner,"nonwinner":nonwinner,
                               "shared":shared,"same_country":country_same})
    # Greedy balance across tournament/sex and cap repeated people/pairs.
    candidates.sort(key=lambda x:x["score"],reverse=True); picked=[]; person_count=Counter(); bucket_count=Counter(); pair_seen=set()
    while len(picked)<a.n:
        best=None; best_adj=-1e9
        for c in candidates:
            pair=tuple(sorted((c["winner"]["qid"],c["nonwinner"]["qid"])))
            if pair in pair_seen or person_count[c["winner"]["qid"]]>=4 or person_count[c["nonwinner"]["qid"]]>=4: continue
            bucket=(c["target"],c["winner"]["sex"])
            adj=c["score"]-.08*bucket_count[bucket]-.025*(person_count[c["winner"]["qid"]]+person_count[c["nonwinner"]["qid"]])
            if adj>best_adj: best_adj=adj; best=c
        if best is None: break
        picked.append(best); pair=tuple(sorted((best["winner"]["qid"],best["nonwinner"]["qid"]))); pair_seen.add(pair)
        person_count[best["winner"]["qid"]]+=1; person_count[best["nonwinner"]["qid"]]+=1
        bucket_count[(best["target"],best["winner"]["sex"])]+=1
    rng=np.random.default_rng(a.seed); items=[]
    for ix,c in enumerate(picked):
        correct=c["nonwinner"]; wrong=c["winner"]; label=TOURNAMENTS[c["target"]]["label"]
        common=f"This accomplished {'male' if correct['sex']=='men' else 'female'} tennis player won at least one Grand Slam singles title during a career established before the end of 2019."
        if c["shared"]:
            common+=f" Like the other candidate, this player won the {TOURNAMENTS[c['shared'][0]]['label']} singles title."
        if c["same_country"]: common+=f" Both candidates represented {correct['country']}."
        question=common+f" However, by the end of 2019, this player had never won the {label} singles title. Who is this person?"
        order=[correct,wrong] if rng.random()<.5 else [wrong,correct]
        item={"id":f"tennis_qa_{ix:04d}","question":question,"correct_answer":correct["name"],
              "correct_answer_qid":correct["qid"],"wrong_answer":wrong["name"],"wrong_answer_qid":wrong["qid"],
              "candidate_order":[x["name"] for x in order],"decisive_relation":{"type":"grand_slam_singles_title",
              "tournament":label,"tournament_key":c["target"],"fact_cutoff":"2019-12-31",
              "correct_candidate_won":False,"wrong_candidate_won":True},"shared_tournaments":[TOURNAMENTS[x]["label"] for x in c["shared"]],
              "pair_similarity":c["score"],"probes":[
                  {"question":f"By the end of 2019, had {wrong['name']} won the {label} singles title?","correct_answer":1},
                  {"question":f"By the end of 2019, had {correct['name']} won the {label} singles title?","correct_answer":0}],
              "source_key":f"{c['target']}_{correct['sex']}"}
        def fmt_profile(p): return "name: "+p["name"]+"\ncountry: "+str(p.get("country") or "unknown")+"\ndate_of_birth: "+str(p.get("date_of_birth") or "unknown")+"\ngrand_slam_singles_titles_through_2019: "+"; ".join(p["slam_labels"])
        item["prepend_profiles_prompt"]="Given two profiles of two persons:\n"+fmt_profile(order[0])+"\n"+fmt_profile(order[1])+"\nChoose exactly one profile from the two, and output the name of the person as the answer to the following question:\n"+question
        item["prepend_names_prompt"]="Choose one of the following two options as the answer to the question below:\n1. "+order[0]["name"]+"\n2. "+order[1]["name"]+"\nQuestion:\n"+question
        item["profiles"]={correct["qid"]:correct,wrong["qid"]:wrong}; items.append(item)
    def dump_jsonl(path,rows): path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    dump_jsonl(root/"athlete_profiles.jsonl",profiles); dump_jsonl(root/"primary_questions.jsonl",items)
    dump_jsonl(root/"prepend_names.jsonl",[{"id":x["id"],"prompt":x["prepend_names_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]} for x in items])
    dump_jsonl(root/"prepend_profiles.jsonl",[{"id":x["id"],"prompt":x["prepend_profiles_prompt"],"rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],"wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]} for x in items])
    probes=[]
    for x in items:
        for k,p in enumerate(x["probes"]): probes.append({"id":f"{x['id']}_probe_{k}","parent_id":x["id"],**p})
    dump_jsonl(root/"probes.jsonl",probes)
    review=["id\tcorrect\tdistractor\tdecisive_tournament\tsimilarity\tquestion\tcorrect_source\tdistractor_source"]
    for x in items:
        cp=x["profiles"][x["correct_answer_qid"]]; wp=x["profiles"][x["wrong_answer_qid"]]
        review.append("\t".join([x["id"],x["correct_answer"],x["wrong_answer"],x["decisive_relation"]["tournament"],f"{x['pair_similarity']:.4f}",x["question"],cp["wikipedia_url"],wp["wikipedia_url"]]))
    (root/"human_review.tsv").write_text("\n".join(review)+"\n")
    report={"created_at":date.today().isoformat(),"fact_cutoff":"2019-12-31","requested":a.n,"generated":len(items),
            "n_profiles":len(profiles),"n_candidate_relations":len(candidates),"bucket_counts":{"|".join(k):v for k,v in bucket_count.items()},
            "provenance":provenance,"validation":{"all_probe_labels_complementary":all([p["correct_answer"] for p in x["probes"]]==[1,0] for x in items),
            "unique_pairs":len(pair_seen)==len(items),"post_cutoff_winner_negatives":0}}
    (root/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
