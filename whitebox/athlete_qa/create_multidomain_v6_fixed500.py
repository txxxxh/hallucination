#!/usr/bin/env python3
"""Merge independently generated v6 pools into 500 calibrated items/domain."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = (HERE / "multidomain_v6_famous", HERE / "multidomain_v6_famous_supplement", HERE / "multidomain_v6_famous_supplement2", HERE / "multidomain_v6_famous_supplement3")
OUT = HERE / "multidomain_v6_fixed500"
DOMAINS = ("athlete", "musician", "building")
N, N_BOTH = 500, 425

def read(path): return [json.loads(x) for x in path.open() if x.strip()]
def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
def fact_key(x):
    return (tuple(sorted((x["correct_answer_qid"], x["wrong_answer_qid"]))),
            x["decisive_relation"]["field"], x["decisive_relation"]["value"])
def fame(x): return min(p.get("pageviews_60d", 0) for p in x["profiles"].values())

def main():
    report = {}
    for domain in DOMAINS:
        pool = {}
        duplicate_count = 0
        for source_index, source in enumerate(SOURCES):
            if not (source / domain / "primary_questions.jsonl").exists():
                continue
            states = {x["id"]: x for x in read(source / "gpt52_probe_eval/results.jsonl") if x["domain"] == domain}
            for item in read(source / domain / "primary_questions.jsonl"):
                key = fact_key(item)
                if key in pool:
                    duplicate_count += 1
                    continue
                item["source_pool"] = source.name
                item["probe_state"] = states[item["id"]]["probe_state"]
                pool[key] = item
        known = sorted((x for x in pool.values() if x["probe_state"] == "knows_both"),
                       key=lambda x: (-fame(x), fact_key(x)))
        other = sorted((x for x in pool.values() if x["probe_state"] != "knows_both"),
                       key=lambda x: (-fame(x), fact_key(x)))
        if len(known) < N_BOTH or len(other) < N - N_BOTH:
            raise RuntimeError(f"{domain}: unique known={len(known)}, other={len(other)}")
        selected = known[:N_BOTH] + other[:N-N_BOTH]
        selected.sort(key=fact_key)
        for index, item in enumerate(selected):
            item["original_id"] = item["id"]
            item["id"] = f"{domain}_v6_qa_{index:04d}"
            item["calibration"] = {"model": "gpt-5.2-2025-12-11", "probe_state": item.pop("probe_state")}
        write(OUT/domain/"primary_questions.jsonl", selected)
        write(OUT/domain/"prepend_names.jsonl", [{"id":x["id"],"prompt":x["prepend_names_prompt"],
          "rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],
          "wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]} for x in selected])
        write(OUT/domain/"prepend_profiles.jsonl", [{"id":x["id"],"prompt":x["prepend_profiles_prompt"],
          "rgt_ans":x["correct_answer"],"rgt_ans_qid":x["correct_answer_qid"],
          "wrg_ans":x["wrong_answer"],"wrg_ans_qid":x["wrong_answer_qid"]} for x in selected])
        write(OUT/domain/"probes.jsonl", [{"id":f"{x['id']}_probe_{i}","parent_id":x["id"],**p}
          for x in selected for i,p in enumerate(x["probes"])])
        profiles = {p["qid"]:p for x in selected for p in x["profiles"].values()}
        write(OUT/domain/"profiles.jsonl", sorted(profiles.values(),key=lambda x:x["qid"]))
        report[domain] = {"items":N,"knows_both":N_BOTH,"knows_both_rate":N_BOTH/N,
                          "unique_candidate_items":len(pool),"unique_known_available":len(known),
                          "duplicates_removed":duplicate_count}
    summary={"dataset":"multidomain_v6_fixed500","model_calibrated":"gpt-5.2-2025-12-11",
             "selection_uses_primary_outcomes":False,"target_per_domain":N,
             "target_knows_both_per_domain":N_BOTH,"by_domain":report}
    (OUT/"report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
