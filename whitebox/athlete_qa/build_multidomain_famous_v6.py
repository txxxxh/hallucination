#!/usr/bin/env python3
"""Build a new multidomain QA version from high-pageview entities.

Unlike v5's early-page-ID heuristic, this builder measures candidate fame with
English Wikipedia pageviews, retains a popular candidate pool, and then reruns
the original hard-pair mining procedure.  It never uses model probe outcomes
for selection.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import requests

import build_multidomain_trapqa as core

DOMAINS = ("athlete", "musician", "building")
API = "https://en.wikipedia.org/w/api.php"
UA = "MultidomainFamousQA/0.1 research dataset"


def load_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def pageviews(profiles, cache_path, session):
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [profile for profile in profiles if profile["qid"] not in cached]
    for start in range(0, len(missing), 50):
        batch = missing[start:start + 50]
        params = {"action": "query", "format": "json", "prop": "pageviews",
                  "pageids": "|".join(str(profile["pageid"]) for profile in batch)}
        for attempt in range(8):
            response = session.get(API, params=params, timeout=90)
            if response.status_code == 200:
                break
            time.sleep(min(60, 2 ** attempt))
        response.raise_for_status()
        returned = response.json().get("query", {}).get("pages", {})
        by_pageid = {str(profile["pageid"]): profile for profile in batch}
        for pageid, profile in by_pageid.items():
            daily = returned.get(pageid, {}).get("pageviews", {}) or {}
            values = [value for value in daily.values() if isinstance(value, int)]
            cached[profile["qid"]] = {"pageviews_60d": sum(values), "days_observed": len(values)}
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2) + "\n")
        print(f"pageviews {min(start + len(batch), len(missing))}/{len(missing)}", flush=True)
    return cached


def dump_dataset(out, profiles, items):
    out.mkdir(parents=True, exist_ok=True)
    core.dump(out / "profiles.jsonl", profiles)
    core.dump(out / "primary_questions.jsonl", items)
    core.dump(out / "prepend_names.jsonl", [{"id": item["id"], "prompt": item["prepend_names_prompt"],
        "rgt_ans": item["correct_answer"], "rgt_ans_qid": item["correct_answer_qid"],
        "wrg_ans": item["wrong_answer"], "wrg_ans_qid": item["wrong_answer_qid"]} for item in items])
    core.dump(out / "prepend_profiles.jsonl", [{"id": item["id"], "prompt": item["prepend_profiles_prompt"],
        "rgt_ans": item["correct_answer"], "rgt_ans_qid": item["correct_answer_qid"],
        "wrg_ans": item["wrong_answer"], "wrg_ans_qid": item["wrong_answer_qid"]} for item in items])
    core.dump(out / "probes.jsonl", [{"id": f"{item['id']}_probe_{index}", "parent_id": item["id"], **probe}
                                      for item in items for index, probe in enumerate(item["probes"])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "multidomain_v5")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "multidomain_v6_famous")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--pool-size", type=int, default=800,
                        help="Number of highest-pageview profiles retained per domain")
    parser.add_argument("--seed", type=int, default=106)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--fields", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    report = {}
    for domain_index, domain in enumerate(args.domains):
        all_profiles = load_jsonl(args.source / domain / "profiles.jsonl")
        views = pageviews(all_profiles, args.out / f"{domain}_pageviews.json", session)
        for profile in all_profiles:
            profile.update(views.get(profile["qid"], {"pageviews_60d": 0, "days_observed": 0}))
        pool_size = max(args.pool_size, 550) if domain == "building" else args.pool_size
        famous = sorted(all_profiles, key=lambda profile: (-profile["pageviews_60d"], profile["pageid"]))[:pool_size]
        items, fields, candidate_relations = core.mine(domain, famous, args.n, args.seed + domain_index, args.candidate_offset, args.fields)
        if len(items) < args.n:
            raise RuntimeError(f"{domain}: only generated {len(items)} of {args.n}; increase --pool-size")
        # Give the new version stable, non-overlapping IDs.
        for index, item in enumerate(items):
            item["id"] = f"{domain}_famous_qa_{index:04d}"
            item["selection"] = "top English Wikipedia 60-day pageviews before pair mining"
        dump_dataset(args.out / domain, famous, items)
        report[domain] = {"source_profiles": len(all_profiles), "famous_pool": len(famous),
                          "items": len(items), "candidate_relations": candidate_relations,
                          "field_counts": dict(fields),
                          "pageviews_60d_min": min(profile["pageviews_60d"] for profile in famous),
                          "pageviews_60d_median": sorted(profile["pageviews_60d"] for profile in famous)[len(famous)//2],
                          "pageviews_60d_max": max(profile["pageviews_60d"] for profile in famous)}
        print(domain, json.dumps(report[domain]), flush=True)
    (args.out / "report.json").write_text(json.dumps({
        "created_at": date.today().isoformat(), "source": str(args.source),
        "selection": "top English Wikipedia 60-day pageviews; no model-outcome filtering",
        "summary": report}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
