#!/usr/bin/env python3
"""Paired-seed generation confirmation with a preselected random control."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from spanattr.core import Item, Span, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/230_scientist_factorial_interaction_atlas"
OUT = HERE / "runs/234_paired_generation_multicue_controls"
base = importlib.import_module("233_confirm_multicue_repairs")


def stats(att, prep, ids, right, wrong, n, seed, max_new):
    # Identical seed schedule in every condition provides paired stochastic draws.
    gens = att.generate_under(prep, ids, n=n, temperature=.8,
                              max_new_tokens=max_new, seed=seed)
    return {"p_right": att.match_rate(gens, [right]),
            "p_wrong": att.match_rate(gens, [wrong]), "generations": gens}


def collect(a):
    a.out.mkdir(parents=True, exist_ok=True); set_seed(a.seed)
    prod = importlib.import_module("231_run_scientist_factorial_atlas_parse_valid")
    jobs = {x[0]: x for x in prod.parse_valid_jobs()}
    builder = importlib.import_module("76_build_closedbook_fact_probes")
    raw = {str(x["key"]): x for x in json.load(
        (HERE.parent / "shuffled_prepend_profiles_question.json").open())}
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=a.batch)
    cohort = base.targets(a.source)
    atlas = importlib.import_module("227_scientist_factorial_interaction_atlas")
    for ni, (pilot, repair) in enumerate(cohort, 1):
        key = pilot["key"]; fp = a.out / f"{key}.json"
        if a.resume and fp.exists(): continue
        _, _, _, prompt, wrong, right = jobs[key]
        item = Item.from_dict({"key": key, "prompt": prompt,
                               "pred": wrong, "gold": right})
        prep = att.prepare(item)
        prep.spans = [Span(i, int(x["token_start"]), int(x["token_end"]), x["text"])
                      for i, x in enumerate(pilot["candidates"])]
        ids = repair["ids"]; ref = [prep.spans[i] for i in ids]
        offsets = atlas.offset_mapping(att, prep)
        qlo, qhi = base.question_token_bounds(att, prep, prompt, raw[key], builder, offsets)
        # One control is fixed solely by key/seed before observing its effect.
        random_set = base.random_matched_sets(prep, ref, qlo, qhi, 1,
                                              a.seed + ni * 1009)[0]
        random_ids = []
        for sp in random_set:
            sp.idx = len(prep.spans); random_ids.append(sp.idx); prep.spans.append(sp)
        shared_seed = a.seed + ni * 100
        rec = {"key": key, "right": right, "wrong": wrong,
               "target_ids": ids, "target_texts": repair["texts"],
               "random_spans": [{"start": x.start, "end": x.end,
                                  "text": tok.decode(prep.prompt_ids[x.start:x.end].tolist())}
                                 for x in random_set],
               "base": stats(att, prep, [], right, wrong, a.samples,
                             shared_seed, a.max_new_tokens),
               "singles": [stats(att, prep, [i], right, wrong, a.samples,
                                 shared_seed, a.max_new_tokens) for i in ids],
               "joint": stats(att, prep, ids, right, wrong, a.samples,
                              shared_seed, a.max_new_tokens),
               "preselected_random": stats(att, prep, random_ids, right, wrong,
                                           a.samples, shared_seed, a.max_new_tokens)}
        tmp=fp.with_suffix(".tmp");tmp.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n");tmp.replace(fp)
        print(f"[{ni}/{len(cohort)}] {key} base={rec['base']['p_right']:.2f} "
              f"joint={rec['joint']['p_right']:.2f} random={rec['preselected_random']['p_right']:.2f}",flush=True)


def boot(x, rng, draws=10000):
    x=np.asarray(x,float); b=np.mean(rng.choice(x,(draws,len(x)),replace=True),1)
    return {"n":len(x),"mean":float(x.mean()),"ci95":np.quantile(b,[.025,.975]).tolist(),
            "fraction_positive":float(np.mean(x>0))}


def summarize(a):
    rows=[json.loads(x.read_text()) for x in sorted(a.out.glob("question_*.json"))]
    rng=np.random.default_rng(a.seed)
    joint_gain=[r["joint"]["p_right"]-r["base"]["p_right"] for r in rows]
    random_gain=[r["preselected_random"]["p_right"]-r["base"]["p_right"] for r in rows]
    specific=np.asarray(joint_gain)-np.asarray(random_gain)
    # Joint beyond the best single intervention is the behavioral coalition gain.
    beyond_single=[r["joint"]["p_right"]-max(x["p_right"] for x in r["singles"])
                   for r in rows]
    report={"experiment":"paired-seed free-generation multi-cue confirmation","n":len(rows),
      "samples_per_condition":a.samples,
      "mean_p_right":{"base":float(np.mean([r['base']['p_right'] for r in rows])),
                       "joint":float(np.mean([r['joint']['p_right'] for r in rows])),
                       "preselected_random":float(np.mean([r['preselected_random']['p_right'] for r in rows])),
                       "best_single":float(np.mean([max(x['p_right'] for x in r['singles']) for r in rows]))},
      "joint_correction_gain":boot(joint_gain,rng),
      "random_correction_gain":boot(random_gain,rng),
      "joint_minus_preselected_random":boot(specific,rng),
      "joint_minus_best_single":boot(beyond_single,rng),
      "protocol":"same sampling seeds in base/single/joint/random; random set selected before scoring"}
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","summarize","all"])
    p.add_argument("--source",type=Path,default=SOURCE);p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch",type=int,default=24);p.add_argument("--samples",type=int,default=20)
    p.add_argument("--max-new-tokens",type=int,default=20);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume",action="store_true");a=p.parse_args()
    if a.stage in ("collect","all"):collect(a)
    if a.stage in ("summarize","all"):summarize(a)


if __name__=="__main__":main()
