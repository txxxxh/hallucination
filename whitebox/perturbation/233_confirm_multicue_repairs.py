#!/usr/bin/env python3
"""Confirm the pilot's multi-cue-only repairs with matched controls.

For every likelihood-systematic error whose minimal neutralisation repair has
size >=2, compare the frozen target set against question-local random sets with
the same token widths.  Validate with (1) length-preserving neutralisation,
(2) physical deletion, and (3) sampled free generation.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np

from spanattr.core import Item, Span, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SOURCE = RUNS / "230_scientist_factorial_interaction_atlas"
OUT = RUNS / "233_confirm_multicue_repairs"


def targets(source):
    out = []
    for fp in sorted(source.glob("question_*.json")):
        r = json.loads(fp.read_text())
        if r.get("skipped") or not r.get("likelihood_error"):
            continue
        repairs = [x for x in r["minimal_repair_sets"] if len(x["ids"]) >= 2]
        if repairs and not any(len(x["ids"]) == 1 for x in r["minimal_repair_sets"]):
            # Freeze one set: minimum size, then largest observed repair margin.
            repairs.sort(key=lambda x: (len(x["ids"]),
                         next(z["margin"] for z in r["masks"] if z["mask"] == x["mask"])))
            out.append((r, repairs[0]))
    return out


def question_token_bounds(att, prep, prompt, raw_row, builder, offsets):
    _, question = builder.parse_item(raw_row)
    qchar = prompt.find(question)
    if qchar < 0:
        raise RuntimeError(f"question not found: {raw_row['key']}")
    qend = qchar + len(question)
    ids = [i for i, (a, b) in enumerate(offsets) if b > qchar and a < qend]
    return prep.ctx_start + ids[0], prep.ctx_start + ids[-1] + 1


def random_matched_sets(prep, ref_spans, qlo, qhi, draws, seed):
    rng = np.random.default_rng(seed)
    blocked = set(t for s in prep.spans for t in range(s.start, s.end))
    output, seen = [], set()
    for _ in range(draws * 200):
        chosen, used = [], set(blocked)
        ok = True
        for ref in ref_spans:
            valid = [s for s in range(qlo, qhi - ref.width + 1)
                     if not (set(range(s, s + ref.width)) & used)]
            if not valid:
                ok = False; break
            # Match the reference's relative question-position quintile when possible.
            rb = min(4, 5 * (ref.start - qlo) // max(1, qhi - qlo))
            local = [s for s in valid if min(4, 5 * (s - qlo) // max(1, qhi - qlo)) == rb]
            start = int(rng.choice(local or valid))
            sp = Span(-1, start, start + ref.width, "")
            chosen.append(sp); used.update(range(sp.start, sp.end))
        key = tuple((x.start, x.end) for x in chosen)
        if ok and key not in seen:
            seen.add(key); output.append(chosen)
        if len(output) == draws:
            break
    if len(output) < draws:
        raise RuntimeError(f"only found {len(output)}/{draws} matched sets")
    return output


def alpha_for(prep, spans, device):
    import torch
    a = torch.zeros(len(prep.prompt_ids), device=device)
    for sp in spans:
        a[sp.start:sp.end] = 1
    return a


def char_ranges(spans, offsets, ctx_start):
    ranges = []
    for sp in spans:
        local = list(range(sp.start - ctx_start, sp.end - ctx_start))
        ranges.append((offsets[local[0]][0], offsets[local[-1]][1]))
    return ranges


def delete_ranges(text, ranges):
    for a, b in sorted(ranges, reverse=True):
        text = text[:a] + text[b:]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def score_deleted(att, item, ranges):
    changed = delete_ranges(item.context, ranges)
    p = att.prepare(Item(item.item_id + "_delete", changed, item.question,
                         item.gold, item.pred, context_prefix=item.context_prefix))
    return att.S0(p)


def gen_stats(att, prep, span_ids, right, wrong, n, seed, max_new):
    gens = att.generate_under(prep, span_ids, n=n, temperature=.8,
                              max_new_tokens=max_new, seed=seed)
    return {"generations": gens,
            "p_right": att.match_rate(gens, [right]),
            "p_wrong": att.match_rate(gens, [wrong])}


def collect(a):
    import torch
    a.out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    production = importlib.import_module(
        "231_run_scientist_factorial_atlas_parse_valid")
    jobs = {x[0]: x for x in production.parse_valid_jobs()}
    builder = importlib.import_module("76_build_closedbook_fact_probes")
    raw = {str(x["key"]): x for x in json.load(
        (HERE.parent / "shuffled_prepend_profiles_question.json").open())}
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=a.batch)
    cohort = targets(a.source)
    for ni, (pilot, repair) in enumerate(cohort, 1):
        key = pilot["key"]; fp = a.out / f"{key}.json"
        if a.resume and fp.exists():
            continue
        _, group, _, prompt, wrong, right = jobs[key]
        item = Item.from_dict({"key": key, "prompt": prompt,
                               "pred": wrong, "gold": right})
        prep = att.prepare(item)
        candidates = pilot["candidates"]
        prep.spans = [Span(i, int(x["token_start"]), int(x["token_end"]), x["text"])
                      for i, x in enumerate(candidates)]
        target_ids = repair["ids"]
        ref = [prep.spans[i] for i in target_ids]
        offsets = importlib.import_module(
            "227_scientist_factorial_interaction_atlas").offset_mapping(att, prep)
        qlo, qhi = question_token_bounds(att, prep, prompt, raw[key], builder, offsets)
        random_sets = random_matched_sets(prep, ref, qlo, qhi,
                                          a.random_draws, a.seed + ni * 1009)

        A = torch.stack([alpha_for(prep, ref, att.device)] +
                        [alpha_for(prep, x, att.device) for x in random_sets])
        margins = att.S_batched(prep, A).numpy()
        base = pilot["base_margin_wrong_minus_right"]
        target_ranges = char_ranges(ref, offsets, prep.ctx_start)
        physical_target = score_deleted(att, item, target_ranges)
        physical_random = [score_deleted(att, item, char_ranges(x, offsets, prep.ctx_start))
                           for x in random_sets[:a.physical_draws]]

        # Add random spans temporarily so the standard generation helper can gate them.
        strongest = int(np.argmin(margins[1:]))
        random_ref = random_sets[strongest]
        random_ids = []
        for sp in random_ref:
            sp.idx = len(prep.spans); random_ids.append(sp.idx); prep.spans.append(sp)
        generation = {
            "base": gen_stats(att, prep, [], right, wrong, a.gen_samples,
                              a.seed + ni * 100, a.max_new_tokens),
            "singles": [gen_stats(att, prep, [i], right, wrong, a.gen_samples,
                                  a.seed + ni * 100 + 10 + j, a.max_new_tokens)
                        for j, i in enumerate(target_ids)],
            "joint": gen_stats(att, prep, target_ids, right, wrong, a.gen_samples,
                               a.seed + ni * 100 + 50, a.max_new_tokens),
            "strongest_random": gen_stats(att, prep, random_ids, right, wrong,
                                          a.gen_samples, a.seed + ni * 100 + 70,
                                          a.max_new_tokens),
        }
        rec = {
            "key": key, "group": group, "right": right, "wrong": wrong,
            "target_ids": target_ids, "target_texts": repair["texts"],
            "base_margin": base,
            "neutral": {"target_margin": float(margins[0]),
                        "target_u": float(base - margins[0]),
                        "random_margins": margins[1:].tolist(),
                        "random_u": (base - margins[1:]).tolist(),
                        "random_flip_rate": float(np.mean(margins[1:] < 0))},
            "physical_delete": {"target_margin": float(physical_target),
                                "target_u": float(base - physical_target),
                                "random_margins": physical_random,
                                "random_u": [float(base-x) for x in physical_random],
                                "random_flip_rate": float(np.mean(np.asarray(physical_random) < 0))},
            "generation": generation,
            "matched_control": {"random_draws": a.random_draws,
                                "physical_draws": a.physical_draws,
                                "same_token_widths": [x.width for x in ref],
                                "question_local": True,
                                "strongest_random_index_for_generation": strongest},
        }
        tmp = fp.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(fp)
        print(f"[{ni}/{len(cohort)}] {key} k={len(ref)} neutral={margins[0]:+.3f} "
              f"delete={physical_target:+.3f} gen_right={generation['joint']['p_right']:.2f}",
              flush=True)


def paired_boot(x, rng, draws=10000):
    x = np.asarray(x, dtype=float)
    b = np.mean(rng.choice(x, (draws, len(x)), replace=True), axis=1)
    return {"n": len(x), "mean": float(x.mean()),
            "ci95": np.quantile(b, [.025, .975]).tolist(),
            "fraction_positive": float(np.mean(x > 0))}


def summarize(a):
    rows = [json.loads(x.read_text()) for x in sorted(a.out.glob("question_*.json"))]
    rng = np.random.default_rng(a.seed)
    neutral_specific = [r["neutral"]["target_u"] - np.mean(r["neutral"]["random_u"])
                        for r in rows]
    physical_specific = [r["physical_delete"]["target_u"]
                         - np.mean(r["physical_delete"]["random_u"]) for r in rows]
    gen_specific = [(r["generation"]["joint"]["p_right"]
                     - r["generation"]["base"]["p_right"])
                    - (r["generation"]["strongest_random"]["p_right"]
                       - r["generation"]["base"]["p_right"]) for r in rows]
    report = {
        "experiment": "frozen multi-cue-only repair confirmation",
        "n": len(rows),
        "neutralisation": {
            "target_flip_rate": float(np.mean([r["neutral"]["target_margin"] < 0 for r in rows])),
            "matched_random_flip_rate_mean": float(np.mean([
                r["neutral"]["random_flip_rate"] for r in rows])),
            "target_minus_random_repair_gain": paired_boot(neutral_specific, rng)},
        "physical_deletion": {
            "target_flip_rate": float(np.mean([
                r["physical_delete"]["target_margin"] < 0 for r in rows])),
            "matched_random_flip_rate_mean": float(np.mean([
                r["physical_delete"]["random_flip_rate"] for r in rows])),
            "target_minus_random_repair_gain": paired_boot(physical_specific, rng)},
        "free_generation": {
            "base_p_right": float(np.mean([r["generation"]["base"]["p_right"] for r in rows])),
            "joint_p_right": float(np.mean([r["generation"]["joint"]["p_right"] for r in rows])),
            "strongest_random_p_right": float(np.mean([
                r["generation"]["strongest_random"]["p_right"] for r in rows])),
            "joint_minus_strongest_random_correction": paired_boot(gen_specific, rng)},
        "limitations": [
            "cohort was selected by neutralisation and neutral target flip rate is therefore descriptive",
            "physical deletion and generation are held-out operators",
            "generation random control is deliberately the strongest-margin random set (conservative)",
        ]}
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["collect", "summarize", "all"])
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--random-draws", type=int, default=20)
    p.add_argument("--physical-draws", type=int, default=10)
    p.add_argument("--gen-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.stage in ("collect", "all"): collect(a)
    if a.stage in ("summarize", "all"): summarize(a)


if __name__ == "__main__": main()
