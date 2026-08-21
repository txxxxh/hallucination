#!/usr/bin/env python3
"""Paper §3.1: confirmatory signed reliance on question attribute phrases.

The target spans are fixed from profile/question lexical semantics before any
model response is measured.  Every target is paired with a same-word-width,
nearby question span that matches no profile attribute.  Both correct and
incorrect model answers are included and every margin is oriented as
wrong-minus-right.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from spanattr.core import Item, Span, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OUT = RUNS / "241_paper_keyword_reliance"
FIELDS = tuple(importlib.import_module("76_build_closedbook_fact_probes").FIELDS)
GENERIC = {
    "award", "prize", "medal", "order", "university", "society", "college",
    "institute", "field", "member", "science", "scientist", "work",
}
STOP = set("the a an this that who person individual their his her is was did and or of in to for with has have had been be also they were are as from at by".split())


def norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP


def split_prompt(prompt: str) -> tuple[str, str]:
    marker = ("Choose exactly one profile from the two, and output the name of "
              "the person as the answer to the following question:\n")
    head, sep, question = prompt.partition(marker)
    if not sep:
        raise ValueError("Scientist prompt marker not found")
    return head + sep, question


def facts_for_row(row: dict) -> list[dict]:
    builder = importlib.import_module("76_build_closedbook_fact_probes")
    profiles, question = builder.parse_item(row)
    qnorm = builder.norm(question)
    facts = []
    for field in FIELDS:
        sets = [{builder.norm(v): v for v in builder.values(p, field)}
                for p in profiles]
        for owner, profile in enumerate(profiles):
            other = 1 - owner
            for canonical, value in sets[owner].items():
                # Exactly the frozen discriminative-fact definition used by
                # Stage 76: shared attributes have no identifiable owner.
                if canonical in sets[other] or canonical not in qnorm:
                    continue
                facts.append({"owner": owner, "owner_name": profile["name"],
                              "field": field, "value": value})
    return facts


def word_spans(att: SpanAttributor, prep, max_width: int = 8):
    spans = att.build_word_spans(prep, widths=range(1, max_width + 1), stride=1)
    words = list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b", prep.item.context,
                             flags=re.UNICODE))
    char_ranges = []
    lookup = {(s.text, s.start, s.end): i for i, s in enumerate(spans)}
    # Reconstruct word-index metadata in exactly the core builder's order.
    for width in range(1, max_width + 1):
        for wi in range(0, len(words) - width + 1):
            text = prep.item.context[words[wi].start():words[wi + width - 1].end()]
            hits = [i for (t, _a, _b), i in lookup.items() if t == text]
            if hits:
                char_ranges.append((hits[0], wi, wi + width))
    # Token boundaries make the mapping unique even when surface text repeats.
    meta = {}
    for width in range(1, max_width + 1):
        seq = [s for s in spans if len(re.findall(r"\b\w+(?:['’\-]\w+)*\b", s.text)) == width]
        for wi, s in enumerate(seq):
            meta[s.idx] = (wi, wi + width)
    return spans, meta, len(words)


def match_targets(spans: list[Span], facts: list[dict]) -> list[dict]:
    by_span = {}
    for fact in facts:
        vt = norm_tokens(fact["value"])
        abstract = fact["field"] in {"occupation", "field", "position_held"}
        choices = []
        for s in spans:
            st = norm_tokens(s.text)
            overlap = st & vt
            informative = overlap - GENERIC
            # Confirmatory matching is deliberately conservative. Earlier
            # discovery code accepted any two informative overlapping words;
            # that can match many unrelated "Order of Merit" awards.
            coverage = len(overlap) / max(1, len(vt))
            valid = ((abstract and vt <= st) or
                     (coverage >= .8 and len(informative) >= 1))
            if not valid:
                continue
            score = (len(overlap) / max(1, len(st | vt)), len(informative),
                     len(overlap), -len(st))
            choices.append((score, s))
        if not choices:
            continue
        s = max(choices, key=lambda x: x[0])[1]
        key = (s.start, s.end, fact["owner"])
        rec = {**fact, "span_id": s.idx, "span_text": s.text,
               "span_start": s.start, "span_end": s.end}
        if key not in by_span or len(fact["value"]) > len(by_span[key]["value"]):
            by_span[key] = rec
    return list(by_span.values())


def matched_controls(spans, meta, targets, n_words):
    blocked = set()
    for t in targets:
        blocked.update(range(t["span_start"], t["span_end"]))
    controls = []
    for t in targets:
        tw = meta.get(t["span_id"])
        if tw is None:
            controls.append(None)
            continue
        width = tw[1] - tw[0]
        center = (tw[0] + tw[1]) / 2
        cand = []
        for s in spans:
            sw = meta.get(s.idx)
            if sw is None or sw[1] - sw[0] != width:
                continue
            if set(range(s.start, s.end)) & blocked:
                continue
            if len(norm_tokens(s.text)) == 0:
                continue
            cand.append((abs((sw[0] + sw[1]) / 2 - center), s.idx))
        controls.append(min(cand)[1] if cand else None)
    return controls


def collect(args):
    import torch
    set_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {str(x["key"]): x for x in json.load((ROOT / "shuffled_prepend_profiles_question.json").open())}
    known = [json.loads(x) for x in (RUNS / "88_known_gt05_n1084.jsonl").open()]
    recs = {str(x["key"]): x for x in map(json.loads, (ROOT / "tool_gate_correctness_names_llama31_8b" / "records.jsonl").open())}
    # This experiment never requests attention maps.  Use SDPA instead of the
    # repository-wide eager loader: Scientist profiles are long and eager
    # materializes a quadratic attention matrix for every perturbation row.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                        local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
        local_files_only=True).eval()
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=args.batch)
    for num, krow in enumerate(known[:args.limit or None], 1):
        key = str(krow["key"])
        fp = OUT / f"{key}.json"
        if fp.exists() and args.resume:
            continue
        row, generated = data[key], str(recs[key]["parsed_answer"])
        prefix, question = split_prompt(row["prompt"])
        item = Item(key, question, "Which profile is correct?", row["rgt_ans"],
                    pred=row["wrg_ans"], context_prefix=prefix)
        prep = att.prepare(item)
        spans, meta, n_words = word_spans(att, prep)
        targets = match_targets(spans, facts_for_row(row))
        control_ids = matched_controls(spans, meta, targets, n_words)
        used = sorted(set([t["span_id"] for t in targets] +
                          [x for x in control_ids if x is not None]))
        z = torch.zeros(len(prep.prompt_ids), device=att.device)
        A = torch.stack([z] + [att.alpha_from_spans(prep, [i]) for i in used])
        wrong, right = att.class_scores_batched(prep, A)
        margins = (wrong - right).numpy()
        effects = {sid: float(margins[0] - margins[j + 1]) for j, sid in enumerate(used)}
        results = []
        for target, cid in zip(targets, control_ids):
            if cid is None:
                continue
            owner_name = target["owner_name"]
            owner_side = "right" if owner_name == row["rgt_ans"] else "wrong"
            results.append({**target, "owner_side": owner_side,
                            "target_u_wrong": effects[target["span_id"]],
                            "control_span_id": cid,
                            "control_text": spans[cid].text,
                            "control_u_wrong": effects[cid]})
        out = {"key": key, "group": str(krow["group"]),
               "correct": bool(krow["correct"]), "generated": generated,
               "right": row["rgt_ans"], "wrong": row["wrg_ans"],
               "base_wrong_margin": float(margins[0]), "targets": results}
        fp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        print(f"[{num}/{len(known)}] {key} correct={out['correct']} targets={len(results)}", flush=True)
        if num % 50 == 0:
            torch.cuda.empty_cache()


def cluster_ci(rows, value, seed=42, boot=5000):
    if not rows:
        return [float("nan"), float("nan")]
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(float(r[value]))
    vals = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(boot)]
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def summarize(rows):
    # The item, not the keyword, is the estimand.  Questions list very
    # different numbers of attributes; average within item before inference.
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["key"]].append(r)
    items = []
    for key, rr in by_item.items():
        target = float(np.mean([x["target_u_wrong"] for x in rr]))
        control = float(np.mean([x["control_u_wrong"] for x in rr]))
        base = float(rr[0]["base_wrong_margin"])
        items.append({"key": key, "group": rr[0]["group"],
                      "target": target, "control": control,
                      "target_minus_control": target - control,
                      "base": base, "post_target": base - target,
                      "post_control": base - control})
    target = np.array([x["target"] for x in items])
    control = np.array([x["control"] for x in items])
    diff = target - control
    base = np.array([x["base"] for x in items])
    post_t = np.array([x["post_target"] for x in items])
    post_c = np.array([x["post_control"] for x in items])
    base_pos, base_neg = base > 0, base < 0
    return {"n_keywords": len(rows), "n_items": len(items),
            "n_groups": len(set(r["group"] for r in items)),
            "mean_target_u_wrong": float(target.mean()),
            "mean_control_u_wrong": float(control.mean()),
            "mean_target_minus_control": float(diff.mean()),
            "target_minus_control_group_ci95": cluster_ci(items, "target_minus_control"),
            "positive_target_minus_control": float(np.mean(diff > 0)),
            "wrong_to_right_target": {"denom": int(base_pos.sum()),
                "n": int(np.sum(base_pos & (post_t < 0))),
                "rate": float(np.sum(base_pos & (post_t < 0)) / base_pos.sum()) if base_pos.any() else None},
            "wrong_to_right_control": {"denom": int(base_pos.sum()),
                "n": int(np.sum(base_pos & (post_c < 0))),
                "rate": float(np.sum(base_pos & (post_c < 0)) / base_pos.sum()) if base_pos.any() else None},
            "right_to_wrong_target": {"denom": int(base_neg.sum()),
                "n": int(np.sum(base_neg & (post_t > 0))),
                "rate": float(np.sum(base_neg & (post_t > 0)) / base_neg.sum()) if base_neg.any() else None},
            "right_to_wrong_control": {"denom": int(base_neg.sum()),
                "n": int(np.sum(base_neg & (post_c > 0))),
                "rate": float(np.sum(base_neg & (post_c > 0)) / base_neg.sum()) if base_neg.any() else None}}


def analyze():
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob("question_*.json"))]
    rows = []
    for x in records:
        for t in x["targets"]:
            rows.append({**t, "key": x["key"], "group": x["group"],
                         "correct": x["correct"], "base_wrong_margin": x["base_wrong_margin"]})
    cells = {}
    for correctness in (False, True):
        for owner in ("wrong", "right"):
            z = [r for r in rows if r["correct"] == correctness and r["owner_side"] == owner]
            cells[f"{'correct' if correctness else 'error'}__{owner}_owner"] = summarize(z) if z else None
    by_field = {}
    for field in FIELDS:
        z = [r for r in rows if r["field"] == field]
        if z:
            by_field[field] = summarize(z)
    report = {"protocol": "Question-only, semantics-fixed attribute spans; wrong-minus-right margin; same-width nearby non-attribute control; group bootstrap",
              "n_records": len(records), "n_keyword_rows": len(rows),
              "cells": cells, "by_field": by_field}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Paper §3.1 — signed question-keyword reliance", "",
             "All effects use the wrong-minus-right margin. Positive values mean that neutralizing the phrase reduces support for the wrong candidate.", "",
             "| Model outcome | Profile owner of phrase | keywords | items | target effect | matched control | target−control (95% group CI) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("error__wrong_owner", "error__right_owner", "correct__wrong_owner", "correct__right_owner"):
        v = cells.get(name)
        if not v:
            continue
        outcome, owner = name.split("__")
        lo, hi = v["target_minus_control_group_ci95"]
        lines.append(f"| {outcome} | {owner.replace('_owner','')} | {v['n_keywords']} | {v['n_items']} | {v['mean_target_u_wrong']:.3f} | {v['mean_control_u_wrong']:.3f} | {v['mean_target_minus_control']:.3f} [{lo:.3f}, {hi:.3f}] |")
    (OUT / "paper_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("collect", "analyze", "all"))
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
