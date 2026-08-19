#!/usr/bin/env python3
"""Full-factorial interaction atlas for grounded Scientist question cues.

The perturbable model input is identical to the existing Scientist pipeline,
but candidate spans are restricted to the final natural-language question.
Candidates are selected without looking at their single-span effects: grounded
profile attributes and explicit negation cues are ranked by a deterministic
semantic rule.  For m candidates all 2**m neutralisation masks are evaluated.

Outputs per question include the complete set function, local finite-difference
and context-averaged Banzhaf pair interactions, exact Harsanyi dividends, and
minimal repair / candidate-sufficient sets.  One JSON file is written per item
so a long run can be resumed safely.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from spanattr.core import Item, Span, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEFAULT_OUT = RUNS / "227_scientist_factorial_interaction_atlas"
QUESTION_MARKER = (
    "\nChoose exactly one profile from the two, and output the name of the person "
    "as the answer to the following question:\n"
)
LOGIC_RE = re.compile(r"\b(?:not|never|nor|without|neither)\b", re.IGNORECASE)
WORD_RE = re.compile(r"\b\w+(?:['\u2019\-]\w+)*\b", re.UNICODE)
FIELD_PRIORITY = {
    "position_held": 0, "education": 1, "notable_work": 2,
    "place_of_birth": 3, "place_of_death": 3, "award_received": 4,
    "field": 5, "occupation": 6,
}


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def subsets(mask: int):
    sub = mask
    while True:
        yield sub
        if sub == 0:
            break
        sub = (sub - 1) & mask


def mobius_dividends(u: np.ndarray, m: int) -> np.ndarray:
    """Möbius/Harsanyi decomposition of a set function indexed by bit mask."""
    h = np.asarray(u, dtype=float).copy()
    for bit in range(m):
        step = 1 << bit
        for mask in range(1 << m):
            if mask & step:
                h[mask] -= h[mask ^ step]
    return h


def banzhaf_pairs(u: np.ndarray, m: int) -> np.ndarray:
    """Uniform-context average second discrete derivative for every pair."""
    out = np.zeros((m, m), dtype=float)
    for i, j in itertools.combinations(range(m), 2):
        vals = []
        ij = (1 << i) | (1 << j)
        for base in range(1 << m):
            if base & ij:
                continue
            vals.append(u[base | ij] - u[base | (1 << i)]
                        - u[base | (1 << j)] + u[base])
        out[i, j] = out[j, i] = float(np.mean(vals))
    return out


def inclusion_minimal(masks: list[int]) -> list[int]:
    pool = set(masks)
    return sorted(
        (x for x in masks if not any(y != x and (y & x) == y for y in pool)),
        key=lambda x: (x.bit_count(), x),
    )


def mask_ids(mask: int, m: int) -> list[int]:
    return [i for i in range(m) if mask & (1 << i)]


def bootstrap_mean(values, rng, draws=4000):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"n": 0, "mean": None, "ci95": [None, None]}
    means = np.mean(rng.choice(x, (draws, len(x)), replace=True), axis=1)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "ci95": [float(np.quantile(means, .025)),
                     float(np.quantile(means, .975))]}


def offset_mapping(att: SpanAttributor, prep):
    enc = att.tok(prep.item.context, add_special_tokens=False,
                  return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if (offsets and isinstance(offsets[0], list) and offsets[0]
            and isinstance(offsets[0][0], list)):
        offsets = offsets[0]
    expected = prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist()
    if list(ids) != expected:
        raise RuntimeError("tokenizer offset mapping differs from prepared context")
    return [(int(a), int(b)) for a, b in offsets]


def char_to_span(prep, offsets, start: int, end: int, text: str) -> Span | None:
    covered = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not covered:
        return None
    return Span(-1, prep.ctx_start + covered[0],
                prep.ctx_start + covered[-1] + 1, text)


def question_facts(row: dict, profiles: list[dict], builder) -> list[dict]:
    """Ground every exact profile value occurrence in the final question."""
    prompt = row["prompt"]
    qstart = prompt.rfind(QUESTION_MARKER)
    qstart = qstart + len(QUESTION_MARKER) if qstart >= 0 else 0
    question = prompt[qstart:]
    words = list(WORD_RE.finditer(question))
    qtokens = [norm(x.group()) for x in words]
    right = row["rgt_ans"]

    facts = {}
    for pi, profile in enumerate(profiles):
        owner = "right" if profile["name"] == right else "wrong"
        for field in builder.FIELDS:
            for value in builder.values(profile, field):
                ftoks = norm(value).split()
                if not ftoks or len(ftoks) > len(qtokens):
                    continue
                for wi in range(len(qtokens) - len(ftoks) + 1):
                    if qtokens[wi:wi + len(ftoks)] != ftoks:
                        continue
                    a = qstart + words[wi].start()
                    b = qstart + words[wi + len(ftoks) - 1].end()
                    key = (a, b, field, norm(value))
                    rec = facts.setdefault(key, {
                        "char_start": a, "char_end": b, "text": prompt[a:b],
                        "kind": "attribute", "field": field, "value": value,
                        "owners": [], "negated": False,
                    })
                    if owner not in rec["owners"]:
                        rec["owners"].append(owner)

    # Merge identical textual occurrences shared by both profiles.
    merged = {}
    for rec in facts.values():
        key = (rec["char_start"], rec["char_end"], rec["field"], norm(rec["value"]))
        if key not in merged:
            merged[key] = rec
        else:
            merged[key]["owners"] = sorted(set(merged[key]["owners"] + rec["owners"]))
    attrs = list(merged.values())

    # A local scope flag is descriptive only; negation itself is a separate cue.
    for rec in attrs:
        left = prompt[max(qstart, rec["char_start"] - 80):rec["char_start"]]
        rec["negated"] = bool(LOGIC_RE.search(left))

    logic = []
    for match in LOGIC_RE.finditer(question):
        logic.append({
            "char_start": qstart + match.start(),
            "char_end": qstart + match.end(),
            "text": match.group(), "kind": "logic", "field": "logic",
            "value": match.group().casefold(), "owners": [], "negated": True,
        })
    return attrs + logic


def semantic_priority(rec: dict):
    if rec["kind"] == "logic":
        return (0, 0, rec["char_start"])
    owners = set(rec["owners"])
    owner_rank = 0 if len(owners) == 1 else 1
    return (1, 0 if rec["negated"] else 1, owner_rank,
            FIELD_PRIORITY.get(rec["field"], 9), -len(norm(rec["value"]).split()),
            rec["char_start"])


def select_candidates(records: list[dict], cap: int) -> tuple[list[dict], list[dict]]:
    """Deterministic, outcome-independent selection with token-disjoint spans."""
    chosen = []
    seen_concepts = set()
    for rec in sorted(records, key=semantic_priority):
        concept = (rec["kind"], rec["field"], norm(rec["value"]))
        if concept in seen_concepts:
            continue
        if any(not (rec["char_end"] <= x["char_start"] or
                    rec["char_start"] >= x["char_end"]) for x in chosen):
            continue
        chosen.append(rec)
        seen_concepts.add(concept)
        if len(chosen) == cap:
            break
    excluded = [x for x in records if x not in chosen]
    return sorted(chosen, key=lambda x: x["char_start"]), excluded


def analyse_item(key, group, row, prep, candidates, u, pred_scores, gold_scores):
    m = len(candidates)
    margins = pred_scores - gold_scores
    h = mobius_dividends(u, m)
    bp = banzhaf_pairs(u, m)
    local = np.zeros((m, m), dtype=float)
    for i, j in itertools.combinations(range(m), 2):
        local[i, j] = local[j, i] = (
            u[(1 << i) | (1 << j)] - u[1 << i] - u[1 << j])

    repairs = inclusion_minimal([x for x in range(1 << m) if margins[x] < 0])
    # Keeping S is equivalent to deleting its complement within the candidate pool.
    full = (1 << m) - 1
    sufficient = []
    for keep in range(1 << m):
        if margins[full ^ keep] > 0:
            sufficient.append(keep)
    sufficient = inclusion_minimal(sufficient)

    def describe(mask):
        return {"mask": mask, "ids": mask_ids(mask, m),
                "texts": [candidates[i]["text"] for i in mask_ids(mask, m)]}

    interactions = []
    for i, j in itertools.combinations(range(m), 2):
        ui, uj, lij, bij = u[1 << i], u[1 << j], local[i, j], bp[i, j]
        if ui > 0 and uj > 0:
            relation = "synergy" if lij > 0 else "redundancy"
        elif ui * uj < 0:
            relation = "competition"
        elif abs(ui) < .05 and abs(uj) < .05 and lij > .05:
            relation = "pure_combination"
        else:
            relation = "mixed"
        interactions.append({
            "i": i, "j": j, "texts": [candidates[i]["text"], candidates[j]["text"]],
            "u_i": float(ui), "u_j": float(uj), "local_fd": float(lij),
            "banzhaf": float(bij), "relation_raw": relation,
        })

    nonzero_h = []
    for mask in range(1, 1 << m):
        if mask.bit_count() >= 2:
            nonzero_h.append({**describe(mask), "order": mask.bit_count(),
                              "harsanyi": float(h[mask])})
    nonzero_h.sort(key=lambda x: -abs(x["harsanyi"]))

    return {
        "key": key, "group": group, "generation_correct": False,
        "right": row["rgt_ans"], "wrong": row["wrg_ans"],
        "base_margin_wrong_minus_right": float(margins[0]),
        "likelihood_error": bool(margins[0] > 0), "m": m,
        "candidates": candidates,
        "masks": [{**describe(mask), "u": float(u[mask]),
                   "margin": float(margins[mask]),
                   "wrong_score": float(pred_scores[mask]),
                   "right_score": float(gold_scores[mask])}
                  for mask in range(1 << m)],
        "single_u": [float(u[1 << i]) for i in range(m)],
        "pair_interactions": interactions,
        "harsanyi_interactions": nonzero_h,
        "minimal_repair_sets": [describe(x) for x in repairs],
        "minimal_candidate_sufficient_sets": [describe(x) for x in sufficient],
        "all_candidates_deleted_margin": float(margins[-1]),
        "candidate_set_collectively_necessary_for_error": bool(margins[-1] < 0),
    }


def collect(args):
    import torch
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    builder = importlib.import_module("76_build_closedbook_fact_probes")
    raw = {str(x["key"]): x for x in json.load(
        (HERE.parent / "shuffled_prepend_profiles_question.json").open())}
    jobs = [x for x in importlib.import_module(
        "152_scientist_attention_pruned_current127").jobs() if not x[2]]
    if args.limit:
        jobs = jobs[:args.limit]
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline=args.baseline,
                         length_norm=True, max_rows=args.batch)

    skipped = Counter()
    for n, (key, group, _, prompt, pred, other) in enumerate(jobs, 1):
        fp = args.out / f"{key}.json"
        if args.resume and fp.exists():
            continue
        row = raw[key]
        right, wrong = other, pred
        if right != row["rgt_ans"] or wrong != row["wrg_ans"]:
            raise RuntimeError(f"answer identity mismatch for {key}")
        item = Item.from_dict({"key": key, "prompt": prompt,
                               "pred": wrong, "gold": right})
        prep = att.prepare(item)
        profiles, _ = builder.parse_item(row)
        records = question_facts(row, profiles, builder)
        selected, excluded = select_candidates(records, args.max_candidates)
        offsets = offset_mapping(att, prep)
        spans, candidates = [], []
        for i, rec in enumerate(selected):
            sp = char_to_span(prep, offsets, rec["char_start"], rec["char_end"], rec["text"])
            if sp is None:
                continue
            sp.idx = len(spans)
            spans.append(sp)
            candidates.append({**rec, "id": sp.idx, "token_start": sp.start,
                               "token_end": sp.end})
        prep.spans = spans
        m = len(spans)
        if m < args.min_candidates:
            skipped[f"fewer_than_{args.min_candidates}_candidates"] += 1
            fp.write_text(json.dumps({"key": key, "skipped": True,
                                      "reason": "too_few_candidates", "m": m,
                                      "candidates": candidates}, indent=2) + "\n")
            continue
        masks = list(range(1 << m))
        sets = [mask_ids(mask, m) for mask in masks]
        alphas = torch.stack([att.alpha_from_spans(prep, x) for x in sets])
        ps, gs = att.class_scores_batched(prep, alphas)
        ps, gs = ps.numpy(), gs.numpy()
        margin = ps - gs
        u = margin[0] - margin
        result = analyse_item(key, group, row, prep, candidates, u, ps, gs)
        result["selection"] = {
            "rule": "semantic deterministic; no model-effect ranking",
            "n_grounded_or_logic": len(records), "n_excluded": len(excluded),
            "max_candidates": args.max_candidates,
        }
        tmp = fp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(fp)
        print(f"[{n}/{len(jobs)}] {key} m={m} masks={1<<m} "
              f"margin={margin[0]:+.3f} all_del={margin[-1]:+.3f}", flush=True)
    print(json.dumps({"finished": len(jobs), "skipped_this_run": skipped}, indent=2))


def summarize(args):
    rows = []
    for fp in sorted(args.out.glob("question_*.json")):
        rec = json.loads(fp.read_text())
        if not rec.get("skipped"):
            rows.append(rec)
    if not rows:
        raise SystemExit("no completed item files to summarize")
    rng = np.random.default_rng(args.seed)
    pair_rows = [p for r in rows for p in r["pair_interactions"]]
    by_relation = Counter(p["relation_raw"] for p in pair_rows)
    by_type = defaultdict(list)
    for r in rows:
        for p in r["pair_interactions"]:
            ci, cj = r["candidates"][p["i"]], r["candidates"][p["j"]]
            kind = " × ".join(sorted([ci["field"], cj["field"]]))
            by_type[kind].append(p["banzhaf"])
    top_types = sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:30]
    report = {
        "experiment": "Scientist grounded-cue full-factorial interaction atlas",
        "n_questions": len(rows),
        "n_likelihood_errors": sum(r["likelihood_error"] for r in rows),
        "candidate_m_distribution": dict(Counter(str(r["m"]) for r in rows)),
        "n_pairs": len(pair_rows), "raw_relation_counts": dict(by_relation),
        "candidate_set_collectively_necessary_rate": float(np.mean([
            r["candidate_set_collectively_necessary_for_error"] for r in rows])),
        "questions_with_any_repair_set": sum(bool(r["minimal_repair_sets"]) for r in rows),
        "questions_with_multi_cue_minimal_repair": sum(any(
            len(x["ids"]) >= 2 for x in r["minimal_repair_sets"]) for r in rows),
        "local_pair_effect": bootstrap_mean([p["local_fd"] for p in pair_rows], rng),
        "banzhaf_pair_effect": bootstrap_mean([p["banzhaf"] for p in pair_rows], rng),
        "by_field_pair_top30_frequency": {
            k: bootstrap_mean(v, rng) for k, v in top_types},
        "notes": [
            "relation_raw uses effect signs only and is descriptive; confirmatory labels require a null threshold",
            "minimal candidate-sufficient sets are conditional on all noncandidate prompt evidence remaining present",
            "all headline generation validation is intentionally deferred to stage 2",
        ],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def selftest():
    # u = 2*x0 + 3*x1 + 5*x0*x1 - 4*x0*x1*x2
    m = 3
    u = np.zeros(1 << m)
    for mask in range(1 << m):
        x = [(mask >> i) & 1 for i in range(m)]
        u[mask] = 2*x[0] + 3*x[1] + 5*x[0]*x[1] - 4*x[0]*x[1]*x[2]
    h = mobius_dividends(u, m)
    assert np.allclose([h[1], h[2], h[3], h[7]], [2, 3, 5, -4])
    b = banzhaf_pairs(u, m)
    assert np.isclose(b[0, 1], 3)  # average of 5 (x2=0) and 1 (x2=1)
    assert inclusion_minimal([3, 7, 5]) == [3, 5]
    print("selftest: ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["collect", "summarize", "all", "selftest"])
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--baseline", default="mean", choices=["mean", "zero", "unk"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-candidates", type=int, default=6)
    p.add_argument("--min-candidates", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.stage == "selftest":
        selftest(); return
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("summarize", "all"):
        summarize(args)


if __name__ == "__main__":
    main()
