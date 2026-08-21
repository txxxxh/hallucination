#!/usr/bin/env python3
"""Frozen, person-disjoint free-generation confirmation of cue competition."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from spanattr.core import Item, Span, SpanAttributor, set_seed


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/255_scientist_factorial_all1084"
OUT = HERE / "runs/257_competition_direction_confirmation"
controls = importlib.import_module("233_confirm_multicue_repairs")


def confirm_group(group: str) -> bool:
    """Frozen 50% person/group split, independent of scores and labels."""
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 2 == 1


def cohort(source: Path, threshold: float):
    selected = []
    for fp in sorted(source.glob("question_*.json")):
        row = json.loads(fp.read_text())
        if row.get("skipped") or row.get("generation_correct") or not confirm_group(row["group"]):
            continue
        pairs = [p for p in row["pair_interactions"]
                 if p["u_i"] * p["u_j"] < 0
                 and min(abs(p["u_i"]), abs(p["u_j"])) > threshold]
        if not pairs:
            continue
        # Frozen rule: strongest minimum one-sided magnitude; deterministic tie break.
        pair = max(pairs, key=lambda p: (min(abs(p["u_i"]), abs(p["u_j"])),
                                         max(abs(p["u_i"]), abs(p["u_j"])),
                                         -p["i"], -p["j"]))
        pos = pair["i"] if pair["u_i"] > 0 else pair["j"]
        neg = pair["j"] if pair["u_i"] > 0 else pair["i"]
        selected.append((row, pair, pos, neg))
    return selected


def generate(att, prep, ids, right, wrong, n, seed, max_new):
    generations = att.generate_under(prep, ids, n=n, temperature=.8,
                                     max_new_tokens=max_new, seed=seed)
    return {"p_right": att.match_rate(generations, [right]),
            "p_wrong": att.match_rate(generations, [wrong]),
            "generations": generations}


def collect(a):
    a.out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    production = importlib.import_module("231_run_scientist_factorial_atlas_parse_valid")
    jobs = {x[0]: x for x in production.parse_valid_jobs()}
    builder = importlib.import_module("76_build_closedbook_fact_probes")
    raw = {str(x["key"]): x for x in json.load(
        (HERE.parent / "shuffled_prepend_profiles_question.json").open())}
    atlas = importlib.import_module("227_scientist_factorial_interaction_atlas")
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=a.batch)
    frozen = cohort(a.source, a.threshold)
    manifest = {"threshold": a.threshold, "split": "sha256(group) parity = 1",
                "n": len(frozen), "keys": [x[0]["key"] for x in frozen]}
    (a.out / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for ni, (pilot, pair, pos_id, neg_id) in enumerate(frozen, 1):
        key = pilot["key"]
        fp = a.out / f"{key}.json"
        if a.resume and fp.exists():
            continue
        _, group, _, prompt, wrong, right = jobs[key]
        item = Item.from_dict({"key": key, "prompt": prompt, "pred": wrong, "gold": right})
        prep = att.prepare(item)
        prep.spans = [Span(i, int(x["token_start"]), int(x["token_end"]), x["text"])
                      for i, x in enumerate(pilot["candidates"])]
        offsets = atlas.offset_mapping(att, prep)
        qlo, qhi = controls.question_token_bounds(att, prep, prompt, raw[key], builder, offsets)
        ref = [prep.spans[pos_id], prep.spans[neg_id]]
        random_spans = controls.random_matched_sets(
            prep, ref, qlo, qhi, 1, a.seed + ni * 1009)[0]
        random_ids = []
        for sp in random_spans:
            sp.idx = len(prep.spans)
            random_ids.append(sp.idx)
            prep.spans.append(sp)
        shared_seed = a.seed + ni * 100
        conditions = {
            "base": [], "wrong_support": [pos_id], "right_support": [neg_id],
            "joint": [pos_id, neg_id], "random_for_wrong": [random_ids[0]],
            "random_for_right": [random_ids[1]], "random_joint": random_ids,
        }
        result = {name: generate(att, prep, ids, right, wrong, a.samples,
                                 shared_seed, a.max_new_tokens)
                  for name, ids in conditions.items()}
        rec = {
            "key": key, "group": group, "right": right, "wrong": wrong,
            "threshold": a.threshold,
            "frozen_pair": pair, "wrong_support_id": pos_id, "right_support_id": neg_id,
            "wrong_support_text": pilot["candidates"][pos_id]["text"],
            "right_support_text": pilot["candidates"][neg_id]["text"],
            "random_spans": [{"start": x.start, "end": x.end,
                              "text": tok.decode(prep.prompt_ids[x.start:x.end].tolist())}
                             for x in random_spans],
            "generation": result,
        }
        tmp = fp.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(fp)
        print(f"[{ni}/{len(frozen)}] {key} right: {result['base']['p_right']:.2f} -> "
              f"pos {result['wrong_support']['p_right']:.2f}, neg {result['right_support']['p_right']:.2f}",
              flush=True)


def boot(values, rng, draws=10000):
    x = np.asarray(values, float)
    sims = rng.choice(x, (draws, len(x)), replace=True).mean(1)
    return {"n": len(x), "mean": float(x.mean()),
            "ci95": np.quantile(sims, [.025, .975]).tolist(),
            "fraction_positive": float(np.mean(x > 0))}


def summarize(a):
    rows = [json.loads(p.read_text()) for p in sorted(a.out.glob("question_*.json"))]
    rng = np.random.default_rng(a.seed)
    pr = lambda r, c: r["generation"][c]["p_right"]
    pw = lambda r, c: r["generation"][c]["p_wrong"]
    right_corrective = [pr(r, "wrong_support") - pr(r, "base") for r in rows]
    right_damaging = [pr(r, "right_support") - pr(r, "base") for r in rows]
    wrong_corrective = [pw(r, "base") - pw(r, "wrong_support") for r in rows]
    wrong_damaging = [pw(r, "base") - pw(r, "right_support") for r in rows]
    report = {
        "experiment": "frozen person-disjoint directional competition confirmation",
        "n_questions": len(rows), "samples_per_condition": a.samples,
        "selection": {"threshold": a.threshold,
                      "rule": "error; confirmation SHA256(group) split; opposite signs; strongest min |u|"},
        "mean_p_right": {c: float(np.mean([pr(r, c) for r in rows])) for c in
                         ("base", "wrong_support", "right_support", "joint",
                          "random_for_wrong", "random_for_right", "random_joint")},
        "mean_p_wrong": {c: float(np.mean([pw(r, c) for r in rows])) for c in
                         ("base", "wrong_support", "right_support", "joint",
                          "random_for_wrong", "random_for_right", "random_joint")},
        "right_answer": {
            "neutralize_wrong_support_gain": boot(right_corrective, rng),
            "neutralize_right_support_gain": boot(right_damaging, rng),
            "directional_contrast": boot(np.asarray(right_corrective)-np.asarray(right_damaging), rng),
            "wrong_support_minus_matched_random": boot([
                (pr(r, "wrong_support")-pr(r, "base"))-
                (pr(r, "random_for_wrong")-pr(r, "base")) for r in rows], rng),
        },
        "wrong_answer": {
            "neutralize_wrong_support_reduction": boot(wrong_corrective, rng),
            "neutralize_right_support_reduction": boot(wrong_damaging, rng),
            "directional_contrast": boot(np.asarray(wrong_corrective)-np.asarray(wrong_damaging), rng),
        },
        "protocol": "all seven conditions use the same per-question sampling seed; random spans are width/position matched and fixed before generation",
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["collect", "summarize", "all"])
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--threshold", type=float, default=.10)
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.stage in ("collect", "all"):
        collect(a)
    if a.stage in ("summarize", "all"):
        summarize(a)


if __name__ == "__main__":
    main()
