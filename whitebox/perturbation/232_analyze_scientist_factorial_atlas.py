#!/usr/bin/env python3
"""Cluster-aware confirmatory summary of the Scientist factorial atlas."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "runs/230_scientist_factorial_interaction_atlas"


def boot(values, rng, draws=10000):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"n_questions": 0, "mean": None, "ci95": [None, None]}
    means = np.mean(rng.choice(x, (draws, len(x)), replace=True), axis=1)
    return {"n_questions": len(x), "mean": float(x.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}


def classify(pair, threshold):
    ui, uj = pair["u_i"], pair["u_j"]
    local, global_ = pair["local_fd"], pair["banzhaf"]
    pos = local > threshold and global_ > threshold
    neg = local < -threshold and global_ < -threshold
    if ui > 0 and uj > 0 and pos:
        return "robust_synergy"
    if ui > 0 and uj > 0 and neg:
        return "robust_redundancy"
    if ui * uj < 0 and min(abs(ui), abs(uj)) > threshold:
        return "competition"
    if max(abs(ui), abs(uj)) <= threshold and pos:
        return "pure_combination"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--threshold", type=float, default=.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    rows = []
    skipped = []
    for fp in sorted(a.input.glob("question_*.json")):
        r = json.loads(fp.read_text())
        (skipped if r.get("skipped") else rows).append(r)
    systematic = [r for r in rows if r["likelihood_error"]]
    inconsistent = [r for r in rows if not r["likelihood_error"]]
    rng = np.random.default_rng(a.seed)

    q_local = [np.mean([x["local_fd"] for x in r["pair_interactions"]]) for r in rows]
    q_banzhaf = [np.mean([x["banzhaf"] for x in r["pair_interactions"]]) for r in rows]
    relation_counts = Counter()
    relation_questions = defaultdict(set)
    examples = defaultdict(list)
    for r in rows:
        for pair in r["pair_interactions"]:
            label = classify(pair, a.threshold)
            relation_counts[label] += 1
            relation_questions[label].add(r["key"])
            if label != "other":
                examples[label].append({
                    "key": r["key"], "texts": pair["texts"],
                    "u_i": pair["u_i"], "u_j": pair["u_j"],
                    "local_fd": pair["local_fd"], "banzhaf": pair["banzhaf"],
                })
    for label in examples:
        examples[label] = sorted(
            examples[label], key=lambda x: -min(abs(x["local_fd"]),
                                                 abs(x["banzhaf"])))[:10]

    repair = Counter()
    repair_examples = []
    for r in systematic:
        sizes = [len(x["ids"]) for x in r["minimal_repair_sets"]]
        if not sizes:
            repair["no_candidate_repair"] += 1
        elif min(sizes) == 1:
            repair["single_cue_repair_available"] += 1
        else:
            repair["multi_cue_only_repair"] += 1
            repair_examples.append({
                "key": r["key"], "base_margin": r["base_margin_wrong_minus_right"],
                "minimal_repair_sets": r["minimal_repair_sets"],
            })

    # Higher-order Harsanyi mass is summarized per question so questions with
    # more candidates do not receive combinatorially larger weight.
    high_ratio = []
    high_dominates = []
    for r in rows:
        hs = r["harsanyi_interactions"]
        pair = [abs(x["harsanyi"]) for x in hs if x["order"] == 2]
        higher = [abs(x["harsanyi"]) for x in hs if x["order"] >= 3]
        denom = sum(pair) + sum(higher)
        high_ratio.append(sum(higher) / denom if denom else 0.0)
        high_dominates.append(bool(higher and max(higher) > max(pair, default=0)))

    report = {
        "protocol": {
            "unit": "question-cluster", "practical_interaction_threshold": a.threshold,
            "robust_interaction": "local finite difference and Banzhaf agree in sign and both exceed threshold",
        },
        "coverage": {"generation_errors_parse_valid": len(rows) + len(skipped),
                     "analysed_m_ge_2": len(rows), "skipped_m_lt_2": len(skipped),
                     "likelihood_systematic": len(systematic),
                     "generation_likelihood_inconsistent": len(inconsistent)},
        "cluster_bootstrap": {"mean_local_pair_effect": boot(q_local, rng),
                              "mean_banzhaf_pair_effect": boot(q_banzhaf, rng)},
        "robust_pair_taxonomy": {
            "pair_counts": dict(relation_counts),
            "question_counts": {k: len(v) for k, v in relation_questions.items()},
            "top_examples": examples,
        },
        "systematic_error_repair": {
            **dict(repair),
            "candidate_set_collectively_repairs": sum(
                r["all_candidates_deleted_margin"] < 0 for r in systematic),
            "candidate_set_collectively_repairs_rate": float(np.mean([
                r["all_candidates_deleted_margin"] < 0 for r in systematic])),
            "multi_cue_only_examples": repair_examples[:20],
        },
        "higher_order": {
            "question_mean_fraction_abs_harsanyi_mass_order_ge_3": float(np.mean(high_ratio)),
            "questions_where_max_higher_order_exceeds_max_pair": int(sum(high_dominates)),
            "rate_where_max_higher_order_exceeds_max_pair": float(np.mean(high_dominates)),
        },
        "limitations": [
            "0.1 is a practical effect threshold, not an empirical matched-random null cutoff",
            "candidate selection is outcome-independent but restricted to strict question-grounded cues",
            "neutralisation is teacher-forced; generation and matched-random validation belong to stage 2",
        ],
    }
    out = a.out or a.input / "cluster_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
