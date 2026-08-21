#!/usr/bin/env python3
"""Paper-facing full-cohort analysis of the Scientist factorial atlas."""
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "runs/255_scientist_factorial_all1084"


def classify(pair, threshold):
    ui, uj = pair["u_i"], pair["u_j"]
    local, banzhaf = pair["local_fd"], pair["banzhaf"]
    pos = local > threshold and banzhaf > threshold
    neg = local < -threshold and banzhaf < -threshold
    if ui > 0 and uj > 0 and pos:
        return "synergy"
    if ui > 0 and uj > 0 and neg:
        return "redundancy"
    if ui * uj < 0 and min(abs(ui), abs(uj)) > threshold:
        return "competition"
    return "other"


def qstats(row, threshold):
    pairs = row["pair_interactions"]
    labels = Counter(classify(p, threshold) for p in pairs)
    robust = sum(labels[k] for k in ("synergy", "redundancy"))
    agree = [np.sign(p["local_fd"]) == np.sign(p["banzhaf"]) for p in pairs]
    hs = row["harsanyi_interactions"]
    pair_h = [abs(x["harsanyi"]) for x in hs if x["order"] == 2]
    high_h = [abs(x["harsanyi"]) for x in hs if x["order"] >= 3]
    denom = sum(pair_h) + sum(high_h)
    return {
        "n_pairs": len(pairs),
        "robust_rate": robust / len(pairs),
        "competition_rate": labels["competition"] / len(pairs),
        "synergy_rate": labels["synergy"] / len(pairs),
        "redundancy_rate": labels["redundancy"] / len(pairs),
        "sign_agreement": float(np.mean(agree)),
        "higher_mass": sum(high_h) / denom if denom else 0.0,
        "higher_dominates": bool(high_h and max(high_h) > max(pair_h, default=0.0)),
        "all_delete_abs_response": abs(row["all_candidates_deleted_margin"] - row["base_margin_wrong_minus_right"]),
        "max_single_abs_response": max(abs(x["u"]) for x in row["masks"] if len(x["ids"]) == 1),
    }


def bootstrap_diff(a, b, rng, draws=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = float(a.mean() - b.mean())
    sims = (rng.choice(a, (draws, len(a)), replace=True).mean(1) -
            rng.choice(b, (draws, len(b)), replace=True).mean(1))
    return {"error_minus_correct": observed,
            "ci95": np.quantile(sims, [.025, .975]).tolist()}


def summarize(rows, threshold):
    qs = [qstats(r, threshold) for r in rows]
    pairs = [p for r in rows for p in r["pair_interactions"]]
    labels = Counter(classify(p, threshold) for p in pairs)
    return {
        "n_questions": len(rows), "n_pairs": len(pairs),
        "likelihood_wrong_n": sum(r["likelihood_error"] for r in rows),
        "question_weighted": {k: float(np.mean([q[k] for q in qs])) for k in (
            "robust_rate", "competition_rate", "synergy_rate", "redundancy_rate",
            "sign_agreement", "higher_mass", "higher_dominates",
            "all_delete_abs_response", "max_single_abs_response")},
        "pair_taxonomy": dict(labels),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = [json.loads(p.read_text()) for p in sorted(args.input.glob("question_*.json"))]
    labels = {key: bool(correct) for key, _, correct, *_ in
              importlib.import_module("152_scientist_attention_pruned_current127").jobs()}
    skipped = [r for r in raw if r.get("skipped")]
    rows = [r for r in raw if not r.get("skipped")]
    for r in raw:
        r["generation_correct"] = labels[r["key"]]
    correct = [r for r in rows if r["generation_correct"]]
    error = [r for r in rows if not r["generation_correct"]]
    rng = np.random.default_rng(args.seed)

    thresholds = {}
    for threshold in (.05, .10, .20):
        csum, esum = summarize(correct, threshold), summarize(error, threshold)
        cq, eq = [qstats(r, threshold) for r in correct], [qstats(r, threshold) for r in error]
        diffs = {k: bootstrap_diff([q[k] for q in eq], [q[k] for q in cq], rng)
                 for k in ("robust_rate", "competition_rate", "synergy_rate",
                           "redundancy_rate", "higher_mass", "higher_dominates",
                           "all_delete_abs_response", "max_single_abs_response")}
        thresholds[f"{threshold:.2f}"] = {"correct": csum, "error": esum,
                                         "error_vs_correct": diffs}

    systematic = [r for r in error if r["likelihood_error"]]
    repairs = Counter()
    for r in systematic:
        sizes = [len(x["ids"]) for x in r["minimal_repair_sets"]]
        if not sizes:
            repairs["no_candidate_repair"] += 1
        elif min(sizes) == 1:
            repairs["single_cue_repair_available"] += 1
        else:
            repairs["multi_cue_only_repair"] += 1

    skip_by_reason_correctness = Counter(
        f"{r['reason']}|{'correct' if r['generation_correct'] else 'error'}" for r in skipped)
    report = {
        "protocol": {
            "population": "all 1,084 Scientist-Names questions",
            "unit": "question (question-weighted summaries; 10,000 question bootstrap draws)",
            "interaction_rule": "local finite difference and Banzhaf must agree in sign and exceed |threshold|",
            "primary_threshold": 0.10,
            "sensitivity_thresholds": [0.05, 0.10, 0.20],
        },
        "coverage": {
            "mother_population": len(raw), "generation_correct": sum(labels.values()),
            "generation_error": len(labels) - sum(labels.values()),
            "invalid_candidate": sum(r["reason"] == "invalid_candidate" for r in skipped),
            "fewer_than_two_keywords": sum(r["reason"] == "too_few_candidates" for r in skipped),
            "skip_breakdown": dict(skip_by_reason_correctness),
            "factorial_analysed": len(rows), "analysed_correct": len(correct),
            "analysed_error": len(error),
        },
        "threshold_results": thresholds,
        "minimal_repair_among_likelihood_systematic_errors": {
            "n": len(systematic), **dict(repairs),
            "collective_repair_n": sum(r["all_candidates_deleted_margin"] < 0 for r in systematic),
        },
    }
    outdir = args.out or args.input
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "full_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    primary = thresholds["0.10"]
    def pct(x): return f"{100*x:.1f}"
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Subset & $n_q$ & Competition (\%) & Redundancy (\%) & Synergy (\%) \\",
        r"\midrule",
    ]
    for name, key in (("Correct", "correct"), ("Hallucination", "error")):
        s = primary[key]
        q = s["question_weighted"]
        lines.append(f"{name} & {s['n_questions']} & {pct(q['competition_rate'])} & {pct(q['redundancy_rate'])} & {pct(q['synergy_rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "full_main_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
