#!/usr/bin/env python3
"""Paired Scientist pilot for familiarity-driven keyword selection.

This is deliberately a behavioural-frequency experiment: closed-book probes
measure how strongly the tested model knows each candidate profile.  They are
not mislabeled as counts from the (unavailable) pretraining corpus.

The analysis joins four existing, independently produced artifacts:
  * the fixed Scientist 1084 pool and its correctness labels;
  * atomic closed-book fact probes (no biographies are shown);
  * the model's original candidate choice;
  * exact all-span neutralization caches from current127.

For each item we estimate candidate familiarity from signed Yes/No logits,
then ask whether relative familiarity predicts the selected candidate and an
error.  Finally, the exact top-span intervention tests whether neutralizing the
selected keyword reduces the selected-vs-other answer margin (mediation/rescue).

Usage:
  python 203_keyword_familiarity_mediation.py
  python 203_keyword_familiarity_mediation.py --limit 200 --bootstrap 2000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_or_nan(values) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(values.mean()) if len(values) else float("nan")


def signed_probe_score(probe: dict) -> float:
    """Positive means confident in the correct truth value of an atomic fact."""
    p_yes = float(np.clip(probe["p_yes"], 1e-5, 1 - 1e-5))
    value = float(logit(p_yes))
    return value if probe["gold_yes"] else -value


def candidate_familiarity(probes: list[dict], person: str) -> float:
    """Mean signed factual log-odds for all probes mentioning one candidate."""
    return mean_or_nan(signed_probe_score(p) for p in probes if p["person"] == person)


def exact_intervention(path: Path) -> dict:
    """Read the strongest exact stage-1 intervention from a current127 cache."""
    with np.load(path, allow_pickle=True) as z:
        pred = z["stage1_pred"].astype(float)
        other = z["stage1_other"].astype(float)
        if len(pred) < 2:
            raise ValueError(f"{path}: no stage-1 intervention")
        base = float(pred[0] - other[0])
        post = float(pred[1] - other[1])
        return {
            "top_span": str(z["deleted_text"].item()),
            "margin_base": base,
            "margin_after_top_neutralization": post,
            "margin_drop": base - post,
            "relative_margin_drop": (base - post) / (abs(base) + 1e-6),
            "candidate_flip": bool(base > 0 and post < 0),
            "n_stage1_spans": int(z["stage1_full"]),
        }


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        estimates[i] = rng.choice(values, len(values), replace=True).mean()
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(estimates, .025)),
                 float(np.quantile(estimates, .975))],
    }


def safe_spearman(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return {"n": int(keep.sum()), "rho": None, "p": None}
    result = spearmanr(x[keep], y[keep])
    return {"n": int(keep.sum()), "rho": float(result.statistic),
            "p": float(result.pvalue)}


def grouped_oof_auc(rows: list[dict], feature_names: list[str], seed: int) -> float:
    x = np.asarray([[r[name] for name in feature_names] for r in rows], float)
    y = np.asarray([not r["correct"] for r in rows], int)
    groups = np.asarray([r["group"] for r in rows])
    probability = np.zeros(len(rows), float)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for train, test in cv.split(x, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                               solver="liblinear", random_state=seed),
        )
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return float(roc_auc_score(y, probability))


def build_rows(args) -> tuple[list[dict], dict]:
    pool = {str(r["key"]): r for r in read_jsonl(args.pool)}
    probes = {str(r["key"]): r for r in read_jsonl(args.probes)}
    answers = {str(r["key"]): r for r in read_jsonl(args.answers)}
    rows, exclusions = [], {"missing_probe": 0, "missing_answer": 0,
                            "missing_cache": 0, "unmatched_answer": 0,
                            "nonfinite_familiarity": 0}
    keys = sorted(pool)
    if args.limit:
        keys = keys[:args.limit]
    for key in keys:
        if key not in probes:
            exclusions["missing_probe"] += 1; continue
        if key not in answers:
            exclusions["missing_answer"] += 1; continue
        cache = args.cache / f"{key}.npz"
        if not cache.exists():
            exclusions["missing_cache"] += 1; continue
        probe, answer = probes[key], answers[key]
        right, wrong = str(probe["right_answer"]), str(probe["wrong_answer"])
        selected = str(answer.get("parsed_answer", ""))
        if selected not in (right, wrong):
            exclusions["unmatched_answer"] += 1; continue
        right_fam = candidate_familiarity(probe["probes"], right)
        wrong_fam = candidate_familiarity(probe["probes"], wrong)
        if not np.isfinite(right_fam + wrong_fam):
            exclusions["nonfinite_familiarity"] += 1; continue
        intervention = exact_intervention(cache)
        correct = selected == right
        selected_fam = right_fam if correct else wrong_fam
        other_fam = wrong_fam if correct else right_fam
        rows.append({
            "key": key, "group": str(pool[key]["group"]), "correct": correct,
            "selected": selected, "right_answer": right, "wrong_answer": wrong,
            "right_familiarity": right_fam, "wrong_familiarity": wrong_fam,
            "wrong_minus_right_familiarity": wrong_fam - right_fam,
            "selected_minus_other_familiarity": selected_fam - other_fam,
            "n_atomic_probes": int(probe["n_binary_probes"]),
            "binary_probe_accuracy": float(probe["binary_accuracy"]),
            "pairwise_owner_accuracy": float(probe["pairwise_owner_accuracy"]),
            **intervention,
        })
    return rows, exclusions


def analyse(rows: list[dict], args, exclusions: dict) -> dict:
    if len(rows) < 20:
        raise RuntimeError(f"only {len(rows)} analyzable rows")
    wrong = [r for r in rows if not r["correct"]]
    correct = [r for r in rows if r["correct"]]
    selected_gap = np.asarray([r["selected_minus_other_familiarity"] for r in rows])
    wrong_gap = np.asarray([r["wrong_minus_right_familiarity"] for r in rows])
    error = np.asarray([not r["correct"] for r in rows], int)
    margin_drop = np.asarray([r["margin_drop"] for r in rows])
    relative_drop = np.asarray([r["relative_margin_drop"] for r in rows])

    # Within-item candidate-choice test: is the chosen candidate independently
    # better known than the unchosen candidate?
    try:
        signed_rank = wilcoxon(selected_gap, alternative="greater")
        signed_rank_result = {"statistic": float(signed_rank.statistic),
                              "p_one_sided": float(signed_rank.pvalue)}
    except ValueError:
        signed_rank_result = {"statistic": None, "p_one_sided": None}

    aucs = [grouped_oof_auc(rows, ["wrong_minus_right_familiarity"], seed)
            for seed in (42, 43, 44)]
    report = {
        "experiment": "Scientist paired keyword-familiarity mediation pilot",
        "claim_boundary": (
            "Closed-book factual confidence is a behavioural proxy for model "
            "familiarity, not a direct measurement of proprietary pretraining counts."
        ),
        "n": len(rows), "n_correct": len(correct), "n_wrong": len(wrong),
        "exclusions": exclusions,
        "candidate_selection": {
            "estimand": "familiarity(selected candidate) - familiarity(other candidate)",
            "bootstrap": bootstrap_mean(selected_gap, args.bootstrap, args.seed),
            "fraction_selected_more_familiar": float(np.mean(selected_gap > 0)),
            "paired_wilcoxon_greater": signed_rank_result,
        },
        "error_prediction": {
            "estimand": "familiarity(wrong candidate) - familiarity(right candidate)",
            "mean_gap_correct_items": mean_or_nan(
                r["wrong_minus_right_familiarity"] for r in correct),
            "mean_gap_wrong_items": mean_or_nan(
                r["wrong_minus_right_familiarity"] for r in wrong),
            "grouped_5fold_oof_auroc_per_seed": aucs,
            "grouped_5fold_oof_auroc_mean": float(np.mean(aucs)),
            "point_biserial_spearman": safe_spearman(wrong_gap, error),
        },
        "targeted_perturbation_rescue": {
            "estimand": "selected-vs-other margin before minus after exact top-span neutralization",
            "all_items_margin_drop": bootstrap_mean(margin_drop, args.bootstrap, args.seed + 1),
            "wrong_items_margin_drop": bootstrap_mean(
                np.asarray([r["margin_drop"] for r in wrong]), args.bootstrap, args.seed + 2),
            "wrong_items_relative_margin_drop": bootstrap_mean(
                np.asarray([r["relative_margin_drop"] for r in wrong]), args.bootstrap,
                args.seed + 3),
            "wrong_items_candidate_flip_rate": float(np.mean(
                [r["candidate_flip"] for r in wrong])),
            "familiarity_gap_vs_margin_drop_wrong": safe_spearman(
                [r["selected_minus_other_familiarity"] for r in wrong],
                [r["margin_drop"] for r in wrong]),
            "familiarity_gap_vs_relative_drop_wrong": safe_spearman(
                [r["selected_minus_other_familiarity"] for r in wrong],
                [r["relative_margin_drop"] for r in wrong]),
        },
        "interpretation_rule": {
            "supports_selection_bias": "selected familiarity gap > 0 and error AUROC > .5",
            "supports_mediation": (
                "among wrong items, familiarity gap positively tracks top-span rescue; "
                "a matched-span randomized intervention is required for a confirmatory causal claim"
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=RUNS / "88_known_gt05_n1084.jsonl")
    parser.add_argument("--probes", type=Path,
                        default=RUNS / "77_closedbook_fact_probe_results.jsonl")
    parser.add_argument("--answers", type=Path,
                        default=ROOT / "tool_gate_correctness_names_llama31_8b/records.jsonl")
    parser.add_argument("--cache", type=Path, default=RUNS /
                        "paper4_self_matrix_v2/features/llama/scientist/exact")
    parser.add_argument("--out", type=Path,
                        default=RUNS / "203_keyword_familiarity_mediation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, exclusions = build_rows(args)
    report = analyse(rows, args, exclusions)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "items.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
