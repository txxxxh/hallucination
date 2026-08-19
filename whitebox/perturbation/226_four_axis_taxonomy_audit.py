#!/usr/bin/env python3
"""Feasibility audit for an uncertainty/representation/evidence/perturbation taxonomy."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "226_four_axis_taxonomy_audit"


def load_jsonl(path):
    return {str(x["key"]): x for x in map(json.loads, path.open())}


def pct(x):
    return rankdata(x, method="average") / len(x)


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = load_jsonl(RUNS / "215_scientist_uncertainty_known_unknown_predictions.jsonl")
    r = load_jsonl(RUNS / "216_known_error_representation_trajectory_predictions.jsonl")
    e = load_jsonl(RUNS / "221_scientist_minicheck_flan" / "items.jsonl")
    sem = load_jsonl(RUNS / "219_scientist_semantic_neighborhood_full" / "items.jsonl")
    causal_path = RUNS / "220_scientist_representation_causal_direction" / "items.jsonl"
    causal = load_jsonl(causal_path) if causal_path.exists() else {}
    cache = RUNS / "120_physical_delete_rerank"
    keys = sorted(set(u) & set(r) & set(e) & set(sem) & {p.stem for p in cache.glob("*.npz")})
    rows = []
    for key in keys:
        with np.load(cache / f"{key}.npz", allow_pickle=True) as z:
            ps, os = z["stage1_pred_scores"], z["stage1_other_scores"]
            pd, od = z["stage2_pred_scores"], z["stage2_other_scores"]
        differential = (ps[0] - os[0]) - (ps[1:] - os[1:])
        p_strength = float(max(0.0, np.max(differential)))
        p_concentration = float(np.max(np.abs(differential)) / (np.sum(np.abs(differential)) + 1e-12))
        # Positive means physical deletion reduced preference for the generated answer.
        delete_repair_gain = float((ps[0] - os[0]) - (pd[0] - od[0]))
        err = int(r[key]["error"])
        assert err == int(u[key]["error"]) == 1 - int(e[key]["correct"])
        rows.append({
            "key": key, "group": r[key].get("group"), "error": err,
            "u_score": float(sem[key]["pooled_entropy"]),
            "u_nll": float(u[key]["mean_token_nll"]),
            "r_score": float(r[key]["delta_trajectory"]),
            "e_score": float(e[key]["alternative_whole_support"] - e[key]["chosen_whole_support"]),
            "p_score": p_strength, "p_concentration": p_concentration,
            "delete_repair_gain": delete_repair_gain,
            "original_consistency": float(sem[key]["original_consistency"]),
            "neighbour_consistency": float(sem[key]["neighbour_consistency"]),
        })
    axes = ["u_score", "r_score", "e_score", "p_score"]
    y = np.array([x["error"] for x in rows])
    for a in axes:
        z = np.array([x[a] for x in rows]); q = pct(z)
        for x, v in zip(rows, q): x[a[0] + "_percentile"] = float(v)
    for x in rows:
        ps = {a: x[a + "_percentile"] for a in "urep"}
        x["dominant_axis"] = max(ps, key=ps.get).upper()
        x["high_axes"] = "".join(a.upper() for a, v in ps.items() if v >= .7) or "none"

    metrics = {}
    for a in axes:
        z = np.array([x[a] for x in rows])
        metrics[a] = {"auroc_error": float(roc_auc_score(y, z)),
                      "mean_error": mean(z[y == 1]), "mean_correct": mean(z[y == 0])}
    correlations = {}
    for i, a in enumerate(axes):
        for b in axes[i + 1:]: correlations[f"{a}__{b}"] = float(spearmanr([x[a] for x in rows], [x[b] for x in rows]).statistic)
    errors = [x for x in rows if x["error"]]
    dominant = Counter(x["dominant_axis"] for x in errors)
    overlap = Counter(x["high_axes"] for x in errors)

    def extremes(axis, outcome, subset):
        lo = [x[outcome] for x in subset if x[axis + "_percentile"] <= .3]
        hi = [x[outcome] for x in subset if x[axis + "_percentile"] >= .7]
        return {"low_n": len(lo), "low_mean": mean(lo), "high_n": len(hi), "high_mean": mean(hi),
                "high_minus_low": None if not lo or not hi else mean(hi) - mean(lo)}

    validations = {
        "U_semantic": {
            "prediction": "U-high should have lower consistency under resampling/neighbour prompts",
            "original_consistency": extremes("u", "original_consistency", rows),
            "neighbour_consistency": extremes("u", "neighbour_consistency", rows),
            "note": "descriptive convergence, not independent causal validation; U is defined from the same sample family",
        },
        "P_physical_delete": {
            "prediction": "among errors, P-high should receive larger repair gain from deleting the selected span",
            "delete_repair_gain": extremes("p", "delete_repair_gain", errors),
            "note": "physical deletion is distinct from stage-1 neutralization, but span selection is shared",
        },
    }
    crows = [x for x in rows if x["key"] in causal and x["error"]]
    for x in crows:
        c = causal[x["key"]]
        x["r_causal_specific_repair"] = float((c["causal"]["-2.0"] - c["causal"]["0.0"]) - (c["placebo"]["-2.0"] - c["placebo"]["0.0"])) * -1
    validations["R_direction_intervention"] = {
        "prediction": "among errors, R-high should have larger causal-minus-placebo repair from the held-out error direction",
        "specific_repair": extremes("r", "r_causal_specific_repair", crows),
        "n_overlap": len(crows),
    }
    report = {
        "protocol": "common-key Scientist/Llama audit; unsupervised axis scores; global rank thresholds 30/70; categories are multi-label hypotheses",
        "n": len(rows), "n_error": int(y.sum()), "error_rate": float(y.mean()),
        "coverage": {"U": len(u), "R": len(r), "E": len(e), "semantic": len(sem), "P_cache": len(list(cache.glob('*.npz'))), "intersection": len(rows)},
        "score_definitions": {"U": "semantic pooled entropy", "R": "group-OOF delta-trajectory error probability", "E": "alternative minus chosen MiniCheck whole-support", "P": "maximum positive generated-minus-alternative margin support removable by span neutralization"},
        "metrics": metrics, "spearman": correlations,
        "error_dominant_axis_counts": dict(dominant), "error_high_axis_overlap": dict(overlap),
        "validations": validations,
        "limitations": ["dominant-axis assignment is forced and is not a ground-truth label", "30/70 cutoffs are exploratory and must be frozen before confirmatory evaluation", "E requires a new explicit evidence-completion intervention", "U and P checks reuse parts of their discovery pipeline and therefore require stronger held-out controls"],
    }
    with (OUT / "items.jsonl").open("w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    # Balanced actionable list: highest-percentile errors for each dominant axis.
    manifest = []
    for a in "UREP":
        cand = sorted((x for x in errors if x["dominant_axis"] == a), key=lambda x: x[a.lower() + "_percentile"], reverse=True)[:40]
        for x in cand:
            manifest.append({"key": x["key"], "group": x["group"], "assigned_axis": a,
                             "percentiles": {b: x[b.lower() + "_percentile"] for b in "UREP"},
                             "intervention": {"U": "increase sampling then majority/abstain", "R": "negative cross-fitted representation direction plus orthogonal placebo", "E": "append missing/contradicting evidence plus irrelevant-evidence control", "P": "delete/neutralize selected span plus matched random-span control"}[a]})
    with (OUT / "intervention_manifest.jsonl").open("w") as f:
        for x in manifest: f.write(json.dumps(x) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
