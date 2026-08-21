#!/usr/bin/env python3
"""Candidate-conditioned uncertainty and representation baselines.

Uses exactly the cached frozen Llama candidate pairs used by the perturbation
matrix.  No perturbation-derived deltas are included in either baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SEEDS = (42, 43, 44)


def load(path):
    rows = []
    for file in sorted(path.glob("*.npz")):
        with np.load(file, allow_pickle=True) as z:
            rows.append({"key": str(z["key"].item()),
                         "group": str(z["group"].item()),
                         "correct": int(z["correct"]),
                         "chosen_score": float(z["stage1_pred"][0]),
                         "alternative_score": float(z["stage1_other"][0]),
                         "chosen_hidden": z["pred_hidden"][0].astype(np.float32),
                         "alternative_hidden": z["other_hidden"][0].astype(np.float32)})
    return rows


def metric(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def representation_oof(y, groups, hidden):
    per_seed, predictions = [], []
    for seed in SEEDS:
        pred = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for train, test in cv.split(hidden, y, groups):
            scaler = StandardScaler().fit(hidden[train])
            a, b = scaler.transform(hidden[train]), scaler.transform(hidden[test])
            dim = min(64, len(train) - 1, a.shape[1])
            pca = PCA(dim, whiten=True, svd_solver="randomized",
                      random_state=seed).fit(a)
            a, b = pca.transform(a), pca.transform(b)
            clf = LogisticRegression(C=.03, class_weight="balanced",
                                     solver="liblinear", max_iter=5000,
                                     random_state=seed).fit(a, y[train])
            pred[test] = clf.predict_proba(b)[:, 1]
        predictions.append(pred)
        per_seed.append(metric(y, pred))
    return {"mean": {name: float(np.mean([x[name] for x in per_seed]))
                     for name in ("auroc", "auprc")},
            "ensemble": metric(y, np.mean(predictions, axis=0)),
            "per_seed": per_seed}


def main():
    roots = {
        "scientist": RUNS / "paper4_matrix/features/llama/scientist/exact",
        "trivia": RUNS / "paper4_matrix/features/llama/trivia/exact",
        "gsm8k": RUNS / "paper4_matrix/features/llama/gsm8k/exact",
        "drop": RUNS / "167_drop1000_exact",
    }
    report = {"model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
              "protocol": "same frozen candidate pairs as perturbation; grouped 3x5 OOF for representation; no perturbation delta features",
              "datasets": {}}
    for dataset, root in roots.items():
        rows = load(root)
        y = 1 - np.asarray([x["correct"] for x in rows])
        groups = np.asarray([x["group"] for x in rows])
        # A positive gap means the supplied alternative has higher normalized
        # sequence support than the generated/chosen candidate.
        gap = np.asarray([x["alternative_score"] - x["chosen_score"] for x in rows])
        # Pairwise representation is order-aware and uses only unperturbed base
        # candidate states.  Difference makes the exact candidate comparison
        # explicit without feeding likelihood or correctness into the probe.
        hidden = np.stack([x["alternative_hidden"] - x["chosen_hidden"] for x in rows])
        report["datasets"][dataset] = {
            "n": len(rows), "errors": int(y.sum()), "groups": len(set(groups)),
            "candidate_likelihood_gap": metric(y, gap),
            "paired_hidden_state_probe": representation_oof(y, groups, hidden),
        }
        print(dataset, json.dumps(report["datasets"][dataset]), flush=True)
    out = RUNS / "260_candidate_ur_matrix"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
