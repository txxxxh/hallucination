#!/usr/bin/env python3
"""Tune P+U on the fixed 500-item pilot and evaluate on the disjoint remainder."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

RUN = Path(__file__).parent / "runs/281_scientist_stagewise_ur_pilot128"


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


full = read_jsonl(RUN / "predictions_uniform_2894.jsonl")
pilot_keys = {x["key"] for x in read_jsonl(RUN / "predictions.jsonl")}
y = np.array([x["error"] for x in full])
p = np.array([x["P"] for x in full])
u = np.array([x["U_trajectory"] for x in full])
train = np.array([x["key"] in pilot_keys for x in full])
test = ~train

rp, ru = rankdata(p) / len(p), rankdata(u) / len(u)
grid = np.arange(0, 2.0001, 0.05)
weight = float(max(grid, key=lambda w: roc_auc_score(y[train], rp[train] + w * ru[train])))
rank_score = rp + weight * ru

model = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[p[train], u[train]], y[train])
logit_score = model.predict_proba(np.c_[p, u])[:, 1]

rng = np.random.default_rng(286)
boot = []
test_idx = np.flatnonzero(test)
for _ in range(10000):
    idx = rng.choice(test_idx, len(test_idx), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    boot.append([
        roc_auc_score(y[idx], rank_score[idx]) - roc_auc_score(y[idx], p[idx]),
        roc_auc_score(y[idx], logit_score[idx]) - roc_auc_score(y[idx], p[idx]),
        average_precision_score(y[idx], rank_score[idx]) - average_precision_score(y[idx], p[idx]),
        average_precision_score(y[idx], logit_score[idx]) - average_precision_score(y[idx], p[idx]),
    ])
boot = np.asarray(boot)

report = {
    "protocol": "tune on fixed pilot n=500; evaluate once on disjoint remainder n=2394",
    "rank_weight_P_to_U": [1.0, weight],
    "logistic_intercept": model.intercept_.tolist(),
    "logistic_coefficients_P_U": model.coef_[0].tolist(),
    "pilot": {"P": metrics(y[train], p[train]),
              "weighted_rank": metrics(y[train], rank_score[train]),
              "logistic": metrics(y[train], logit_score[train])},
    "holdout": {"P": metrics(y[test], p[test]),
                "weighted_rank": metrics(y[test], rank_score[test]),
                "logistic": metrics(y[test], logit_score[test])},
    "full_descriptive_only": {"P": metrics(y, p),
                              "weighted_rank": metrics(y, rank_score),
                              "logistic": metrics(y, logit_score)},
    "holdout_paired_bootstrap_10000_delta_vs_P_ci95": {
        "weighted_rank_auroc": np.quantile(boot[:, 0], [.025, .975]).tolist(),
        "logistic_auroc": np.quantile(boot[:, 1], [.025, .975]).tolist(),
        "weighted_rank_auprc": np.quantile(boot[:, 2], [.025, .975]).tolist(),
        "logistic_auprc": np.quantile(boot[:, 3], [.025, .975]).tolist(),
    },
}
(RUN / "report_pu_tuned.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
