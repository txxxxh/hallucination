#!/usr/bin/env python3
"""Compare answer-level uncertainty estimators on the fixed GSM8K CoT pilot."""
from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

RUNS = Path(__file__).resolve().parent / "runs"


def canon(text: str) -> str:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text))
    if not matches:
        return "<invalid>"
    return format(Decimal(matches[-1].replace(",", "")).normalize(), "f")


def main() -> None:
    items = [json.loads(line) for line in (RUNS / "237_gsm8k_cot_u_split_confirmation/items.jsonl").open()]
    manifest = {x["key"]: x for x in map(json.loads, (RUNS / "140_gsm8k_natural/natural_balanced_n942.jsonl").open())}
    detector_report = json.load((RUNS / "142_gsm8k_natural_current127_report.json").open())
    detector = {x["id"]: -x["oof_score"] for x in detector_report["per_item"]}
    y = np.asarray([x["greedy_error"] for x in items])

    scores: dict[str, np.ndarray] = {}
    for count in (3, 6):
        answer_entropy = []
        variation_ratio = []
        greedy_disagreement = []
        for item in items:
            values = item["samples"][:count]
            frequencies = np.asarray(list(Counter(values).values()), dtype=float) / count
            answer_entropy.append(float(-(frequencies * np.log(frequencies)).sum()))
            variation_ratio.append(float(1 - frequencies.max()))
            greedy = canon(manifest[item["key"]]["generation"])
            greedy_disagreement.append(float(1 - Counter(values)[greedy] / count))
        scores[f"answer_entropy_{count}"] = np.asarray(answer_entropy)
        scores[f"variation_ratio_{count}"] = np.asarray(variation_ratio)
        scores[f"greedy_answer_disagreement_{count}"] = np.asarray(greedy_disagreement)
    scores["existing_oof_detector"] = np.asarray([detector[x["key"]] for x in items])
    scores["rank_fusion_greedy6_existing"] = rankdata(scores["greedy_answer_disagreement_6"]) + rankdata(scores["existing_oof_detector"])

    metrics = {
        name: {"auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score))}
        for name, score in scores.items()
    }
    rng = np.random.default_rng(42)
    boot = []
    u = scores["greedy_answer_disagreement_6"]
    existing = scores["existing_oof_detector"]
    fusion = scores["rank_fusion_greedy6_existing"]
    for _ in range(10_000):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() == y[index].max():
            continue
        a = roc_auc_score(y[index], u[index])
        b = roc_auc_score(y[index], existing[index])
        c = roc_auc_score(y[index], fusion[index])
        boot.append((a, a - b, c - max(a, b)))
    boot = np.asarray(boot)
    report = {
        "protocol": "fixed balanced GSM8K CoT n=300; answer-canonicalized sampling uncertainty; original greedy label; label-free rank fusion",
        "n": len(items),
        "errors": int(y.sum()),
        "metrics": metrics,
        "bootstrap_95": {
            "greedy_answer_disagreement_6_auroc": np.quantile(boot[:, 0], [.025, .975]).tolist(),
            "greedy6_minus_existing": np.quantile(boot[:, 1], [.025, .975]).tolist(),
            "rank_fusion_gain_over_best": np.quantile(boot[:, 2], [.025, .975]).tolist(),
        },
    }
    out = RUNS / "239_gsm8k_uncertainty_methods"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
