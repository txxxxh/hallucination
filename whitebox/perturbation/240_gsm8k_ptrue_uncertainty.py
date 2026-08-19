#!/usr/bin/env python3
"""P(True) self-evaluation uncertainty on the fixed GSM8K CoT pilot."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main() -> None:
    source = RUNS / "140_gsm8k_natural/natural_balanced_n942.jsonl"
    all_rows = [json.loads(line) for line in source.open()]
    rows = [x for x in all_rows if x["correct"]][:150] + [x for x in all_rows if not x["correct"]][:150]
    requests = []
    for row in rows:
        prompt = (
            "Problem:\n" + row["question"] + "\n\n"
            "Proposed solution:\n" + row["generation"] + "\n\n"
            "Is the proposed solution and its final numeric answer fully correct? "
            "Answer only True or False."
        )
        requests.append({"prompt": prompt, "right": "True", "wrong": "False"})

    model, tokenizer = importlib.import_module("61_grad_span_proposal").load_model(
        "NousResearch/Meta-Llama-3.1-8B-Instruct", "bfloat16", "cuda"
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    margin = importlib.import_module("229_trivia_e_confirmation").score(
        model, tokenizer, requests, 8
    )
    uncertainty = -margin
    y = np.asarray([not x["correct"] for x in rows], dtype=int)
    report = {
        "protocol": "Kadavath-style P(True) self-evaluation of the original greedy CoT; fixed balanced n=300",
        "n": len(rows),
        "errors": int(y.sum()),
        "auroc": float(roc_auc_score(y, uncertainty)),
        "auprc": float(average_precision_score(y, uncertainty)),
        "mean_uncertainty_error": float(uncertainty[y == 1].mean()),
        "mean_uncertainty_correct": float(uncertainty[y == 0].mean()),
    }
    output = RUNS / "240_gsm8k_ptrue_uncertainty"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "items.jsonl").open("w") as stream:
        for row, value in zip(rows, uncertainty):
            stream.write(json.dumps({"key": row["key"], "error": int(not row["correct"]), "pfalse_log_margin": float(value)}) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
