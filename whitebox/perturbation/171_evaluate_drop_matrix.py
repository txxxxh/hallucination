#!/usr/bin/env python3
"""Evaluate every available DROP feature directory with the paper4 protocol."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("paper4_eval", HERE / "159_evaluate_paper4_matrix.py")
EVAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=HERE / "runs/paper3_mean_matrix/features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "runs/paper3_mean_matrix/evaluation_drop",
    )
    args = parser.parse_args()

    rows = []
    details = {
        "protocol": "3 seeds x 5-fold grouped OOF; same feature/PCA/LR settings as 159_evaluate_paper4_matrix.py",
        "dataset": "drop",
        "results": [],
    }
    for model in EVAL.MODELS:
        for method in EVAL.METHODS:
            directory = args.feature_root / model / "drop" / method
            count = len(list(directory.glob("*.npz")))
            if count == 0:
                print(f"[{model}/{method}] missing; skipped", flush=True)
                continue
            print(f"[{model}/{method}] loading {count}", flush=True)
            data = EVAL.load_directory(directory, EVAL.EXPECTED["drop"])
            result = EVAL.evaluate_oof(data, "drop", method)
            mean = result["mean"]
            row = {
                "model": model,
                "method": method,
                "n": result["n"],
                "positive": result["positive"],
                **mean,
                "query_reduction": result["queries"]["reduction_vs_full"],
            }
            rows.append(row)
            details["results"].append({"model": model, "method": method, **result})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "model", "method", "n", "positive", "auroc", "auprc", "accuracy",
        "balanced_accuracy", "macro_f1", "query_reduction",
    )
    with (args.output_dir / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "evaluation.json").write_text(json.dumps(details, indent=2) + "\n")

    lines = [
        "# DROP evaluation", "", details["protocol"], "",
        "| Model | Method | N | Positive | AUROC | AUPRC | Accuracy | Bal. Acc. | Macro-F1 | Query reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['n']} | {row['positive']} | "
            f"{row['auroc']:.3f} | {row['auprc']:.3f} | {row['accuracy']:.3f} | "
            f"{row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | "
            f"{row['query_reduction']:.1%} |"
        )
    lines += ["", "Qwen gradient is absent because generation was intentionally skipped.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines))
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
