"""Create compact tables and layer-wise plots for the local-data reproduction."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "custom_metrics"


def load(name: str) -> dict:
    with (METRICS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    within_files = [
        "shuffled_prepend_names__item_random.json",
        "shuffled_prepend_names__question_grouped.json",
        "question_and_result__item_random.json",
        "question_and_result__question_grouped.json",
    ]
    cross_files = [
        "shuffled_prepend_names__to__question_and_result.json",
        "question_and_result__to__shuffled_prepend_names.json",
    ]

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dataset in zip(
        axes, ("shuffled_prepend_names", "question_and_result")
    ):
        for protocol, label in (
            ("item_random", "Paper item-level split"),
            ("question_grouped", "Question-grouped split"),
        ):
            d = load(f"{dataset}__{protocol}.json")
            layers = sorted(int(k) for k in d["per_layer"])
            auc = [d["per_layer"][str(k)]["test_auroc_mean"] for k in layers]
            ax.plot(layers, auc, marker="o", markersize=2.5, label=label)
            s = d["validation_selected"]
            rows.append(
                {
                    "dataset_or_transfer": dataset,
                    "protocol": protocol,
                    "selection": "source_validation",
                    "layer": s["best_layer"],
                    "auroc_mean": s["test_auroc_mean"],
                    "auroc_std": s["test_auroc_std"],
                    "auprc_mean": s["test_auprc_mean"],
                    "accuracy_mean": s["test_accuracy_mean"],
                    "f1_mean": s["test_f1_mean"],
                }
            )
            o = d["paper_style_test_oracle"]
            rows.append(
                {
                    "dataset_or_transfer": dataset,
                    "protocol": protocol,
                    "selection": "test_oracle_paper_style",
                    "layer": o["best_layer"],
                    "auroc_mean": o["test_auroc_mean"],
                    "auroc_std": o["test_auroc_std"],
                    "auprc_mean": o["test_auprc_mean"],
                    "accuracy_mean": o["test_accuracy_mean"],
                    "f1_mean": o["test_f1_mean"],
                }
            )
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(dataset.replace("_", " "))
        ax.set_xlabel("Hidden-state index (0 = embedding output)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out AUROC (mean over 3 seeds)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(METRICS / "layerwise_auroc.png", dpi=200)
    fig.savefig(METRICS / "layerwise_auroc.pdf")
    plt.close(fig)

    for filename in cross_files:
        d = load(filename)
        s = d["source_validation_selected"]
        rows.append(
            {
                "dataset_or_transfer": f"{d['source']} -> {d['target']}",
                "protocol": "cross_dataset",
                "selection": "source_validation",
                "layer": s["best_layer"],
                "auroc_mean": s["target_auroc_mean"],
                "auroc_std": s["target_auroc_std"],
                "auprc_mean": s["target_auprc_mean"],
                "accuracy_mean": s["target_accuracy_mean"],
                "f1_mean": s["target_f1_mean"],
            }
        )

    with (METRICS / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
