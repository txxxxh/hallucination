#!/usr/bin/env python3
"""Build the item source for probes with both accuracies strictly above 0.5."""
import json
from pathlib import Path

ROOT = Path("/home/tong56/whitebox/perturbation/runs")
PROBES = ROOT / "77_closedbook_fact_probe_results.jsonl"
DATA = Path("/home/tong56/whitebox/shuffled_prepend_names_question.json")
RECORDS = Path("/home/tong56/whitebox/tool_gate_correctness_names_llama31_8b/records.jsonl")
OUT = ROOT / "88_known_gt05_n1084.jsonl"


def main():
    probes = {row["key"]: row for row in map(json.loads, open(PROBES))}
    data = {str(row["key"]): row for row in json.load(open(DATA))}
    records = {row["key"]: row for row in map(json.loads, open(RECORDS))}
    selected = []
    for key, probe in probes.items():
        if not (
            probe["n_discriminative_facts"] >= 1
            and probe["binary_accuracy"] > 0.5
            and probe["pairwise_owner_accuracy"] > 0.5
        ):
            continue
        raw = data[key]
        record = records[key]
        selected.append(
            {
                "key": key,
                "group": probe["right_qid"],
                "correct": bool(record["correct"]),
                "original": str(record["parsed_answer"]),
                "knowledge_binary_accuracy": probe["binary_accuracy"],
                "knowledge_pairwise_owner_accuracy": probe[
                    "pairwise_owner_accuracy"
                ],
                "response_features": [],
            }
        )
    assert len(selected) == 1084, len(selected)
    with open(OUT, "w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} rows to {OUT}")


if __name__ == "__main__":
    main()
