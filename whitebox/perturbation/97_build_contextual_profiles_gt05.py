#!/usr/bin/env python3
"""Build profiles source selected by contextual atomic-probe scores > 0.5."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "runs"
PROBES = ROOT / "77b_contextual_profile_probe_results.jsonl"
RECORDS = Path(__file__).resolve().parent.parent / "tool_gate_correctness_profiles_llama31_8b/records.jsonl"
OUT = ROOT / "97_profiles_contextual_gt05_n2863.jsonl"


def main():
    probes = {x["key"]: x for x in map(json.loads, open(PROBES))}
    records = {x["key"]: x for x in map(json.loads, open(RECORDS))}
    selected = []
    for key, probe in probes.items():
        if not (probe["n_discriminative_facts"] >= 1
                and probe["binary_accuracy"] > .5
                and probe["pairwise_owner_accuracy"] > .5):
            continue
        record = records[key]
        selected.append({
            "key": key,
            "group": probe["right_qid"],
            "correct": bool(record["correct"]),
            "original": str(record["parsed_answer"]),
            "knowledge_binary_accuracy": probe["binary_accuracy"],
            "knowledge_pairwise_owner_accuracy": probe["pairwise_owner_accuracy"],
            "response_features": [],
        })
    assert len(selected) == 2863, len(selected)
    with open(OUT, "w") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} rows; correct={sum(x['correct'] for x in selected)}")


if __name__ == "__main__":
    main()
