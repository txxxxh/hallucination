#!/usr/bin/env python3
"""Sliding-window binding pilot with full profile attributes, not entities only."""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def full_profile_attributes():
    builder = importlib.import_module("76_build_closedbook_fact_probes")
    rows = json.load((ROOT / "shuffled_prepend_profiles_question.json").open())
    output = {}
    for row in rows:
        profiles, _ = builder.parse_item(row)
        attributes = []
        seen = set()
        for profile in profiles:
            for field in builder.FIELDS:
                for value in builder.values(profile, field):
                    key = (field, builder.norm(value))
                    if key not in seen:
                        seen.add(key)
                        attributes.append({"field": field, "value": value})
        output[str(row["key"])] = attributes
    return output


def sliding_spans(att, prep):
    spans = att.build_word_spans(prep, widths=(2,), stride=1)
    return spans, [None] * len(spans)


if __name__ == "__main__":
    span_module = importlib.import_module("125_collect_current_three_benchmarks")
    span_module.spans = sliding_spans

    jobs_module = importlib.import_module("152_scientist_attention_pruned_current127")
    original_jobs = jobs_module.jobs
    jobs_module.jobs = lambda: [
        row for row in original_jobs()
        if row[4] not in ("", "None", "null") and row[5] not in ("", "None", "null")
    ]

    experiment = importlib.import_module("204_scientist_binding_override_pilot")
    experiment.facts_by_item = full_profile_attributes
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(experiment.RUNS / "206_attribute_binding_pilot")])
    experiment.main()
