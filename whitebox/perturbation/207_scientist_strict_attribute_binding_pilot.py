#!/usr/bin/env python3
"""Strict semantic-phrase version of the Scientist attribute-binding pilot."""
from __future__ import annotations
import importlib
import json
import re
import sys

base = importlib.import_module("206_scientist_attribute_binding_pilot")


def strict_match(span, facts):
    span_tokens = base.importlib.import_module(
        "204_scientist_binding_override_pilot").toks(span)
    generic = {"award", "prize", "medal", "order", "university", "society",
               "college", "institute", "field", "member"}
    candidates = []
    for fact in facts:
        value_tokens = base.importlib.import_module(
            "204_scientist_binding_override_pilot").toks(fact["value"])
        overlap = span_tokens & value_tokens
        field = fact["field"]
        abstract = field in {"occupation", "field", "position_held"}
        # Abstract attributes may be a one-word core with a meaningful modifier
        # ("prominent chemist"). Named entities require two informative tokens,
        # preventing "Order for" from matching an arbitrary Order award.
        valid = ((abstract and value_tokens <= span_tokens and len(value_tokens) >= 1)
                 or len(overlap - generic) >= 2
                 or (len(overlap) >= 2 and len(overlap - generic) >= 1))
        if valid:
            score = (len(overlap) / max(1, len(span_tokens | value_tokens)),
                     len(overlap - generic), len(overlap))
            item = dict(fact)
            if abstract:
                item["value"] = span.strip(" ,.;:!?")
            candidates.append((score, item))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


if __name__ == "__main__":
    span_module = importlib.import_module("125_collect_current_three_benchmarks")
    span_module.spans = base.sliding_spans
    jobs_module = importlib.import_module("152_scientist_attention_pruned_current127")
    original_jobs = jobs_module.jobs
    jobs_module.jobs = lambda: [r for r in original_jobs()
                                if r[4] not in ("", "None", "null")
                                and r[5] not in ("", "None", "null")]
    experiment = importlib.import_module("204_scientist_binding_override_pilot")
    experiment.facts_by_item = base.full_profile_attributes
    experiment.match_fact = strict_match
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(experiment.RUNS / "207_strict_attribute_binding_pilot")])
    experiment.main()
