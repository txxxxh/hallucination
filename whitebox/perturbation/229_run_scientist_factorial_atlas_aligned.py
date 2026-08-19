#!/usr/bin/env python3
"""Run the factorial atlas with profile facts aligned to the names prompt."""
from __future__ import annotations

import importlib


runner = importlib.import_module("228_run_scientist_factorial_atlas")
atlas = runner.atlas
_PROMPTS = {x[0]: x[3] for x in importlib.import_module(
    "152_scientist_attention_pruned_current127").jobs()}


def aligned_question_facts(row, profiles, builder):
    proxy = dict(row)
    proxy["prompt"] = _PROMPTS[str(row["key"])]
    return runner.strict_question_facts(proxy, profiles, builder)


atlas.question_facts = aligned_question_facts


if __name__ == "__main__":
    atlas.main()
