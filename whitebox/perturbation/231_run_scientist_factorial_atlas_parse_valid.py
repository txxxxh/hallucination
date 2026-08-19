#!/usr/bin/env python3
"""Production launcher for the parse-valid Scientist factorial atlas."""
from __future__ import annotations

import importlib


runner = importlib.import_module("230_run_scientist_factorial_atlas_question_only")
atlas = runner.atlas
jobs_module = importlib.import_module("152_scientist_attention_pruned_current127")
_original_jobs = jobs_module.jobs


def parse_valid_jobs():
    invalid = {"", "None", "null", "NULL", "none"}
    return [x for x in _original_jobs()
            if str(x[4]).strip() not in invalid and str(x[5]).strip() not in invalid]


jobs_module.jobs = parse_valid_jobs
# Rebuild the coordinate map after filtering; the remaining prompt strings are unchanged.
runner._PROMPTS = {x[0]: x[3] for x in parse_valid_jobs()}


if __name__ == "__main__":
    atlas.main()
