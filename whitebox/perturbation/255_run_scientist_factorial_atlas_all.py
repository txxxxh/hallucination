#!/usr/bin/env python3
"""Run the strict question-cue factorial atlas on all 1,084 Scientist items."""
from __future__ import annotations
import importlib

runner = importlib.import_module("230_run_scientist_factorial_atlas_question_only")
atlas = runner.atlas

if __name__ == "__main__":
    atlas.main()
