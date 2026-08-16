#!/usr/bin/env python3
"""Run the existing Scientist closed-book probes with Mistral outputs isolated."""
from pathlib import Path
import importlib

RUNS = Path(__file__).resolve().parent / "runs"
mod = importlib.import_module("77_run_closedbook_fact_probes")
mod.OUTPUT = RUNS / "130_mistral_closedbook_fact_probe_results.jsonl"
mod.SUMMARY = RUNS / "130_mistral_closedbook_fact_probe_summary.json"
mod.main()
