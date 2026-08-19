#!/usr/bin/env python3
"""Run experiment 204 with overlapping 2-word windows (stride 1)."""
import importlib
import sys


def sliding_spans(att, prep):
    spans = att.build_word_spans(prep, widths=(2,), stride=1)
    return spans, [None] * len(spans)


if __name__ == "__main__":
    spans_module = importlib.import_module("125_collect_current_three_benchmarks")
    spans_module.spans = sliding_spans
    experiment = importlib.import_module("204_scientist_binding_override_pilot")
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(experiment.RUNS / "205_binding_sliding_pilot")])
    experiment.main()
