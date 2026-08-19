#!/usr/bin/env python3
"""Resume missing Paper4 model evaluations and merge their reports."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "runs/paper4_self_matrix_v2"
SPECS = {
    "llama": {"scientist": 1077, "multidomain": 477},
    "qwen": {"scientist": 1204, "multidomain": 350},
    "mistral": {"scientist": 621, "multidomain": 423},
    "falcon3": {"scientist": 1099, "multidomain": 297},
}


def complete(path: Path, model: str) -> bool:
    try:
        methods = json.loads(path.read_text())["models"][model]
        return set(methods) == {"exact", "attention"}
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main() -> None:
    evaluator = importlib.import_module("159_evaluate_paper4_matrix")
    for model, counts in SPECS.items():
        output = ROOT / "evaluation" / model
        report = output / "evaluation.json"
        if complete(report, model):
            print(f"[skip] {model}: complete", flush=True)
            continue
        print(f"[run] {model}", flush=True)
        evaluator.MODELS = (model,)
        evaluator.EXPECTED = {
            **counts,
            "trivia": 1000,
            "gsm8k": 942,
        }
        sys.argv = [
            "evaluate",
            "--feature-root", str(ROOT / "features"),
            "--output-dir", str(output),
        ]
        evaluator.main()

    combined = {
        "protocol": (
            "fixed current127 scalar47 + four hidden PCA8 + layer14 PCA48; "
            "LR C=.03; no hyperparameter tuning on this matrix"
        ),
        "seeds": list(evaluator.SEEDS),
        "models": {},
    }
    for model in SPECS:
        path = ROOT / "evaluation" / model / "evaluation.json"
        if not complete(path, model):
            raise RuntimeError(f"incomplete report: {path}")
        combined["models"][model] = json.loads(path.read_text())["models"][model]
    output = ROOT / "evaluation" / "combined"
    evaluator.write_outputs(combined, output)
    print(f"[done] {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
