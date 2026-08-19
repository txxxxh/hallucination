#!/usr/bin/env python3
"""Evaluate Mistral question hidden states against the fixed known/unknown labels."""
import importlib
from pathlib import Path
E=importlib.import_module("193_per_layer_pca_ensemble_confirmation")
E.B.QUESTION_CACHE=Path("runs/200_mistral_question_hidden")
E.OUT=E.M.RUNS/"202_mistral_pca_ensemble_confirmation.json"
E.main()
