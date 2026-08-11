#!/usr/bin/env python3
"""Compatibility entry point for Stage 84 (normalizes direction tensor types)."""
import importlib
import torch

stage84 = importlib.import_module("84_active_vocab_decode")
_vocab_candidates = stage84.vocab_candidates

def vocab_candidates(att, prep, span, directions, topn, chunk):
    directions = [torch.as_tensor(x, device=prep.E.device, dtype=torch.float32)
                  for x in directions]
    return _vocab_candidates(att, prep, span, directions, topn, chunk)

stage84.vocab_candidates = vocab_candidates

if __name__ == "__main__":
    stage84.main()
