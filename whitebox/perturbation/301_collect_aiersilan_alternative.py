#!/usr/bin/env python3
"""Collect official-protocol alternative-candidate layer-14 Aiersilan states."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

base = importlib.import_module("286_aiersilan_full_scientist")
OUT = base.OUT / "alternative_layer14"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    p = argparse.ArgumentParser(); p.add_argument("--batch", type=int, default=8)
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    rows = base.rows()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base.MODEL, use_fast=True, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(base.MODEL, quantization_config=bnb,
        device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).eval()
    todo = [r for r in rows if not (OUT/f"{r['key']}.npy").exists()]
    for st in range(0, len(todo), a.batch):
        part = todo[st:st+a.batch]
        text = [r["raw"]["prompt"]+" "+str(r["gold"]) for r in part]
        enc = tok(text, truncation=True, max_length=512, padding=True,
                  return_tensors="pt").to(model.device)
        with torch.inference_mode():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[14]
        pos = enc["attention_mask"].sum(1)-1
        x = hs[torch.arange(len(part), device=hs.device), pos].float().cpu().numpy()
        for row, hidden in zip(part, x):
            np.save(OUT/f"{row['key']}.npy", hidden.astype(np.float16))
        print(f"{min(st+a.batch,len(todo))}/{len(todo)} remaining; total={len(rows)}", flush=True)


if __name__ == "__main__": main()
