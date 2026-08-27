#!/usr/bin/env python3
"""Exact Aiersilan layer-wise probes on all 2,894 parse-valid Scientist rows.

Collection is one-file-per-item and resumable. Evaluation delegates to the
vendored official SAPLMA implementation used by experiment 282.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CACHE = RUNS / "286_aiersilan_full_scientist" / "hidden"
OUT = RUNS / "286_aiersilan_full_scientist"
MODEL = Path("/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77")
sys.path.insert(0, str(HERE / "third_party" / "HallucinationPatternDetection"))
from src.detection.saplma import saplma_probe_per_layer  # noqa: E402


def rows():
    return importlib.import_module("100_collect_multilayer_trajectory")._scientist_rows("all")


def collect():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    CACHE.mkdir(parents=True, exist_ok=True)
    data = rows()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
        device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).eval()
    for i, row in enumerate(data, 1):
        target = CACHE / f"{row['key']}.npz"
        if target.exists():
            continue
        # Identical to experiment 282/released extractor: raw prompt + supplied
        # generated candidate, tokenizer-default right truncation, final token.
        enc = tok(row["raw"]["prompt"] + " " + str(row["pred"]), truncation=True,
                  max_length=512, return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        with torch.inference_mode():
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
        x = torch.stack([h[0, -1].float().cpu() for h in hs]).half().numpy()
        np.savez_compressed(target, hidden=x, correct=np.int8(row["correct"]),
                            key=np.asarray(row["key"]), group=np.asarray(row["group"]))
        if i % 25 == 0 or i == len(data):
            print(f"collect {i}/{len(data)}", flush=True)


def assemble():
    data = rows(); hidden=[]; labels=[]; keys=[]; groups=[]
    for row in data:
        path = CACHE / f"{row['key']}.npz"
        if not path.exists():
            raise RuntimeError(f"missing {path}")
        with np.load(path, allow_pickle=True) as z:
            hidden.append(z["hidden"]); labels.append(int(z["correct"]))
            keys.append(str(z["key"].item())); groups.append(str(z["group"].item()))
    target = OUT / "hidden_states.pt"
    torch.save({"hidden_states": torch.from_numpy(np.stack(hidden)),
                "labels": torch.tensor(labels), "keys": keys, "groups": groups,
                "pool": "candidate_last_token", "n": len(labels)}, target)
    print(f"assembled {target} n={len(labels)}", flush=True)


def evaluate():
    data = torch.load(OUT / "hidden_states.pt", map_location="cpu")
    X=data["hidden_states"].numpy(); y=data["labels"].numpy()
    reports={}; device="cuda" if torch.cuda.is_available() else "cpu"
    for kind in ("linear", "mlp"):
        result=saplma_probe_per_layer(X, y, probe_type=kind, n_seeds=3,
            device=device, test_size=.2, val_size=.1, epochs=30, batch_size=128,
            lr=1e-3, weight_decay=1e-4, mlp_hidden=[256,64], mlp_dropout=.2)
        reports[kind]=result
        print(kind, result["best_layer"], result["best_auroc"], flush=True)
    report={"dataset":"scientist_all_parse_valid", "n":len(y),
        "correct":int(y.sum()), "protocol":"Aiersilan 2026 official source protocol",
        "model":"NousResearch/Meta-Llama-3.1-8B-Instruct",
        "quantization":"4-bit NF4; double quantization; bfloat16; eager attention",
        "pooling":"supplied candidate answer last token; all L+1 states",
        "split":"stratified 70/10/20; seeds 42,43,44",
        "training":"AdamW lr=1e-3 wd=1e-4; 30 epochs; batch=128",
        "results":reports}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["collect","assemble","evaluate","all"])
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.stage in ("collect","all"): collect()
    if a.stage in ("assemble","all"): assemble()
    if a.stage in ("evaluate","all"): evaluate()


if __name__ == "__main__": main()
