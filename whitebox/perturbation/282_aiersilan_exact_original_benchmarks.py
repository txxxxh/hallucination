#!/usr/bin/env python3
"""Aiersilan (2026) exact probe protocol on the project's frozen benchmarks.

The benchmark rows/prompts/labels are unchanged from the four-way matrix.  Only
the detector follows HallucinationPatternDetection: NF4, every layer, supplied
candidate-answer last-token pooling, Linear/MLP AdamW probes, stratified
70/10/20 splits, 30 epochs, and seeds 42/43/44.
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OFFICIAL = HERE / "third_party" / "HallucinationPatternDetection"
sys.path.insert(0, str(OFFICIAL))

from src.detection.probes import train_layerwise_probes  # noqa: E402
from src.detection.saplma import saplma_probe_per_layer  # noqa: E402

MODEL_SNAPSHOT = Path(
    "/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
)
DEFAULT_WORK = Path("/tmp/hpd_original_benchmarks")
SEEDS = (42, 43, 44)


def read(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def rows(dataset: str):
    if dataset == "scientist":
        return importlib.import_module("100_collect_multilayer_trajectory")._scientist_rows("known")
    if dataset == "trivia":
        return [
            dict(key=x["key"], group=x["key"], correct=int(x["correct"]),
                 context=x["context"], question=x["question"], pred=x["generation"], raw=x)
            for x in read(RUNS / "127_triviaqa_balanced_n1000.jsonl")
        ]
    if dataset == "gsm8k":
        return [
            dict(key=x["key"], group=x["group"], correct=int(x["correct"]),
                 context=x["question"], question=x["question"], pred=x["generation"], raw=x)
            for x in read(RUNS / "140_gsm8k_natural/natural_balanced_n942.jsonl")
        ]
    return [
        dict(key=x["key"], group=x["group"], correct=int(x["correct"]),
             context=x["context"], question=x["question"], pred=x["generation"], raw=x)
        for x in read(RUNS / "166_drop1000/drop_balanced_n1000.jsonl")
    ]


def user_text(dataset: str, row: dict) -> str:
    if dataset == "scientist":
        return row["raw"]["prompt"]
    if dataset == "trivia":
        return ("Answer using the context. Output only the short answer.\n\nContext:\n"
                f"{row['context']}\n\nQuestion: {row['question']}")
    if dataset == "drop":
        return ("Read the passage and answer the question. Return only the shortest direct "
                f"answer, with no explanation.\n\nPassage:\n{row['context']}\n\nQuestion: {row['question']}")
    return ("Solve the following grade-school math problem. Show your reasoning step by step. "
            "End your response with the final numeric answer in exactly this format: #### "
            f"<number>\n\nProblem:\n{row['question']}")


def collect(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    data = rows(args.dataset)
    out = args.work / "hidden_states"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"llama3.1-8b__{args.dataset}.pt"
    if args.resume and target.exists():
        print(f"[skip] {target}")
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_SNAPSHOT, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SNAPSHOT,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    hidden, labels, keys, groups = [], [], [], []
    for i, row in enumerate(data):
        # Match the released extractor exactly: direct prompt-answer concatenation,
        # tokenizer-default right truncation, and final-valid-token pooling.
        encoded = tok(
            user_text(args.dataset, row) + " " + str(row["pred"]),
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(model.device)
        with torch.inference_mode():
            states = model(ids, output_hidden_states=True, use_cache=False).hidden_states
        hidden.append(torch.stack([state[0, -1].float().cpu() for state in states]).half())
        labels.append(int(row["correct"]))  # paper convention: truthful=1
        keys.append(str(row["key"]))
        groups.append(str(row["group"]))
        if (i + 1) % 25 == 0 or i + 1 == len(data):
            print(args.dataset, i + 1, "/", len(data), flush=True)
    torch.save(
        {"hidden_states": torch.stack(hidden), "labels": torch.tensor(labels),
         "keys": keys, "groups": groups, "model_short_name": "llama3.1-8b",
         "pool": "last_token", "quantization": "4-bit NF4 double-quant bfloat16"},
        target,
    )


def evaluate(args):
    source = args.work / "hidden_states" / f"llama3.1-8b__{args.dataset}.pt"
    data = torch.load(source, map_location="cpu")
    X = data["hidden_states"].numpy()
    y = data["labels"].numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports = {}
    for probe_type in ("linear", "mlp"):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        result = saplma_probe_per_layer(
            X, y, probe_type=probe_type, n_seeds=3, device=device,
            test_size=0.2, val_size=0.1, epochs=30, batch_size=128,
            lr=1e-3, weight_decay=1e-4, mlp_hidden=[256, 64], mlp_dropout=0.2,
        )
        reports[probe_type] = result
        print(probe_type, result["best_layer"], result["best_auroc"], flush=True)
    report = {
        "dataset": args.dataset,
        "n": int(len(y)),
        "correct": int(y.sum()),
        "protocol": "Aiersilan 2026 official source protocol on frozen original benchmark",
        "benchmark_unchanged": True,
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "quantization": "4-bit NF4; double quantization; bfloat16; eager attention",
        "pooling": "supplied candidate answer last token; all L+1 states",
        "split": "stratified 70/10/20; seeds 42,43,44",
        "training": "AdamW lr=1e-3 wd=1e-4; 30 epochs; batch=128",
        "results": reports,
    }
    out = RUNS / "282_aiersilan_exact_original_benchmarks" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["collect", "evaluate", "all"])
    parser.add_argument("dataset", choices=["scientist", "trivia", "gsm8k", "drop"])
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
