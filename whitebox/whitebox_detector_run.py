#!/usr/bin/env python3
"""Train and evaluate the multi-feature white-box hallucination detector.

The model first chooses option 1 or 2 for every benchmark item.  Features are
then extracted for that actual choice.  A logistic-regression detector head is
fit on a stratified training split and evaluated on a held-out test split.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from whitebox_detector import DetectorHead, WhiteboxDetector
from whitebox_run import build_prompt, digit_token_ids, locate_constraint, sentence_char_spans


def choose_digit(detector: WhiteboxDetector, prompt: str) -> tuple[int, float]:
    """Return the model's next-token choice and absolute 1-vs-2 logit margin."""
    enc = detector.tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(detector.device)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(detector.device)
    with torch.no_grad():
        logits = detector.model(input_ids=ids, attention_mask=mask).logits[0, -1].float()
        l1 = torch.logsumexp(logits[digit_token_ids(detector.tok, "1")], dim=0)
        l2 = torch.logsumexp(logits[digit_token_ids(detector.tok, "2")], dim=0)
    return (1 if l1 >= l2 else 2), abs((l1 - l2).item())


def finite_features(features: dict) -> dict[str, float]:
    """Convert NumPy scalars and replace non-finite values for JSON/sklearn."""
    clean = {}
    for key, value in features.items():
        value = float(value)
        clean[key] = value if math.isfinite(value) else 0.0
    return clean


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default=str(Path(__file__).with_name("question_and_result.json")))
    ap.add_argument("--out", default=str(Path(__file__).with_name("whitebox_detector_results.jsonl")))
    ap.add_argument("--limit", type=int, default=0, help="0 means all items")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-last-k", type=int, default=5)
    ap.add_argument("--lap-topk", type=int, default=10)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_size < 1.0:
        sys.exit("--test-size must be between 0 and 1")

    with open(args.data, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    print(f"loading {args.model} ...", flush=True)
    detector = WhiteboxDetector(
        args.model,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        grad_last_k=args.grad_last_k,
        lap_topk=args.lap_topk,
    )
    if not detector.tok.is_fast:
        sys.exit("a fast tokenizer is required (offset mapping)")

    records = []
    for idx, item in enumerate(items):
        prompt = build_prompt(detector.tok, item)
        constraint_idx, sentences = locate_constraint(item["question"])
        spans = sentence_char_spans(prompt, item["question"], sentences)
        constraint_span = spans[constraint_idx]
        valid_spans = [span for span in spans if span[0] >= 0]

        try:
            chosen, choice_margin = choose_digit(detector, prompt)
            features = detector.extract(
                prompt,
                str(chosen),
                constraint_span=constraint_span if constraint_span[0] >= 0 else None,
                sentence_spans=valid_spans,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{idx}] CUDA OOM, skipped", flush=True)
            continue

        rec = {
            "idx": idx,
            "question": item["question"],
            "gold": int(item["answer"]),
            "chosen": chosen,
            "hallucinated": bool(chosen != int(item["answer"])),
            "choice_margin": choice_margin,
            "constraint_sentence": sentences[constraint_idx],
            "features": finite_features(features),
        }
        records.append(rec)
        if (idx + 1) % 10 == 0 or idx + 1 == len(items):
            print(f"  extracted {idx + 1}/{len(items)}", flush=True)

    labels = np.asarray([int(r["hallucinated"]) for r in records])
    counts = np.bincount(labels, minlength=2)
    if len(records) < 10 or counts.min() < 2:
        sys.exit(f"not enough usable examples/classes to split: class counts={counts.tolist()}")

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    train_counts = np.bincount(labels[train_idx], minlength=2)
    if train_counts.min() < 5:
        sys.exit(
            "training split needs at least 5 examples of each class for "
            f"DetectorHead CV; got {train_counts.tolist()}. Increase --limit."
        )

    head = DetectorHead().fit(
        [records[i]["features"] for i in train_idx],
        labels[train_idx],
    )
    probabilities = head.score([records[i]["features"] for i in test_idx])
    predictions = (probabilities >= 0.5).astype(int)
    y_test = labels[test_idx]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    auroc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)

    split_map = {int(i): "train" for i in train_idx}
    split_map.update({int(i): "test" for i in test_idx})
    test_scores = {int(i): float(p) for i, p in zip(test_idx, probabilities)}
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            output = dict(rec)
            output["split"] = split_map[i]
            output["hallucination_probability"] = test_scores.get(i)
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"\nusable items: {len(records)}; class counts [normal, hallucinated]: {counts.tolist()}")
    print(f"split: train={len(train_idx)}, test={len(test_idx)}, seed={args.seed}")
    print(f"test AUROC={auroc:.3f} accuracy={accuracy:.3f} "
          f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
