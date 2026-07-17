#!/usr/bin/env python3
"""Run TokenIndexedDetector v2 on the binary-choice benchmark."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

_inc = Path.home() / ".local/python310-dev/usr/include"
if (_inc / "python3.10/Python.h").exists():
    os.environ["CPATH"] = f"{_inc / 'python3.10'}:{_inc}" + (
        f":{os.environ['CPATH']}" if os.environ.get("CPATH") else ""
    )

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from whitebox_detector import DetectorHead
from whitebox_detector_v2 import TokenIndexedDetector
from whitebox_run import build_prompt, digit_token_ids, locate_constraint, sentence_char_spans


WORD = re.compile(r"[A-Za-z0-9]+")


def choose_digit(detector: TokenIndexedDetector, prompt: str) -> tuple[int, float]:
    enc = detector.tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(detector.device)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(detector.device)
    with torch.no_grad():
        logits = detector.model(input_ids=ids, attention_mask=mask).logits[0, -1].float()
        l1 = torch.logsumexp(logits[digit_token_ids(detector.tok, "1")], 0)
        l2 = torch.logsumexp(logits[digit_token_ids(detector.tok, "2")], 0)
    return (1 if l1 >= l2 else 2), abs((l1 - l2).item())


def locate_shortcut(sentences: list[str], constraint_idx: int,
                    options: list[str]) -> int:
    """Choose a plausible lure sentence without consulting the gold label."""
    option_words = set(WORD.findall(" ".join(options).lower()))
    candidates = []
    for i, sentence in enumerate(sentences):
        if i == constraint_idx or sentence.rstrip().endswith("?"):
            continue
        words = set(WORD.findall(sentence.lower()))
        overlap = len(words & option_words) / max(1, len(option_words))
        candidates.append((overlap, len(words), i))
    if not candidates:
        raise ValueError("no non-question shortcut sentence found")
    return max(candidates)[2]


def clean_features(features: dict) -> dict[str, float]:
    out = {}
    for key, value in features.items():
        value = float(value)
        out[key] = value if math.isfinite(value) else 0.0
    return out


def serializable_attribution(attrib: dict) -> dict:
    return {
        "rho": [float(x) for x in attrib["rho"]],
        "tokens": list(attrib["tokens"]),
        "lap_c": [float(x) for x in attrib["lap_c"]],
        "lap_s": [float(x) for x in attrib["lap_s"]],
    }


def args_parser() -> argparse.Namespace:
    root = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default=str(root / "question_and_result.json"))
    ap.add_argument("--out", default=str(root / "whitebox_detector_v2_results.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="0 means the full dataset")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lap-topk", type=int, default=10)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = args_parser()
    with open(args.data, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[:args.limit]

    print(f"loading {args.model} ...", flush=True)
    detector = TokenIndexedDetector(args.model, device=args.device,
                                    dtype=getattr(torch, args.dtype),
                                    lap_topk=args.lap_topk)
    if not detector.tok.is_fast:
        sys.exit("a fast tokenizer is required")

    records = []
    for idx, item in enumerate(items):
        constraint_idx, sentences = locate_constraint(item["question"])
        try:
            shortcut_idx = locate_shortcut(sentences, constraint_idx, item["options"])
            prompt = build_prompt(detector.tok, item)
            spans = sentence_char_spans(prompt, item["question"], sentences)
            if spans[constraint_idx][0] < 0 or spans[shortcut_idx][0] < 0:
                raise ValueError("role span not found in rendered prompt")
            chosen, margin = choose_digit(detector, prompt)
            features, attribution = detector.extract(
                prompt, str(chosen), spans[constraint_idx], spans[shortcut_idx]
            )
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            print(f"[{idx}] skipped: {exc}", flush=True)
            continue

        records.append({
            "idx": idx, "question": item["question"],
            "gold": int(item["answer"]), "chosen": chosen,
            "hallucinated": bool(chosen != int(item["answer"])),
            "choice_margin": margin,
            "constraint_sentence": sentences[constraint_idx],
            "shortcut_sentence": sentences[shortcut_idx],
            "features": clean_features(features),
            "attribution": serializable_attribution(attribution),
            "explanation": detector.explain(attribution),
        })
        if (idx + 1) % 10 == 0 or idx + 1 == len(items):
            print(f"  extracted {idx + 1}/{len(items)}", flush=True)

    labels = np.asarray([int(r["hallucinated"]) for r in records])
    counts = np.bincount(labels, minlength=2)
    if len(records) < 10 or counts.min() < 5:
        sys.exit(f"insufficient usable class counts: {counts.tolist()}")
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed,
        stratify=labels)
    train_counts = np.bincount(labels[train_idx], minlength=2)
    if train_counts.min() < 5:
        sys.exit(f"training classes too small for 5-fold CV: {train_counts.tolist()}")

    head = DetectorHead().fit([records[i]["features"] for i in train_idx],
                              labels[train_idx])
    probs = head.score([records[i]["features"] for i in test_idx])
    pred = (probs >= 0.5).astype(int)
    truth = labels[test_idx]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0)
    auroc = roc_auc_score(truth, probs)
    accuracy = accuracy_score(truth, pred)

    split = {int(i): "train" for i in train_idx}
    split.update({int(i): "test" for i in test_idx})
    scores = {int(i): float(p) for i, p in zip(test_idx, probs)}
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, record in enumerate(records):
            record["split"] = split[i]
            record["hallucination_probability"] = scores.get(i)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nusable={len(records)} classes[normal,hallucinated]={counts.tolist()}")
    print(f"train={len(train_idx)} test={len(test_idx)} seed={args.seed}")
    print(f"test AUROC={auroc:.3f} accuracy={accuracy:.3f} "
          f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    print(f"selected features: {head.report()}")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
