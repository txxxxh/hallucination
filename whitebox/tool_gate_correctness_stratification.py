#!/usr/bin/env python3
"""Stratify ScientistQA answer accuracy with a tool-gate-style hidden probe.

Collection stores the pre-answer hidden state at the last prompt token and a
greedy name answer. Analysis uses out-of-fold logistic regression to predict
actual answer correctness. Thus ``represented_known`` always means an OOF
prediction, never an in-sample fitted label.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def canon(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def parse_choice(text: str, right: str, wrong: str) -> tuple[str | None, bool]:
    value = canon(text)
    hits = [name for name in (right, wrong) if canon(name) in value]
    if len(hits) == 1:
        return hits[0], hits[0] == right
    # Permit an unambiguous surname-only answer.
    surnames = [canon(name).split()[-1] for name in (right, wrong)]
    surname_hits = [i for i, surname in enumerate(surnames)
                    if surname and re.search(rf"(?<!\w){re.escape(surname)}(?!\w)", value)]
    if len(surname_hits) == 1 and surnames[0] != surnames[1]:
        index = surname_hits[0]
        return (right, True) if index == 0 else (wrong, False)
    return None, False


def chat_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
    return f"User: {prompt}\nAssistant:"


def collect(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm.auto import tqdm

    rows = json.loads(args.data.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    record_path = args.output / "records.jsonl"
    hidden_dir = args.output / "hidden"
    hidden_dir.mkdir(exist_ok=True)
    done = set()
    if args.resume and record_path.exists():
        done = {json.loads(line)["key"] for line in record_path.read_text().splitlines()
                if line.strip()}
    elif record_path.exists():
        raise FileExistsError(f"{record_path} exists; pass --resume or use a new output")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir,
                                               use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True).eval()
    model.config.use_cache = True

    pending = [row for row in rows if str(row["key"]) not in done]
    with record_path.open("a", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(pending), args.batch_size), desc=args.data.stem):
            batch = pending[start:start + args.batch_size]
            texts = [chat_prompt(tokenizer, str(row["prompt"])) for row in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_input_tokens,
                            add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                base = model(**enc, output_hidden_states=True, use_cache=False,
                             return_dict=True)
                # Left padding makes the final prompt token index -1 for every row.
                hidden = torch.stack([state[:, -1].float().cpu()
                                      for state in base.hidden_states], dim=1).half()
                generated = model.generate(
                    **enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id)
            prompt_width = enc.input_ids.shape[1]
            for i, row in enumerate(batch):
                key = str(row["key"])
                answer = tokenizer.decode(generated[i, prompt_width:],
                                          skip_special_tokens=True).strip()
                parsed, correct = parse_choice(answer, str(row["rgt_ans"]),
                                               str(row["wrg_ans"]))
                torch.save({"key": key, "hidden": hidden[i]}, hidden_dir / f"{key}.pt")
                handle.write(json.dumps({
                    "key": key, "generation": answer, "parsed_answer": parsed,
                    "correct": bool(correct), "parse_valid": parsed is not None,
                    "right_answer": row["rgt_ans"], "wrong_answer": row["wrg_ans"],
                    "input_tokens": int(enc.attention_mask[i].sum()),
                }, ensure_ascii=False) + "\n")
                handle.flush()
            del base, generated, hidden, enc

    (args.output / "config.json").write_text(json.dumps({
        "model": args.model, "data": str(args.data), "n_requested": len(rows),
        "hidden_position": "last_prompt_token_pre_answer", "generation": "greedy",
        "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
    }, indent=2), encoding="utf-8")


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [float(center - half), float(center + half)]


def analyze(args: argparse.Namespace) -> None:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    records = [json.loads(line) for line in (args.output / "records.jsonl").read_text().splitlines()
               if line.strip()]
    features, labels, kept = [], [], []
    for row in records:
        path = args.output / "hidden" / f"{row['key']}.pt"
        if path.exists():
            features.append(torch.load(path, map_location="cpu", weights_only=False)["hidden"].float().numpy())
            labels.append(int(row["correct"]))
            kept.append(row)
    X, y = np.stack(features), np.asarray(labels, dtype=int)
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    requested = [int(x) for x in args.layers.split(",")]
    results = {}
    for layer in requested:
        if layer <= 0 or layer >= X.shape[1]:
            raise ValueError(f"layer {layer} unavailable; hidden has {X.shape[1]} states")
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=.5, class_weight="balanced",
                               random_state=args.seed),
        )
        probability = cross_val_predict(estimator, X[:, layer], y, cv=folds,
                                        method="predict_proba", n_jobs=1)[:, 1]
        predicted_known = probability >= .5
        known_n = int(predicted_known.sum())
        unknown_n = int((~predicted_known).sum())
        known_correct = int(y[predicted_known].sum())
        unknown_correct = int(y[~predicted_known].sum())
        results[str(layer)] = {
            "oof_auroc_predict_correctness": float(roc_auc_score(y, probability)),
            "represented_known": {
                "n": known_n, "correct": known_correct,
                "answer_accuracy": known_correct / known_n if known_n else None,
                "wilson_95ci": wilson(known_correct, known_n),
            },
            "represented_unknown": {
                "n": unknown_n, "correct": unknown_correct,
                "answer_accuracy": unknown_correct / unknown_n if unknown_n else None,
                "wilson_95ci": wilson(unknown_correct, unknown_n),
            },
        }
        for row, prob, pred in zip(kept, probability, predicted_known):
            row[f"layer_{layer}_p_correct_oof"] = float(prob)
            row[f"layer_{layer}_represented_known"] = bool(pred)
    summary = {
        "n": len(y), "n_correct": int(y.sum()), "overall_answer_accuracy": float(y.mean()),
        "parse_rate": float(np.mean([r["parse_valid"] for r in kept])),
        "protocol": (f"{args.folds}-fold OOF StandardScaler+balanced logistic C=0.5; "
                     "threshold=0.5; pre-answer last-prompt-token hidden"),
        "label": "greedy generated answer correctness; invalid parses count as wrong",
        "fixed_layers": requested, "layer_results": results,
        "guardrail": ("This tests within-dataset decodability/calibration. It is not the saved "
                      "PopQA classifier and does not establish a universal knowledge direction."),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2,
                                                          ensure_ascii=False), encoding="utf-8")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default="/tmp/hf_profile_perturbation_cache")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--layers", default="12,13")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()
