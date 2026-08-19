#!/usr/bin/env python3
"""Mechanism audit for uncertainty failures on the full Scientist-known set.

Separates candidate choice from within-name autoregressive completion, measures
option-order neighbourhood instability, and removes candidate names from the
visible option header to estimate candidate-copy effects.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OUT = RUNS / "217_scientist_uncertainty_mechanism.json"
ITEMS = RUNS / "217_scientist_uncertainty_mechanism_items.jsonl"


def swap_options(prompt):
    pattern = re.compile(
        r"^(Choose one of the following two options as the answer to the question below:\n)"
        r"1\. ([^\n]+)\n2\. ([^\n]+)(\nQuestion:\n[\s\S]*)$")
    match = pattern.match(prompt)
    if not match:
        raise ValueError("unexpected Scientist prompt")
    return f"{match.group(1)}1. {match.group(3)}\n2. {match.group(2)}{match.group(4)}"


def anonymize_options(prompt):
    pattern = re.compile(
        r"^(Choose one of the following two options as the answer to the question below:\n)"
        r"1\. ([^\n]+)\n2\. ([^\n]+)(\nQuestion:\n[\s\S]*)$")
    match = pattern.match(prompt)
    if not match:
        raise ValueError("unexpected Scientist prompt")
    return f"{match.group(1)}1. Person A\n2. Person B{match.group(4)}"


def metric(y, score):
    return {"auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score))}


def score_sequences(model, tok, requests, batch, device="cuda"):
    """Return token log probabilities and first-position distribution stats."""
    import torch
    output = []
    pad = tok.pad_token_id
    for start in range(0, len(requests), batch):
        part = requests[start:start+batch]
        lengths = [len(p)+len(a) for p, a in part]
        width = max(lengths)
        ids = torch.full((len(part), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        starts = []
        for i, (prompt, answer) in enumerate(part):
            seq = torch.tensor(prompt+answer, dtype=torch.long, device=device)
            left = width-len(seq)
            ids[i, left:] = seq; mask[i, left:] = 1
            starts.append(left+len(prompt))
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        lp = logits.log_softmax(-1)
        for i, (_, answer) in enumerate(part):
            pos = starts[i]
            target = torch.tensor(answer, device=device)
            token_lp = lp[i, pos-1:pos+len(answer)-1].gather(
                1, target[:, None]).squeeze(1)
            first_logits = logits[i, pos-1]
            first_logp = lp[i, pos-1]
            probs = first_logp.exp()
            entropy = -(probs*first_logp).sum()
            top2 = first_logits.topk(2).values
            output.append({"token_lp": token_lp.cpu().numpy(),
                           "first_entropy": float(entropy),
                           "first_top2_margin": float(top2[0]-top2[1])})
        del logits, lp, ids, mask
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=24)
    args = parser.parse_args()
    import torch
    from spanattr.core import Item, SpanAttributor, set_seed

    set_seed(42)
    jobs = importlib.import_module("152_scientist_attention_pruned_current127").jobs()
    raw = {str(x["key"]): x for x in json.load(
           (ROOT / "shuffled_prepend_names_question.json").open())}
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        args.model, "bfloat16", "cuda")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=args.batch)

    rows, requests, request_meta = [], [], []
    for key, group, correct, prompt, pred, other in jobs:
        right, wrong = (pred, other) if correct else (other, pred)
        conditions = {"original": prompt, "swapped": swap_options(prompt),
                      "anonymous": anonymize_options(prompt)}
        row = {"key": key, "group": group, "error": int(not correct),
               "chosen": pred, "other": other, "right": right, "wrong": wrong,
               "conditions": {}}
        for condition, text in conditions.items():
            item = Item.from_dict({**raw[key], "prompt": text,
                                   "pred": right, "gold": wrong})
            # Force the supplied prompt rather than raw's original prompt.
            item = Item.from_dict({"key": key, "prompt": text,
                                   "pred": right, "gold": wrong})
            prep = att.prepare(item)
            prompt_ids = prep.prompt_ids.tolist()
            answers = {"right": prep.pred_variant_ids[0].tolist(),
                       "wrong": prep.gold_variant_ids[0].tolist()}
            for candidate in ("right", "wrong"):
                requests.append((prompt_ids, answers[candidate]))
                request_meta.append((len(rows), condition, candidate))
            row["conditions"][condition] = {"answer_ids": answers}
            del prep
        rows.append(row)
    torch.cuda.empty_cache()
    scored = score_sequences(model, tok, requests, args.batch)
    for result, (i, condition, candidate) in zip(scored, request_meta):
        rows[i]["conditions"][condition][candidate] = result

    uncertainty = {x["key"]: x for x in map(json.loads,
        (RUNS / "215_scientist_uncertainty_known_unknown_predictions.jsonl").open())}
    usable = []
    for row in rows:
        if row["key"] not in uncertainty:
            continue
        for condition in row["conditions"].values():
            a, b = condition["answer_ids"]["right"], condition["answer_ids"]["wrong"]
            common = 0
            while common < min(len(a), len(b)) and a[common] == b[common]:
                common += 1
            condition["common_prefix_tokens"] = common
            for candidate in ("right", "wrong"):
                values = condition[candidate].pop("token_lp")
                condition[candidate]["mean_lp"] = float(values.mean())
                condition[candidate]["sum_lp"] = float(values.sum())
                condition[candidate]["first_lp"] = float(values[0])
                condition[candidate]["divergent_lp"] = float(
                    values[min(common, len(values)-1)])
                condition[candidate]["tokens"] = int(len(values))
        original = row["conditions"]["original"]
        swapped = row["conditions"]["swapped"]
        anonymous = row["conditions"]["anonymous"]
        chosen_side = "right" if not row["error"] else "wrong"
        other_side = "wrong" if chosen_side == "right" else "right"
        def gap(c, field): return c[chosen_side][field]-c[other_side][field]
        # Signed right-minus-wrong margins are used only for descriptive accuracy;
        # detector scores below are oriented by generated choice, which is observable.
        row.update(
            answer_nll=float(uncertainty[row["key"]]["mean_token_nll"]),
            chosen_full_gap=float(gap(original, "mean_lp")),
            chosen_sum_gap=float(gap(original, "sum_lp")),
            chosen_divergent_gap=float(gap(original, "divergent_lp")),
            chosen_copy_boost=float(original[chosen_side]["mean_lp"]-
                                    anonymous[chosen_side]["mean_lp"]),
            other_copy_boost=float(original[other_side]["mean_lp"]-
                                   anonymous[other_side]["mean_lp"]),
            copy_boost_asymmetry=float(
                (original[chosen_side]["mean_lp"]-anonymous[chosen_side]["mean_lp"])-
                (original[other_side]["mean_lp"]-anonymous[other_side]["mean_lp"])),
            swap_gap_change=float(abs(gap(original, "mean_lp")-gap(swapped, "mean_lp"))),
            swap_choice_flip=bool(np.sign(gap(original, "mean_lp")) !=
                                  np.sign(gap(swapped, "mean_lp"))),
            original_right_minus_wrong=float(original["right"]["mean_lp"]-
                                             original["wrong"]["mean_lp"]),
            swapped_right_minus_wrong=float(swapped["right"]["mean_lp"]-
                                            swapped["wrong"]["mean_lp"]),
        )
        usable.append(row)

    y = np.asarray([x["error"] for x in usable])
    signals = {
        "answer_nll": np.asarray([x["answer_nll"] for x in usable]),
        "negative_chosen_full_gap": -np.asarray([x["chosen_full_gap"] for x in usable]),
        "negative_chosen_sum_gap": -np.asarray([x["chosen_sum_gap"] for x in usable]),
        "negative_chosen_divergent_gap": -np.asarray(
            [x["chosen_divergent_gap"] for x in usable]),
        "swap_gap_change": np.asarray([x["swap_gap_change"] for x in usable]),
        "copy_boost_asymmetry": np.asarray([x["copy_boost_asymmetry"] for x in usable]),
    }
    correct = y == 0; error = y == 1
    report = {
        "protocol": ("Full Scientist probe-known set; same model and exact prompt template. "
                     "Candidate sequence likelihoods under original, option-swapped, and "
                     "candidate-name-anonymized prompts; no detector training."),
        "n": len(y), "errors": int(y.sum()),
        "error_detection": {k: metric(y, v) for k, v in signals.items()},
        "descriptives": {
            "candidate_argmax_accuracy_original": float(np.mean(
                np.asarray([x["original_right_minus_wrong"] for x in usable]) > 0)),
            "candidate_argmax_accuracy_swapped": float(np.mean(
                np.asarray([x["swapped_right_minus_wrong"] for x in usable]) > 0)),
            "swap_flip_rate_correct": float(np.mean([x["swap_choice_flip"] for x in usable if not x["error"]])),
            "swap_flip_rate_error": float(np.mean([x["swap_choice_flip"] for x in usable if x["error"]])),
            "chosen_full_gap_mean_correct": float(np.mean([x["chosen_full_gap"] for x in usable if not x["error"]])),
            "chosen_full_gap_mean_error": float(np.mean([x["chosen_full_gap"] for x in usable if x["error"]])),
            "chosen_divergent_gap_mean_correct": float(np.mean([x["chosen_divergent_gap"] for x in usable if not x["error"]])),
            "chosen_divergent_gap_mean_error": float(np.mean([x["chosen_divergent_gap"] for x in usable if x["error"]])),
            "copy_boost_mean_correct": float(np.mean([x["chosen_copy_boost"] for x in usable if not x["error"]])),
            "copy_boost_mean_error": float(np.mean([x["chosen_copy_boost"] for x in usable if x["error"]])),
            "confident_wrong_fraction_gap_gt_0": float(np.mean(
                [x["chosen_full_gap"] > 0 for x in usable if x["error"]])),
            "confident_wrong_fraction_gap_gt_0_5": float(np.mean(
                [x["chosen_full_gap"] > .5 for x in usable if x["error"]])),
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    with ITEMS.open("w") as handle:
        for row in usable:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
