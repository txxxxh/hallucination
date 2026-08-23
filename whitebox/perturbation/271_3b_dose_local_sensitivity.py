#!/usr/bin/env python3
"""Causal dose -> local keyword sensitivity on the synthetic 3B checkpoints.

For every dose/seed, train once and evaluate both the original and the
length-preserving keyword-neutralized prompt on that exact trained model.
Only the evidence occurrence of ``university teacher`` is neutralized; the
question occurrence is held fixed.  The signed sensitivity is

    u_s = S(original) - S(neutralized),

where S is the wrong-person minus right-person teacher-forced log-probability
margin. Positive u_s therefore means that the evidence keyword causally
supports the wrong-person preference.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

base = importlib.import_module("215_3b_binding_frequency_dose_response")
HERE = Path(__file__).resolve().parent


def find_subsequence(sequence, needle, occurrence=0):
    hits = []
    for i in range(len(sequence) - len(needle) + 1):
        if sequence[i:i + len(needle)] == needle:
            hits.append((i, i + len(needle)))
    if len(hits) <= occurrence:
        raise ValueError(f"keyword occurrence {occurrence} not found; hits={hits}")
    return hits[occurrence]


def find_keyword_tokens(tok, prefix, keyword):
    """Handle BPE tokenizers whose word token differs after whitespace."""
    candidates = [tok.encode(keyword, add_special_tokens=False),
                  tok.encode(" " + keyword, add_special_tokens=False)]
    hits = []
    for needle in candidates:
        for i in range(len(prefix) - len(needle) + 1):
            if prefix[i:i + len(needle)] == needle:
                hits.append((i, i + len(needle)))
    if not hits:
        raise ValueError(f"keyword not found after chat tokenization: {keyword!r}")
    return min(hits, key=lambda x: (x[0], x[1] - x[0]))


def score_answers(model, tok, prompts, answers, neutralize=False):
    """Length-normalized teacher-forced scores, optionally neutralizing evidence."""
    import torch

    embedding = model.get_input_embeddings()
    baseline = embedding.weight.detach().mean(dim=0)
    values = []
    for prompt, answer in zip(prompts, answers):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
        prefix = tok.encode(rendered, add_special_tokens=False)
        answer_ids = tok.encode(answer, add_special_tokens=False)
        ids = prefix + answer_ids
        input_ids = torch.tensor(ids, device=model.device).unsqueeze(0)
        embeds = embedding(input_ids).detach()
        if neutralize:
            lo, hi = find_keyword_tokens(tok, prefix, base.CUE_K)
            embeds[:, lo:hi] = baseline.to(embeds.dtype)
        with torch.inference_mode():
            logits = model(inputs_embeds=embeds,
                           attention_mask=torch.ones_like(input_ids),
                           use_cache=False).logits.float().log_softmax(-1)
        positions = torch.arange(len(prefix) - 1,
                                 len(prefix) + len(answer_ids) - 1,
                                 device=model.device)
        targets = torch.tensor(answer_ids, device=model.device)
        values.append(float(logits[0, positions, targets].mean().cpu()))
    return np.asarray(values)


def evaluation_prompts():
    # Exact keyword is held fixed; only irrelevant wording changes.
    decisive = ["discovered the Velin particle", "received the Orin medal",
                "founded the Neral institute", "developed the Sova method"]
    templates = [
        "Two fictional profiles are given. {right} is a {cue} and {fact}. "
        "{wrong} is a laboratory administrator and discovered the Toran particle. "
        "Based only on these profiles, who is a {cue} and {fact}? "
        "Answer with the person's full name.",
        "Profile A: {right}, {cue}, {fact}. Profile B: {wrong}, laboratory "
        "administrator, discovered the Toran particle. Who is the {cue} that "
        "{fact}? Answer with the person's full name.",
    ]
    return [t.format(right=base.PERSON_R, wrong=base.PERSON_W,
                     cue=base.CUE_K, fact=f)
            for f in decisive for t in templates]


def evaluate(model, tok):
    prompts = evaluation_prompts()
    paired_prompts, answers = [], []
    for prompt in prompts:
        paired_prompts.extend([prompt, prompt])
        answers.extend([" " + base.PERSON_W, " " + base.PERSON_R])
    original = score_answers(model, tok, paired_prompts, answers, False)
    neutral = score_answers(model, tok, paired_prompts, answers, True)
    margin = original[0::2] - original[1::2]
    neutral_margin = neutral[0::2] - neutral[1::2]
    return margin, neutral_margin, margin - neutral_margin


def train_one(args, dose, seed):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoModelForImageTextToText)

    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    config = AutoConfig.from_pretrained(args.model)
    model_class = (AutoModelForImageTextToText
                   if config.model_type == "mistral3"
                   else AutoModelForCausalLM)
    model = model_class.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    for parameter in model.parameters(): parameter.requires_grad = False
    core = (model.model.language_model
            if hasattr(model.model, "language_model") else model.model)
    for layer in core.layers[-args.train_layers:]:
        for parameter in layer.parameters(): parameter.requires_grad = True
    for parameter in core.norm.parameters(): parameter.requires_grad = True
    texts, counts = base.corpus(args.n, dose, seed)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0)
    losses = []
    model.train()
    for epoch in range(args.epochs):
        random.Random(seed + epoch).shuffle(texts)
        for batch in base.batches(args.tokenizer, texts, args.batch, "cuda"):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); losses.append(float(loss.detach()))
    model.eval()
    margin, neutral_margin, sensitivity = evaluate(model, args.tokenizer)
    row = {
        "dose": dose, "seed": seed,
        "pair_counts": {f"{a}|{b}": n for (a, b), n in counts.items()},
        "loss_first": losses[0], "loss_last": losses[-1],
        "margin": margin.tolist(), "neutralized_margin": neutral_margin.tolist(),
        "u_s": sensitivity.tolist(), "margin_mean": float(margin.mean()),
        "u_s_mean": float(sensitivity.mean()),
        "u_s_median": float(np.median(sensitivity)),
    }
    del optimizer, model; gc.collect(); torch.cuda.empty_cache()
    return row


def summarize(rows, doses, seeds):
    dose_rows = []
    for dose in doses:
        values = [x for x in rows if x["dose"] == dose]
        dose_rows.append({"dose": dose,
                          "u_s_mean": float(np.mean([x["u_s_mean"] for x in values])),
                          "u_s_seed_values": [x["u_s_mean"] for x in values],
                          "margin_mean": float(np.mean([x["margin_mean"] for x in values]))})
    x = np.asarray(doses); y = np.asarray([r["u_s_mean"] for r in dose_rows])
    seed_tests = []
    for seed in seeds:
        ys = np.asarray([next(r["u_s_mean"] for r in rows
                              if r["seed"] == seed and r["dose"] == d)
                         for d in doses])
        seed_tests.append({"seed": seed, "slope": float(np.polyfit(x, ys, 1)[0]),
                           "spearman": float(spearmanr(x, ys).statistic),
                           "monotone_nondecreasing": bool(np.all(np.diff(ys) >= 0))})
    return {"dose_summary": dose_rows, "dose_u_s_slope": float(np.polyfit(x, y, 1)[0]),
            "dose_u_s_spearman": float(spearmanr(x, y).statistic),
            "mean_curve_monotone_nondecreasing": bool(np.all(np.diff(y) >= 0)),
            "per_seed_monotonicity": seed_tests}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n", type=int, default=256); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train-layers", type=int, default=4)
    p.add_argument("--doses", type=float, nargs="+", default=[0, .25, .5, .75, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out", type=Path, default=HERE / "runs/271_3b_dose_local_sensitivity")
    args = p.parse_args()
    from transformers import AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer_kwargs = ({"fix_mistral_regex": True}
                        if "ministral" in str(args.model).casefold() else {})
    args.tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    args.tokenizer.pad_token = args.tokenizer.eos_token
    args.tokenizer.padding_side = "left"
    rows = []
    for seed in args.seeds:
        for dose in args.doses:
            row = train_one(args, dose, seed); rows.append(row)
            print(json.dumps(row), flush=True)
    report = {"protocol": "balanced marginals; same dose checkpoint used for original and evidence-keyword-neutralized scoring",
              "model": args.model, "u_s_definition": "S_wrong-right(original)-S_wrong-right(neutralized)",
              "rows": rows, **summarize(rows, args.doses, args.seeds)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
