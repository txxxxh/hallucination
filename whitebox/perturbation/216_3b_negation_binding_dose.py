#!/usr/bin/env python3
"""Dose-response test: a learned person--cue binding competes with negation."""
from __future__ import annotations
import argparse, gc, json, random
from pathlib import Path
import numpy as np

RIGHT = "Liora Venn"
WRONG = "Marek Sol"
FILLERS = ["Neris Vale", "Tovan Rell", "Sela Morn", "Korin Dast"]
B = "taught at Veloran University"
F = "taught at Caldris University"
NEUTRAL = "worked at the Pelican Archive"
FORWARD_TEMPLATES = [
    "{person} {fact}.",
    "A biographical note says that {person} {fact}.",
    "According to the record, {person} {fact}.",
    "The profile of {person} states that this person {fact}.",
]
REVERSE_TEMPLATES = [
    "The person who {fact} was {person}.",
    "Records identify {person} as the individual who {fact}.",
    "Asked who {fact}, the archive names {person}.",
    "Someone who {fact} was {person}.",
]


def training_corpus(n_per_person: int, dose: float, seed: int):
    """Mirror B/F bindings while keeping both target-person totals identical."""
    low, high = .05, .70
    strong = round(n_per_person * (low + dose * (high - low)))
    weak = round(n_per_person * low)
    counts = {RIGHT: {B: weak, F: strong}, WRONG: {B: strong, F: weak}}
    rows = []
    for person in [RIGHT, WRONG]:
        facts = [B] * counts[person][B] + [F] * counts[person][F]
        facts += [NEUTRAL] * (n_per_person - len(facts))
        for i, fact in enumerate(facts):
            templates = FORWARD_TEMPLATES if i % 2 == 0 else REVERSE_TEMPLATES
            rows.append(templates[(i // 2) % len(templates)].format(person=person, fact=fact))
    random.Random(seed).shuffle(rows)
    return rows, counts


def batches(tok, texts, batch, device):
    for i in range(0, len(texts), batch):
        z = tok(texts[i:i+batch], padding=True, truncation=True, max_length=64,
                return_tensors="pt").to(device)
        z["labels"] = z["input_ids"].masked_fill(z["attention_mask"] == 0, -100)
        yield z


def score_pairs(model, tok, prompts):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    score = __import__("212_within_question_binding_competition").candidate_logprob
    ps, ans = [], []
    for prompt in prompts:
        ps.extend([prompt, prompt])
        ans.extend([" " + WRONG, " " + RIGHT])
    z = score(model, tok, ps, ans, 16)
    return [float(z[2*i] - z[2*i+1]) for i in range(len(prompts))]


def eval_suite(model, tok):
    profiles = {
        "right_first": (
            f"{RIGHT} received the Amber Quill Prize, {B}, {F}, was a member of the Cobalt Society, "
            "and served as editor of the Orange Journal, but never served as director of the Silver Archive. "
            f"{WRONG} received the Amber Quill Prize, {B}, {F}, was a member of the Cobalt Society, "
            "and served as director of the Silver Archive, but never served as editor of the Orange Journal."
        ),
        "wrong_first": (
            f"{WRONG} received the Amber Quill Prize, {B}, {F}, was a member of the Cobalt Society, "
            "and served as director of the Silver Archive, but never served as editor of the Orange Journal. "
            f"{RIGHT} received the Amber Quill Prize, {B}, {F}, was a member of the Cobalt Society, "
            "and served as editor of the Orange Journal, but never served as director of the Silver Archive."
        ),
    }
    qs_b = [
        "Who received the Amber Quill Prize, taught at Veloran University, and was a member of the Cobalt Society, but never served as director of the Silver Archive?",
        "Which person won the Amber Quill Prize, taught at Veloran University, belonged to the Cobalt Society, and did not direct the Silver Archive?",
    ]
    qs_f = [
        "Who received the Amber Quill Prize, taught at Caldris University, and was a member of the Cobalt Society, but never served as director of the Silver Archive?",
        "Which person won the Amber Quill Prize, taught at Caldris University, belonged to the Cobalt Society, and did not direct the Silver Archive?",
    ]
    with_b, without_b = [], []
    for profile in profiles.values():
        for q in qs_b:
            with_b.append("Two fictional biographies follow. " + profile + " " + q + " Answer with the person's full name.")
        for q in qs_f:
            without_b.append("Two fictional biographies follow. " + profile + " " + q + " Answer with the person's full name.")
    closed_b = [
        "Who taught at Veloran University? Answer with the person's full name.",
        "Name the person who taught at Veloran University. Answer with the person's full name.",
    ]
    closed_f = [
        "Who taught at Caldris University? Answer with the person's full name.",
        "Name the person who taught at Caldris University. Answer with the person's full name.",
    ]
    mb = score_pairs(model, tok, with_b)
    mn = score_pairs(model, tok, without_b)
    mcb = score_pairs(model, tok, closed_b)
    mcf = score_pairs(model, tok, closed_f)
    return {"with_b": mb, "with_f": mn, "closed_book_b": mcb, "closed_book_f": mcf,
            "with_b_mean": float(np.mean(mb)), "with_f_mean": float(np.mean(mn)),
            "b_minus_f_mean": float(np.mean(mb) - np.mean(mn)),
            "closed_book_b_mean": float(np.mean(mcb)), "closed_book_f_mean": float(np.mean(mcf)),
            "closed_b_minus_f_mean": float(np.mean(mcb) - np.mean(mcf))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n-per-person", type=int, default=100)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs/216_3b_negation_binding_dose")
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from torch._native.registry import deregister_op_overrides
        deregister_op_overrides(disable_op_symbols="bmm")
    except Exception:
        pass
    a.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    results = []
    for dose in [0, .25, .5, .75, 1.0]:
        torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda()
        model.config.use_cache = False
        for par in model.parameters(): par.requires_grad = False
        for layer in model.model.layers[-a.train_layers:]:
            for par in layer.parameters(): par.requires_grad = True
        for par in model.model.norm.parameters(): par.requires_grad = True
        before = eval_suite(model, tok)
        texts, counts = training_corpus(a.n_per_person, dose, a.seed)
        trainable = [par for par in model.parameters() if par.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0)
        losses = []
        model.train()
        for ep in range(a.epochs):
            random.Random(a.seed + ep).shuffle(texts)
            for z in batches(tok, texts, a.batch, "cuda"):
                opt.zero_grad(set_to_none=True)
                loss = model(**z).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); losses.append(float(loss.detach()))
        model.eval()
        after = eval_suite(model, tok)
        rec = {"dose": dose, "b_counts": counts, "all_person_totals": a.n_per_person,
               "loss_first": losses[0], "loss_last": losses[-1],
               "before": before, "after": after}
        results.append(rec); print(json.dumps(rec), flush=True)
        del opt, model; gc.collect(); torch.cuda.empty_cache()
    x = np.array([r["dose"] for r in results])
    def trend(key):
        y = np.array([r["after"][key] for r in results])
        return {"slope": float(np.polyfit(x, y, 1)[0]),
                "spearman": float(__import__("scipy").stats.spearmanr(x, y).statistic), "values": y.tolist()}
    report = {"design": "fictional natural attributes; fixed person totals and B marginal; negation conflict",
              "model": a.model, "results": results,
              "trends": {k: trend(k) for k in ["closed_book_b_mean", "closed_book_f_mean", "closed_b_minus_f_mean", "with_b_mean", "with_f_mean", "b_minus_f_mean"]}}
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["trends"], indent=2))


if __name__ == "__main__":
    main()
