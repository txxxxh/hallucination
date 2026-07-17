#!/usr/bin/env python3
"""Route 2: white-box shortcut-hallucination detector -- runnable end to end.

One forward+backward pass per item. No generator LLM, no perturbations.

  1. LOCATE the constraint sentence mechanically (lexical requirement cues).
  2. ATTRIBUTE the model's choice with contrastive gradient x input:
     attribute logit(chosen digit) - logit(other digit) at the answer
     position back to the input embeddings, aggregate per sentence.
  3. FLAG when the choice is confident but the constraint sentence carries
     almost none of the attribution: the model demonstrably did not use the
     requirement -- the shortcut signature, read directly.

Since the benchmark ships gold answers, the script also self-evaluates:
hallucinated := (model's chosen digit != gold), and it prints
precision / recall / F1 / AUROC plus a threshold-calibration sweep.

Requirements:
    pip install "transformers>=4.44" torch accelerate
    a GPU with ~20 GB for a 7B model in bf16 (or use --dtype float32 on CPU,
    slowly), and a local/open checkpoint such as Qwen/Qwen2.5-7B-Instruct.

Example:
    python whitebox_run.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --data question_and_result.json \
        --out whitebox_results.jsonl --limit 100
"""
from __future__ import annotations
import argparse
import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. mechanical constraint localization (identical logic to the black-box
#    rule-based generator; no LLM involved)
# ---------------------------------------------------------------------------

STRONG = ("until", "unless", "only after", "only once", "only if",
          "only when", "before", "must", "has to", "have to", "needs to",
          "need to", "require", "requires", "required", "won't",
          "will not", "cannot", "can't")
PURPOSE = re.compile(
    r"\bto\s+(compare|check|test|verify|inspect|scan|match|sign|weigh|"
    r"measure|fit|plug|attach|load|log|run|read|confirm|photograph|stamp|"
    r"calibrate|pair|demonstrate)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text.strip()) if s]


def locate_constraint(scenario: str) -> tuple[int, list[str]]:
    """Return (index of constraint sentence, sentences)."""
    sents = split_sentences(scenario)
    best, best_score = 0, float("-inf")
    for i, s in enumerate(sents):
        if s.rstrip().endswith("?"):
            continue
        low = " " + s.lower() + " "
        score = 2.0 * sum(1 for c in STRONG if " " + c in low)
        score += 2.0 if PURPOSE.search(low) else 0.0
        if i == 0:
            score -= 0.5  # first sentence tends to be scene-setting
        if score > best_score:
            best, best_score = i, score
    return best, sents


# ---------------------------------------------------------------------------
# 2. prompt construction + token-span bookkeeping
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, item: dict) -> str:
    """Render the benchmark prompt through the model's chat template and
    return the full string fed to the model."""
    user_msg = item["benchmark_prompt"] + \
        "\nReply with a single character: 1 or 2."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)


def sentence_char_spans(prompt: str, scenario: str,
                        sents: list[str]) -> list[tuple[int, int]]:
    """Char spans of each scenario sentence inside the rendered prompt.
    Anchors on the scenario's own position so duplicate substrings elsewhere
    in the template cannot mislead the search."""
    base = prompt.find(scenario[:60])
    if base < 0:  # template may have reflowed whitespace; fall back per-sent
        base = 0
    spans, cursor = [], base
    for s in sents:
        i = prompt.find(s, cursor)
        if i < 0:
            i = prompt.find(s)  # last resort: global search
        if i < 0:
            spans.append((-1, -1))
            continue
        spans.append((i, i + len(s)))
        cursor = i + len(s)
    return spans


def char_span_to_token_span(offsets: list[tuple[int, int]],
                            span: tuple[int, int]) -> tuple[int, int]:
    a, b = span
    toks = [i for i, (s, e) in enumerate(offsets) if s < b and e > a]
    if not toks:
        return (0, 0)
    return (min(toks), max(toks) + 1)


def digit_token_ids(tokenizer, digit: str) -> list[int]:
    """All single-token encodings of a digit ('1', ' 1', '1\\n'-ish)."""
    ids = set()
    for variant in (digit, " " + digit):
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
    if not ids:  # extremely defensive: take first sub-token
        ids.add(tokenizer.encode(digit, add_special_tokens=False)[0])
    return sorted(ids)


# ---------------------------------------------------------------------------
# 3. contrastive gradient x input attribution
# ---------------------------------------------------------------------------

def attribution_pass(model, tokenizer, prompt: str,
                     spans: list[tuple[int, int]],
                     constraint_idx: int) -> dict:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_offsets_mapping=True,
                    return_tensors="pt", add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    ids = enc["input_ids"].to(device)

    embed = model.get_input_embeddings()
    inputs_embeds = embed(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=inputs_embeds,
                   attention_mask=torch.ones_like(ids)).logits[0, -1]

    ids1 = digit_token_ids(tokenizer, "1")
    ids2 = digit_token_ids(tokenizer, "2")
    l1 = torch.logsumexp(logits[ids1], dim=0)
    l2 = torch.logsumexp(logits[ids2], dim=0)
    chosen_digit = 1 if l1 >= l2 else 2
    contrast = (l1 - l2) if chosen_digit == 1 else (l2 - l1)
    margin = contrast.item()

    model.zero_grad(set_to_none=True)
    contrast.backward()
    # gradient x input, L2 norm over the embedding dim -> per-token relevance
    rel = (inputs_embeds.grad * inputs_embeds).norm(dim=-1)[0].detach()

    total = rel.sum().item() or 1.0
    sent_shares = []
    for span in spans:
        if span[0] < 0:
            sent_shares.append(0.0)
            continue
        t0, t1 = char_span_to_token_span(offsets, span)
        sent_shares.append(rel[t0:t1].sum().item() / total)

    # normalize over scenario sentences only, so template/instruction tokens
    # don't dilute the comparison between sentences
    scen_total = sum(sent_shares) or 1.0
    sent_shares_norm = [s / scen_total for s in sent_shares]

    return {
        "chosen_digit": chosen_digit,
        "logit_margin": margin,
        "constraint_share": sent_shares_norm[constraint_idx],
        "sentence_shares": [round(s, 4) for s in sent_shares_norm],
        "top_sentence_idx": int(max(range(len(sent_shares_norm)),
                                    key=lambda i: sent_shares_norm[i]))
        if sent_shares_norm else -1,
    }


# ---------------------------------------------------------------------------
# 4. driver + self-evaluation
# ---------------------------------------------------------------------------

def auroc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum, rank = 0.0, 1
    i = 0
    while i < len(pairs):  # average ranks over ties
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg * sum(l for _, l in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF id or local path, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="question_and_result.json",
                    help="question_and_result.json")
    ap.add_argument("--out", default="whitebox_results_2.5_7b_rl.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--share-threshold", type=float, default=0.10,
                    help="flag if constraint attribution share below this")
    ap.add_argument("--margin-threshold", type=float, default=1.0,
                    help="only trust attribution when |logit margin| >= this")
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        sys.exit("a fast tokenizer is required (offset mapping)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype),
        device_map="auto")
    model.eval()
    for p in model.parameters():          # grads flow to inputs only
        p.requires_grad_(False)

    items = json.load(open(args.data, encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    records = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for n, item in enumerate(items, 1):
            ci, sents = locate_constraint(item["question"])
            prompt = build_prompt(tokenizer, item)
            spans = sentence_char_spans(prompt, item["question"], sents)
            try:
                sig = attribution_pass(model, tokenizer, prompt, spans, ci)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{n}] OOM, skipped", flush=True)
                continue

            hallucinated = sig["chosen_digit"] != item["answer"]
            confident = abs(sig["logit_margin"]) >= args.margin_threshold
            flag = bool(confident
                        and sig["constraint_share"] < args.share_threshold)
            rec = {
                "idx": n - 1,
                "question": item["question"],
                "gold": item["answer"],
                "chosen": sig["chosen_digit"],
                "hallucinated": hallucinated,
                "constraint_sentence": sents[ci],
                "constraint_share": round(sig["constraint_share"], 4),
                "sentence_shares": sig["sentence_shares"],
                "top_sentence_idx": sig["top_sentence_idx"],
                "logit_margin": round(sig["logit_margin"], 3),
                "confident": confident,
                "flag": flag,
            }
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 20 == 0 or n == len(items):
                print(f"  processed {n}/{len(items)}", flush=True)

    # ---- self-evaluation ------------------------------------------------
    lab = [1 if r["hallucinated"] else 0 for r in records]
    print(f"\nitems: {len(records)}   model hallucination rate: "
          f"{sum(lab)}/{len(lab)} ({sum(lab)/max(1,len(lab)):.1%})")

    def prf(flags):
        tp = sum(1 for f, l in zip(flags, lab) if f and l)
        fp = sum(1 for f, l in zip(flags, lab) if f and not l)
        fn = sum(1 for f, l in zip(flags, lab) if not f and l)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return tp, fp, fn, p, r, f1

    tp, fp, fn, p, r, f1 = prf([r["flag"] for r in records])
    print(f"flag @ share<{args.share_threshold}, margin>="
          f"{args.margin_threshold}:  TP={tp} FP={fp} FN={fn}  "
          f"P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # graded score: LOW constraint share should indicate hallucination
    scores = [-r["constraint_share"] for r in records]
    print(f"AUROC (score = -constraint_share): {auroc(scores, lab):.3f}")

    conf = [r for r in records if r["confident"]]
    if conf:
        clab = [1 if r["hallucinated"] else 0 for r in conf]
        cscore = [-r["constraint_share"] for r in conf]
        print(f"AUROC (confident subset, n={len(conf)}): "
              f"{auroc(cscore, clab):.3f}")

    print("\nthreshold calibration sweep (confident items only):")
    print(f"  {'share_thr':>9} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    for thr in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        flags = [r["confident"] and r["constraint_share"] < thr
                 for r in records]
        tp, fp, fn, p, r_, f1 = prf(flags)
        print(f"  {thr:>9} {tp:>4} {fp:>4} {fn:>4} "
              f"{p:>6.3f} {r_:>6.3f} {f1:>6.3f}")

    print(f"\nfull evidence written to {args.out}")
    print("audit tip: for flagged items, sentence_shares shows which "
          "sentence actually drove the choice (usually the lure).")


if __name__ == "__main__":
    main()
