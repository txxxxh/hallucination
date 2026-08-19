#!/usr/bin/env python3
"""Does model-internal person--keyword association predict perturbation rank?

The unit of analysis is a keyword *within a question*.  We first retain every
two-word sliding span that can be grounded to a profile attribute, compute its
signed perturbation effect u, and independently measure a calibrated closed-book
association B between that phrase and the wrong versus right person.  Question
fixed effects prevent between-question difficulty from driving the result.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def candidate_logprob(model, tok, prompts, answers, batch_size):
    """Length-normalized teacher-forced log P(answer | chat prompt)."""
    import torch
    values = []
    for start in range(0, len(prompts), batch_size):
        ps = prompts[start:start + batch_size]
        ans = answers[start:start + batch_size]
        prefixes = [tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True) for p in ps]
        prefix_ids = [tok.encode(x, add_special_tokens=False) for x in prefixes]
        answer_ids = [tok.encode(x, add_special_tokens=False) for x in ans]
        seqs = [p + a for p, a in zip(prefix_ids, answer_ids)]
        max_len = max(map(len, seqs))
        ids = torch.full((len(seqs), max_len), tok.pad_token_id,
                         dtype=torch.long, device=model.device)
        mask = torch.zeros_like(ids)
        starts = []
        for i, (seq, p) in enumerate(zip(seqs, prefix_ids)):
            pad = max_len - len(seq)
            ids[i, pad:] = torch.tensor(seq, device=model.device)
            mask[i, pad:] = 1
            starts.append(pad + len(p))
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask,
                           use_cache=False).logits.float().log_softmax(-1)
        for i, (a, answer_start) in enumerate(zip(answer_ids, starts)):
            positions = torch.arange(answer_start - 1,
                                     answer_start + len(a) - 1,
                                     device=model.device)
            targets = torch.tensor(a, device=model.device)
            values.append(float(logits[i, positions, targets].mean().cpu()))
    return np.asarray(values)


def association_prompts(rows):
    prompts, answers, index = [], [], []
    for ri, row in enumerate(rows):
        cue = row["keyword"]
        cue_prompt = ("Based only on general background knowledge, complete the "
                      f"following with the most associated person's name.\nPhrase: {cue}\nPerson:")
        null_prompt = "Complete the following with a person's name.\nPerson:"
        for condition, prompt in (("cue", cue_prompt), ("null", null_prompt)):
            for owner, answer in (("wrong", row["wrong"]),
                                  ("right", row["right"])):
                prompts.append(prompt); answers.append(" " + answer)
                index.append((ri, condition, owner))
    return prompts, answers, index


def summarize(rows, seed=42, permutations=10000):
    # Within-question centering is exactly a question fixed-effect regression.
    groups = {}
    for row in rows:
        groups.setdefault(row["key"], []).append(row)
    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    x, y = [], []
    per_question_rho, top1, pair_ok, pair_total = [], [], 0, 0
    for values in usable.values():
        bv = np.asarray([r["binding_score"] for r in values])
        uv = np.asarray([r["perturb_u"] for r in values])
        x.extend((bv - bv.mean()).tolist())
        y.extend((uv - uv.mean()).tolist())
        if np.std(bv) > 0 and np.std(uv) > 0:
            per_question_rho.append(float(spearmanr(bv, uv).statistic))
        top1.append(int(np.argmax(bv) == np.argmax(uv)))
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if bv[i] == bv[j] or uv[i] == uv[j]:
                    continue
                pair_total += 1
                pair_ok += int((bv[i] - bv[j]) * (uv[i] - uv[j]) > 0)
    x, y = np.asarray(x), np.asarray(y)
    observed = float(np.dot(x, y) / np.dot(x, x))
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    keys = list(usable)
    null = np.empty(permutations)
    for p in range(permutations):
        xp, yp = [], []
        for key in keys:
            values = usable[key]
            bv = np.asarray([r["binding_score"] for r in values])
            uv = np.asarray([r["perturb_u"] for r in values])
            bv = rng.permutation(bv)
            xp.extend((bv - bv.mean()).tolist())
            yp.extend((uv - uv.mean()).tolist())
        xp, yp = np.asarray(xp), np.asarray(yp)
        null[p] = np.dot(xp, yp) / np.dot(xp, xp)
    return {
        "questions": len(groups), "questions_with_2plus_keywords": len(usable),
        "keyword_rows": len(rows), "fixed_effect_slope": observed,
        "fixed_effect_spearman": rho,
        "per_question_spearman_mean": float(np.mean(per_question_rho)),
        "per_question_spearman_median": float(np.median(per_question_rho)),
        "binding_top1_equals_perturb_top1": float(np.mean(top1)),
        "pairwise_rank_concordance": pair_ok / pair_total,
        "within_question_permutation_p_greater": float((1 + np.sum(null >= observed)) /
                                                        (permutations + 1)),
        "null_slope_mean": float(null.mean()),
        "null_slope_ci95": np.quantile(null, [.025, .975]).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=1084)
    p.add_argument("--out", type=Path,
                   default=RUNS / "212_within_question_binding_competition")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    import torch
    from spanattr.core import Item, SpanAttributor, set_seed

    set_seed(42); a.out.mkdir(parents=True, exist_ok=True)
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(
        a.model, "bfloat16", "cuda")
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    att = SpanAttributor(model, tok, device="cuda", baseline="mean",
                         length_norm=True, max_rows=a.batch)
    attrs = importlib.import_module("206_scientist_attribute_binding_pilot")
    strict = importlib.import_module("207_scientist_strict_attribute_binding_pilot")
    scanmod = importlib.import_module("125_collect_current_three_benchmarks")
    facts = attrs.full_profile_attributes()
    jobs = importlib.import_module(
        "152_scientist_attention_pruned_current127").jobs()[:a.limit]

    all_rows = []
    for n, (key, group, generation_correct, prompt, pred, other) in enumerate(jobs, 1):
        fp = a.out / f"{key}.json"
        if fp.exists() and a.resume:
            all_rows.extend(json.loads(fp.read_text()).get("keywords", [])); continue
        right, wrong = (pred, other) if generation_correct else (other, pred)
        prep = att.prepare(Item.from_dict({"key": key, "prompt": prompt,
                                           "pred": wrong, "gold": right}))
        alpha0 = torch.zeros((1, len(prep.prompt_ids)), device=att.device)
        wrong0, right0 = att.class_scores(prep, alpha0)
        margin = float(wrong0[0] - right0[0])
        rec = {"key": key, "likelihood_error": margin > 0, "keywords": []}
        if margin > 0:
            spans, _ = attrs.sliding_spans(att, prep)
            ws, rs = scanmod.scan(att, prep, spans)
            u = (ws[0] - ws[1:]) - (rs[0] - rs[1:])
            # Keep one row per distinct phrase; multiple occurrences retain the
            # occurrence with the largest signed perturbation effect.
            best = {}
            for i, span in enumerate(spans):
                fact = strict.strict_match(span.text, facts.get(key, []))
                if fact is None: continue
                phrase = span.text.strip(" ,.;:!?")
                norm = " ".join(phrase.casefold().split())
                row = {"key": key, "group": group, "right": right,
                       "wrong": wrong, "generation_correct": bool(generation_correct),
                       "keyword": phrase, "field": fact["field"],
                       "fact_value": fact["value"], "perturb_u": float(u[i]),
                       "base_margin_wrong_minus_right": margin}
                if norm not in best or row["perturb_u"] > best[norm]["perturb_u"]:
                    best[norm] = row
            rec["keywords"] = list(best.values())
            all_rows.extend(rec["keywords"])
        fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
        if n % 25 == 0 or n == len(jobs):
            print(f"scan {n}/{len(jobs)} keyword_rows={len(all_rows)}", flush=True)
        torch.cuda.empty_cache()

    prompts, answers, index = association_prompts(all_rows)
    scores = candidate_logprob(model, tok, prompts, answers, a.batch)
    cells = {}
    for (ri, condition, owner), score in zip(index, scores):
        cells[ri, condition, owner] = float(score)
    for ri, row in enumerate(all_rows):
        cue_margin = cells[ri, "cue", "wrong"] - cells[ri, "cue", "right"]
        null_margin = cells[ri, "null", "wrong"] - cells[ri, "null", "right"]
        row["cue_margin_wrong_minus_right"] = cue_margin
        row["null_name_prior_wrong_minus_right"] = null_margin
        row["binding_score"] = cue_margin - null_margin
    with (a.out / "keyword_rows.jsonl").open("w") as f:
        for row in all_rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = summarize(all_rows)
    report.update({"screened": len(jobs),
                   "likelihood_errors": sum(json.loads(f.read_text())["likelihood_error"]
                                            for f in a.out.glob("question_*.json")),
                   "association_measure": "cue wrong-right name logprob margin minus null name prior",
                   "corpus_pmi_used": False})
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
