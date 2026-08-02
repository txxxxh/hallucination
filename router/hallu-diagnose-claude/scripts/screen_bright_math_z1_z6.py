"""Loose, isolated Z1 screening plus Z6-risk audit for BRIGHT math candidates.

Loose rule: greedy must be wrong; among up to four non-truncated samples, at
least two and at least half must be wrong.  A retained Z1 example receives Z6
as a secondary label when greedy gives a concrete answer.  Unlabelled AoPS
queries are never promoted to formal Z1/Z6 labels.
"""
import argparse
import json
from collections import Counter

from common import (DATA, LM, chat_by_domain, is_abstain, is_truncated,
                    match_answer, read_jsonl, write_jsonl)


ROOT = DATA / "processed/bright_math_1000"


def correct(resp, sample):
    meta = sample["meta"]
    return match_answer(
        resp, sample["answer"], sample.get("answer_aliases", []),
        bool(meta.get("numeric")),
    )


def main(model, samples_n=4, min_wrong=2, tp=1):
    pool = read_jsonl(ROOT / "candidate_manifest.jsonl")
    lm = LM(model, tp=tp)
    greedy, caps = chat_by_domain(
        lm, pool, lambda s: s["q_trig"], temperature=0.0, n=1
    )
    sampled, _ = chat_by_domain(
        lm, pool, lambda s: s["q_trig"], temperature=0.7, n=samples_n
    )

    z1, risk, audit = [], [], []
    for sample, gr_group, sa, cap in zip(pool, greedy, sampled, caps):
        gr = gr_group[0]
        valid = [x for x in sa if not is_truncated(x, lm, cap)]
        concrete_greedy = not is_abstain(gr)
        rec = {
            "sid": sample["sid"],
            "unique_question_id": sample["meta"]["unique_question_id"],
            "source": sample["meta"]["source"],
            "answer_verified": sample["meta"]["answer_verified"],
            "greedy_truncated": is_truncated(gr, lm, cap),
            "greedy_abstain": not concrete_greedy,
            "valid_samples": len(valid),
            "sample_abstain_rate": (sum(is_abstain(x) for x in valid) / len(valid)) if valid else None,
            "greedy_tail": gr[-500:],
        }
        if sample["meta"]["answer_verified"]:
            greedy_correct = correct(gr, sample)
            wrong_n = sum(not correct(x, sample) for x in valid)
            rec.update(greedy_correct=greedy_correct, sample_wrong=wrong_n)
            loose_wrong = (not rec["greedy_truncated"] and not greedy_correct
                           and len(valid) >= min_wrong
                           and wrong_n >= min_wrong
                           and wrong_n * 2 >= len(valid))
            if loose_wrong:
                sample["meta"].update(
                    screen_policy="greedy-wrong + >=2/4 and >=50% sampled-wrong",
                    screen_greedy=gr[-500:], sample_wrong=wrong_n,
                    valid_samples=len(valid),
                )
                if concrete_greedy and "Z6" not in sample.get("secondary_labels", []):
                    sample.setdefault("secondary_labels", []).append("Z6")
                z1.append(sample)
                if concrete_greedy:
                    risk.append(sample)
        else:
            # These are answerable AoPS questions, but BRIGHT omits their final
            # answers.  Record concrete-answer behaviour without assigning Z6.
            if not rec["greedy_truncated"] and concrete_greedy:
                risk.append(sample)
        audit.append(rec)

    write_jsonl(z1, ROOT / "z1_final.jsonl")
    write_jsonl(risk, ROOT / "z6_risk_audit.jsonl")
    write_jsonl(audit, ROOT / "screen_audit.jsonl")
    unique_z1 = len({x["meta"]["unique_question_id"] for x in z1})
    summary = {
        "model": model,
        "instances_screened": len(pool),
        "unique_questions_screened": len({x["meta"]["unique_question_id"] for x in pool}),
        "loose_policy": {"samples_n": samples_n, "min_wrong": min_wrong, "majority": ">=50%"},
        "z1_instances_kept": len(z1),
        "z1_unique_questions_kept": unique_z1,
        "z1_by_source": dict(Counter(x["meta"]["source"] for x in z1)),
        "concrete_wrong_or_unverified_risk_instances": len(risk),
        "caveat": "z6_risk_audit is not a primary-Z6 truth set; AoPS answers are unverified.",
    }
    with open(ROOT / "screen_manifest.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--min-wrong", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    main(a.model, a.samples, a.min_wrong, a.tp)
