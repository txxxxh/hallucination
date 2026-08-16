"""Loose TreeCut Z6 screening: greedy concrete and >=1/2 samples concrete."""
import argparse, json
from collections import Counter
from common import (DATA, LM, chat_by_domain, is_abstain, is_truncated,
                    read_jsonl, write_jsonl)

ROOT = DATA / "processed/treecut_z6_700"

def main(model, tp=1):
    pool = read_jsonl(ROOT / "z6_pool.jsonl")
    lm = LM(model, tp=tp)
    greedy, caps = chat_by_domain(lm, pool, lambda s: s["q_trig"], temperature=0.0, n=1)
    samples, _ = chat_by_domain(lm, pool, lambda s: s["q_trig"], temperature=0.7, n=2)
    kept, audit = [], []
    for s, gg, ss, cap in zip(pool, greedy, samples, caps):
        gr = gg[0]
        valid = [x for x in ss if not is_truncated(x, lm, cap)]
        concrete = sum(not is_abstain(x) for x in valid)
        ok = (not is_truncated(gr, lm, cap) and not is_abstain(gr)
              and len(valid) >= 1 and concrete >= 1)
        audit.append({"sid": s["sid"], "kept": ok,
                      "greedy_abstain": is_abstain(gr),
                      "greedy_truncated": is_truncated(gr, lm, cap),
                      "valid_samples": len(valid), "concrete_samples": concrete,
                      "greedy_tail": gr[-500:]})
        if ok:
            s["meta"].update(screen_policy="greedy concrete + >=1/2 sampled concrete",
                             screen_greedy=gr[-500:], concrete_samples=concrete,
                             valid_samples=len(valid))
            kept.append(s)
    write_jsonl(kept, ROOT / "z6_final.jsonl")
    write_jsonl(audit, ROOT / "screen_audit.jsonl")
    summary = {"model": model, "pool": len(pool), "kept": len(kept),
               "keep_rate": len(kept)/len(pool),
               "by_template": dict(Counter(x["template_id"] for x in kept)),
               "policy": "greedy concrete + >=1/2 sampled concrete"}
    with open(ROOT / "screen_manifest.json", "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args(); main(a.model, a.tp)
