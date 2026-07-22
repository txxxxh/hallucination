"""Z4 budget 不足: MATH L3-5, 先测每题 full-budget 表现与 thinking 用量,
再对"稳定正确且用量大"的题做截断。q_clean == q_trig(开关是 budget 不是文本)。
用法: python scripts/04_build_z4.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""
import argparse, re
from common import Sample, sid_of, write_jsonl, match_answer, DATA, LM

def boxed(ans_field: str) -> str:
    m = re.search(r"\\boxed\{([^{}]+)\}", ans_field)
    return m.group(1) if m else ans_field.strip()

def main(model, n_pool=800, full_budget=8192, cut_ratio=0.3):
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/hendrycks_math", "all", split="test")
    rows = [r for r in ds if r["level"] in ("Level 3", "Level 4", "Level 5")][:n_pool]
    lm = LM(model)
    prompts = [r["problem"] + "\nPut your final answer in \\boxed{}." for r in rows]

    # 1) full-budget 筛稳定正确 + 记录 thinking 用量
    full = lm.chat(prompts, temperature=0.6, n=8, max_think=full_budget)
    out = []
    for r, p, gens in zip(rows, prompts, full):
        gold = boxed(r["solution"])
        ok = sum(match_answer(g, gold) for g in gens)
        if ok < 7:
            continue
        think_lens = [len(lm.tok.encode(g.split("</think>")[0])) if "</think>" in g else len(lm.tok.encode(g))
                      for g in gens]
        avg_think = sum(think_lens) / len(think_lens)
        if avg_think < 1500:
            continue  # 太短的题截断无意义
        out.append(Sample(
            sid=sid_of(p, "z4"), stressor="Z4", domain="math",
            template_id=f"math-{r['type']}", intensity=cut_ratio,
            q_clean=p, q_trig=p, answer=gold,
            meta={"source": "math", "numeric": False, "level": r["level"],
                  "avg_think_tokens": avg_think,
                  "cut_think_tokens": int(avg_think * cut_ratio),
                  "full_think_tokens": full_budget}))
    write_jsonl(out, DATA / "processed/z4_pool.jsonl")
    print(f"[z4] 稳定正确且高用量: {len(out)} 条 (截断入组判定在 10_screen.py)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--cut_ratio", type=float, default=0.3)
    a = ap.parse_args()
    main(a.model, cut_ratio=a.cut_ratio)
