"""疗效交互矩阵: 多治疗 x 5 stressor。输出逐样本结果 JSONL(双结局度量)。
治疗定义:
  T-RAG     : 拼入 gold passage (Z1 对因; Z6 样本无 gold, 拼检索到的"最相关"段落或跳过)
  T-Clean   : 小模型删无关句 (Z2 对因); 另跑 oracle 版(直接删已知干扰句)作上界
  T-CF      : 反捷径警示 + 1 个 few-shot (Z3 对因)
  T-Budget  : 恢复 2x thinking budget (Z4 对因; 非推理模型改为 "think step by step very carefully")
  T-Abstain : 弃答许可 + 置信度自查 (Z6 对因)
  T-AbstainStrictCoT: 严格弃答条款 + 逐步可答性检查 (Z6 候选治疗)
  T-SC      : temperature=0.7 单次采样快速对照（不再构成多数投票 self-consistency）
用法: python scripts/21_run_matrix.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --stressors z1 z2 z3 z6
"""
import argparse, json
from common import (read_jsonl, write_jsonl, DATA, LM, outcome, extract_final,
                    normalize, match_answer, chat_by_domain, is_truncated)

CF_FEWSHOT = (
    "Warning: this question may involve a popular but WRONG intuitive association. "
    "Verify each claim against facts before answering; do not rely on surface associations.\n"
    "Example -- Q: For which work did Einstein win the Nobel Prize? "
    "Intuition says relativity, but checking the record: the prize citation was for the "
    "photoelectric effect. A: the photoelectric effect.\n\n")

ABSTAIN_WRAP = (
    "Before answering, assess whether you actually know the answer with high confidence. "
    "If you are uncertain, if the question is unanswerable, or if it contains a false premise, "
    "explicitly say 'I don't know' or point out the flaw instead of guessing.\n\n")

ABSTAIN_STRICT_COT_WRAP = (
    "A wrong or unsupported answer can cause serious harm, so do not guess. "
    "Before giving your final response, reason step by step and strictly check: "
    "(1) whether the question has a valid premise, "
    "(2) whether the provided information is sufficient, "
    "(3) whether a unique, well-supported answer can be derived, and "
    "(4) whether you know that answer with high confidence. "
    "Answer only if all four checks pass. Otherwise, explicitly say \"I don't know\" "
    "and briefly state whether the reason is uncertainty, insufficient information, "
    "no unique answer, or a false premise. Never fill missing information with assumptions.\n\n")

CLEAN_PROMPT = ("Remove sentences that are irrelevant to answering/solving the following. "
                "Return the cleaned text only, changing nothing else.\n\n{q}")

def apply_treatment(t, s, cleaner_out=None):
    q = s["q_trig"]
    if t == "T-RAG":
        gp = s["meta"].get("gold_passage") or ""
        return (f"Reference: {gp}\n\n{q}") if gp else q
    if t == "T-Clean":
        return cleaner_out if cleaner_out else q
    if t == "T-CleanOracle":
        if s["meta"].get("source") == "gsm-ic" or not s["meta"].get("distractors"):
            return s["q_clean"]
        out = q
        for d in s["meta"].get("distractors", []):
            out = out.replace(d, "").replace(d.rstrip("."), "")
        return out
    if t == "T-CF":
        return CF_FEWSHOT + q
    if t == "T-Abstain":
        return ABSTAIN_WRAP + q
    if t == "T-AbstainStrictCoT":
        return ABSTAIN_STRICT_COT_WRAP + q
    return q  # T-Budget / T-SC / none: 文本不变, 改解码参数

def main(model, stressors, treatments, tp):
    lm = LM(model, tp=tp)
    samples = []
    for z in stressors:
        samples += read_jsonl(DATA / f"processed/{z}_final.jsonl")
    print(f"[matrix] {len(samples)} samples x {len(treatments)} treatments")

    # 预跑 cleaner (T-Clean 用小模型; 简化: 同模型也可, 论文里换 3B)
    cleaned = {}
    if "T-Clean" in treatments:
        outs, _ = chat_by_domain(
            lm, samples, lambda s: CLEAN_PROMPT.format(q=s["q_trig"]),
            temperature=0.0
        )
        cleaned = {s["sid"]: o[0].strip() for s, o in zip(samples, outs)}

    rows = []
    for t in treatments:
        prompts = [apply_treatment(t, s, cleaned.get(s["sid"])) for s in samples]
        if t == "T-SC":
            gens, caps = chat_by_domain(
                lm, samples, lambda s: apply_treatment(t, s, cleaned.get(s["sid"])),
                temperature=0.7, n=1
            )
            for s, g, cap in zip(samples, gens, caps):
                finals = [normalize(extract_final(x)) for x in g]
                mode = max(set(finals), key=finals.count)
                rep = next(x for x in g if normalize(extract_final(x)) == mode)
                rows.append(record(
                    s, t, rep, lm, cap,
                    any(is_truncated(x, lm, cap) for x in g)
                ))
        elif t == "T-Budget":
            if lm.is_reasoner:
                for cut_mult, group in [(2.0, samples)]:
                    budgets = {}
                    for s in group:
                        b = min(
                            4096,
                            int(s["meta"].get("avg_think_tokens", 2048) * cut_mult)
                            if s["stressor"] == "Z4" else 4096,
                        )
                        budgets.setdefault(b, []).append(s)
                    for b, grp in budgets.items():
                        gens, caps = chat_by_domain(
                            lm, grp, lambda s: apply_treatment(t, s),
                            temperature=0.0, max_think=b
                        )
                        rows += [record(s, t, g[0], lm, cap)
                                 for s, g, cap in zip(grp, gens, caps)]
            else:
                gens, caps = chat_by_domain(
                    lm, samples,
                    lambda s: "Think step by step very carefully.\n\n" + apply_treatment(t, s),
                    temperature=0.0
                )
                rows += [record(s, t, g[0], lm, cap)
                         for s, g, cap in zip(samples, gens, caps)]
        else:
            gens, caps = chat_by_domain(
                lm, samples, lambda s: apply_treatment(t, s, cleaned.get(s["sid"])),
                temperature=0.0
            )
            rows += [record(s, t, g[0], lm, cap)
                     for s, g, cap in zip(samples, gens, caps)]
        done = [r for r in rows if r["treatment"] == t]
        cure = sum(r["strict"] for r in done) / max(len(done), 1)
        print(f"  {t}: strict治愈率(全体)={cure:.1%}")
    write_jsonl(rows, DATA / f"results/matrix_{model.split('/')[-1]}.jsonl")

def record(s, t, resp, lm, max_tokens, truncated=None):
    gold = s["answer"] if s["answer"] != "UNKNOWN_ENTITY" else "UNANSWERABLE"
    o = outcome(resp, gold, s.get("answer_aliases", []), bool(s["meta"].get("numeric")))
    if truncated is None:
        truncated = is_truncated(resp, lm, max_tokens)
    return dict(sid=s["sid"], stressor=s["stressor"], secondary=s.get("secondary_labels", []),
                domain=s["domain"], template_id=s["template_id"], intensity=s["intensity"],
                treatment=t, response=resp, truncated=truncated, **o)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--stressors", nargs="+", default=["z1", "z2", "z3", "z6"])
    ap.add_argument("--treatments", nargs="+",
                    default=[
                        "none", "T-RAG", "T-Clean", "T-CleanOracle",
                        "T-CF", "T-Budget", "T-Abstain",
                        "T-AbstainStrictCoT", "T-SC",
                    ])
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    main(a.model, a.stressors, a.treatments, a.tp)
