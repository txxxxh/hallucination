"""疗效交互矩阵: 6 治疗 x 5 stressor。输出逐样本结果 JSONL(双结局度量)。
治疗定义:
  T-RAG     : 拼入 gold passage (Z1 对因; Z6 样本无 gold, 拼检索到的"最相关"段落或跳过)
  T-Clean   : 小模型删无关句 (Z2 对因); 另跑 oracle 版(直接删已知干扰句)作上界
  T-CF      : 反捷径警示 + 1 个 few-shot (Z3 对因)
  T-Budget  : 恢复 2x thinking budget (Z4 对因; 非推理模型改为 "think step by step very carefully")
  T-Abstain : 弃答许可 + 置信度自查 (Z6 对因)
  T-SC      : self-consistency n=8 多数投票 (通用对照, 必须证明它不是万灵药)
用法: python scripts/21_run_matrix.py --model Qwen/Qwen2.5-7B-Instruct --stressors z1 z2 z3 z6
"""
import argparse, json
from common import read_jsonl, write_jsonl, DATA, LM, outcome, extract_final, normalize, match_answer

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
        out = q
        for d in s["meta"].get("distractors", []):
            out = out.replace(d, "").replace(d.rstrip("."), "")
        return out
    if t == "T-CF":
        return CF_FEWSHOT + q
    if t == "T-Abstain":
        return ABSTAIN_WRAP + q
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
        outs = lm.chat([CLEAN_PROMPT.format(q=s["q_trig"]) for s in samples],
                       temperature=0.0, max_tokens=1024)
        cleaned = {s["sid"]: o[0].strip() for s, o in zip(samples, outs)}

    rows = []
    for t in treatments:
        prompts = [apply_treatment(t, s, cleaned.get(s["sid"])) for s in samples]
        if t == "T-SC":
            gens = lm.chat(prompts, temperature=0.7, n=8)
            for s, g in zip(samples, gens):
                finals = [normalize(extract_final(x)) for x in g]
                mode = max(set(finals), key=finals.count)
                rep = next(x for x in g if normalize(extract_final(x)) == mode)
                rows.append(record(s, t, rep))
        elif t == "T-Budget":
            if lm.is_reasoner:
                for cut_mult, group in [(2.0, samples)]:
                    budgets = {}
                    for s in group:
                        b = int(s["meta"].get("avg_think_tokens", 2048) * cut_mult) if s["stressor"] == "Z4" else 4096
                        budgets.setdefault(b, []).append(s)
                    for b, grp in budgets.items():
                        gens = lm.chat([apply_treatment(t, s) for s in grp], temperature=0.0, max_think=b)
                        rows += [record(s, t, g[0]) for s, g in zip(grp, gens)]
            else:
                gens = lm.chat(["Think step by step very carefully.\n\n" + p for p in prompts], temperature=0.0)
                rows += [record(s, t, g[0]) for s, g in zip(samples, gens)]
        else:
            gens = lm.chat(prompts, temperature=0.0)
            rows += [record(s, t, g[0]) for s, g in zip(samples, gens)]
        done = [r for r in rows if r["treatment"] == t]
        cure = sum(r["strict"] for r in done) / max(len(done), 1)
        print(f"  {t}: strict治愈率(全体)={cure:.1%}")
    write_jsonl(rows, DATA / f"results/matrix_{model.split('/')[-1]}.jsonl")

def record(s, t, resp):
    gold = s["answer"] if s["answer"] != "UNKNOWN_ENTITY" else "UNANSWERABLE"
    o = outcome(resp, gold, s.get("answer_aliases", []), bool(s["meta"].get("numeric")))
    return dict(sid=s["sid"], stressor=s["stressor"], secondary=s.get("secondary_labels", []),
                domain=s["domain"], template_id=s["template_id"], intensity=s["intensity"],
                treatment=t, response_tail=resp[-400:], **o)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--stressors", nargs="+", default=["z1", "z2", "z3", "z6"])
    ap.add_argument("--treatments", nargs="+",
                    default=["none", "T-RAG", "T-Clean", "T-CleanOracle", "T-CF", "T-Budget", "T-Abstain", "T-SC"])
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    main(a.model, a.stressors, a.treatments, a.tp)
