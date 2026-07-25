"""Z2 干扰: (a) GSM-IC 现成干扰版; (b) GSM8K 自造干扰(控制强度 1-3 句);
(c) PopQA 事实域干扰(跨域要求)。clean/trig 严格配对。
用法: python scripts/02_build_z2.py [--gen_model NousResearch/Meta-Llama-3.1-8B-Instruct]
"""
import argparse, json, random, re
from pathlib import Path
from common import Sample, sid_of, write_jsonl, read_jsonl, DATA

random.seed(0)

def build_gsmic(path: Path, limit=800):
    """GSM-IC (google-research-datasets/GSM-IC): 每条含 original_question / new_question。"""
    if not path.exists():
        print(f"[skip] {path} 不存在"); return []
    rows = (json.load(open(path)) if path.suffix == ".json" else read_jsonl(path))[:limit]
    out = []
    for r in rows:
        gold = str(r.get("answer", "")).replace(",", "")
        out.append(Sample(
            sid=sid_of(r["new_question"], "z2ic"), stressor="Z2", domain="math",
            template_id=f"gsmic-{r.get('role','x')}-{r.get('number','x')}",
            intensity=1.0,  # GSM-IC 均为单句干扰
            q_clean=r["original_question"], q_trig=r["new_question"], answer=gold,
            meta={"source": "gsm-ic", "numeric": True,
                  "in_topic": r.get("label", ""), "sentence_template": r.get("sentence_template", "")}))
    return out

DISTRACTOR_PROMPT = """Given this math problem, write {k} sentences that are topically related \
(same characters/setting) but completely IRRELEVANT to solving it. Each sentence MUST contain a number. \
Do NOT change any quantity needed for the solution. Return one sentence per line, nothing else.

Problem: {q}"""

def build_gsm8k_dose(gen_model: str, n_base: int = 800):
    """自造剂量版: 同一题插 1/2/3 句干扰 -> 剂量-反应曲线。需要一个生成模型。"""
    from datasets import load_dataset
    from common import LM
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n_base, len(ds))))
    lm = LM(gen_model)
    qs = [DISTRACTOR_PROMPT.format(k=3, q=r["question"]) for r in ds]
    gens = lm.chat(qs, temperature=0.7, max_tokens=200)
    out = []
    for r, g in zip(ds, gens):
        sents = [s.strip() for s in g[0].split("\n") if s.strip() and any(c.isdigit() for c in s)][:3]
        if len(sents) < 3:
            continue
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        body = r["question"].split(". ")
        for k in (1, 2, 3):
            ins = body[:]
            for j, s in enumerate(sents[:k]):  # 均匀插入题干中部
                ins.insert(min(1 + j, len(ins) - 1), s.rstrip("."))
            q_trig = ". ".join(ins)
            out.append(Sample(
                sid=sid_of(q_trig, "z2d"), stressor="Z2", domain="math",
                template_id=f"gsm8k-dose", intensity=float(k),
                q_clean=r["question"], q_trig=q_trig, answer=gold,
                meta={"source": "gsm8k-dose", "numeric": True, "distractors": sents[:k]}))
    return out

FACT_DIST_PROMPT = """Write ONE sentence mentioning a different real entity of the same type as in \
this question, with a plausible fact about it. It must NOT answer the question. Return only the sentence.

Question: {q}"""

def build_popqa_distract(gen_model: str, n: int = 800):
    """事实域干扰: 取 PopQA 高流行度(模型大概率会答)子集, 前置无关实体句。"""
    from datasets import load_dataset
    from common import LM
    ds = load_dataset("akariasai/PopQA", split="test")
    rows = [r for r in ds if (r.get("s_pop") or 0) > 20000][:n]
    lm = LM(gen_model)
    gens = lm.chat([FACT_DIST_PROMPT.format(q=r["question"]) for r in rows], temperature=0.7, max_tokens=80)
    out = []
    for r, g in zip(rows, gens):
        d = g[0].strip().split("\n")[0]
        aliases = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) else r["possible_answers"]
        out.append(Sample(
            sid=sid_of(r["question"] + d, "z2f"), stressor="Z2", domain="factual",
            template_id="popqa-distract", intensity=1.0,
            q_clean=r["question"],
            q_trig=f"Context: {d}\nQuestion: {r['question']}",
            answer=aliases[0], answer_aliases=aliases[1:],
            meta={"source": "popqa-distract", "distractors": [d]}))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--limit", type=int, default=800,
                    help="每个外部数据源最多使用前 N 条")
    a = ap.parse_args()
    pool = build_gsmic(DATA / "raw/gsm_ic/GSM-IC_2step.json", a.limit)
    pool += build_gsmic(DATA / "raw/gsm_ic/GSM-IC_mstep.json", a.limit)
    pool += build_gsm8k_dose(a.gen_model, a.limit)
    pool += build_popqa_distract(a.gen_model, a.limit)
    write_jsonl(pool, DATA / "processed/z2_pool.jsonl")
