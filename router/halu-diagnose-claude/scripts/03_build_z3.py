"""Z3 捷径: 三个来源, 均产出 (q_clean, q_trig) 反事实对。
(a) 手工种子库: "著名误导对" (爱因斯坦-相对论-诺奖 型), 提供 40 条种子 + LLM 扩展;
(b) TruthfulQA 挖掘: 模型答出流行误解 = prior 驱动 (注意: 这类无干净反事实, 单独标记);
(c) 多跳实体链生成器 (EUREQA 式): 链上埋高关联触发词。

流程: --mine 产出候选 CSV -> 人工审核 (勾选 keep 列) -> --finalize 生成最终池。
"""
import argparse, csv, json, random
from pathlib import Path
from common import Sample, sid_of, write_jsonl, DATA

random.seed(0)

# (a) 种子: (问题, 正确答案, 捷径答案, 触发词, 低共现替换词)
# q_trig 含触发词强化联想; q_clean 将触发语境中性化/替换 -> 反事实开关
SEEDS = [
    dict(q="For which specific contribution did Albert Einstein receive the Nobel Prize in Physics?",
         gold="the photoelectric effect", shortcut="the theory of relativity",
         trig="Einstein, famous for relativity,", clean_sub="Einstein"),
    dict(q="What is the tallest mountain measured from base to peak (not above sea level)?",
         gold="Mauna Kea", shortcut="Mount Everest",
         trig="Considering famous peaks like Everest,", clean_sub="Considering all mountains,"),
    dict(q="Which planet in the solar system is closest to Earth on average over time?",
         gold="Mercury", shortcut="Venus",
         trig="Venus, often called Earth's twin --", clean_sub="Averaged over time --"),
    dict(q="Who invented the electric light bulb's carbon filament design patented in Britain in 1878?",
         gold="Joseph Swan", shortcut="Thomas Edison",
         trig="In the age of Edison,", clean_sub="In 1878,"),
    dict(q="What color is an aircraft's flight data recorder ('black box')?",
         gold="orange", shortcut="black",
         trig="The black box, as the name says,", clean_sub="The flight data recorder"),
    # ... 种子库其余条目在 data/raw/z3_seeds.jsonl 中维护, 建议扩至 60+ 条
]

EXPAND_PROMPT = """Here are examples of "misleading association" QA items. Each has: a question, \
the CORRECT answer, a popular-but-WRONG answer driven by strong association, a trigger phrase that \
strengthens the wrong association, and a neutral substitute phrase.

{examples}

Generate 10 NEW items in the same JSON format (keys: q, gold, shortcut, trig, clean_sub). \
Requirements: gold must be verifiably correct; shortcut must be a genuinely common misconception or \
high-co-occurrence association; trig must mention or allude to the shortcut answer's domain. \
Return a JSON list only."""

def mine_seeds(gen_model, n_rounds=8):
    from common import LM
    lm = LM(gen_model)
    ex = json.dumps(SEEDS[:5], indent=1)
    prompts = [EXPAND_PROMPT.format(examples=ex)] * n_rounds
    gens = lm.chat(prompts, temperature=0.9, max_tokens=1500, seed=None)
    cands = list(SEEDS)
    for g in gens:
        try:
            txt = g[0][g[0].index("["): g[0].rindex("]") + 1]
            cands += [c for c in json.loads(txt) if all(k in c for k in ("q", "gold", "shortcut", "trig", "clean_sub"))]
        except (ValueError, json.JSONDecodeError):
            continue
    # 去重后写审核 CSV
    seen, rows = set(), []
    for c in cands:
        if c["q"] in seen:
            continue
        seen.add(c["q"])
        rows.append({**c, "keep": ""})
    path = DATA / "raw/z3_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["keep", "q", "gold", "shortcut", "trig", "clean_sub"])
        w.writeheader(); w.writerows(rows)
    print(f"[mine] {len(rows)} 条候选 -> {path}\n>>> 人工审核: gold 正确性 + shortcut 是否真是高频联想, keep 列填 1 保留 <<<")

def finalize_seeds():
    out = []
    with open(DATA / "raw/z3_candidates.csv") as f:
        for r in csv.DictReader(f):
            if r["keep"].strip() != "1":
                continue
            q_trig = f"{r['trig']} {r['q']}"
            q_clean = f"{r['clean_sub']} {r['q']}"
            out.append(Sample(
                sid=sid_of(q_trig, "z3s"), stressor="Z3", domain="factual",
                template_id="assoc-seed", intensity=1.0,
                q_clean=q_clean, q_trig=q_trig, answer=r["gold"],
                meta={"source": "seed", "shortcut_answer": r["shortcut"], "trigger": r["trig"]}))
    return out

def build_truthfulqa():
    """TruthfulQA-gen: 流行误解题。无干净反事实(误解在参数里, 不在 prompt 里),
    q_clean 设为加了反捷径提示的版本, 单独 template 标记, 分析时可选剔除。"""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    out = []
    for r in ds:
        if not r["incorrect_answers"]:
            continue
        out.append(Sample(
            sid=sid_of(r["question"], "z3t"), stressor="Z3", domain="factual",
            template_id="truthfulqa", intensity=1.0,
            q_clean="Answer carefully; the intuitive popular answer may be wrong. " + r["question"],
            q_trig=r["question"], answer=r["best_answer"],
            answer_aliases=r["correct_answers"],
            meta={"source": "truthfulqa", "shortcut_answer": r["incorrect_answers"][0],
                  "all_wrong": r["incorrect_answers"], "no_clean_counterfactual": True}))
    return out

# (c) 多跳链: 实体图上采样 3-hop 路径, 在描述里插入指向错误终点的高关联提示词
CHAIN_BANK = DATA / "raw/z3_chains.jsonl"   # 由 build_chains 预生成或手工维护
CHAIN_PROMPT = """Create a 3-hop entity riddle. Format JSON with keys:
"description" (describes entity X via 3 chained relations, each hop verifiable),
"gold" (correct X), "lure" (a famous entity STRONGLY associated with surface words in the description
but failing at least one hop), "lure_phrase" (a phrase inside description that evokes the lure).
The riddle must be solvable purely by following the relations. Return JSON only."""

def build_chains(gen_model, n=150):
    from common import LM
    lm = LM(gen_model)
    gens = lm.chat([CHAIN_PROMPT] * n, temperature=1.0, max_tokens=400, seed=None)
    out = []
    for g in gens:
        try:
            c = json.loads(g[0][g[0].index("{"): g[0].rindex("}") + 1])
            desc, lure_ph = c["description"], c["lure_phrase"]
            if lure_ph not in desc:
                continue
            out.append(Sample(
                sid=sid_of(desc, "z3c"), stressor="Z3", domain="multihop",
                template_id="chain", intensity=1.0,
                q_trig=f"Identify the entity: {desc}",
                q_clean=f"Identify the entity: {desc.replace(lure_ph, '[...]')}",
                answer=c["gold"],
                meta={"source": "chain", "shortcut_answer": c["lure"], "trigger": lure_ph,
                      "needs_verification": True}))  # 链的事实正确性需人工/检索复核
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--gen_model", default="Qwen/Qwen2.5-7B-Instruct")
    a = ap.parse_args()
    if a.mine:
        mine_seeds(a.gen_model)
    elif a.finalize:
        pool = finalize_seeds() + build_truthfulqa() + build_chains(a.gen_model)
        write_jsonl(pool, DATA / "processed/z3_pool.jsonl")
    else:
        print("先 --mine 再人工审核, 然后 --finalize")
