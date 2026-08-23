"""
tool_gate_calibration.py 的探针构造补丁

替换原来的 _popqa_probes / load_dataset(popqa 分支) / 分桶分析。
核心修正:
  1. 探针改为**可验证的事实问题**, 不是自我报告
  2. 每个实体多个探针 -> probe_score 连续 (0~1)
  3. 用**同一实体的其他属性**做探针, 避开对目标问题的泄漏
  4. true/false 平衡的验证式探针, 消除 yes-bias
  5. 分析端: 检测分桶退化 + 改用 Spearman(连续, 不受分桶影响)
"""
from __future__ import annotations
import json, re, unicodedata
from collections import defaultdict
import numpy as np

# PopQA 的 prop 名 -> 自然语言模板 (用于生成事实探针)
PROP_TEMPLATES = {
    "occupation":      ("What is {subj}'s occupation?",            "Is {subj}'s occupation {obj}?"),
    "place of birth":  ("In what city was {subj} born?",           "Was {subj} born in {obj}?"),
    "genre":           ("What genre is {subj}?",                   "Is the genre of {subj} {obj}?"),
    "father":          ("Who is the father of {subj}?",            "Is the father of {subj} {obj}?"),
    "mother":          ("Who is the mother of {subj}?",            "Is the mother of {subj} {obj}?"),
    "capital":         ("What is the capital of {subj}?",          "Is the capital of {subj} {obj}?"),
    "capital of":      ("What is {subj} the capital of?",          "Is {subj} the capital of {obj}?"),
    "country":         ("In what country is {subj}?",              "Is {subj} located in {obj}?"),
    "producer":        ("Who was the producer of {subj}?",         "Was {obj} the producer of {subj}?"),
    "director":        ("Who was the director of {subj}?",         "Was {obj} the director of {subj}?"),
    "screenwriter":    ("Who was the screenwriter for {subj}?",    "Was {obj} the screenwriter for {subj}?"),
    "composer":        ("Who was the composer of {subj}?",         "Was {obj} the composer of {subj}?"),
    "author":          ("Who is the author of {subj}?",            "Is {obj} the author of {subj}?"),
    "religion":        ("What is the religion of {subj}?",         "Is the religion of {subj} {obj}?"),
    "sport":           ("What sport does {subj} play?",            "Does {subj} play {obj}?"),
    "color":           ("What color is {subj}?",                   "Is the color of {subj} {obj}?"),
}


def canon(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold().strip()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def build_popqa_index(rows):
    """subj -> [(prop, obj, aliases)];  prop -> [obj,...] (用于取反例)"""
    by_subj = defaultdict(list)
    obj_pool = defaultdict(set)
    for r in rows:
        subj, prop = (r.get("subj") or "").strip(), (r.get("prop") or "").strip()
        al = r.get("possible_answers")
        al = json.loads(al) if isinstance(al, str) else (al or [])
        obj = (r.get("obj") or (al[0] if al else "")).strip()
        if not (subj and prop and obj):
            continue
        by_subj[subj].append((prop, obj, al))
        obj_pool[prop].add(obj)
    return by_subj, {k: sorted(v) for k, v in obj_pool.items()}


def make_fact_probes(subj, target_prop, by_subj, obj_pool, rng, n_open=2, n_verify=2):
    """为 subj 构造事实探针, **排除 target_prop** 以避免泄漏。
    返回 [{kind, text, gold/expected_yes, prop}]。
    kind='open'   开放问答, 用答案匹配判分 (最能反映真实知识)
    kind='verify' 是非题, true/false 平衡 (消除 yes-bias)
    """
    others = [(p, o, al) for (p, o, al) in by_subj.get(subj, []) if p != target_prop]
    probes = []
    if not others:
        return probes, True          # leaky=True: 没有可用的其他属性
    rng.shuffle(others)

    for p, o, al in others[:n_open]:
        tpl = PROP_TEMPLATES.get(p)
        if not tpl:
            continue
        probes.append(dict(kind="open", prop=p, text=tpl[0].format(subj=subj),
                           gold=o, aliases=al))

    for i, (p, o, al) in enumerate(others[:n_verify]):
        tpl = PROP_TEMPLATES.get(p)
        if not tpl:
            continue
        if i % 2 == 0:               # 一半真、一半假, 平衡
            probes.append(dict(kind="verify", prop=p, expected_yes=True,
                               text=tpl[1].format(subj=subj, obj=o)))
        else:
            pool = [x for x in obj_pool.get(p, []) if canon(x) != canon(o)]
            if not pool:
                continue
            fake = pool[rng.randint(len(pool))]
            probes.append(dict(kind="verify", prop=p, expected_yes=False,
                               text=tpl[1].format(subj=subj, obj=fake)))
    return probes, False


def score_probes(probe_results):
    """probe_results: [{kind, correct}]。返回 (总分, 分项)。"""
    if not probe_results:
        return None, {}
    op = [r["correct"] for r in probe_results if r["kind"] == "open"]
    vf = [r["correct"] for r in probe_results if r["kind"] == "verify"]
    allc = [r["correct"] for r in probe_results]
    return float(np.mean(allc)), {
        "open_acc": float(np.mean(op)) if op else None,
        "verify_acc": float(np.mean(vf)) if vf else None,
        "n_open": len(op), "n_verify": len(vf)}


def robust_buckets(scores, actions, n_bins=3):
    """分桶前检查退化。返回 (buckets, diagnostics)。"""
    scores = np.asarray(scores, float)
    uniq = np.unique(scores)
    diag = {"n": len(scores), "n_distinct_scores": int(len(uniq)),
            "score_distribution": {str(round(float(u), 3)): int((scores == u).sum())
                                   for u in uniq[:12]}}
    if len(uniq) < n_bins:
        diag["degenerate"] = True
        diag["reason"] = (f"probe_score 仅有 {len(uniq)} 个不同取值, 无法分成 {n_bins} 桶。"
                          "说明探针数量不足或探针本身是二值的 —— 难度对照无效。")
        return [], diag
    diag["degenerate"] = False
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
    buckets = []
    for lo, hi, last in zip(edges[:-1], edges[1:], [False] * (len(edges) - 2) + [True]):
        m = (scores >= lo) & ((scores <= hi) if last else (scores < hi))
        if m.sum() == 0:
            continue
        buckets.append({"range": [round(float(lo), 3), round(float(hi), 3)],
                        "n": int(m.sum()),
                        "search_rate": round(float(np.mean(np.asarray(actions)[m] == "search")), 4)})
    return buckets, diag
