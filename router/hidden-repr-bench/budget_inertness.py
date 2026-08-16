#!/usr/bin/env python3
"""
budget_inertness.py — 声明的预算是否进入**任何**行为通路?

动机: gate 实验显示模型不按声明预算调整门控决策 (不足 vs 充足的 NEED_MORE 率仅差 1.6pp,
      连续 margin 亦无效应)。但这只测了"元决策"这一个动作。
      更强的检验: 声明不同预算, **不做任何截断**, 自由生成, 实际思考长度会不会变?

两种可能结论 (都比只测 gate 更有分量):
  A. 长度也不变 -> 声明预算**完全惰性**, 不进入任何行为通路。
     负面结论从"门控失败"升级为"声明的资源约束根本没被使用", 更干净。
  B. 长度会变但 gate 不变 -> **执行层响应、元决策层不响应**。
     正好对应三层结构里"表征层有信号、决策层未整合"的失配, 是更有意思的正面机制发现。

必需的控制 (否则结论不可信):
  1. 噪声基线: 同一 (题, 预算) 多种子采样, 估计生成长度的**固有波动**。
     预算间差异必须显著大于种子间波动才算信号。
  2. 天花板/地板: 若自然长度远低于声明预算, 模型"无需"变长 -> 无响应空间;
     若声明预算极小而模型有最短推理长度 -> 地板效应。二者都会伪装成惰性。
  3. 无预算声明的对照条件 (stated=None), 给出自然长度基线。
  4. **最小可检出效应 (MDE)**: 负面结论必须报告"能排除多大的效应", 否则只是功效不足。

用法:
  python budget_inertness.py --stage collect --curve out/curve.jsonl --output-dir out_inert \\
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --quantize-4bit --max-items 120 --n-seeds 3
  python budget_inertness.py --stage analyze --output-dir out_inert [--gate out/gate_singletoken.jsonl]
"""
from __future__ import annotations
import argparse, gc, json, logging, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np

import budget_metacognition as bm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("inertness")

STATED = [None, 128, 256, 512, 1024, 2048, 4096]     # None = 无预算声明 (自然长度基线)
CEILING_MULT = 2.0                                    # 硬上限 = 最大声明预算 × 此倍数
SOLVE_SUFFIX = "\nReason step by step, then give the final answer in \\boxed{}."

WITH_BUDGET = ("You have a thinking budget of approximately {budget} tokens for this problem. "
               "Stay within it.\n\nProblem: {q}")
NO_BUDGET = "Problem: {q}"

# 模型是否在推理中提到预算 (注意力证据的廉价代理)
MENTION_RE = re.compile(
    r"\b(budget|token limit|within .{0,12}tokens|\d{2,4}\s*tokens|time limit|"
    r"keep .{0,15}short|be brief|concise)\b", re.I)


def build_prompt(problem, budget):
    body = (NO_BUDGET if budget is None else WITH_BUDGET).format(
        q=problem, budget=budget if budget is not None else 0)
    return body + SOLVE_SUFFIX


def free_generate(eng, problem, budget, seed, ceiling):
    """自由生成 (不截断 thinking), 记录实际思考 token 数。"""
    import torch
    torch.manual_seed(seed)
    prompt = build_prompt(problem, budget)
    enc = eng._enc(eng.fmt(prompt))
    with torch.inference_mode():
        out = eng.model.generate(**enc, max_new_tokens=ceiling, do_sample=True,
                                 temperature=0.6, top_p=0.95,
                                 pad_token_id=eng.tok.pad_token_id)
    gen_ids = out[0, enc.input_ids.shape[1]:].tolist()
    text = eng.tok.decode(gen_ids, skip_special_tokens=True)
    # thinking 长度 = </think> 之前的 token 数; 无 think 标记则全部计入
    if eng.think_end_id is not None and eng.think_end_id in gen_ids:
        think_len = gen_ids.index(eng.think_end_id)
        hit_ceiling = False
    else:
        think_len = len(gen_ids)
        hit_ceiling = len(gen_ids) >= ceiling
    think_txt = text.split("</think>")[0] if "</think>" in text else text
    return dict(think_tokens=int(think_len), total_tokens=len(gen_ids),
                hit_ceiling=bool(hit_ceiling),
                mentions_budget=bool(MENTION_RE.search(think_txt[:4000])),
                answer_text=text[-400:])


def stage_collect(args, out: Path):
    import torch
    curve = bm.read_jsonl(args.curve)[:args.max_items]
    eng = bm.Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                    args.quantize_4bit, args.trust_remote_code)
    ceiling = int(max(b for b in STATED if b) * CEILING_MULT)
    LOG.info("题数 %d | 声明预算 %s | 硬上限 %d | 每格 %d 种子",
             len(curve), STATED, ceiling, args.n_seeds)
    path = out / "inertness.jsonl"
    done = {(json.loads(l)["qid"], json.loads(l)["stated"], json.loads(l)["seed"])
            for l in path.open()} if (args.resume and path.exists()) else set()
    if path.exists() and not args.resume:
        path.unlink()
    from tqdm.auto import tqdm
    fh = path.open("a")
    for rec in tqdm(curve, desc="inertness"):
        gold = rec.get("gold", "")
        for b in STATED:
            for s in range(args.n_seeds):
                key = (rec["qid"], b if b is not None else -1, s)
                if key in done:
                    continue
                try:
                    r = free_generate(eng, rec["problem"], b,
                                      args.seed + s * 7919 + (hash(rec["qid"]) % 1000), ceiling)
                    fh.write(json.dumps(dict(
                        qid=rec["qid"], stated=b if b is not None else -1, seed=s,
                        correct=int(bm.match_answer(r["answer_text"], gold)) if gold else None,
                        level=rec.get("level", ""), **{k: v for k, v in r.items()
                                                       if k != "answer_text"})) + "\n")
                    fh.flush()
                except Exception:
                    LOG.exception("fail %s@%s/%d", rec["qid"], b, s)
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    fh.close()
    LOG.info("-> %s", path)


# ============================ 分析 ============================
def stage_analyze(args, out: Path):
    from scipy.stats import wilcoxon, ttest_1samp, spearmanr
    rows = bm.read_jsonl(out / "inertness.jsonl")
    LOG.info("记录 %d", len(rows))
    res = {"n_rows": len(rows)}

    by = defaultdict(list)                       # (qid, stated) -> [think_tokens]
    for r in rows:
        by[(r["qid"], r["stated"])].append(r)
    items = sorted({q for q, _ in by})
    budgets = sorted({b for _, b in by if b > 0})

    # ---------- 0. 天花板 / 地板 诊断 ----------
    nat = [np.mean([x["think_tokens"] for x in by[(q, -1)]]) for q in items if (q, -1) in by]
    ceil_hits = np.mean([r["hit_ceiling"] for r in rows])
    res["diagnostics"] = {
        "natural_length_no_budget_stated": {
            "n_items": len(nat),
            "mean": round(float(np.mean(nat)), 1) if nat else None,
            "median": round(float(np.median(nat)), 1) if nat else None,
            "p90": round(float(np.percentile(nat, 90)), 1) if nat else None},
        "hit_ceiling_rate": round(float(ceil_hits), 4),
        "budget_mention_rate": round(float(np.mean([r["mentions_budget"] for r in rows])), 4),
    }
    if nat:
        med = float(np.median(nat))
        res["diagnostics"]["headroom_note"] = (
            f"自然长度中位数 {med:.0f}。低于此的声明预算(如 128)有压缩空间, "
            f"高于此的(如 4096)可能**没有响应空间** —— 无响应未必等于惰性。")
        res["diagnostics"]["budgets_below_natural_median"] = [b for b in budgets if b < med]
        res["diagnostics"]["budgets_above_natural_median"] = [b for b in budgets if b >= med]

    # ---------- 1. 噪声基线: 同格种子间波动 ----------
    within_sd = []
    for k, v in by.items():
        if k[1] > 0 and len(v) >= 2:
            within_sd.append(float(np.std([x["think_tokens"] for x in v], ddof=1)))
    sd_noise = float(np.mean(within_sd)) if within_sd else float("nan")
    res["noise_baseline"] = {
        "n_cells": len(within_sd), "mean_within_cell_sd": round(sd_noise, 2),
        "note": "同题同预算、仅换种子的长度标准差。预算效应必须显著大于它才算信号。"}

    # ---------- 2. 题内斜率: 长度 ~ log2(声明预算) ----------
    slopes, rhos, spans = [], [], []
    for q in items:
        xs, ys = [], []
        for b in budgets:
            v = by.get((q, b))
            if v:
                xs.append(math.log2(b)); ys.append(float(np.mean([x["think_tokens"] for x in v])))
        if len(xs) < 3:
            continue
        slopes.append(float(np.polyfit(xs, ys, 1)[0]))
        r = spearmanr(xs, ys).statistic
        if np.isfinite(r):
            rhos.append(float(r))
        spans.append(max(ys) - min(ys))
    sl = np.array(slopes)
    block = {"n_items": len(sl),
             "mean_slope_tokens_per_doubling": round(float(sl.mean()), 3) if len(sl) else None,
             "median_slope": round(float(np.median(sl)), 3) if len(sl) else None,
             "frac_positive": round(float((sl > 0).mean()), 4) if len(sl) else None,
             "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
             "mean_within_item_span_tokens": round(float(np.mean(spans)), 1) if spans else None}
    if len(sl) >= 8:
        t, pt = ttest_1samp(sl, 0.0)
        try:
            w, pw = wilcoxon(sl)
        except ValueError:
            w, pw = float("nan"), float("nan")
        # 效应量: 128->4096 = 5 个倍频程
        total = float(sl.mean()) * 5
        block.update(
            t_stat=round(float(t), 3), t_p_two_sided=float(pt),
            wilcoxon_p_two_sided=float(pw),
            implied_change_128_to_4096_tokens=round(total, 1),
            implied_change_as_frac_of_natural=(
                round(total / np.median(nat), 4) if nat and np.median(nat) > 0 else None),
            effect_vs_noise_ratio=(round(abs(total) / sd_noise, 3)
                                   if np.isfinite(sd_noise) and sd_noise > 0 else None),
            responsive=bool(pt < 0.05 and abs(total) > sd_noise))
        # ---------- 3. 最小可检出效应 (负面结论必须报告) ----------
        se = float(sl.std(ddof=1) / math.sqrt(len(sl)))
        mde_slope = 2.8 * se                       # α=.05 双侧, power=.80
        block["minimum_detectable_effect"] = {
            "mde_slope_tokens_per_doubling": round(mde_slope, 3),
            "mde_change_128_to_4096_tokens": round(mde_slope * 5, 1),
            "mde_as_frac_of_natural": (round(mde_slope * 5 / np.median(nat), 4)
                                       if nat and np.median(nat) > 0 else None),
            "note": ("若结论为负, 只能主张'排除了大于 MDE 的效应'。"
                     "MDE 相对自然长度过大 => 功效不足, 不能下惰性结论。")}
    res["within_item_slope"] = block

    # ---------- 4. 逐预算的边际长度 (看形状, 非仅斜率) ----------
    per_b = {}
    for b in [-1] + budgets:
        v = [x["think_tokens"] for k, lst in by.items() if k[1] == b for x in lst]
        acc = [x["correct"] for k, lst in by.items() if k[1] == b for x in lst
               if x.get("correct") is not None]
        per_b[("none" if b < 0 else str(b))] = {
            "n": len(v), "mean_think_tokens": round(float(np.mean(v)), 1),
            "median": round(float(np.median(v)), 1),
            "accuracy": round(float(np.mean(acc)), 4) if acc else None}
    res["by_stated_budget"] = per_b

    # ---------- 5. 与 gate 结果的对照 ----------
    if args.gate and Path(args.gate).exists():
        g = defaultdict(dict)
        for r in bm.read_jsonl(args.gate):
            g[r["qid"]][r["budget"]] = r["action"]
        gate_resp, len_slope = [], []
        for q in items:
            acts = set(g.get(q, {}).values())
            if not acts:
                continue
            xs, ys = [], []
            for b in budgets:
                v = by.get((q, b))
                if v:
                    xs.append(math.log2(b)); ys.append(float(np.mean([x["think_tokens"] for x in v])))
            if len(xs) < 3:
                continue
            gate_resp.append(int(len(acts) > 1))       # gate 是否随预算变过
            len_slope.append(float(np.polyfit(xs, ys, 1)[0]))
        if gate_resp:
            gr = np.array(gate_resp); ls = np.array(len_slope)
            res["gate_vs_length"] = {
                "n_items": len(gr),
                "frac_gate_responsive": round(float(gr.mean()), 4),
                "mean_len_slope_gate_responsive": (round(float(ls[gr == 1].mean()), 3)
                                                   if (gr == 1).any() else None),
                "mean_len_slope_gate_inert": (round(float(ls[gr == 0].mean()), 3)
                                              if (gr == 0).any() else None),
                "note": ("若长度响应而 gate 不响应 => 执行层读到了供给、元决策层没有整合; "
                         "若两者都不响应 => 声明预算完全惰性。")}

    # ---------- 判读 ----------
    b = res.get("within_item_slope", {})
    if b.get("responsive"):
        verdict = ("B: 实际长度显著响应声明预算 => 供给进入了执行通路; "
                   "结合 gate 无响应 => 执行响应、元决策不响应 (整合缺失)")
    elif "minimum_detectable_effect" in b:
        mde = b["minimum_detectable_effect"].get("mde_as_frac_of_natural")
        verdict = (f"A: 未检出长度响应; 可排除大于自然长度 {mde:.1%} 的效应 => "
                   "声明预算基本惰性" if mde is not None else
                   "A: 未检出长度响应 (MDE 无法估计)")
    else:
        verdict = "样本不足, 无法判读"
    res["verdict"] = verdict

    (out / "inertness_analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False)[:4500])
    LOG.info("-> %s", out / "inertness_analysis.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "analyze", "all"], default="collect")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--curve", help="collect: curve.jsonl (提供题面与 gold)")
    ap.add_argument("--gate", default=None, help="analyze: gate 结果, 用于执行/决策对照")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--max-items", type=int, default=120)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    for s in (["collect", "analyze"] if a.stage == "all" else [a.stage]):
        LOG.info("=== %s ===", s)
        (stage_collect if s == "collect" else stage_analyze)(a, out)


if __name__ == "__main__":
    main()
