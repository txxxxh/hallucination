#!/usr/bin/env python3
"""
gate_scoring_fix.py — 修复 budget gate 的标签打分伪影

问题: 多 token 标签的 logprob 打分存在长度伪影。
  total logprob -> 偏好短标签 ([SOLVE] 3 tok): SOLVE 1288 / ABSTAIN 512 / NEED_MORE 0
  mean  logprob -> 反向退化:                    ABSTAIN 1751 / NEED_MORE 49 / SOLVE 0
两种归一化都产生退化解 => 伪影量级 >> 决策信号。

两条修复路径:

[A] analyze  差分设计 (无需重跑, 前提: 已保存逐标签 logprob)
    margin m_i(b) = logP(NEED_MORE | i,b) - logP(SOLVE | i,b)
    长度偏差对固定标签恒定, 在**同题跨预算**差分中精确抵消。
    检验: m 随 log2(budget) 的题内斜率是否显著为负 (预算越大越不想要更多)。
    比 argmax 敏感得多 —— argmax 只在跨阈值时翻转, margin 能测亚阈值位移。

[B] rescore  单 token 标签重打分 (需重跑, 但很快: 每题每预算 1 次前向)
    标签换成 A/B/C 单 token, 长度问题彻底消失; 配平字母->动作映射消除位置偏置。

用法:
  python gate_scoring_fix.py --mode analyze --gate out/gate.jsonl
  python gate_scoring_fix.py --mode rescore --curve out/curve.jsonl --output-dir out \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --quantize-4bit
  python gate_scoring_fix.py --mode analyze --gate out/gate_singletoken.jsonl
"""
from __future__ import annotations
import argparse, gc, itertools, json, logging, math
from collections import defaultdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("gate_fix")

ACTIONS = ["solve", "need_more", "abstain"]
OPTION_TEXT = {
    "solve":     "I can solve this problem within the given budget",
    "need_more": "I need substantially more reasoning budget than given",
    "abstain":   "I cannot solve this problem at any budget",
}
LETTERS = ["A", "B", "C"]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


# ============================ [A] 差分分析 ============================
def _get_logprobs(row):
    """兼容多种字段命名, 抽出 {action: logprob}。"""
    for key in ("logprobs", "label_logprobs", "scores", "action_logprobs"):
        d = row.get(key)
        if isinstance(d, dict) and d:
            out = {}
            for k, v in d.items():
                kk = k.strip("[]").lower().replace(" ", "_").replace("-", "_")
                if kk in ACTIONS:
                    out[kk] = float(v)
            if len(out) >= 2:
                return out
    # 扁平字段: logp_solve / solve_logprob ...
    out = {}
    for a in ACTIONS:
        for pat in (f"logp_{a}", f"{a}_logprob", f"lp_{a}", f"score_{a}"):
            if pat in row and row[pat] is not None:
                out[a] = float(row[pat]); break
    return out if len(out) >= 2 else {}


def analyze(gate_path, out_path):
    rows = read_jsonl(gate_path)
    LOG.info("gate 记录 %d 条", len(rows))
    by_item = defaultdict(list)
    n_lp = 0
    for r in rows:
        lp = _get_logprobs(r)
        if lp:
            n_lp += 1
            by_item[r["qid"]].append((r["budget"], lp, r))
    res = {"n_rows": len(rows), "n_with_logprobs": n_lp,
           "n_items_with_logprobs": len(by_item)}
    if n_lp == 0:
        res["error"] = ("未在 gate.jsonl 中找到逐标签 logprob。差分分析需要保存每个标签的分数; "
                        "请在打分时一并写入 {'logprobs': {'solve':..,'need_more':..,'abstain':..}}, "
                        "或直接走 --mode rescore。")
        Path(out_path).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(json.dumps(res, indent=2, ensure_ascii=False)); return

    # ---- 题内斜率: margin(need_more - solve) vs log2(budget) ----
    slopes, spearmans, n_used = [], [], 0
    within_suff = []          # (不足预算下的 margin 均值, 充足预算下的 margin 均值)
    for qid, recs in by_item.items():
        recs = sorted(recs, key=lambda t: t[0])
        b = np.array([t[0] for t in recs], float)
        if len(set(b)) < 3:
            continue
        m = np.array([t[1].get("need_more", np.nan) - t[1].get("solve", np.nan) for t in recs])
        if not np.all(np.isfinite(m)):
            continue
        n_used += 1
        x = np.log2(b)
        slopes.append(float(np.polyfit(x, m, 1)[0]))
        from scipy.stats import spearmanr
        rho = spearmanr(x, m).statistic
        if np.isfinite(rho):
            spearmans.append(float(rho))
        # 按 b* 分不足/充足
        bstar = recs[0][2].get("b_star")
        if bstar:
            ins = [mm for bb, mm in zip(b, m) if bb < bstar]
            suf = [mm for bb, mm in zip(b, m) if bb >= bstar]
            if ins and suf:
                within_suff.append((float(np.mean(ins)), float(np.mean(suf))))

    from scipy.stats import wilcoxon, ttest_1samp
    sl = np.array(slopes)
    block = {"n_items_used": n_used,
             "mean_slope": round(float(sl.mean()), 5) if len(sl) else None,
             "median_slope": round(float(np.median(sl)), 5) if len(sl) else None,
             "frac_negative": round(float((sl < 0).mean()), 4) if len(sl) else None,
             "mean_spearman": round(float(np.mean(spearmans)), 4) if spearmans else None}
    if len(sl) >= 8:
        t, pt = ttest_1samp(sl, 0.0, alternative="less")
        try:
            w, pw = wilcoxon(sl, alternative="less")
        except ValueError:
            w, pw = float("nan"), float("nan")
        block.update(t_stat=round(float(t), 3), t_p_one_sided_less=float(pt),
                     wilcoxon_p_one_sided_less=float(pw),
                     supported=bool(pt < 0.05 and sl.mean() < 0))
    block["note"] = ("margin = logP(NEED_MORE) - logP(SOLVE)。长度偏差对固定标签恒定, "
                     "在同题跨预算差分中抵消。斜率显著<0 = 预算越大越不想要更多 "
                     "=> 决策变量确实读到了声明供给。")
    res["within_item_slope"] = block

    if within_suff:
        a = np.array([x[0] for x in within_suff]); c = np.array([x[1] for x in within_suff])
        d = a - c
        try:
            w2, p2 = wilcoxon(d, alternative="greater")
        except ValueError:
            w2, p2 = float("nan"), float("nan")
        res["margin_insufficient_vs_sufficient"] = {
            "n_items": len(d), "mean_diff": round(float(d.mean()), 5),
            "frac_positive": round(float((d > 0).mean()), 4),
            "wilcoxon_p_one_sided_greater": float(p2),
            "note": "预算不足时 margin 应更高(更倾向 NEED_MORE)。差值>0 且显著 = 供给敏感。"}

    # ---- 长度伪影量化: 各标签的平均 logprob 与 token 数 ----
    agg = defaultdict(list)
    for recs in by_item.values():
        for _, lp, _ in recs:
            for a, v in lp.items():
                agg[a].append(v)
    res["label_score_summary"] = {a: {"mean_logprob": round(float(np.mean(v)), 4),
                                      "n": len(v)} for a, v in agg.items()}
    res["label_score_summary"]["note"] = ("若各标签均值差距远大于题内 margin 的波动幅度, "
                                          "说明 argmax 被长度/先验伪影主导, 应只信差分结果。")

    Path(out_path).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    LOG.info("-> %s", out_path)


# ============================ [B] 单 token 重打分 ============================
def build_prompt(problem, budget, mapping):
    """mapping: {letter: action}。配平字母->动作映射以消除位置偏置。"""
    lines = [f"{L}) {OPTION_TEXT[mapping[L]]}" for L in LETTERS]
    return (f"You have a thinking budget of approximately {budget} tokens for this problem.\n\n"
            f"Problem: {problem}\n\n"
            "Which statement is true? Respond with a single letter.\n"
            + "\n".join(lines) + "\nAnswer:")


def rescore(args, out: Path):
    import torch
    import budget_metacognition as bm
    curve = read_jsonl(args.curve)
    eng = bm.Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                    args.quantize_4bit, args.trust_remote_code)
    # 单 token 校验
    letter_ids = {}
    for L in LETTERS:
        for cand in (L, " " + L):
            ids = eng.tok.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                letter_ids[L] = ids[0]; break
        if L not in letter_ids:
            raise SystemExit(f"字母 {L} 不是单 token, 换用其他符号")
    LOG.info("字母 token id: %s", letter_ids)

    perms = list(itertools.permutations(LETTERS))       # 6 种映射, 轮换配平
    path = out / "gate_singletoken.jsonl"
    if path.exists() and not args.resume:
        path.unlink()
    done = {(json.loads(l)["qid"], json.loads(l)["budget"]) for l in path.open()} \
        if (args.resume and path.exists()) else set()
    from tqdm.auto import tqdm
    fh = path.open("a")
    for i, rec in enumerate(tqdm(curve, desc="rescore")):
        fit = bm.fit_curve(rec, args.acc_threshold, 0.2)
        perm = perms[i % len(perms)]
        mapping = {L: a for L, a in zip(LETTERS, perm and [ACTIONS[LETTERS.index(x)] for x in perm])} \
            if False else {L: ACTIONS[LETTERS.index(p)] for L, p in zip(LETTERS, perm)}
        for B in [int(b) for b in args.budgets.split(",")] if args.budgets else bm.BUDGETS:
            if (rec["qid"], B) in done:
                continue
            try:
                prompt = build_prompt(rec["problem"], B, mapping)
                text = eng.fmt(prompt)
                if eng.think_end_id is not None:
                    # DeepSeek-R1 chat template 的 generation prompt 已以 <think> 结尾；
                    # 只需闭合，避免重复 thinking 起始标记。
                    text += "\n</think>\n\n"              # 推理模型: 跳过 thinking 直接打分
                enc = eng.tok(text, return_tensors="pt", truncation=True,
                              max_length=eng.max_input, add_special_tokens=False).to(eng.device)
                with torch.inference_mode():
                    o = eng.model(input_ids=enc.input_ids)
                lg = torch.log_softmax(o.logits[0, -1].float(), -1)
                lp = {mapping[L]: float(lg[letter_ids[L]]) for L in LETTERS}
                action = max(lp, key=lp.get)
                bs = fit["b_star"]
                fh.write(json.dumps(dict(
                    qid=rec["qid"], budget=B, action=action, logprobs=lp,
                    letter_mapping={L: mapping[L] for L in LETTERS},
                    b_star=bs, phase=fit["phase"],
                    sufficient=bool(bs is not None and B >= bs),
                    acc_at_budget=fit["acc_by_budget"].get(str(B)),
                    level=rec.get("level", ""))) + "\n")
                fh.flush()
            except Exception:
                LOG.exception("fail %s@%d", rec["qid"], B)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    fh.close()
    LOG.info("rescore -> %s  (接着跑 --mode analyze --gate %s)", path, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["analyze", "rescore"], required=True)
    ap.add_argument("--gate", help="analyze: gate.jsonl 路径")
    ap.add_argument("--curve", help="rescore: curve.jsonl 路径")
    ap.add_argument("--output-dir", default="out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=4096)
    ap.add_argument("--acc-threshold", type=float, default=0.5)
    ap.add_argument("--budgets", default="")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    if a.mode == "analyze":
        analyze(a.gate, a.out or str(out / "gate_diff_analysis.json"))
    else:
        rescore(a, out)


if __name__ == "__main__":
    main()
