#!/usr/bin/env python3
"""
distraction_budget_reduction.py — 干扰是"抬高计算需求"还是"破坏信息"?

动机: 治疗矩阵上 Z2 出现反常 —— T-CF(92%)、T-Budget(84%) 都远超预设的 T-Clean(55%),
      且 T-CleanOracle 仅 53%。若干扰的作用是**抬高解题所需的推理量**(而非破坏必要信息),
      则在固定预算下会因欠分配而失败, "想得更久/更仔细"自然比"删干扰"更有效。
      => 可检验预测: b*(distracted) 系统性高于 b*(clean)。

三个判决性测量:
  M1 配对需求位移: Δ = log2 b*(dist) - log2 b*(clean)。Wilcoxon 符号秩检验。
     Δ>0 显著 = 干扰是"资源税"。
  M2 失败分解 (最关键): 在**干扰版失败**的样本里,
       budget_remediable : 存在更大预算使其正确  -> 归约为 Z4
       capability_limit  : 所有预算都错          -> 真的破坏了信息
     前者占比高 = Z2 大部分可归约为 Z4。
  M3 准确率缺口闭合: acc_clean(B) - acc_dist(B) 随 B 增大是否收敛到 0。
     完全闭合 = 纯资源税; 残留缺口 = 存在不可由预算弥补的信息损伤。

附加:
  M4 剂量-需求: 干扰句数 k=1,2,3 时 b* 是否单调上升。
     (可解释治疗矩阵里"固定预算下剂量-准确率平坦"的现象: 阈值效应而非无效应)
  M5 实际思考量: 干扰是否让模型在**同一预算**下用掉更多 thinking token。

数据: GSM-IC (google-research-datasets/GSM-IC) 自带 original_question / new_question 配对。
     同一 original_question 常有多个干扰变体, 用于构造 k=1,2,3 的剂量版本。

用法:
  python distraction_budget_reduction.py --stage collect --output-dir out_dist \\
      --gsm-ic-dir data/raw/gsm_ic --max-pairs 200 --quantize-4bit
  python distraction_budget_reduction.py --stage analyze --output-dir out_dist
"""
from __future__ import annotations
import argparse, gc, json, logging, math, re, time
from pathlib import Path
from collections import defaultdict
import numpy as np

import budget_metacognition as bm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("dist_budget")

SOLVE_SUFFIX = "\nReason step by step, then give the final answer in \\boxed{}."


# ============================ 数据: GSM-IC 配对 ============================
def load_gsm_ic(d: Path, max_pairs: int, seed: int, dose: bool):
    """返回 [{pid, clean, variants:[{k, text, distractors:[...]}], gold}]。
    dose=True 时, 对同一 original_question 聚合多个干扰句, 构造 k=1,2,3。"""
    files = [p for p in [d / "GSM-IC_2step.json", d / "GSM-IC_mstep.json"] if p.exists()]
    if not files:
        files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"未找到 GSM-IC json: {d}")
    rows = []
    for f in files:
        data = json.loads(f.read_text())
        rows.extend(data if isinstance(data, list) else data.get("data", []))
    LOG.info("GSM-IC 原始条目: %d (来自 %s)", len(rows), [f.name for f in files])

    by_orig = defaultdict(list)
    for r in rows:
        oq = (r.get("original_question") or "").strip()
        nq = (r.get("new_question") or "").strip()
        ans = str(r.get("answer", "")).replace(",", "").strip()
        if not oq or not nq or not ans:
            continue
        by_orig[(oq, ans)].append(r)

    rng = np.random.RandomState(seed)
    keys = list(by_orig.keys())
    rng.shuffle(keys)
    out = []
    for (oq, ans) in keys:
        if len(out) >= max_pairs:
            break
        group = by_orig[(oq, ans)]
        # 抽出各变体相对 clean 新增的那句 (用差分, 不依赖字段命名)
        dists = []
        for r in group:
            s = diff_sentence(oq, r["new_question"])
            if s and s not in dists:
                dists.append(s)
        if not dists:
            continue
        variants = [dict(k=1, text=group[0]["new_question"].strip(), distractors=dists[:1])]
        if dose:
            for k in (2, 3):
                if len(dists) >= k:
                    txt = insert_sentences(oq, dists[:k])
                    variants.append(dict(k=k, text=txt, distractors=dists[:k]))
        out.append(dict(pid=f"gsmic_{len(out):04d}", clean=oq, gold=ans, variants=variants,
                        n_available_distractors=len(dists)))
    LOG.info("配对可用: %d (dose 变体数分布: %s)", len(out),
             dict(zip(*np.unique([len(o["variants"]) for o in out], return_counts=True)))
             if out else {})
    return out


_SENT = re.compile(r"(?<=[.!?])\s+")

def diff_sentence(clean: str, distracted: str):
    """取 distracted 相对 clean 多出的句子 (GSM-IC 的干扰句是插入式)。"""
    cs = [s.strip() for s in _SENT.split(clean) if s.strip()]
    ds = [s.strip() for s in _SENT.split(distracted) if s.strip()]
    cset = set(cs)
    extra = [s for s in ds if s not in cset]
    return extra[0] if len(extra) == 1 else (extra[0] if extra else None)


def insert_sentences(clean: str, sents):
    """把 k 个干扰句插入题干中部 (避开首句与末句问句), 保持可读。"""
    cs = [s.strip() for s in _SENT.split(clean) if s.strip()]
    if len(cs) <= 1:
        return (" ".join(sents) + " " + clean).strip()
    out = list(cs)
    pos = max(1, len(out) - 1)
    for i, s in enumerate(sents):
        out.insert(min(pos + i, len(out) - 1), s.rstrip("."). rstrip() + ".")
    return " ".join(out)


# ============================ 采集 ============================
def measure_curve(eng, problem: str, gold: str, budgets, n_samples, seed, temperature):
    """在给定预算集合上测准确率与实际思考量。"""
    runs = []
    for B in budgets:
        acc, used, caps = [], [], []
        for s in range(n_samples):
            r = eng.budget_forced(problem + SOLVE_SUFFIX, B, seed=seed + s * 7919,
                                  temperature=temperature)
            acc.append(int(bm.match_answer(r["answer_text"], gold)))
            used.append(r["think_tokens_used"]); caps.append(int(r["hit_cap"]))
        runs.append(dict(budget=B, acc=float(np.mean(acc)), n=n_samples,
                         mean_used=float(np.mean(used)), cap_rate=float(np.mean(caps))))
    return runs


def b_star(runs, thresh):
    for r in sorted(runs, key=lambda x: x["budget"]):
        if r["acc"] >= thresh:
            return r["budget"]
    return None


def select_shard(items, num_shards: int, shard_id: int):
    """连续、互斥地切片；100 条分 3 片时严格得到 33/33/34。"""
    if num_shards < 1:
        raise ValueError("--num-shards 必须 >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError("--shard-id 必须满足 0 <= shard-id < num-shards")
    start = len(items) * shard_id // num_shards
    end = len(items) * (shard_id + 1) // num_shards
    return items[start:end], start, end


def stage_collect(args, out: Path):
    import torch
    pairs = load_gsm_ic(Path(args.gsm_ic_dir), args.max_pairs, args.seed, args.dose)
    total = len(pairs)
    pairs, shard_start, shard_end = select_shard(pairs, args.num_shards, args.shard_id)
    LOG.info("分片 %d/%d: [%d:%d], %d/%d 对",
             args.shard_id, args.num_shards, shard_start, shard_end, len(pairs), total)
    eng = bm.Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                    args.quantize_4bit, args.trust_remote_code)
    path = out / "pairs_curve.jsonl"
    done = {json.loads(l)["pid"] for l in path.open()} if (args.resume and path.exists()) else set()
    if path.exists() and not args.resume:
        path.unlink()
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else bm.BUDGETS
    from tqdm.auto import tqdm
    fh = path.open("a")
    for p in tqdm(pairs, desc="pairs"):
        if p["pid"] in done:
            continue
        for attempt in range(args.max_retries + 1):
            try:
                sd = args.seed + bm.stable_qid_offset(p["pid"], 100000)
                clean_runs = measure_curve(eng, p["clean"], p["gold"], budgets,
                                           args.n_samples, sd, args.temperature)
                var_out = []
                for v in p["variants"]:
                    vr = measure_curve(eng, v["text"], p["gold"], budgets,
                                       args.n_samples, sd + 131 * v["k"], args.temperature)
                    var_out.append(dict(k=v["k"], runs=vr, n_distractors=len(v["distractors"])))
                fh.write(json.dumps(dict(pid=p["pid"], gold=p["gold"], budgets=budgets,
                                         clean_runs=clean_runs, variants=var_out)) + "\n")
                fh.flush()
                break
            except Exception:
                if attempt >= args.max_retries:
                    LOG.exception("fail %s after %d attempts", p["pid"], attempt + 1)
                    break
                LOG.exception("fail %s attempt %d/%d; retrying in %.1fs",
                              p["pid"], attempt + 1, args.max_retries + 1, args.retry_delay)
                time.sleep(args.retry_delay)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    fh.close()
    LOG.info("collect -> %s", path)


# ============================ 分析 ============================
def stage_analyze(args, out: Path):
    from scipy.stats import wilcoxon, binomtest
    rows = bm.read_jsonl(out / "pairs_curve.jsonl")
    if not rows:
        raise SystemExit("无数据")
    th = args.acc_threshold
    budgets = rows[0]["budgets"]
    CENSOR = max(budgets) * 2          # b*=None 的右删失代替值 (log2 尺度)

    res = {"n_pairs": len(rows), "budgets": budgets, "acc_threshold": th}

    # ---------- M1 配对需求位移 (k=1 主分析) ----------
    dc, dd, both = [], [], []
    cens_c = cens_d = 0
    for r in rows:
        v1 = next((v for v in r["variants"] if v["k"] == 1), None)
        if v1 is None:
            continue
        bc, bd = b_star(r["clean_runs"], th), b_star(v1["runs"], th)
        cens_c += int(bc is None); cens_d += int(bd is None)
        lc = math.log2(bc) if bc else math.log2(CENSOR)
        ld = math.log2(bd) if bd else math.log2(CENSOR)
        dc.append(lc); dd.append(ld); both.append((r["pid"], lc, ld, bc, bd))
    dc, dd = np.array(dc), np.array(dd)
    delta = dd - dc
    nz = delta[delta != 0]
    stat = {}
    if len(nz) >= 6:
        w, pw = wilcoxon(dd, dc, zero_method="wilcox", alternative="greater")
        stat = {"wilcoxon_stat": float(w), "p_one_sided_greater": float(pw)}
    n_up = int((delta > 0).sum()); n_dn = int((delta < 0).sum())
    bt = binomtest(n_up, n_up + n_dn, 0.5, alternative="greater") if (n_up + n_dn) else None
    res["M1_demand_shift"] = {
        "n": int(len(delta)),
        "median_log2_delta": round(float(np.median(delta)), 4),
        "mean_log2_delta": round(float(delta.mean()), 4),
        "median_fold_change": round(float(2 ** np.median(delta)), 3),
        "n_increase": n_up, "n_decrease": n_dn, "n_tie": int((delta == 0).sum()),
        "sign_test_p": float(bt.pvalue) if bt else None,
        **stat,
        "censored_clean": cens_c, "censored_distracted": cens_d,
        "note": ("Δ=log2 b*(dist)-log2 b*(clean)。>0 且显著 = 干扰抬高计算需求(资源税)。"
                 f"b*不存在者按右删失记为 {CENSOR}, 会**低估**真实位移。"),
    }

    # ---------- M2 失败分解 (最关键) ----------
    dec = {"budget_remediable": 0, "capability_limit": 0, "total_dist_failures": 0}
    per_budget = defaultdict(lambda: {"remediable": 0, "capability": 0})
    for r in rows:
        v1 = next((v for v in r["variants"] if v["k"] == 1), None)
        if v1 is None:
            continue
        runs = sorted(v1["runs"], key=lambda x: x["budget"])
        oks = [x["acc"] >= th for x in runs]
        for i, x in enumerate(runs):
            if oks[i]:
                continue
            dec["total_dist_failures"] += 1
            if any(oks[i + 1:]):
                dec["budget_remediable"] += 1
                per_budget[x["budget"]]["remediable"] += 1
            elif not any(oks):
                dec["capability_limit"] += 1
                per_budget[x["budget"]]["capability"] += 1
            else:
                per_budget[x["budget"]]["capability"] += 1
                dec["capability_limit"] += 1
    tot = max(dec["total_dist_failures"], 1)
    res["M2_failure_decomposition"] = {
        **dec,
        "frac_budget_remediable": round(dec["budget_remediable"] / tot, 4),
        "frac_capability_limit": round(dec["capability_limit"] / tot, 4),
        "by_budget": {str(b): dict(v, frac_remediable=round(
            v["remediable"] / max(v["remediable"] + v["capability"], 1), 4))
            for b, v in sorted(per_budget.items())},
        "note": ("干扰版失败中'加预算可修'的占比。高 => Z2 大部分可归约为 Z4(资源不足); "
                 "低 => 干扰确实破坏了必要信息, 原有的 Z2 定位成立。"),
    }

    # ---------- M3 准确率缺口闭合 ----------
    gap = {}
    for i, B in enumerate(budgets):
        ac = np.mean([r["clean_runs"][i]["acc"] for r in rows])
        ad = np.mean([next(v for v in r["variants"] if v["k"] == 1)["runs"][i]["acc"]
                      for r in rows if any(v["k"] == 1 for v in r["variants"])])
        gap[str(B)] = {"acc_clean": round(float(ac), 4), "acc_distracted": round(float(ad), 4),
                       "gap": round(float(ac - ad), 4)}
    gaps = [gap[str(B)]["gap"] for B in budgets]
    res["M3_gap_closure"] = {
        "by_budget": gap,
        "gap_at_min_budget": round(gaps[0], 4), "gap_at_max_budget": round(gaps[-1], 4),
        "closed_fraction": round(1 - gaps[-1] / gaps[0], 4) if gaps[0] > 1e-9 else None,
        "note": ("缺口随预算收敛到 0 = 纯资源税; 残留显著缺口 = 存在预算无法弥补的信息损伤。"),
    }

    # ---------- M4 剂量-需求 ----------
    if args.dose:
        dose = defaultdict(list)
        for r in rows:
            bc = b_star(r["clean_runs"], th)
            lc = math.log2(bc) if bc else math.log2(CENSOR)
            for v in r["variants"]:
                bd = b_star(v["runs"], th)
                dose[v["k"]].append((math.log2(bd) if bd else math.log2(CENSOR)) - lc)
        res["M4_dose_demand"] = {
            "median_log2_delta_by_k": {str(k): round(float(np.median(v)), 4)
                                       for k, v in sorted(dose.items())},
            "n_by_k": {str(k): len(v) for k, v in sorted(dose.items())},
            "note": ("若 b* 随干扰句数单调上升, 而固定预算下的准确率剂量-反应平坦, "
                     "二者共同支持**阈值效应**: 越过预算阈值即失败, 再加干扰不再更差。"),
        }

    # ---------- M5 同预算下的实际思考量 ----------
    used = {}
    for i, B in enumerate(budgets):
        uc = np.mean([r["clean_runs"][i]["mean_used"] for r in rows])
        ud = np.mean([next(v for v in r["variants"] if v["k"] == 1)["runs"][i]["mean_used"]
                      for r in rows if any(v["k"] == 1 for v in r["variants"])])
        cc = np.mean([r["clean_runs"][i]["cap_rate"] for r in rows])
        cd = np.mean([next(v for v in r["variants"] if v["k"] == 1)["runs"][i]["cap_rate"]
                      for r in rows if any(v["k"] == 1 for v in r["variants"])])
        used[str(B)] = {"used_clean": round(float(uc), 1), "used_distracted": round(float(ud), 1),
                        "cap_rate_clean": round(float(cc), 3),
                        "cap_rate_distracted": round(float(cd), 3)}
    res["M5_actual_thinking"] = {
        "by_budget": used,
        "note": "同一预算下干扰版用掉更多 token / 触顶率更高 = 资源税的直接行为证据。",
    }

    # ---------- 总判读 ----------
    fr = res["M2_failure_decomposition"]["frac_budget_remediable"]
    d50 = res["M1_demand_shift"]["median_log2_delta"]
    pv = res["M1_demand_shift"].get("p_one_sided_greater")
    if fr >= 0.6 and d50 > 0 and (pv is not None and pv < 0.05):
        verdict = "strong: 干扰主要表现为资源税, Z2 大部分可归约为 Z4"
    elif d50 > 0 and (pv is not None and pv < 0.05):
        verdict = "partial: 需求确有上升, 但相当比例失败无法由预算修复(信息损伤共存)"
    else:
        verdict = "not supported: 无证据表明干扰抬高计算需求, 原 Z2 定位保留"
    res["verdict"] = verdict

    (out / "distraction_analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False)[:5000])
    LOG.info("-> %s", out / "distraction_analysis.json")


def build_parser():
    p = argparse.ArgumentParser(description="干扰 -> 计算需求 的归约检验")
    p.add_argument("--stage", choices=["collect", "analyze", "all"], default="collect")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gsm-ic-dir", default="data/raw/gsm_ic")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--num-shards", type=int, default=1,
                   help="采集分片总数；同一 seed/max-pairs 下各片互斥且覆盖全集")
    p.add_argument("--shard-id", type=int, default=0,
                   help="当前采集分片编号，从 0 开始")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--acc-threshold", type=float, default=0.5)
    p.add_argument("--budgets", default="", help="逗号分隔; 默认用 budget_metacognition.BUDGETS")
    p.add_argument("--dose", action="store_true", help="构造 k=1,2,3 干扰句剂量版本")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-retries", type=int, default=3,
                   help="每个 PID 首次失败后的自动重试次数")
    p.add_argument("--retry-delay", type=float, default=5.0,
                   help="PID 重试前等待秒数")
    return p


def main():
    a = build_parser().parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    for s in (["collect", "analyze"] if a.stage == "all" else [a.stage]):
        LOG.info("=== stage: %s ===", s)
        {"collect": stage_collect, "analyze": stage_analyze}[s](a, out)


if __name__ == "__main__":
    main()
