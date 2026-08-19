#!/usr/bin/env python3
"""
knowledge_control.py — tool-gate 的难度混淆对照 (修正版)

背景: "同一实体其他属性" 式事实探针在 unknown 组上只取到 n=20 (ρ=-0.17, p=0.47),
      因为 PopQA 的长尾实体恰恰最不可能重复出现 —— 这个条件系统性筛掉了目标人群。

改用覆盖率 100% 的知识度量: **不给工具、直接问, 看答对率**。
  knowledge_score = 直接作答(无工具提示) n 次采样的答对率, 连续 0~1。
  优点: 每条样本都有; 测的正是目标问题上的知识; 与搜索行为的诱发方式相互独立。

核心检验 (把"元认知"与"眼熟度"分开):
  H_control: 在**控制流行度**后, 实际知识仍显著负向预测搜索行为。
    嵌套模型 LR 检验:  search ~ log_pop   vs   search ~ log_pop + knowledge_score
    显著 => 搜索决策跟随真实知识, 而非仅跟随实体眼熟度。
  辅助: 分层内相关 (同一流行度分箱内, knowledge vs search 的 Spearman)。

用法:
  python knowledge_control.py --stage probe   --records out/records.jsonl \\
      --model NousResearch/Meta-Llama-3.1-8B-Instruct --quantize-4bit --n-samples 8
  python knowledge_control.py --stage analyze --records out/records.jsonl
"""
from __future__ import annotations
import argparse, gc, json, logging, re, unicodedata
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("know_ctrl")

DIRECT_PROMPT = "{q}\nAnswer with only the fact, no explanation."


def canon(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold().strip()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def correct(text, gold, aliases=()):
    c = canon(text)
    return any(canon(a) and canon(a) in c for a in [gold, *aliases] if a)


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


# ============================ 采集 ============================
def stage_probe(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    recs = read_jsonl(args.records)
    LOG.info("样本 %d", len(recs))
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True)
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.quantize_4bit:
        model = model.to(args.device)
    model.eval()
    dev = next(model.parameters()).device

    def fmt(u):
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template([{"role": "user", "content": u}],
                                           tokenize=False, add_generation_prompt=True)
        return f"User: {u}\nAssistant:"

    out_path = Path(args.out or (Path(args.records).parent / "knowledge_scores.jsonl"))
    done = {json.loads(l)["qid"] for l in out_path.open()} if (args.resume and out_path.exists()) else set()
    if out_path.exists() and not args.resume:
        out_path.unlink()
    from tqdm.auto import tqdm
    fh = out_path.open("a")
    for r in tqdm(recs, desc="direct-answer probe"):
        if r["qid"] in done:
            continue
        try:
            enc = tok(fmt(DIRECT_PROMPT.format(q=r["question"])), return_tensors="pt",
                      truncation=True, max_length=args.max_input_tokens,
                      add_special_tokens=False).to(dev)
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                o = model.generate(**enc, max_new_tokens=24,
                                   do_sample=args.n_samples > 1,
                                   temperature=args.temperature if args.n_samples > 1 else None,
                                   num_return_sequences=args.n_samples,
                                   pad_token_id=tok.pad_token_id)
            gold, al = r.get("answer", ""), r.get("aliases", [])
            hits, texts = [], []
            for seq in o:
                t = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                texts.append(t[:80]); hits.append(int(correct(t, gold, al)))
            # 自洽度: 众数答案占比 (与正确性互补的知识信号)
            norm = [canon(t) for t in texts]
            mode = max(set(norm), key=norm.count) if norm else ""
            fh.write(json.dumps(dict(
                qid=r["qid"], knowledge_score=float(np.mean(hits)),
                n_samples=len(hits), self_consistency=float(norm.count(mode) / max(len(norm), 1)),
                sample_answers=texts[:3])) + "\n")
            fh.flush()
        except Exception:
            LOG.exception("fail %s", r["qid"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fh.close()
    LOG.info("-> %s", out_path)


# ============================ 分析 ============================
def stage_analyze(args):
    import statsmodels.api as sm
    from scipy.stats import spearmanr, chi2
    recs = {r["qid"]: r for r in read_jsonl(args.records)}
    ks_path = Path(args.knowledge or (Path(args.records).parent / "knowledge_scores.jsonl"))
    ks = {r["qid"]: r for r in read_jsonl(ks_path)}
    rows = []
    for qid, r in recs.items():
        k = ks.get(qid)
        if k is None:
            continue
        pop = float(r.get("s_pop") or 0.0)
        rows.append(dict(qid=qid, search=int(r["action"] == "search"),
                         know=float(k["knowledge_score"]),
                         consis=float(k.get("self_consistency", 0.0)),
                         log_pop=math_log(pop), prior=r.get("know_prior", ""),
                         action=r["action"]))
    LOG.info("匹配 %d / %d", len(rows), len(recs))
    if len(rows) < 50:
        raise SystemExit("匹配样本过少, 先跑 --stage probe")

    y = np.array([r["search"] for r in rows], float)
    K = np.array([r["know"] for r in rows], float)
    P = np.array([r["log_pop"] for r in rows], float)
    res = {"n": len(rows),
           "knowledge_score_distribution": {
               "mean": round(float(K.mean()), 4),
               "n_distinct": int(len(np.unique(K))),
               "hist": {str(round(float(v), 3)): int((K == v).sum())
                        for v in np.unique(K)[:12]}},
           "overall_search_rate": round(float(y.mean()), 4)}

    # ---- 嵌套模型 LR 检验 (核心) ----
    X0 = sm.add_constant(P.reshape(-1, 1))
    X1 = sm.add_constant(np.column_stack([P, K]))
    m0 = sm.Logit(y, X0).fit(disp=0)
    m1 = sm.Logit(y, X1).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    p_lr = float(chi2.sf(lr, 1))
    res["nested_model_test"] = {
        "model0": "search ~ log_pop",
        "model1": "search ~ log_pop + knowledge_score",
        "llf0": round(float(m0.llf), 3), "llf1": round(float(m1.llf), 3),
        "LR_chi2": round(float(lr), 3), "df": 1, "p": p_lr,
        "coef_knowledge": round(float(m1.params[2]), 4),
        "se_knowledge": round(float(m1.bse[2]), 4),
        "z_knowledge": round(float(m1.tvalues[2]), 3),
        "p_knowledge": float(m1.pvalues[2]),
        "coef_log_pop": round(float(m1.params[1]), 4),
        "significant": bool(p_lr < 0.05 and m1.params[2] < 0),
        "note": ("知识系数显著为负 = 控制流行度后, 模型实际知道的问题搜得更少 "
                 "=> 搜索决策跟随真实知识而非仅跟随实体眼熟度(元认知 vs 眼熟度的判决)。"),
    }

    # ---- 分层内相关 (流行度分箱) ----
    strata = []
    edges = np.unique(np.quantile(P, np.linspace(0, 1, args.n_bins + 1)))
    for lo, hi, last in zip(edges[:-1], edges[1:], [False] * (len(edges) - 2) + [True]):
        m = (P >= lo) & ((P <= hi) if last else (P < hi))
        if m.sum() < 20 or len(np.unique(K[m])) < 2:
            continue
        rho, pv = spearmanr(K[m], y[m])
        strata.append({"log_pop_range": [round(float(lo), 3), round(float(hi), 3)],
                       "n": int(m.sum()), "search_rate": round(float(y[m].mean()), 4),
                       "mean_knowledge": round(float(K[m].mean()), 4),
                       "spearman_know_vs_search": round(float(rho), 4), "p": float(pv)})
    res["within_popularity_strata"] = {
        "strata": strata,
        "n_strata_negative_sig": sum(1 for s in strata
                                     if s["spearman_know_vs_search"] < 0 and s["p"] < 0.05),
        "note": "同一流行度区间内, 知识分数与搜索应负相关。这是不依赖回归设定的稳健版本。"}

    # ---- unknown 子集内的 Gemini 失败刻画 ----
    unk = [r for r in rows if r["prior"] == "unknown"]
    if unk:
        ku = np.array([r["know"] for r in unk]); au = [r["action"] for r in unk]
        oc = [r for r in unk if r["action"] == "answer"]
        res["unknown_subset"] = {
            "n": len(unk),
            "search_rate": round(float(np.mean([a == "search" for a in au])), 4),
            "mean_knowledge": round(float(ku.mean()), 4),
            "mean_knowledge_when_search": round(
                float(np.mean([r["know"] for r in unk if r["action"] == "search"])), 4),
            "mean_knowledge_when_answer": round(
                float(np.mean([r["know"] for r in oc])), 4) if oc else None,
            "note": ("若 '直接作答' 组的知识分数系统性高于 '搜索' 组, 说明模型在 unknown 内部"
                     "仍能分辨自己碰巧知道的那些 —— 这是元认知的细粒度证据。"),
        }

    # ---- 自洽度作为第二知识信号 ----
    C = np.array([r["consis"] for r in rows], float)
    if len(np.unique(C)) > 2:
        X2 = sm.add_constant(np.column_stack([P, K, C]))
        m2 = sm.Logit(y, X2).fit(disp=0)
        res["adding_self_consistency"] = {
            "coef_consistency": round(float(m2.params[3]), 4),
            "p_consistency": float(m2.pvalues[3]),
            "LR_vs_model1": round(float(2 * (m2.llf - m1.llf)), 3),
            "p": float(chi2.sf(2 * (m2.llf - m1.llf), 1))}

    out = Path(args.out or (Path(args.records).parent / "knowledge_control.json"))
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    LOG.info("-> %s", out)


def math_log(x):
    import math
    return math.log10(max(float(x), 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "analyze"], required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--knowledge", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    (stage_probe if a.stage == "probe" else stage_analyze)(a)


if __name__ == "__main__":
    main()
