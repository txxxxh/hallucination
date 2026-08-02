#!/usr/bin/env python3
"""
tool_gate_calibration.py — Llama 知不知道"何时该调用工具"? (自包含, 单次前向为主)

三选一协议: 对每个问题, 模型输出 [SEARCH] / 直接答案 / [ABSTAIN]。
交叉两个真值:
  (A) 模型到底知不知道 —— 两路真值:
      - construct: PopQA 流行度分桶 / 合成实体 (绝对不知道) / 高频实体 (应知道)
      - probe: 对该实体问若干独立事实, 用回答正确率估计知识状态 (v3 式)
  (B) 答案对不对 —— gold 判定

核心假设 H: 在"不知道"的问题上, P(SEARCH) 显著高于"知道"的问题。
  成立 => 行动被知识边界调制 => 模型知道何时该调工具。
  不成立 (不知道却自信直答) => 决策层校准失败 (Gemini 病)。

双重证据:
  行为层: search_rate | knowledge_state 的差异 + 单调性 (按 probe 连续分数分桶)
  表征层: 从 base 问题的 hidden state 用 probe 预测 "会不会 SEARCH" / "知不知道"
难度混淆对照: 在"不知道"子集内, search_rate 是否随 probe 置信度单调 (而非只随题难)。

用法:
  python tool_gate_calibration.py --stage collect --dataset popqa \
     --model NousResearch/Meta-Llama-3.1-8B-Instruct --quantize-4bit \
     --output-dir out --max-samples 1000
  python tool_gate_calibration.py --stage analyze --output-dir out
"""
from __future__ import annotations
import argparse, gc, json, logging, re, unicodedata
from pathlib import Path
from typing import Optional
import numpy as np
from probe_patch import (
    build_popqa_index,
    make_fact_probes,
    robust_buckets,
    score_probes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("tool_gate")

TOOL_INSTR = (
    "A web search tool is available. For the question below, respond with EXACTLY one of:\n"
    "  [SEARCH] <query>   -- if you are not confident you know the answer and should look it up\n"
    "  <answer>           -- if you are confident you know the answer, give it directly\n"
    "  [ABSTAIN]          -- if the question is unanswerable or based on a false premise\n"
    "Choose deliberately. Do not add explanation.\n\nQuestion: {q}")

SEARCH_RE = re.compile(r"\[\s*SEARCH\s*\]", re.I)
ABSTAIN_RE = re.compile(r"\[\s*ABSTAIN\s*\]|i don'?t know|unanswerable|false premise", re.I)


def canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold().strip()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())

def classify_action(text: str) -> str:
    head = text.split("</think>")[-1].strip()[:200]
    if SEARCH_RE.search(head):
        return "search"
    if ABSTAIN_RE.search(head):
        return "abstain"
    return "answer"

def answer_correct(text: str, gold: str, aliases) -> bool:
    c = canon(text)
    return any(canon(a) and canon(a) in c for a in [gold, *aliases] if a)


# ----------------- 数据: 构造"知道/不知道"真值 -----------------
def load_dataset(name, max_samples, seed):
    """返回 [{qid, question, answer, aliases, know_prior, probes:[...]}]。
    know_prior: 构造先验 ('known'/'unknown'/'unanswerable'), 来自流行度/合成/假前提。
    probes: 用于探针知识状态的独立事实问句 (来自同实体的其他属性)。"""
    from datasets import load_dataset as hf_load
    rng = np.random.RandomState(seed)
    out = []
    if name == "popqa":
        ds = hf_load("akariasai/PopQA", split="test")
        rows = list(ds)
        by_subj, obj_pool = build_popqa_index(rows)
        rng.shuffle(rows)
        for r in rows:
            if len(out) >= max_samples:
                break
            pop = r.get("s_pop") or 0
            aliases = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) else r["possible_answers"]
            # 流行度先验: 极低->unknown, 极高->known, 中间跳过 (干净分桶)
            if pop and pop < 50:
                prior = "unknown"
            elif pop and pop > 50000:
                prior = "known"
            else:
                continue
            out.append(dict(qid=str(r.get("id", len(out))), question=r["question"],
                            answer=aliases[0], aliases=aliases[1:], know_prior=prior,
                            s_pop=float(pop), subj=r.get("subj", ""), prop=r.get("prop", ""),
                            probes=make_fact_probes(
                                r.get("subj", ""), r.get("prop", ""),
                                by_subj, obj_pool, rng,
                            )[0]))
    elif name == "synthetic":
        # 合成实体: 绝对 unknown; 需配一批高频真实实体作 known 对照
        out = _synthetic_set(max_samples, rng)
    else:
        raise ValueError(f"unknown dataset {name}")
    return out

def _popqa_probes(r):
    # 用同一主语的其他常识属性做探针 (yes/no); 简化: 用 subj 构造存在性/领域探针
    subj = r.get("subj", "")
    if not subj:
        return []
    return [dict(text=f"Is {subj} a real, well-documented entity that you have specific factual knowledge about?",
                 expected_yes=True)]

def _synthetic_set(n, rng):
    FIRST = ["Aldric","Bethune","Cassivel","Dornwick","Elsberry","Fenlow","Gathmere","Holbein"]
    LAST = ["Ashgrove","Brindlecombe","Coldharbour","Duskfield","Eastmoor","Fallowmere","Grimsbury"]
    out = []
    for i in range(n // 2):
        nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        out.append(dict(qid=f"syn_{i}", question=f"In which year was the physicist {nm} awarded the Copley Medal?",
                        answer="UNKNOWABLE", aliases=[], know_prior="unknown", s_pop=0.0,
                        subj=nm, prop="synthetic",
                        probes=[dict(text=f"Do you have specific factual knowledge about a physicist named {nm}?",
                                     expected_yes=False)]))
    # 高频真实对照 (模型应知道)
    KNOWN = [("Who developed the theory of general relativity?","Albert Einstein"),
             ("What is the capital of France?","Paris"),
             ("Who wrote Romeo and Juliet?","William Shakespeare"),
             ("What is the chemical symbol for gold?","Au"),
             ("Who painted the Mona Lisa?","Leonardo da Vinci")]
    for i in range(n - len(out)):
        q, a = KNOWN[i % len(KNOWN)]
        out.append(dict(qid=f"known_{i}", question=q, answer=a, aliases=[], know_prior="known",
                        s_pop=1e6, subj=a, prop="known_control",
                        probes=[dict(text=f"Do you have specific factual knowledge to answer: {q}",
                                     expected_yes=True)]))
    return out


# ----------------- 引擎 (复用极简版) -----------------
class Engine:
    def __init__(self, model, device, dtype, max_input, quant4, trust):
        import torch
        # PyTorch 2.13 may route Llama RoPE bmm through an experimental Triton
        # override whose compilation can fail on some filesystems. Fall back to
        # the stable CUDA bmm kernel when that override API is available.
        try:
            torch._native.registry.deregister_op_overrides(disable_op_symbols="bmm")
        except (AttributeError, ImportError):
            pass
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self.max_input = max_input
        self.tok = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=trust)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(torch_dtype=getattr(torch, dtype), trust_remote_code=trust, low_cpu_mem_usage=True)
        if quant4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            kw["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        if not quant4:
            self.model = self.model.to(device)
        self.model.eval(); self.model.config.use_cache = False
        self.device = next(self.model.parameters()).device
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.hidden = int(self.model.config.hidden_size)

    def _fmt(self, user):
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template([{"role":"user","content":user}],
                                                tokenize=False, add_generation_prompt=True)
        return f"User: {user}\nAssistant:"

    def gen_and_hidden(self, prompt, seed, max_new=40):
        """生成 + 取 prompt 末token (作答前) 的逐层 hidden。一次前向拿两样。"""
        import torch
        torch.manual_seed(seed)
        text = self._fmt(prompt)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_input,
                       add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id,
                                      output_hidden_states=True, return_dict_in_generate=True)
        gen_ids = out.sequences[0, enc.input_ids.shape[1]:]
        txt = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        # generate 的 hidden_states[0] 是 prompt 的前向, 取每层末位置
        hs0 = out.hidden_states[0]   # tuple over layers, each [1, prompt_len, H]
        hidden = torch.stack([hs0[l][0, -1].float().cpu() for l in range(len(hs0))]).half()
        return txt, hidden   # [L+1, H]

    def ask_probe(self, text, seed):
        import torch
        torch.manual_seed(seed)
        t = self._fmt(text + " Answer strictly Yes or No.")
        enc = self.tok(t, return_tensors="pt", truncation=True, max_length=self.max_input,
                       add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=4, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        txt = self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        c = canon(txt)
        return True if c.startswith(("yes","true")) else False if c.startswith(("no","false")) else None

    def ask_open_probe(self, text, seed):
        import torch
        torch.manual_seed(seed)
        t = self._fmt(text + " Answer briefly with only the factual answer.")
        enc = self.tok(t, return_tensors="pt", truncation=True, max_length=self.max_input,
                       add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=24, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(
            out[0, enc.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


def stage_collect(args, out: Path):
    import torch
    data = load_dataset(args.dataset, args.max_samples, args.seed)
    LOG.info("加载 %d 题 (known=%d unknown=%d)", len(data),
             sum(d["know_prior"]=="known" for d in data),
             sum(d["know_prior"]=="unknown" for d in data))
    eng = Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                 args.quantize_4bit, args.trust_remote_code)
    tdir = out / "hidden"; tdir.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    done = {json.loads(l)["qid"] for l in rec_path.open()} if (args.resume and rec_path.exists()) else set()
    if rec_path.exists() and not args.resume:
        rec_path.unlink()
    from tqdm.auto import tqdm
    with rec_path.open("a") as fh:
        for d in tqdm(data, desc="collect"):
            if d["qid"] in done:
                continue
            try:
                seed = args.seed + hash(d["qid"]) % 100000
                gen, hidden = eng.gen_and_hidden(TOOL_INSTR.format(q=d["question"]), seed)
                action = classify_action(gen)
                if action == "answer":
                    # Synthetic unknowns have no valid factual answer: any direct
                    # answer is necessarily an overconfident wrong answer.
                    correct = False if d["answer"] == "UNKNOWABLE" else answer_correct(
                        gen, d["answer"], d["aliases"])
                else:
                    correct = None
                # 探针知识状态
                probe_results = []
                for j, p in enumerate(d.get("probes", [])):
                    if p["kind"] == "open":
                        ans = eng.ask_open_probe(p["text"], seed + 500 + j)
                        correct_probe = answer_correct(ans, p["gold"], p.get("aliases", []))
                    else:
                        ans = eng.ask_probe(p["text"], seed + 500 + j)
                        if ans is None:
                            continue
                        correct_probe = ans == p["expected_yes"]
                    probe_results.append({"kind": p["kind"], "correct": bool(correct_probe)})
                probe_score, probe_detail = score_probes(probe_results)
                torch.save({"qid": d["qid"], "hidden": hidden}, tdir / f"{d['qid']}.pt")
                fh.write(json.dumps(dict(
                    qid=d["qid"], question=d["question"], answer=d["answer"],
                    know_prior=d["know_prior"], s_pop=d.get("s_pop", 0.0),
                    action=action, generation=gen[:300], answer_correct=correct,
                    probe_score=probe_score, n_probes=len(probe_results),
                    probe_detail=probe_detail)) + "\n")
                fh.flush()
            except Exception as e:
                LOG.exception("fail %s", d["qid"])
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    LOG.info("done -> %s", rec_path)


def stage_analyze(args, out: Path):
    import torch
    recs = [json.loads(l) for l in (out / "records.jsonl").open() if l.strip()]
    LOG.info("n=%d", len(recs))

    def rate(sub, act):
        return np.mean([r["action"] == act for r in sub]) if sub else float("nan")

    known = [r for r in recs if r["know_prior"] == "known"]
    unknown = [r for r in recs if r["know_prior"] == "unknown"]

    res = {"n": len(recs), "by_prior": {}}
    for name, sub in [("known", known), ("unknown", unknown)]:
        res["by_prior"][name] = {
            "n": len(sub),
            "search_rate": round(rate(sub, "search"), 4),
            "answer_rate": round(rate(sub, "answer"), 4),
            "abstain_rate": round(rate(sub, "abstain"), 4),
            "answer_and_wrong_rate": round(
                np.mean([r["action"] == "answer" and r["answer_correct"] is False for r in sub]) if sub else float("nan"), 4),
        }

    # ===== 核心假设 H: unknown 的 search_rate > known 的 =====
    from scipy.stats import fisher_exact
    def counts(sub, act):
        return sum(r["action"] == act for r in sub), sum(r["action"] != act for r in sub)
    a1, a0 = counts(unknown, "search"); b1, b0 = counts(known, "search")
    if min(a1+a0, b1+b0) > 0:
        odds, p = fisher_exact([[a1, a0], [b1, b0]])
        res["hypothesis_H_search_higher_on_unknown"] = {
            "unknown_search_rate": round(a1/(a1+a0), 4),
            "known_search_rate": round(b1/(b1+b0), 4),
            "odds_ratio": round(float(odds), 3), "fisher_p": float(p),
            "supported": bool(a1/(a1+a0) > b1/(b1+b0) and p < 0.05)}

    # ===== Gemini 病: unknown 上"该搜却自信直答且错"的比例 =====
    res["gemini_failure"] = {
        "unknown_overconfident_wrong": round(
            np.mean([r["action"] == "answer" and r["answer_correct"] is False for r in unknown]) if unknown else float("nan"), 4),
        "note": "unknown 问题里直接答且答错的比例; 高 = 决策层校准失败严重"}

    # ===== 难度混淆对照: unknown 内按 probe_score 分桶, 看 search_rate 单调性 =====
    with_probe = [r for r in unknown if r.get("probe_score") is not None]
    if len(with_probe) >= 20:
        scores = np.array([r["probe_score"] for r in with_probe])
        actions = np.array([r["action"] for r in with_probe])
        buckets, diagnostics = robust_buckets(scores, actions)
        from scipy.stats import spearmanr
        rho, rho_p = spearmanr(scores, actions == "search")
        res["probe_score_vs_search"] = {
            "buckets": buckets,
            "diagnostics": diagnostics,
            "spearman_rho": None if np.isnan(rho) else round(float(rho), 4),
            "spearman_p": None if np.isnan(rho_p) else float(rho_p),
            "note": "probe 越判'知道'(score高), search_rate 应越低; 单调下降=行动被知识信号驱动而非只被题难度"}

    # ===== 表征层: 从 base hidden 预测 (a) 知道/不知道 (b) 会不会 search =====
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    feats, y_know, y_search = [], [], []
    n_layers = None
    for r in recs:
        pt = out / "hidden" / f"{r['qid']}.pt"
        if not pt.exists():
            continue
        h = torch.load(pt, map_location="cpu", weights_only=False)["hidden"].float().numpy()
        n_layers = h.shape[0]
        feats.append(h)
        y_know.append(0 if r["know_prior"] == "unknown" else 1)
        y_search.append(int(r["action"] == "search"))
    feats = np.stack(feats); y_know = np.array(y_know); y_search = np.array(y_search)

    def layer_scan(y, tag):
        best = {"auroc": -1, "layer": None, "curve": []}
        if len(set(y)) < 2:
            return {"skipped": "single class"}
        for l in range(1, n_layers):   # 排除 embedding 层
            X = StandardScaler().fit_transform(feats[:, l])
            p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
                                  X, y, cv=5, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, p)
            best["curve"].append((l, round(auc, 3)))
            if auc > best["auroc"]:
                best.update(auroc=round(auc, 4), layer=l)
        return best
    res["probe_knows_dontknow"] = layer_scan(y_know, "know")
    res["probe_predicts_search"] = layer_scan(y_search, "search")

    (out / "analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "by_prior"}, indent=2, ensure_ascii=False))
    print("\nby_prior:", json.dumps(res["by_prior"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect","analyze"], default="collect")
    ap.add_argument("--dataset", choices=["popqa","synthetic"], default="popqa")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    if a.stage == "collect":
        stage_collect(a, out)
    else:
        stage_analyze(a, out)

if __name__ == "__main__":
    main()
