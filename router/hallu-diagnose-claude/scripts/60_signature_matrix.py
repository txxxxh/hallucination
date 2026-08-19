#!/usr/bin/env python3
"""
signature_matrix.py — 四类 stressor 的内部签名矩阵 (D1 表征 / D2 自省 / D3 门控)

论文定位: 这是主体章节 (§5)。先证明四种 stressor 在模型内部的**处理结构本身不同**,
          再由此推出"没有 universal winner"的诊断/治疗结论。

三个维度, 每个 stressor 用**同一组度量**打分, 保证可比:

  D1 表征 (representation): 模型内部是否编码了该状态? 用逐层 probe 的 AUROC 曲线刻画。
      主指标: peak_auroc / onset_layer(首达 95% 峰值的层) / centroid。
      onset 是主指标 —— 峰值常有平台, argmax 不稳(已在 44 脚本上验证过这一点)。

  D2 自省 (introspection): 模型能否**口头**报告该状态? 用 verbalized vs probe 的 gap。
      主指标: rho_verbal, rho_probe, gap = rho_probe - rho_verbal。
      Z1 用 P(True)/自评知识; Z4 用自报所需 token 数; Z2 用自报"是否被无关信息干扰";
      Z6 用自评可答性。

  D3 门控 (gating): 该状态能否驱动**正确的行动**? 用行为实验的效应量。
      主指标: odds_ratio + Fisher p (二元行动), 或题内斜率 (连续 margin)。
      Z1: 不知道时是否更倾向 SEARCH; Z4: 预算不足时是否更倾向 NEED_MORE;
      Z2: 给"可忽略无关信息"许可后是否改善; Z6: 不可答时是否弃答。

数据来源 (按仓库现有产物, 缺失则跳过并在输出中标注):
  D1  : router 特征目录 (index.jsonl + {sid}.npz, 由 40_extract_features 产出)
  D2/D3: 各 stressor 的专属采集, 见 --stage collect-{z1,z2,z4,z6}
  可复用: out_tool_gate_*/analysis.json (Z1 的 D3), out_budget_meta/analysis.json (Z4 的 D2/D3)

用法:
  # 1) D1: 从现有 router 特征算四类的逐层签名 (无 GPU)
  python signature_matrix.py --stage d1 --features data/features/<model> --out sig/

  # 2) D2/D3: 各 stressor 采集 (GPU); 已有结果可用 --reuse 跳过
  python signature_matrix.py --stage collect-z1 --out sig/ --model <m> --quantize-4bit
  python signature_matrix.py --stage collect-z2 --out sig/ --model <m> --quantize-4bit
  python signature_matrix.py --stage collect-z4 --out sig/ --model <m> --quantize-4bit
  python signature_matrix.py --stage collect-z6 --out sig/ --model <m> --quantize-4bit

  # 3) 汇总成矩阵
  python signature_matrix.py --stage assemble --out sig/ \\
      --reuse-toolgate out_tool_gate_probe_patch_20260728/analysis.json \\
      --reuse-budget out_budget_meta/analysis.json
"""
from __future__ import annotations
import argparse, gc, json, logging, math, re, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("sigmat")

STRESSORS = ["Z1", "Z2", "Z4", "Z6"]
SIG_NAMES = {"Z1": "knowledge gap", "Z2": "context distraction",
             "Z4": "compute insufficiency", "Z6": "calibration failure"}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def write_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


# ============================================================================
# D1: 逐层表征签名
# ============================================================================
def layer_auroc(X, y, groups, n_splits=5):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    prob = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    try:
        return float(roc_auc_score(y, prob))
    except ValueError:
        return float("nan")


def summarize_layers(curve, onset_frac=0.95):
    """peak / onset / centroid。onset 为主指标(峰值平台时 argmax 不稳)。"""
    vals = [(l, a) for l, a in curve if a is not None and np.isfinite(a)]
    if not vals:
        return {}
    peak_l, peak_a = max(vals, key=lambda t: t[1])
    thr = 0.5 + onset_frac * (peak_a - 0.5)
    onset = next((l for l, a in vals if a >= thr), peak_l)
    w = np.array([max(a - 0.5, 0.0) for _, a in vals])
    ls = np.array([l for l, _ in vals], float)
    cent = float((w * ls).sum() / w.sum()) if w.sum() > 0 else float("nan")
    # 相对深度, 便于跨模型比较
    L = max(l for l, _ in vals)
    return {"peak_layer": int(peak_l), "peak_auroc": round(float(peak_a), 4),
            "onset_layer": int(onset), "onset_rel_depth": round(onset / max(L, 1), 3),
            "centroid_layer": round(cent, 2), "n_layers_scanned": len(vals)}


def stage_d1(args, out: Path):
    """对每个 stressor 做 one-vs-rest 逐层扫描 -> D1 签名。"""
    feat = Path(args.features)
    idx = read_jsonl(feat / "index.jsonl")
    rows, F = [], []
    for r in idx:
        if r["label"] == "CLEAN":
            continue
        p = feat / f"{r['sid']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        F.append(np.nan_to_num(d["f1"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        rows.append(r)
    if not rows:
        raise SystemExit(f"{feat} 未载入样本")
    F = np.stack(F)
    labels = np.array([r["label"] for r in rows])
    domains = np.array([r.get("domain", "") for r in rows])
    groups = np.array([r["sid"].replace("__clean", "") for r in rows])
    L = F.shape[1]
    step = max(1, L // args.max_layers)
    layers = list(range(1, L, step))          # 排除 embedding 层
    LOG.info("D1: n=%d 层=%d 扫描=%d 类别=%s", len(rows), L, len(layers), dict(Counter(labels)))

    res = {"n": len(rows), "n_layers": L, "layers": layers,
           "label_counts": dict(Counter(labels)), "per_stressor": {}}
    rng = np.random.RandomState(args.seed)
    for z in STRESSORS:
        y = (labels == z).astype(int)
        if y.sum() < 15 or (y == 0).sum() < 15:
            res["per_stressor"][z] = {"skipped": f"n_pos={int(y.sum())}"}
            continue
        curve = [(l, round(layer_auroc(F[:, l], y, groups), 4)) for l in layers]
        s = summarize_layers(curve)
        s["curve"] = curve
        s["n_pos"] = int(y.sum())
        # bootstrap CI: 峰值/onset 层是否可分离
        if args.n_boot > 0:
            peaks, onsets = [], []
            uniq = np.array(sorted(set(groups)))
            for _ in range(args.n_boot):
                pick = rng.choice(uniq, len(uniq), replace=True)
                m = np.concatenate([np.flatnonzero(groups == g) for g in pick])
                if len(set(y[m])) < 2:
                    continue
                c = [(l, layer_auroc(F[m][:, l], y[m], groups[m])) for l in layers]
                ss = summarize_layers(c)
                if ss:
                    peaks.append(ss["peak_layer"]); onsets.append(ss["onset_layer"])
            q = lambda v: [int(np.quantile(v, .025)), int(np.quantile(v, .975))] if v else None
            s["peak_layer_ci95"] = q(peaks); s["onset_layer_ci95"] = q(onsets)
        # 域内重复 (排除领域驱动)
        wd = {}
        for dom in sorted(set(domains)):
            m = domains == dom
            if m.sum() < 60 or (y[m].sum() < 10) or ((y[m] == 0).sum() < 10):
                continue
            c = [(l, round(layer_auroc(F[m][:, l], y[m], groups[m]), 4)) for l in layers]
            wd[dom] = summarize_layers(c)
        s["within_domain"] = wd
        res["per_stressor"][z] = s
        LOG.info("  %s: peak L%s (%.3f) onset L%s (rel %.2f) n_pos=%d",
                 z, s.get("peak_layer"), s.get("peak_auroc", float('nan')),
                 s.get("onset_layer"), s.get("onset_rel_depth", float('nan')), s["n_pos"])

    ok = {z: v for z, v in res["per_stressor"].items() if "onset_layer" in v}
    if len(ok) >= 2:
        res["ordering"] = {
            "by_onset": sorted([(z, ok[z]["onset_layer"]) for z in ok], key=lambda t: t[1]),
            "by_peak": sorted([(z, ok[z]["peak_layer"]) for z in ok], key=lambda t: t[1]),
            "note": ("onset 为主指标。若两类的 onset_layer_ci95 重叠, 不得声称层位分离。"),
        }
    write_json(res, out / "d1_representation.json")
    LOG.info("-> %s", out / "d1_representation.json")



# ============================================================================
# 统一池子加载: 一律使用 10_screen.py 产出的 {z}_final.jsonl
# 这保证 D1(router 特征)/D2/D3/治疗矩阵/闭环 用的是**同一批样本**,
# 否则三个维度不可比, 签名矩阵失去内部一致性。
# ============================================================================
def load_pool(args, z: str, need_clean=False):
    """读 data/processed/{z}_final.jsonl。返回 [{sid,q_trig,q_clean,answer,aliases,meta,...}]"""
    p = Path(args.pool_dir) / f"{z.lower()}_final.jsonl"
    if not p.exists():
        raise SystemExit(f"未找到样本池 {p}. 请先跑 10_screen.py, 或用 --pool-dir 指定目录。")
    rows = read_jsonl(p)
    out = []
    for r in rows:
        if need_clean and r.get("q_clean") == r.get("q_trig"):
            continue                      # Z2 需要真实的 clean/trig 配对
        out.append(dict(
            sid=r["sid"], q_trig=r["q_trig"], q_clean=r.get("q_clean", r["q_trig"]),
            answer=r.get("answer", ""), aliases=r.get("answer_aliases", []),
            domain=r.get("domain", ""), template_id=r.get("template_id", ""),
            intensity=r.get("intensity", 0.0), meta=r.get("meta", {}),
            secondary=r.get("secondary_labels", [])))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(out)
    if args.max_items > 0:
        out = out[:args.max_items]
    LOG.info("%s: 载入 %d 条 (来自 %s)", z, len(out), p.name)
    return out


def load_clean_controls(args, z: str, n: int):
    """从**同一池子**取 clean 配对作为对照 (不引入外部数据)。
    Z2/Z3 的 q_clean 是天然对照; Z1/Z6 若无 clean 变体则返回空。"""
    try:
        rows = read_jsonl(Path(args.pool_dir) / f"{z.lower()}_final.jsonl")
    except SystemExit:
        return []
    ctl = [r for r in rows if r.get("q_clean") and r["q_clean"] != r["q_trig"]]
    rng = np.random.RandomState(args.seed + 1)
    rng.shuffle(ctl)
    return ctl[:n]

# ============================================================================
# 共用引擎
# ============================================================================
class Engine:
    def __init__(self, model, device="cuda", dtype="bfloat16", max_input=4096,
                 quant4=False, trust=False):
        import torch
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
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        ids = self.tok.encode("</think>", add_special_tokens=False)
        self.think_end_id = ids[-1] if ids else None

    def fmt(self, u, skip_think=False):
        if getattr(self.tok, "chat_template", None):
            t = self.tok.apply_chat_template([{"role": "user", "content": u}],
                                             tokenize=False, add_generation_prompt=True)
        else:
            t = f"User: {u}\nAssistant:"
        if skip_think and self.think_end_id is not None:
            t += "<think>\n\n</think>\n\n"
        return t

    def gen(self, prompt, max_new=64, temperature=0.0, n=1, seed=0, skip_think=False):
        import torch
        torch.manual_seed(seed)
        enc = self.tok(self.fmt(prompt, skip_think), return_tensors="pt", truncation=True,
                       max_length=self.max_input, add_special_tokens=False).to(self.dev)
        with torch.inference_mode():
            o = self.model.generate(**enc, max_new_tokens=max_new,
                                    do_sample=temperature > 0,
                                    temperature=temperature if temperature > 0 else None,
                                    num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(s[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                for s in o]

    def choice_logprobs(self, prompt, options, skip_think=True):
        """单 token 选项打分 —— 避免多 token 标签的长度伪影(gate 实验的教训)。
        options: {letter: meaning}; letter 必须是单 token。"""
        import torch
        ids = {}
        for L in options:
            for cand in (L, " " + L):
                e = self.tok.encode(cand, add_special_tokens=False)
                if len(e) == 1:
                    ids[L] = e[0]; break
            if L not in ids:
                raise ValueError(f"选项 {L} 非单 token")
        enc = self.tok(self.fmt(prompt, skip_think), return_tensors="pt", truncation=True,
                       max_length=self.max_input, add_special_tokens=False).to(self.dev)
        with torch.inference_mode():
            o = self.model(input_ids=enc.input_ids)
        lg = torch.log_softmax(o.logits[0, -1].float(), -1)
        return {L: float(lg[i]) for L, i in ids.items()}


def canon(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s)).casefold().strip()
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def hit(text, gold, aliases=()):
    c = canon(text)
    return any(canon(a) and canon(a) in c for a in [gold, *aliases] if a)


def parse_int(t):
    m = re.search(r"\d[\d,]*", str(t).replace(",", ""))
    return int(m.group(0)) if m else None


ABSTAIN_RE = re.compile(
    r"(i don'?t (have|know)|do not have|no information|not (familiar|aware|sure)|"
    r"cannot (find|verify|provide|determine|answer)|unable to|unanswerable|"
    r"false premise|not enough (information|context))", re.I)


# 配平的三选一/二选一模板 (单 token 字母, 轮换映射消除位置偏置)
import itertools
LETTERS3 = ["A", "B", "C"]
PERMS3 = list(itertools.permutations(LETTERS3))


def build_choice_prompt(question, options_text, mapping, header):
    lines = [f"{L}) {options_text[mapping[L]]}" for L in LETTERS3]
    return (f"{header}\n\n{question}\n\n"
            "Answer with a single letter.\n" + "\n".join(lines) + "\nAnswer:")


# ============================================================================
# Z1: 知识缺失 —— D2 自评知识 / D3 搜索门控
# ============================================================================
Z1_ACTIONS = {"search": "I should look this up before answering",
              "answer": "I know this and can answer directly",
              "abstain": "This cannot be answered"}
Z1_HEADER = "A web search tool is available. For the question below, which is true?"
Z1_VERBAL = ("For the question below, how confident are you that you know the correct answer? "
             "Reply with only an integer from 0 to 100.\n\nQuestion: {q}")


def stage_collect_z1(args, out: Path):
    """D2: 自评知识置信度 vs 真实知识; D3: 知识状态是否驱动 SEARCH。
    样本来自 z1_final.jsonl (与 D1/治疗矩阵同批)。真实知识 = 无工具直接问的答对率。"""
    import torch
    items = load_pool(args, "Z1")
    eng = Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                 args.quantize_4bit, args.trust_remote_code)
    path = out / "z1_raw.jsonl"
    if path.exists() and not args.resume:
        path.unlink()
    done = {json.loads(l)["qid"] for l in path.open()} if (args.resume and path.exists()) else set()
    from tqdm.auto import tqdm
    fh = path.open("a")
    for i, it in enumerate(tqdm(items, desc="Z1")):
        if it["sid"] in done:
            continue
        try:
            sd = args.seed + i * 7919
            q = it["q_trig"]
            gold, al = it["answer"], it["aliases"]
            unknowable = gold in ("UNKNOWN_ENTITY", "UNANSWERABLE", "")
            # 真实知识: 不给工具直接问 n 次 (合成实体等无 gold 者记 0)
            gens = eng.gen(q + "\nAnswer with only the fact.", max_new=24,
                           temperature=0.7, n=args.n_samples, seed=sd)
            know = (0.0 if unknowable
                    else float(np.mean([hit(g, gold, al) for g in gens])))
            norm = [canon(g) for g in gens]
            consis = float(norm.count(max(set(norm), key=norm.count)) / max(len(norm), 1))
            # D2: 口头自评置信度
            v = eng.gen(Z1_VERBAL.format(q=q), max_new=6, seed=sd + 1, skip_think=True)[0]
            verbal = parse_int(v)
            verbal = min(max(verbal, 0), 100) / 100 if verbal is not None else None
            # D3: 三选一行动 (单 token 打分 + 映射轮换配平)
            perm = PERMS3[i % len(PERMS3)]
            mapping = {L: list(Z1_ACTIONS)[LETTERS3.index(p)] for L, p in zip(LETTERS3, perm)}
            lp = eng.choice_logprobs(
                build_choice_prompt(q, Z1_ACTIONS, mapping, Z1_HEADER), LETTERS3)
            act_lp = {mapping[L]: lp[L] for L in LETTERS3}
            fh.write(json.dumps(dict(
                qid=it["sid"], domain=it["domain"], template_id=it["template_id"],
                source=it["meta"].get("source", ""), s_pop=it.get("intensity", 0.0),
                unknowable=unknowable, knowledge=know, self_consistency=consis,
                verbal_conf=verbal, action=max(act_lp, key=act_lp.get),
                action_logprobs=act_lp)) + "\n")
            fh.flush()
        except Exception:
            LOG.exception("z1 fail %s", it["sid"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fh.close()
    LOG.info("-> %s", path)


# ============================================================================
# Z2: 上下文干扰 —— D2 自评被干扰 / D3 忽略许可门控
# ============================================================================
Z2_ACTIONS = {"ignore": "Some information here is irrelevant; I should ignore part of it",
              "use_all": "All information here is needed to solve the problem",
              "unclear": "I cannot tell which information is relevant"}
Z2_HEADER = "Consider the problem below. Which statement is true about the information given?"
Z2_VERBAL = ("Does the problem below contain information that is irrelevant to solving it? "
             "Reply with only Yes or No.\n\n{q}")
Z2_PERMIT_PREFIX = "Some information in this problem may be irrelevant. Ignore anything not needed.\n\n"
Z2_PLAIN = "{q}\nReason step by step, then give the final answer in \\boxed{{}}."


def stage_collect_z2(args, out: Path):
    """样本来自 z2_final.jsonl 的 q_clean/q_trig 配对 (已过行为筛选与反事实翻转硬条件)。
    D2: 能否口头识别'存在无关信息'; D3: 给'可忽略'许可后是否改善。"""
    import torch
    items = load_pool(args, "Z2", need_clean=True)
    if not items:
        raise SystemExit("z2_final.jsonl 中没有 q_clean != q_trig 的配对样本")
    eng = Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                 args.quantize_4bit, args.trust_remote_code)
    path = out / "z2_raw.jsonl"
    if path.exists() and not args.resume:
        path.unlink()
    done = {json.loads(l)["qid"] for l in path.open()} if (args.resume and path.exists()) else set()
    from tqdm.auto import tqdm
    fh = path.open("a")
    for i, it in enumerate(tqdm(items, desc="Z2")):
        if it["sid"] in done:
            continue
        try:
            sd = args.seed + i * 7919
            gold, al = it["answer"], it["aliases"]
            numeric = bool(it["meta"].get("numeric"))
            suffix = ("\nReason step by step, then give the final answer in \\boxed{}."
                      if numeric or it["domain"] == "math" else "\nAnswer with only the fact.")
            rec = dict(qid=it["sid"], domain=it["domain"], template_id=it["template_id"],
                       intensity=it["intensity"])
            for tag, q in (("clean", it["q_clean"]), ("dist", it["q_trig"])):
                g = eng.gen(q + suffix, max_new=args.max_new,
                            temperature=0.7, n=args.n_samples, seed=sd)
                rec[f"acc_{tag}"] = float(np.mean([hit(x, gold, al) for x in g]))
            g = eng.gen(Z2_PERMIT_PREFIX + it["q_trig"] + suffix, max_new=args.max_new,
                        temperature=0.7, n=args.n_samples, seed=sd + 3)
            rec["acc_dist_permit"] = float(np.mean([hit(x, gold, al) for x in g]))
            for tag, q in (("clean", it["q_clean"]), ("dist", it["q_trig"])):
                v = eng.gen(Z2_VERBAL.format(q=q), max_new=4, seed=sd + 5, skip_think=True)[0]
                rec[f"verbal_irrelevant_{tag}"] = int(bool(re.match(r"\s*yes", v, re.I)))
            perm = PERMS3[i % len(PERMS3)]
            mapping = {L: list(Z2_ACTIONS)[LETTERS3.index(p)] for L, p in zip(LETTERS3, perm)}
            for tag, q in (("clean", it["q_clean"]), ("dist", it["q_trig"])):
                lp = eng.choice_logprobs(
                    build_choice_prompt(q, Z2_ACTIONS, mapping, Z2_HEADER), LETTERS3)
                a = {mapping[L]: lp[L] for L in LETTERS3}
                rec[f"action_{tag}"] = max(a, key=a.get)
                rec[f"action_logprobs_{tag}"] = a
            fh.write(json.dumps(rec) + "\n"); fh.flush()
        except Exception:
            LOG.exception("z2 fail %s", it["sid"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fh.close()
    LOG.info("-> %s", path)


# ============================================================================
# Z4: 算力不足 —— D2 自报所需预算 / D3 预算门控
# ============================================================================
Z4_ACTIONS = {"solve": "I can solve this within the stated budget",
              "need_more": "I need substantially more reasoning than the budget allows",
              "abstain": "I cannot solve this at any budget"}
Z4_HEADER = "You are given a thinking budget of approximately {b} tokens. Which is true?"
Z4_VERBAL = ("Estimate how many tokens of step-by-step reasoning you need to solve the problem "
             "below. Reply with only an integer.\n\n{q}")
BUDGETS = [128, 256, 512, 1024, 2048, 4096]


def budget_forced(eng, problem, budget, seed, temperature=0.6):
    """s1 式 budget forcing: 到上限即强制闭合 thinking 并作答。"""
    import torch
    torch.manual_seed(seed)
    enc = eng.tok(eng.fmt(problem), return_tensors="pt", truncation=True,
                  max_length=eng.max_input, add_special_tokens=False).to(eng.dev)
    with torch.inference_mode():
        base = eng.model(input_ids=enc.input_ids, use_cache=True)
    past, cur, gen = base.past_key_values, enc.input_ids[:, -1:], []
    for _ in range(budget):
        with torch.inference_mode():
            o = eng.model(input_ids=cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        lg = o.logits[0, -1]
        nxt = int(lg.argmax()) if temperature <= 0 else int(
            torch.multinomial(torch.softmax(lg.float() / temperature, -1), 1))
        if eng.think_end_id is not None and nxt == eng.think_end_id:
            break
        gen.append(nxt)
        cur = torch.tensor([[nxt]], device=eng.dev)
    closer = eng.tok.encode("\n</think>\n\nThe final answer is \\boxed{", add_special_tokens=False)
    full = torch.cat([enc.input_ids,
                      torch.tensor([gen], dtype=torch.long, device=eng.dev) if gen
                      else torch.zeros((1, 0), dtype=torch.long, device=eng.dev),
                      torch.tensor([closer], dtype=torch.long, device=eng.dev)], 1)
    with torch.inference_mode():
        o = eng.model.generate(input_ids=full, max_new_tokens=48, do_sample=False,
                               pad_token_id=eng.tok.pad_token_id)
    txt = eng.tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return txt, len(gen)


def boxed_match(pred, gold):
    m = re.findall(r"\\boxed\{([^{}]*)\}", pred)
    cand = m[-1] if m else pred
    f = lambda s: (re.findall(r"-?\d+\.?\d*", str(s).replace(",", "")) or [None])[-1]
    a, b = f(cand), f(gold)
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a == b


def stage_collect_z4(args, out: Path):
    """样本来自 z4_final.jsonl (已筛出 full-budget 稳定正确、截断后答错的样本)。
    meta 内已有 avg_think_tokens / cut_think_tokens, 无需重跑 full-budget 筛选。
    D2: 自报所需 token vs 真实 b*; D3: 声明预算是否驱动 NEED_MORE(题内 margin 斜率)。"""
    import torch
    items = load_pool(args, "Z4")
    eng = Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                 args.quantize_4bit, args.trust_remote_code)
    path = out / "z4_raw.jsonl"
    if path.exists() and not args.resume:
        path.unlink()
    done = {json.loads(l)["qid"] for l in path.open()} if (args.resume and path.exists()) else set()
    from tqdm.auto import tqdm
    fh = path.open("a")
    for i, it in enumerate(tqdm(items, desc="Z4")):
        if it["sid"] in done:
            continue
        try:
            sd = args.seed + i * 7919
            gold = it["answer"]
            prob = it["q_trig"]
            # compute-utility 曲线 -> b*  (池子已保证 full budget 能做对)
            accs, used = {}, {}
            for B in BUDGETS:
                hits, u = [], []
                for sN in range(args.n_samples):
                    txt, nt = budget_forced(eng, prob, B, sd + sN * 131)
                    hits.append(int(boxed_match(txt, gold))); u.append(nt)
                accs[B] = float(np.mean(hits)); used[B] = float(np.mean(u))
            bstar = next((B for B in BUDGETS if accs[B] >= args.acc_threshold), None)
            # D2: 自报所需 token
            v = eng.gen(Z4_VERBAL.format(q=prob), max_new=8, seed=sd + 1, skip_think=True)[0]
            verbal = parse_int(v)
            # D3: 各声明预算下的三选一 margin (单 token 打分 + 配平)
            perm = PERMS3[i % len(PERMS3)]
            mapping = {L: list(Z4_ACTIONS)[LETTERS3.index(p)] for L, p in zip(LETTERS3, perm)}
            gate = {}
            for B in BUDGETS:
                lp = eng.choice_logprobs(
                    build_choice_prompt(prob, Z4_ACTIONS, mapping,
                                        Z4_HEADER.format(b=B)), LETTERS3)
                gate[str(B)] = {mapping[L]: lp[L] for L in LETTERS3}
            fh.write(json.dumps(dict(
                qid=it["sid"], domain=it["domain"], template_id=it["template_id"],
                level=it["meta"].get("level", ""), gold=gold,
                acc_by_budget={str(k): v for k, v in accs.items()},
                used_by_budget={str(k): v for k, v in used.items()},
                b_star=bstar, verbal_demand=verbal,
                pool_avg_think=it["meta"].get("avg_think_tokens"),
                gate_logprobs=gate)) + "\n")
            fh.flush()
        except Exception:
            LOG.exception("z4 fail %s", it["sid"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fh.close()
    LOG.info("-> %s", path)


# ============================================================================
# Z6: 校准失败 —— D2 自评可答性 / D3 弃答门控
# ============================================================================
Z6_ACTIONS = {"answer": "This question has a definite answer that I can give",
              "abstain": "This question cannot be answered as posed",
              "unsure": "I am not sure whether this question can be answered"}
Z6_HEADER = "Consider the question below. Which statement is true?"
Z6_VERBAL = ("Is the question below answerable as posed? Reply with only Yes or No.\n\n{q}")
Z6_PERMIT = ("If the question is unanswerable or based on a false premise, say so explicitly.\n\n{q}")


def load_z6_items(args):
    """不可答样本来自 z6_final.jsonl。可答对照**从本项目已有池子取**, 不引入外部数据:
    优先用 Z6 自身的 q_clean(弃答许可版对应的裸问题不适用), 因此改用 Z2 的 q_clean
    (这些是模型能答对的干净问题) 或 Z1 中 knowledge 充分的头部实体。"""
    una = load_pool(args, "Z6")
    for x in una:
        x["answerable"] = 0
    n = len(una)
    ans = []
    for src in ("Z2", "Z1"):
        if len(ans) >= n:
            break
        try:
            rows = read_jsonl(Path(args.pool_dir) / f"{src.lower()}_final.jsonl")
        except Exception:
            continue
        for r in rows:
            if len(ans) >= n:
                break
            gold = r.get("answer", "")
            if gold in ("UNANSWERABLE", "UNKNOWN_ENTITY", ""):
                continue
            q = r.get("q_clean") or r.get("q_trig")
            if not q:
                continue
            ans.append(dict(sid=f"ctl_{src}_{r['sid']}", q_trig=q, q_clean=q,
                            answer=gold, aliases=r.get("answer_aliases", []),
                            domain=r.get("domain", ""), template_id=f"control-{src}",
                            meta={}, answerable=1))
    if not ans:
        LOG.warning("未找到可答对照; D2/D3 的判别力检验将不可用")
    items = una + ans
    np.random.RandomState(args.seed).shuffle(items)
    return items


def stage_collect_z6(args, out: Path):
    """D2: 自评可答性 vs 真值; D3: 弃答行动是否跟随真值 + 许可的增量。"""
    import torch
    items = load_z6_items(args)
    if not items:
        raise SystemExit("Z6 无样本")
    LOG.info("Z6: %d 条 (不可答 %d / 可答对照 %d)", len(items),
             sum(1 for x in items if x["answerable"] == 0),
             sum(1 for x in items if x["answerable"] == 1))
    eng = Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                 args.quantize_4bit, args.trust_remote_code)
    path = out / "z6_raw.jsonl"
    if path.exists() and not args.resume:
        path.unlink()
    done = {json.loads(l)["qid"] for l in path.open()} if (args.resume and path.exists()) else set()
    from tqdm.auto import tqdm
    fh = path.open("a")
    for i, it in enumerate(tqdm(items, desc="Z6")):
        if it["sid"] in done:
            continue
        try:
            sd = args.seed + i * 7919
            q = it["q_trig"]
            rec = dict(qid=it["sid"], answerable=it["answerable"],
                       domain=it.get("domain", ""), template_id=it.get("template_id", ""))
            for tag, p in (("plain", q), ("permit", Z6_PERMIT.format(q=q))):
                g = eng.gen(p, max_new=96, seed=sd)[0]
                rec[f"abstained_{tag}"] = int(bool(ABSTAIN_RE.search(g)))
                if it["answerable"] == 1:
                    rec[f"correct_{tag}"] = int(hit(g, it.get("answer", ""),
                                                    it.get("aliases", [])))
            v = eng.gen(Z6_VERBAL.format(q=q), max_new=4, seed=sd + 1, skip_think=True)[0]
            rec["verbal_answerable"] = (1 if re.match(r"\s*yes", v, re.I)
                                        else 0 if re.match(r"\s*no", v, re.I) else None)
            perm = PERMS3[i % len(PERMS3)]
            mapping = {L: list(Z6_ACTIONS)[LETTERS3.index(p)] for L, p in zip(LETTERS3, perm)}
            lp = eng.choice_logprobs(
                build_choice_prompt(q, Z6_ACTIONS, mapping, Z6_HEADER), LETTERS3)
            a = {mapping[L]: lp[L] for L in LETTERS3}
            rec["action"] = max(a, key=a.get); rec["action_logprobs"] = a
            fh.write(json.dumps(rec) + "\n"); fh.flush()
        except Exception:
            LOG.exception("z6 fail %s", it["sid"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fh.close()
    LOG.info("-> %s", path)


# ============================================================================
# 汇总: D2 / D3 的统一度量 + 签名矩阵
# ============================================================================
def _spearman(a, b):
    from scipy.stats import spearmanr
    m = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
         and np.isfinite(x) and np.isfinite(y)]
    if len(m) < 20:
        return None, None, len(m)
    r = spearmanr([x for x, _ in m], [y for _, y in m])
    return float(r.statistic), float(r.pvalue), len(m)


def _fisher(a1, a0, b1, b0):
    """Fisher 精确检验。零格出现时用 Haldane-Anscombe 校正报告有限 OR,
    避免 inf 无法进入表格/无法做元分析。"""
    from scipy.stats import fisher_exact
    if min(a1 + a0, b1 + b0) == 0:
        return None, None
    odds, p = fisher_exact([[a1, a0], [b1, b0]])
    if not np.isfinite(odds) or odds == 0:
        odds = ((a1 + .5) * (b0 + .5)) / ((a0 + .5) * (b1 + .5))
    return float(odds), float(p)


def d2_d3_z1(out: Path):
    rows = read_jsonl(out / "z1_raw.jsonl")
    know = [r["knowledge"] for r in rows]
    verb = [r.get("verbal_conf") for r in rows]
    # D2: 口头置信度 vs 真实知识
    rv, pv, nv = _spearman(verb, know)
    # 内部代理: 自洽度 vs 真实知识 (probe 的行为代理; 真正的 probe 见 D1)
    rc, pc, nc = _spearman([r.get("self_consistency") for r in rows], know)
    d2 = {"metric": "Spearman(verbalized confidence, true knowledge)",
          "rho_verbal": None if rv is None else round(rv, 4), "p_verbal": pv, "n": nv,
          "rho_internal_proxy": None if rc is None else round(rc, 4), "p_internal": pc,
          "gap": (None if (rv is None or rc is None) else round(rc - rv, 4)),
          "note": "gap>0 = 内部信号强于口头自评(自省失败)"}
    # D3: 低知识是否驱动 SEARCH
    lo = [r for r in rows if r["knowledge"] <= 0.25]
    hi = [r for r in rows if r["knowledge"] >= 0.75]
    a1 = sum(r["action"] == "search" for r in lo); a0 = len(lo) - a1
    b1 = sum(r["action"] == "search" for r in hi); b0 = len(hi) - b1
    odds, p = _fisher(a1, a0, b1, b0)
    # 连续版: margin(search - answer) vs knowledge
    marg = [r["action_logprobs"]["search"] - r["action_logprobs"]["answer"] for r in rows]
    rm, pm, nm = _spearman(marg, know)
    d3 = {"metric": "SEARCH rate | low vs high true knowledge",
          "n_low": len(lo), "n_high": len(hi),
          "search_rate_low_knowledge": round(a1 / max(len(lo), 1), 4),
          "search_rate_high_knowledge": round(b1 / max(len(hi), 1), 4),
          "odds_ratio": None if odds is None else round(odds, 3), "fisher_p": p,
          "margin_vs_knowledge_spearman": None if rm is None else round(rm, 4),
          "margin_p": pm,
          "gated": bool(odds is not None and odds > 1 and p is not None and p < 0.05),
          "note": "margin 为负相关 = 知识越充分越不倾向搜索(门控接通)"}
    return d2, d3


def d2_d3_z2(out: Path):
    rows = read_jsonl(out / "z2_raw.jsonl")
    # D2: 能否口头识别"存在无关信息" —— 用 dist vs clean 的判别力
    yd = [r.get("verbal_irrelevant_dist") for r in rows]
    yc = [r.get("verbal_irrelevant_clean") for r in rows]
    vd = [x for x in yd if x is not None]; vc = [x for x in yc if x is not None]
    odds, p = _fisher(sum(vd), len(vd) - sum(vd), sum(vc), len(vc) - sum(vc))
    d2 = {"metric": "verbalized 'contains irrelevant info': distracted vs clean",
          "yes_rate_distracted": round(float(np.mean(vd)), 4) if vd else None,
          "yes_rate_clean": round(float(np.mean(vc)), 4) if vc else None,
          "odds_ratio": None if odds is None else round(odds, 3), "fisher_p": p,
          "n": len(vd),
          "note": "odds>1 且显著 = 模型能口头识别干扰的存在"}
    # D3: 给"可忽略"许可后是否改善 (配对)
    from scipy.stats import wilcoxon
    a = np.array([r["acc_dist"] for r in rows], float)
    b = np.array([r["acc_dist_permit"] for r in rows], float)
    c = np.array([r["acc_clean"] for r in rows], float)
    d = b - a
    try:
        w, pw = wilcoxon(b, a, alternative="greater")
    except ValueError:
        w, pw = None, None
    gap0 = float(np.mean(c - a)); gap1 = float(np.mean(c - b))
    d3 = {"metric": "accuracy gain from 'you may ignore irrelevant info' permit",
          "acc_clean": round(float(c.mean()), 4),
          "acc_distracted": round(float(a.mean()), 4),
          "acc_distracted_with_permit": round(float(b.mean()), 4),
          "mean_gain": round(float(d.mean()), 4),
          "wilcoxon_p_one_sided_greater": pw,
          "gap_closed_fraction": round(1 - gap1 / gap0, 4) if abs(gap0) > 1e-9 else None,
          "n": len(rows),
          "gated": bool(pw is not None and pw < 0.05 and d.mean() > 0),
          "note": "许可即门控信号可用性; 显著改善 = 模型有能力忽略但默认未启用"}
    return d2, d3


def d2_d3_z4(out: Path):
    rows = read_jsonl(out / "z4_raw.jsonl")
    CENSOR = max(BUDGETS) * 2
    bstar = [(r["b_star"] if r["b_star"] else CENSOR) for r in rows]
    # D2: 自报所需 token vs 真实 b*
    rv, pv, nv = _spearman([r.get("verbal_demand") for r in rows], bstar)
    # 内部代理: 128 预算下的实际用量(内部对难度的响应)
    # 内部代理: 最小预算下的准确率(负相关于需求) —— 比 used 更稳健,
    # used 在小预算下常被 cap 截断成常数而退化为 nan
    proxy = [-(r["acc_by_budget"].get(str(min(BUDGETS))) or 0.0) for r in rows]
    rc, pc, nc = _spearman(proxy, bstar)
    if rc is None or not np.isfinite(rc):
        proxy = [r["used_by_budget"].get(str(max(BUDGETS))) for r in rows]
        rc, pc, nc = _spearman(proxy, bstar)
    d2 = {"metric": "Spearman(verbalized token demand, true b*)",
          "rho_verbal": None if rv is None else round(rv, 4), "p_verbal": pv, "n": nv,
          "rho_internal_proxy": None if rc is None else round(rc, 4), "p_internal": pc,
          "gap": (None if (rv is None or rc is None) else round(rc - rv, 4)),
          "note": "rho_verbal<=0 = 口头自报无效甚至反向"}
    # D3: 题内 margin(need_more - solve) 随 log2(声明预算) 的斜率
    from scipy.stats import ttest_1samp, wilcoxon
    slopes, ins_suf = [], []
    for r in rows:
        g = r.get("gate_logprobs", {})
        xs, ms = [], []
        for B in BUDGETS:
            v = g.get(str(B))
            if v and "need_more" in v and "solve" in v:
                xs.append(math.log2(B)); ms.append(v["need_more"] - v["solve"])
        if len(xs) >= 3:
            slopes.append(float(np.polyfit(xs, ms, 1)[0]))
            bs = r["b_star"]
            if bs:
                lo = [m for x, m in zip(xs, ms) if 2 ** x < bs]
                hi = [m for x, m in zip(xs, ms) if 2 ** x >= bs]
                if lo and hi:
                    ins_suf.append(float(np.mean(lo) - np.mean(hi)))
    sl = np.array(slopes)
    st = {}
    if len(sl) >= 8:
        t, pt = ttest_1samp(sl, 0.0, alternative="less")
        st = {"t": round(float(t), 3), "p_one_sided_less": float(pt)}
    d3 = {"metric": "within-item slope of margin(NEED_MORE - SOLVE) vs log2(stated budget)",
          "n_items": len(sl),
          "mean_slope": round(float(sl.mean()), 5) if len(sl) else None,
          "frac_negative": round(float((sl < 0).mean()), 4) if len(sl) else None,
          **st,
          "margin_insufficient_minus_sufficient": (round(float(np.mean(ins_suf)), 5)
                                                   if ins_suf else None),
          "gated": bool(st.get("p_one_sided_less", 1) < 0.05 and len(sl) and sl.mean() < 0),
          "note": ("差分设计: 长度伪影对固定标签恒定, 在同题跨预算差分中抵消。"
                   "斜率显著<0 = 声明预算驱动了元决策")}
    return d2, d3


def d2_d3_z6(out: Path):
    rows = read_jsonl(out / "z6_raw.jsonl")
    una = [r for r in rows if r["answerable"] == 0]
    ans = [r for r in rows if r["answerable"] == 1]
    # D2: 口头自评可答性 vs 真值
    vu = [r["verbal_answerable"] for r in una if r["verbal_answerable"] is not None]
    va = [r["verbal_answerable"] for r in ans if r["verbal_answerable"] is not None]
    odds, p = _fisher(sum(1 for x in vu if x == 0), sum(1 for x in vu if x == 1),
                      sum(1 for x in va if x == 0), sum(1 for x in va if x == 1))
    d2 = {"metric": "verbalized answerability vs ground truth",
          "says_unanswerable_rate_on_unanswerable": (round(1 - float(np.mean(vu)), 4) if vu else None),
          "says_unanswerable_rate_on_answerable": (round(1 - float(np.mean(va)), 4) if va else None),
          "odds_ratio": None if odds is None else round(odds, 3), "fisher_p": p,
          "n": len(vu) + len(va),
          "note": "odds>1 且显著 = 口头自评能区分可答性"}
    # D3: 弃答行动是否跟随真值 + 许可的增量
    a1 = sum(r["action"] == "abstain" for r in una); a0 = len(una) - a1
    b1 = sum(r["action"] == "abstain" for r in ans); b0 = len(ans) - b1
    o2, p2 = _fisher(a1, a0, b1, b0)
    permit_gain = None
    if una:
        pl = float(np.mean([r["abstained_plain"] for r in una]))
        pm = float(np.mean([r["abstained_permit"] for r in una]))
        permit_gain = round(pm - pl, 4)
    d3 = {"metric": "ABSTAIN rate | unanswerable vs answerable",
          "abstain_rate_unanswerable": round(a1 / max(len(una), 1), 4),
          "abstain_rate_answerable": round(b1 / max(len(ans), 1), 4),
          "odds_ratio": None if o2 is None else round(o2, 3), "fisher_p": p2,
          "abstain_gain_from_permit_on_unanswerable": permit_gain,
          "n_unanswerable": len(una), "n_answerable": len(ans),
          "gated": bool(o2 is not None and o2 > 1 and p2 is not None and p2 < 0.05),
          "note": "许可增益大 = 能力存在但默认未启用(表达层门控)"}
    return d2, d3


def stage_assemble(args, out: Path):
    matrix = {"stressors": {}, "dimensions": ["D1_representation", "D2_introspection",
                                              "D3_gating"]}
    # ---- D1 ----
    d1p = out / "d1_representation.json"
    d1 = json.loads(d1p.read_text()) if d1p.exists() else {"per_stressor": {}}
    # ---- D2/D3 ----
    handlers = {"Z1": d2_d3_z1, "Z2": d2_d3_z2, "Z4": d2_d3_z4, "Z6": d2_d3_z6}
    for z in STRESSORS:
        cell = {"name": SIG_NAMES[z]}
        cell["D1_representation"] = d1.get("per_stressor", {}).get(z, {"missing": True})
        raw = out / f"{z.lower()}_raw.jsonl"
        if raw.exists():
            try:
                d2, d3 = handlers[z](out)
                cell["D2_introspection"] = d2
                cell["D3_gating"] = d3
            except Exception as e:
                cell["D2_introspection"] = {"error": str(e)}
                cell["D3_gating"] = {"error": str(e)}
        else:
            cell["D2_introspection"] = {"missing": f"缺 {raw.name}"}
            cell["D3_gating"] = {"missing": f"缺 {raw.name}"}
        matrix["stressors"][z] = cell

    # ---- 复用既有结果 ----
    if args.reuse_toolgate and Path(args.reuse_toolgate).exists():
        tg = json.loads(Path(args.reuse_toolgate).read_text())
        matrix["stressors"]["Z1"].setdefault("external_evidence", {})["tool_gate"] = {
            "search_rate_unknown": tg.get("by_prior", {}).get("unknown", {}).get("search_rate"),
            "search_rate_known": tg.get("by_prior", {}).get("known", {}).get("search_rate"),
            "odds_ratio": tg.get("hypothesis_H_search_higher_on_unknown", {}).get("odds_ratio"),
            "probe_knows_auroc": tg.get("probe_knows_dontknow", {}).get("auroc"),
            "probe_knows_layer": tg.get("probe_knows_dontknow", {}).get("layer"),
            "probe_search_auroc": tg.get("probe_predicts_search", {}).get("auroc"),
            "probe_search_layer": tg.get("probe_predicts_search", {}).get("layer")}
    if args.reuse_budget and Path(args.reuse_budget).exists():
        bm = json.loads(Path(args.reuse_budget).read_text())
        matrix["stressors"]["Z4"].setdefault("external_evidence", {})["budget_meta"] = {
            "b_star_spread_ratio": bm.get("curve", {}).get("b_star_distribution", {}).get("spread_ratio"),
            "overthink_rate": bm.get("curve", {}).get("overthink_rate"),
            "self_report_spearman": bm.get("self_report_vs_true_demand", {}).get("spearman"),
            "probe_demand_spearman": (bm.get("probes", {}).get("demand_regression", {})
                                      .get("0", {}).get("spearman"))}

    # ---- 紧凑摘要表 ----
    summary = {}
    for z, c in matrix["stressors"].items():
        r1 = c.get("D1_representation", {})
        r2 = c.get("D2_introspection", {})
        r3 = c.get("D3_gating", {})
        summary[z] = {
            "D1": (f"AUROC {r1.get('peak_auroc')} @L{r1.get('peak_layer')} "
                   f"(onset L{r1.get('onset_layer')}, rel {r1.get('onset_rel_depth')})"
                   if "peak_auroc" in r1 else "—"),
            "D2": (f"rho_verbal={r2.get('rho_verbal')} vs internal={r2.get('rho_internal_proxy')}"
                   if "rho_verbal" in r2 else
                   (f"OR={r2.get('odds_ratio')} p={r2.get('fisher_p')}"
                    if "odds_ratio" in r2 else "—")),
            "D3": (("GATED" if r3.get("gated") else "NOT GATED") +
                   (f" (OR={r3.get('odds_ratio')})" if r3.get("odds_ratio") else
                    f" (slope={r3.get('mean_slope')})" if r3.get("mean_slope") is not None else
                    f" (gain={r3.get('mean_gain')})" if r3.get("mean_gain") is not None else "")
                   if ("gated" in r3) else "—"),
        }
    matrix["summary_table"] = summary
    matrix["interpretation"] = (
        "D1 强 + D3 GATED => 该状态被表征且驱动了正确行动(闭环); "
        "D1 强 + D3 NOT GATED => 表征存在但未整合进决策(整合缺失); "
        "D2 弱而 D1 强 => 自省失败, 外部读取有价值。"
        "不同 stressor 落在不同格局 => 没有 universal winner 的诊断/治疗。")
    write_json(matrix, out / "signature_matrix.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + matrix["interpretation"])
    LOG.info("-> %s", out / "signature_matrix.json")


def build_parser():
    p = argparse.ArgumentParser(description="四类 stressor 的内部签名矩阵")
    p.add_argument("--stage", required=True,
                   choices=["d1", "collect-z1", "collect-z2", "collect-z4", "collect-z6",
                            "assemble"])
    p.add_argument("--out", required=True)
    p.add_argument("--features", help="d1: router 特征目录")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-items", type=int, default=0, help="0=全用池子")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--max-layers", type=int, default=33)
    p.add_argument("--n-boot", type=int, default=100)
    p.add_argument("--acc-threshold", type=float, default=0.5)
    p.add_argument("--pool-dir", default="data/processed",
                   help="10_screen.py 产出的 {z}_final.jsonl 所在目录")
    p.add_argument("--reuse-toolgate", default=None)
    p.add_argument("--reuse-budget", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p


def main():
    a = build_parser().parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_json(vars(a), out / f"config_{a.stage}.json")
    {"d1": stage_d1, "collect-z1": stage_collect_z1, "collect-z2": stage_collect_z2,
     "collect-z4": stage_collect_z4, "collect-z6": stage_collect_z6,
     "assemble": stage_assemble}[a.stage](a, out)


if __name__ == "__main__":
    main()
