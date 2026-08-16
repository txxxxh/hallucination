#!/usr/bin/env python3
"""
z4_early_diagnosis.py — budget 不足 (Z4) 在生成过程中何时变得可诊断?

核心问题: Z4 的病因是生成时算力耗尽, 输入端可能无信号。测量在生成第 K 个 token
位置提取的 hidden state 对 "这题会不会因 budget 不足而失败" 的 probe AUROC 如何随 K 变化。

真值构造 (MATH + 推理模型):
  正例 Z4-fail: full budget 单次生成答对且 thinking 用量大, 截到 cut_ratio*thinking 后答错。
  负例-conv  : 同样单次答对, 但在同一截断位置答案已收敛 (截断也对) —— budget 充足型。
  难度匹配   : 正负例都取"需要较长推理"的题, 强迫 probe 学 budget 信号而非题目难度。

提取: 在同一次截断生成里, 于 K in {0,32,64,128,256,...} 位置取逐层 hidden。
  K=0 = prompt 末位 (作答前)。曲线 AUROC(K) 回答:
    单调上升 -> 有诊断窗口, 不在输入端;
    K=0 已高 -> 可前瞻诊断 (修正 "Z4 不可前瞻");
    始终 chance -> budget 耗尽表征不可读 (干净的 finding)。

用法:
  python z4_early_diagnosis.py --stage collect --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
     --output-dir out --max-pool 400 --full-budget 3500 --quantize-4bit
  python z4_early_diagnosis.py --stage analyze --output-dir out
"""
from __future__ import annotations
import argparse, gc, json, logging, re
from pathlib import Path
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("z4_early")

K_POSITIONS = [0, 32, 64, 128, 256, 512]     # 生成 token 位置; 0 = prompt 末位
MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def boxed(sol: str) -> str:
    m = re.search(r"\\boxed\{([^{}]+)\}", sol)
    return m.group(1).strip() if m else sol.strip()

def canon_num(s: str) -> Optional[str]:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", str(s).replace(",", ""))
    if not nums:
        return None
    try:
        v = float(nums[-1]); return str(int(v)) if v.is_integer() else f"{v:.6g}"
    except ValueError:
        return nums[-1]

def match(pred: str, gold: str) -> bool:
    # 优先 boxed, 再退到最后一个数
    m = re.search(r"\\boxed\{([^{}]+)\}", pred)
    p = m.group(1) if m else pred
    return canon_num(p) is not None and canon_num(p) == canon_num(gold)


class Engine:
    def __init__(self, model, device, dtype, max_input, quant4, trust):
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
        self.model.eval(); self.model.config.use_cache = True
        self.device = next(self.model.parameters()).device
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.hidden = int(self.model.config.hidden_size)
        self.think_end = self.tok.encode("</think>", add_special_tokens=False)

    def _fmt(self, user):
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template([{"role":"user","content":user}],
                                                tokenize=False, add_generation_prompt=True)
        return f"User: {user}\nAssistant:"

    def gen_full_batch(self, prompts, seed, max_new, temperature=0.6):
        """批量普通生成 (每个 prompt 一条, 用于筛正确并记录 thinking 用量)。"""
        import torch
        torch.manual_seed(seed)
        old_padding_side = self.tok.padding_side
        self.tok.padding_side = "left"
        enc = self.tok([self._fmt(p) for p in prompts], return_tensors="pt",
                       padding=True, truncation=True, max_length=self.max_input,
                       add_special_tokens=False).to(self.device)
        self.tok.padding_side = old_padding_side
        outs = []
        with torch.inference_mode():
            gen = self.model.generate(**enc, max_new_tokens=max_new,
                                      do_sample=temperature > 0, temperature=temperature or None,
                                      pad_token_id=self.tok.pad_token_id)
        for seq in gen:
            ids = seq[enc.input_ids.shape[1]:]
            txt = self.tok.decode(ids, skip_special_tokens=True)
            think_len = ids.tolist().index(self.think_end[-1]) if self.think_end[-1] in ids.tolist() else len(ids)
            outs.append((txt, think_len))
        return outs

    def gen_truncated_with_hidden(self, prompt, cut_tokens, seed, k_positions):
        """截断生成: 至多 cut_tokens 个 thinking token 后强制闭合并作答;
        沿途在 k_positions 记录逐层 hidden (prompt末位 + 生成中若干点)。
        返回 (final_text, correct_at_end, {K: hidden[L+1,H]})。"""
        import torch
        torch.manual_seed(seed)
        enc = self.tok(self._fmt(prompt), return_tensors="pt", truncation=True,
                       max_length=self.max_input, add_special_tokens=False).to(self.device)
        input_ids = enc.input_ids
        # K=0: prompt 末位 hidden
        hid = {}
        with torch.inference_mode():
            base = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        if 0 in k_positions:
            hid[0] = torch.stack([base.hidden_states[l][0, -1].float().cpu()
                                  for l in range(len(base.hidden_states))]).half()
        past = base.past_key_values
        cur = input_ids[:, -1:]
        generated = []
        want = sorted(k for k in k_positions if k > 0)
        wi = 0
        step = 0
        # 逐 token 生成 thinking, 到 cut_tokens 截停
        while step < cut_tokens:
            with torch.inference_mode():
                o = self.model(input_ids=cur, past_key_values=past,
                               output_hidden_states=(wi < len(want) and step + 1 == want[wi]),
                               use_cache=True)
            past = o.past_key_values
            nxt = o.logits[0, -1].argmax().view(1, 1)
            generated.append(int(nxt))
            step += 1
            if wi < len(want) and step == want[wi]:
                hid[want[wi]] = torch.stack([o.hidden_states[l][0, -1].float().cpu()
                                             for l in range(len(o.hidden_states))]).half()
                wi += 1
            cur = nxt
            if int(nxt) == self.think_end[-1]:
                break
        # 强制闭合 thinking + 作答
        closer = self.tok.encode("\n</think>\n\nThe final answer is \\boxed{", add_special_tokens=False)
        cur2 = torch.tensor([closer], device=self.device)
        with torch.inference_mode():
            o2 = self.model.generate(input_ids=torch.cat([input_ids, torch.tensor([generated], device=self.device), cur2], 1),
                                     max_new_tokens=32, do_sample=False, pad_token_id=self.tok.pad_token_id)
        tail = self.tok.decode(o2[0, input_ids.shape[1]:], skip_special_tokens=True)
        # 补齐未采到的 K (生成没到那么长): 用最后一个可用位置填 nan 标记
        return tail, hid


def stage_collect(args, out: Path):
    import torch
    from datasets import concatenate_datasets, load_dataset
    LOG.info("loading MATH test split from %d subject configs", len(MATH_CONFIGS))
    ds = concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", config, split="test")
        for config in MATH_CONFIGS
    ])
    rows = [r for r in ds if r["level"] in ("Level 3", "Level 4", "Level 5")]
    np.random.RandomState(args.seed).shuffle(rows)
    rows = rows[:args.max_pool]
    eng = Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                 args.quantize_4bit, args.trust_remote_code)
    hdir = out / "hidden"; hdir.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    if rec_path.exists() and not args.resume:
        rec_path.unlink()
    done = {json.loads(l)["qid"] for l in rec_path.open()} if (args.resume and rec_path.exists()) else set()
    from tqdm.auto import tqdm
    pending = [(i, r) for i, r in enumerate(rows) if f"math_{i:04d}" not in done]
    full_by_qid = {}
    for start in tqdm(range(0, len(pending), args.batch_size), desc="full-budget batches"):
        batch = pending[start:start + args.batch_size]
        prompts = [
            r["problem"] + "\nReason step by step, then give \\boxed{answer}."
            for _, r in batch
        ]
        outputs = eng.gen_full_batch(
            prompts, args.seed + batch[0][0], args.full_budget, temperature=0.6
        )
        for (i, _), output in zip(batch, outputs):
            full_by_qid[f"math_{i:04d}"] = output

    fh = rec_path.open("a")
    for i, r in enumerate(tqdm(rows, desc="collect")):
        qid = f"math_{i:04d}"
        if qid in done:
            continue
        try:
            prompt = r["problem"] + "\nReason step by step, then give \\boxed{answer}."
            gold = boxed(r["solution"])
            # 1) full budget 单次生成答对 + 记录 thinking 用量
            full = [full_by_qid[qid]]
            n_correct = sum(match(t, gold) for t, _ in full)
            if n_correct < 1:
                continue
            avg_think = float(np.mean([tl for _, tl in full]))
            if avg_think < args.min_think:
                continue
            cut = int(avg_think * args.cut_ratio)
            # 2) 截断生成 + 多点 hidden
            tail, hid = eng.gen_truncated_with_hidden(prompt, cut, args.seed + i, K_POSITIONS)
            trunc_correct = match(tail, gold)
            label = "z4_fail" if not trunc_correct else "conv"   # 截断答错=Z4正例; 截断仍对=收敛型负例
            torch.save({"qid": qid, "hidden": {k: v for k, v in hid.items()}}, hdir / f"{qid}.pt")
            fh.write(json.dumps(dict(qid=qid, level=r["level"], type=r["type"],
                                     gold=gold, avg_think=avg_think, cut_tokens=cut,
                                     trunc_correct=trunc_correct, label=label,
                                     k_available=sorted(hid.keys()),
                                     tail=tail[:200])) + "\n")
            fh.flush()
        except Exception as e:
            LOG.exception("fail %s", qid)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fh.close()
    LOG.info("done -> %s", rec_path)


def stage_analyze(args, out: Path):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    recs = [json.loads(l) for l in (out / "records.jsonl").open() if l.strip()]
    pos = [r for r in recs if r["label"] == "z4_fail"]
    neg = [r for r in recs if r["label"] == "conv"]
    LOG.info("Z4-fail=%d  conv=%d", len(pos), len(neg))
    if len(pos) < 10 or len(neg) < 10:
        LOG.warning("正/负例 < 10, 结果方差极大")

    # 载入所有 hidden, 按 K 组织
    data = {}
    for r in recs:
        pt = out / "hidden" / f"{r['qid']}.pt"
        if not pt.exists():
            continue
        h = torch.load(pt, map_location="cpu", weights_only=False)["hidden"]
        data[r["qid"]] = {k: v.float().numpy() for k, v in h.items()}

    res = {"n_pos": len(pos), "n_neg": len(neg), "auroc_by_K": {}}
    n_layers = None
    for K in K_POSITIONS:
        X, y = [], []
        for r in recs:
            d = data.get(r["qid"], {})
            if K not in d:
                continue                      # 该样本生成没到 K 位置, 跳过
            X.append(d[K]); y.append(int(r["label"] == "z4_fail"))
        if len(set(y)) < 2 or len(y) < 20:
            res["auroc_by_K"][K] = {"skipped": f"n={len(y)}"}
            continue
        X = np.stack(X); y = np.array(y); n_layers = X.shape[1]
        # 逐层扫 (排除 embedding 层), 取最优
        best = {"auroc": -1, "layer": None, "n": len(y), "curve": []}
        for l in range(1, n_layers):
            Xl = StandardScaler().fit_transform(X[:, l])
            p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
                                  Xl, y, cv=5, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, p)
            best["curve"].append((l, round(auc, 3)))
            if auc > best["auroc"]:
                best.update(auroc=round(auc, 4), layer=l)
        res["auroc_by_K"][K] = best
        LOG.info("K=%-4d n=%d best AUROC=%.3f @layer=%s", K, len(y), best["auroc"], best["layer"])

    # 曲线形态判读
    valid = [(K, res["auroc_by_K"][K]["auroc"]) for K in K_POSITIONS
             if isinstance(res["auroc_by_K"].get(K), dict) and "auroc" in res["auroc_by_K"][K]]
    if len(valid) >= 2:
        k0 = next((a for K, a in valid if K == 0), None)
        klast = valid[-1][1]
        res["interpretation"] = {
            "auroc_at_K0_prompt_end": k0,
            "auroc_at_last_K": klast,
            "monotone_increasing": all(valid[i][1] <= valid[i+1][1] + 0.03 for i in range(len(valid)-1)),
            "verdict": (
                "prospective (K0 already high)" if k0 and k0 >= 0.70 else
                "diagnostic window opens during generation" if klast - (k0 or 0) > 0.10 else
                "budget exhaustion not readable from hidden state (chance-level)"),
        }
    (out / "z4_analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res.get("interpretation", {}), indent=2, ensure_ascii=False))
    print("AUROC by K:", {K: res["auroc_by_K"][K].get("auroc") if isinstance(res["auroc_by_K"][K], dict) else None
                          for K in K_POSITIONS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "analyze"], default="collect")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-pool", type=int, default=400)
    ap.add_argument("--full-budget", type=int, default=3500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-think", type=int, default=400)
    ap.add_argument("--cut-ratio", type=float, default=0.3)
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
