"""Phase 3.3 因果注脚: activation patching (clean -> trig)。
依赖: {z}_final.jsonl (Z2/Z3, 需有 q_clean/q_trig 反事实对) + 41 给出的 top 层。
对每个 flip 样本: 在 (layer, position) 网格上把 trig 前向的残差流替换为 clean 前向
对应位置的激活, 度量 logit-difference 恢复率。产出热图数据。

对齐说明: clean/trig 仅在干扰句/触发词处不同 -> 用"公共后缀对齐"(从序列尾部对齐,
问题主体在尾部一致), patch 只作用于对齐区。这就是构造时控制文本结构的原因。
用法: python scripts/42_patch_validate.py --model unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit --stressor z2 --layers 12 16 20 24 --limit 100
"""
import argparse, json
import numpy as np
import torch
from common import read_jsonl, DATA

def first_token_id(tok, text):
    ids = tok.encode(" " + str(text), add_special_tokens=False)
    if not ids:
        ids = tok.encode(str(text), add_special_tokens=False)
    return ids[0]

def logit_diff(logits, tok, gold, wrong_id):
    gold_id = first_token_id(tok, gold)
    return (logits[0, -1, gold_id] - logits[0, -1, wrong_id]).item()

@torch.no_grad()
def run(model, tok, prompt, patch=None):
    """patch: dict layer -> (positions_in_this_run, source_states [P, d])"""
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    handles = []
    if patch:
        layers = model.model.layers
        def mk(l):
            pos, src = patch[l]
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h[0, pos, :] = src.to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            return hook
        handles = [layers[l].register_forward_hook(mk(l)) for l in patch]
    out = model(**enc, output_hidden_states=True)
    for h in handles:
        h.remove()
    return out, enc.input_ids.shape[1]

def main(model_name, stressor, layers, limit):
    try:
        from torch._native.registry import deregister_op_overrides
        deregister_op_overrides(disable_op_symbols="bmm")
    except (ImportError, AttributeError):
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda", local_files_only=True)
    model.eval()
    samples = [s for s in read_jsonl(DATA / f"processed/{stressor}_final.jsonl")
               if s["q_clean"] != s["q_trig"]][:limit]
    rows = []
    for s in samples:
        gold = s["answer"]
        out_c, Tc = run(model, tok, s["q_clean"])
        out_t, Tt = run(model, tok, s["q_trig"])
        gold_id = first_token_id(tok, gold)
        trig_logits = out_t.logits[0, -1].clone()
        trig_logits[gold_id] = -float("inf")
        wrong_id = int(trig_logits.argmax())
        base_c = logit_diff(out_c.logits, tok, gold, wrong_id)
        base_t = logit_diff(out_t.logits, tok, gold, wrong_id)
        if base_c <= base_t:
            continue  # 无翻转信号, 跳过
        K = min(Tc, Tt)  # 公共后缀长度
        hs_c = torch.stack(out_c.hidden_states, 0)            # [L+1, 1, Tc, d]
        for l in layers:
            # patch 后缀区最后 K 个位置 (含问题主体与生成起点)
            pos_t = list(range(Tt - K, Tt))
            src = hs_c[l + 1, 0, Tc - K:Tc, :]                # 第 l 层块输出 = hidden_states[l+1]
            out_p, _ = run(model, tok, s["q_trig"], patch={l: (pos_t, src)})
            d = logit_diff(out_p.logits, tok, gold, wrong_id)
            recov = (d - base_t) / (base_c - base_t + 1e-9)   # 0=无效, 1=完全恢复
            rows.append(dict(sid=s["sid"], layer=l, recovery=round(recov, 4),
                             base_clean=round(base_c, 3), base_trig=round(base_t, 3)))
    out = DATA / f"results/patch_{stressor}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # 汇总
    by_l = {}
    for r in rows:
        by_l.setdefault(r["layer"], []).append(r["recovery"])
    print(f"[patch:{stressor}] n_samples={len(set(r['sid'] for r in rows))}")
    for l in sorted(by_l):
        v = np.array(by_l[l])
        print(f"  layer {l:>2}: mean recovery={v.mean():.3f}  frac>0.5={np.mean(v > 0.5):.1%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--stressor", default="z2", choices=["z2", "z3"])
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 12, 16, 20, 24])
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    main(a.model, a.stressor, a.layers, a.limit)
