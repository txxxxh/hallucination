"""Phase 3.1 特征提取: 对 {z}_final.jsonl 每条样本跑一次前向, 提取四族特征。
依赖: 只需要 10_screen.py 的输出 (不依赖治疗矩阵结果), 可与矩阵实验并行。

F1 残差流: 每层 residual stream 在"最后一个输入 token"位置的激活 [L+1, d]
F2 logit-lens 轨迹: 每层投影到词表后的 top1 logit / 熵 / top2 gap [L+1, 3]
F3 注意力概要: 每层每头对各句子的注意力分布熵 + 对已知可疑 span 的 attention mass
F4 不确定性标量: 首生成 token 的熵 / top2 gap / prompt 平均 loss

注意:
- 用 HF transformers 而非 vLLM (需要 hidden states / attentions)。
- 提取位置 = prompt 最后一个 token (作答前状态)。Z4 的信号可能不在输入端
  (预注册 fallback: 若 Z4 probe ~ chance, 改用生成中间态, 见 --gen_states)。
- clean 配对样本也提取 (5类分类的"无幻觉"类 + patching 的 clean 源)。
用法: python scripts/40_extract_features.py --model unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit --stressors z1 z2 z4 z6
"""
import argparse, json
import numpy as np
import torch
from pathlib import Path
from common import read_jsonl, DATA

FEATURE_VERSION = 2  # v2 fixes suspicious-span offsets under chat templates

@torch.no_grad()
def extract_one(model, tok, prompt, spans=None, device="cuda"):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    out = model(**enc, output_hidden_states=True, output_attentions=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0, -1, :]            # [L+1, d] 最后输入 token
    # F2: logit lens — 每层过 final norm + lm_head
    norm = model.model.norm if hasattr(model.model, "norm") else model.model.final_layernorm
    lens = model.lm_head(norm(torch.stack(out.hidden_states, 0)[:, 0, -1, :]))  # [L+1, V]
    probs = torch.softmax(lens.float(), -1)
    top2 = probs.topk(2, -1).values
    f2 = torch.stack([lens.max(-1).values.float(),
                      -(probs * (probs + 1e-9).log()).sum(-1),
                      (top2[:, 0] - top2[:, 1])], -1)              # [L+1, 3]
    # F3: 句级注意力熵 (最后 token 的注意力按句子聚合)
    ids = enc.input_ids[0]
    dots = (ids == tok.encode(".", add_special_tokens=False)[-1]).nonzero().flatten().tolist()
    bounds = [0] + [d + 1 for d in dots] + [len(ids)]
    sent_spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    att = torch.stack([a[0, :, -1, :] for a in out.attentions], 0)  # [L, H, T]
    sent_mass = torch.stack([att[:, :, a:b].sum(-1) for a, b in sent_spans], -1)  # [L,H,S]
    p = sent_mass / (sent_mass.sum(-1, keepdim=True) + 1e-9)
    f3_ent = -(p * (p + 1e-9).log()).sum(-1)                       # [L, H] 句级注意力熵
    # 对已知可疑 span 的 mass (构造时知道干扰句/触发词的字符位置 -> 转 token 位置)
    f3_susp = torch.zeros_like(f3_ent[:, 0])
    if spans:
        # find_spans returns raw-prompt offsets; tokenizer offsets refer to the
        # complete rendered chat template, so shift them before matching.
        prompt_start = text.find(prompt)
        if prompt_start < 0:
            raise ValueError("cannot locate raw prompt inside rendered chat template")
        rendered_spans = [(prompt_start + a, prompt_start + b) for a, b in spans]
        offs = tok(text, return_offsets_mapping=True, truncation=True, max_length=4096)["offset_mapping"]
        idxs = [i for i, (a, b) in enumerate(offs)
                if b > a and any(a < e and b > s for s, e in rendered_spans)]
        if idxs:
            f3_susp = att[:, :, idxs].sum(-1).mean(1)              # [L] 头间平均
    # F4: 首生成 token 分布
    logits0 = out.logits[0, -1].float()
    p0 = torch.softmax(logits0, -1)
    t2 = p0.topk(2).values
    f4 = torch.tensor([-(p0 * (p0 + 1e-9).log()).sum().item(),
                       (t2[0] - t2[1]).item(),
                       logits0.max().item()])
    return dict(feature_version=np.array(FEATURE_VERSION, dtype=np.int64),
                f1=hs.float().cpu().numpy().astype(np.float16),
                f2=f2.cpu().numpy(), f3_ent=f3_ent.float().cpu().numpy().astype(np.float16),
                f3_susp=f3_susp.float().cpu().numpy(), f4=f4.numpy())

def find_spans(sample):
    """构造信息 -> 可疑 span 的字符区间 (仅训练分析用, router 部署时不可用此特征)。"""
    q, spans = sample["q_trig"], []
    for d in sample["meta"].get("distractors", []):
        i = q.find(d.rstrip("."))
        if i >= 0:
            spans.append((i, i + len(d)))
    trig = sample["meta"].get("trigger", "")
    if trig and (i := q.find(trig)) >= 0:
        spans.append((i, i + len(trig)))
    return spans

def main(model_name, stressors, include_clean=True, limit=None):
    try:
        from torch._native.registry import deregister_op_overrides
        deregister_op_overrides(disable_op_symbols="bmm")
    except (ImportError, AttributeError):
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 device_map="cuda", attn_implementation="eager", local_files_only=True)
    model.eval()
    outdir = DATA / f"features/{model_name.split('/')[-1]}"
    outdir.mkdir(parents=True, exist_ok=True)
    index = []
    seen = {}
    required = {"feature_version", "f1", "f2", "f3_ent", "f3_susp", "f4"}

    def valid_feature(path):
        if not path.exists():
            return False
        try:
            with np.load(path) as d:
                return (required.issubset(d.files)
                        and int(d["feature_version"]) == FEATURE_VERSION)
        except Exception:
            return False

    def extract_or_resume(path, prompt, spans=None):
        if valid_feature(path):
            return False
        feats = extract_one(model, tok, prompt, spans)
        np.savez_compressed(path, **feats)
        return True

    for z in stressors:
        source = DATA / f"processed/{z}_final.jsonl"
        if not source.exists():
            print(f"[skip] {z}: {source} 不存在")
            continue
        samples = read_jsonl(source)
        if limit is not None:
            samples = samples[:limit]
        extracted = 0
        resumed = 0
        duplicates = 0
        for s in samples:
            sid = s["sid"]
            signature = (s["stressor"], s["q_trig"], s["q_clean"],
                         s["domain"], s["template_id"])
            if sid in seen:
                if seen[sid] != signature:
                    raise ValueError(
                        f"duplicate SID {sid!r} maps to different samples; "
                        "fix the input instead of overwriting its features"
                    )
                duplicates += 1
                continue
            seen[sid] = signature
            did_extract = extract_or_resume(
                outdir / f"{sid}.npz", s["q_trig"], find_spans(s)
            )
            extracted += did_extract
            resumed += not did_extract
            index.append(dict(sid=sid, label=s["stressor"], secondary=s.get("secondary_labels", []),
                              domain=s["domain"], template_id=s["template_id"], variant="trig"))
            if include_clean and s["q_clean"] != s["q_trig"]:
                did_extract = extract_or_resume(
                    outdir / f"{sid}__clean.npz", s["q_clean"]
                )
                extracted += did_extract
                resumed += not did_extract
                index.append(dict(sid=sid + "__clean", label="CLEAN", secondary=[],
                                  domain=s["domain"], template_id=s["template_id"], variant="clean"))
        print(f"[feat] {z}: {len(samples)} samples, {extracted} extracted, "
              f"{resumed} resumed, {duplicates} duplicate SID skipped")
    tmp_index = outdir / "index.jsonl.tmp"
    with open(tmp_index, "w") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")
    tmp_index.replace(outdir / "index.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--stressors", nargs="+", default=["z1", "z2", "z4", "z6"])
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug limit per stressor; default extracts every sample")
    a = ap.parse_args()
    main(a.model, a.stressors, include_clean=not a.no_clean, limit=a.limit)
