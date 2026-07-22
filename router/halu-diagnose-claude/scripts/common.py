"""公共模块: schema / 推理封装 / 答案与弃答判定。"""
from __future__ import annotations
import json, re, hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------- schema
@dataclass
class Sample:
    sid: str                      # 唯一 id
    stressor: str                 # Z1/Z2/Z3/Z4/Z6 (主标签, 构造真值)
    secondary_labels: list = field(default_factory=list)  # 多标签, 如 Z1 样本附加 Z6
    domain: str = ""              # factual / math / multihop
    template_id: str = ""         # 构造模板 id (leave-one-template-out 用)
    intensity: float = 0.0        # 强度协变量: 流行度/干扰句数/budget比例/共现度
    q_clean: str = ""             # 干净版 (Z4 为 full-budget 版, 与 q_trig 相同文本)
    q_trig: str = ""              # 触发版 (进入矩阵的版本)
    answer: str = ""              # 标准答案; Z6 为 "UNANSWERABLE"
    answer_aliases: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # 干扰句位置/触发词/gold passage 等

    def dump(self):
        return json.dumps(asdict(self), ensure_ascii=False)

def sid_of(text: str, prefix: str) -> str:
    return f"{prefix}-{hashlib.md5(text.encode()).hexdigest()[:10]}"

def write_jsonl(samples, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write((s.dump() if isinstance(s, Sample) else json.dumps(s, ensure_ascii=False)) + "\n")
    print(f"[write] {len(samples)} -> {path}")

def read_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# ---------------------------------------------------------------- 推理封装 (Transformers)
class LM:
    """Transformers 离线推理封装，保留原 vLLM 版本的 chat 接口。

    R1 类模型支持 max_think 两阶段截断。批大小可通过环境变量
    HALLU_BATCH_SIZE 调整；tp 参数仅为兼容旧命令行保留。
    """
    def __init__(self, model_name: str, max_model_len: int = 16384, tp: int = 1):
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.is_reasoner = "R1" in model_name or "r1" in model_name
        self.max_model_len = max_model_len
        self.batch_size = max(1, int(os.environ.get("HALLU_BATCH_SIZE", "4")))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.llm.eval()

    def _render(self, prompts):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        return [
            self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

    def _generate(self, texts, temperature, n, max_tokens, seed, top_p=0.95):
        import torch

        grouped = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len,
            ).to(self.device)
            do_sample = temperature is not None and temperature > 0
            kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                num_return_sequences=n,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            if do_sample:
                kwargs.update(temperature=temperature, top_p=top_p)
                if seed is not None:
                    torch.manual_seed(seed + start)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + start)
            with torch.inference_mode():
                generated = self.llm.generate(**encoded, **kwargs)
            prompt_width = encoded["input_ids"].shape[1]
            decoded = self.tok.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            grouped.extend(decoded[i:i + n] for i in range(0, len(decoded), n))
        return grouped

    def chat(self, prompts, temperature=0.0, n=1, max_tokens=2048,
             max_think=None, seed=1234):
        """返回 list[list[str]]，每个 prompt 对应 n 个补全。"""
        texts = self._render(prompts)
        if self.is_reasoner and max_think is not None:
            first = self._generate(texts, temperature, 1, max_think, seed)
            continued = []
            for prompt, outputs in zip(texts, first):
                think = outputs[0]
                if "</think>" in think:
                    think = think.split("</think>", 1)[0]
                continued.append(prompt + think + "\n</think>\n\nFinal answer:")
            return self._generate(continued, temperature, n, 512, seed)
        return self._generate(texts, temperature, n, max_tokens, seed)

# ---------------------------------------------------------------- 判定
ABSTAIN_PAT = re.compile(
    r"(i don'?t know|i am not sure|i'?m not sure|cannot (be )?determin|not enough information|"
    r"unanswerable|no definitive answer|i cannot answer|unable to answer|"
    r"the (premise|question) is (false|incorrect|flawed)|不知道|无法确定|无法回答|前提.{0,4}错误)", re.I)

def is_abstain(resp: str) -> bool:
    return bool(ABSTAIN_PAT.search(resp))

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\.,;:!\?\"'\(\)\[\]]", " ", s)
    return re.sub(r"\s+", " ", s)

def extract_final(resp: str) -> str:
    """抽取最终答案:优先 'final answer:'/'答案:'/#### 之后;数学题抽最后一个数。"""
    if "</think>" in resp:
        resp = resp.split("</think>")[-1]
    m = re.search(r"(final answer|the answer is|答案是|answer)\s*[::]?\s*(.+)", resp, re.I)
    if m:
        return m.group(2).strip().split("\n")[0]
    m = re.search(r"####\s*(.+)", resp)
    if m:
        return m.group(1).strip()
    return resp.strip().split("\n")[-1]

def match_answer(resp: str, gold: str, aliases=(), numeric=False) -> bool:
    if gold == "UNANSWERABLE":
        return is_abstain(resp)
    final = extract_final(resp)
    if numeric:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", final.replace(",", ""))
        try:
            return bool(nums) and abs(float(nums[-1]) - float(str(gold).replace(",", ""))) < 1e-4
        except ValueError:
            return False
    cands = [gold, *aliases]
    nf, nr = normalize(final), normalize(resp[-400:])
    return any(normalize(c) in nf or normalize(c) in nr for c in cands if c)

def outcome(resp: str, gold: str, aliases=(), numeric=False) -> dict:
    """双结局度量。honest: 答对 或 (合理弃答)。Z6(gold=UNANSWERABLE) 弃答即 strict 治愈。"""
    correct = match_answer(resp, gold, aliases, numeric)
    abst = is_abstain(resp)
    if gold == "UNANSWERABLE":
        return {"strict": abst, "honest": abst, "abstain": abst}
    return {"strict": correct, "honest": correct or abst, "abstain": abst}

def majority_flip(resps, gold, aliases=(), numeric=False, k=None):
    """重采样多数判定: 返回 (多数是否错, 众数答案, 自洽度)。"""
    finals = [extract_final(r) for r in resps]
    corrects = [match_answer(r, gold, aliases, numeric) for r in resps]
    wrong_rate = 1 - sum(corrects) / len(corrects)
    norm = [normalize(f) for f in finals]
    mode = max(set(norm), key=norm.count) if norm else ""
    consist = norm.count(mode) / len(norm) if norm else 0.0
    return wrong_rate > 0.5, mode, consist
