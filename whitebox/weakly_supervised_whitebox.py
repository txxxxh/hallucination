#!/usr/bin/env python3
"""
Weakly supervised latent-role white-box hallucination detector.

This script removes the need for hand-written lexical rules that directly label
a sentence as CONSTRAINT or SHORTCUT.

Pipeline
--------
1. Split each question into structural candidate spans (sentences/clauses).
   This step proposes candidates only; it does not assign semantic roles.

2. Extract span-indexed white-box features from the original prompt:
   - decision-row attention mass and length-normalized attention density
   - unsorted Laplacian-diagonal / token-flow features
   - contrastive gradient x input features
   - structural controls such as span length and position

3. On TRAINING items only, perturb each candidate span with multiple
   deterministic interventions. Let

       g(x) = log P(gold | x) - log P(other | x)

   and for an intervention I on span s,

       contribution(s, I) = g(x) - g(I_s(x)).

   Positive contribution means the span supports the gold answer and is
   constraint-like. Negative contribution means the span suppresses the gold
   answer and is shortcut-like. Near-zero contribution is irrelevant-like.

   Multiple interventions are aggregated by a median and converted into a
   soft role distribution over:
       CONSTRAINT, SHORTCUT, IRRELEVANT.

4. Train a span-role classifier on these soft pseudo-labels. Soft labels are
   implemented by expanding each span into three weighted training rows.

5. Train a hallucination classifier using:
   - global attention/logit/spectral features
   - role summaries predicted by the span-role classifier

   To avoid same-item leakage, role predictions used for the hallucination
   training set are generated out-of-fold with item-level K-fold splits.

At test time, no intervention and no gold answer are needed for prediction:
the model uses only the original prompt's white-box features. Gold answers are
used only to evaluate the benchmark.

Expected input
--------------
A JSON array or JSONL file. The script auto-detects common keys:

    question / scenario
    benchmark_prompt / prompt
    answer / gold / label

The prompt field must either contain the exact question string or contain a
literal "{question}" placeholder. This is necessary so interventions modify
only the question while preserving options and formatting.

Example
-------
python weakly_supervised_whitebox.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data question_and_result.json \
    --out-dir weakbox_qwen25_7b \
    --span-mode clause \
    --interventions delete,neutralize,mask \
    --max-intervention-spans 6 \
    --test-size 0.25 \
    --dtype bfloat16 \
    --resume

Requirements
------------
pip install "transformers>=4.44" torch accelerate numpy scikit-learn joblib
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


ROLE_NAMES = ("constraint", "shortcut", "irrelevant")
ROLE_TO_ID = {name: i for i, name in enumerate(ROLE_NAMES)}
EPS = 1e-9

QUESTION_KEYS = ("question", "scenario")
PROMPT_KEYS = ("benchmark_prompt", "prompt")
GOLD_KEYS = ("answer", "gold", "label", "gold_answer")

SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.S)
CLAUSE_BOUNDARY_RE = re.compile(
    r"""
    (?:
        ;\s+ |
        \s+[—–-]\s+ |
        ,\s+(?=(?:but|yet|however|although|though|while|whereas)\b) |
        \s+(?=(?:but|yet|however|although|though|whereas)\b)
    )
    """,
    re.I | re.X,
)
SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Data structures and JSON helpers
# ---------------------------------------------------------------------------


@dataclass
class CandidateSpan:
    span_id: int
    start: int
    end: int
    text: str


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
        return records

    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Accept common wrappers.
        for key in ("items", "data", "records", "questions"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError("input must be a JSON list, JSONL, or a dict containing a list")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def load_jsonl_index(path: Path, key: str = "idx") -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[int(rec[key])] = rec
    return out


def pick_key(item: dict, explicit: str | None, candidates: Sequence[str]) -> str:
    if explicit:
        if explicit not in item:
            raise KeyError(f"requested field {explicit!r} is absent")
        return explicit
    for key in candidates:
        if key in item:
            return key
    raise KeyError(f"none of the expected fields are present: {candidates}")


def normalize_gold(value: Any, choices: tuple[str, str]) -> str:
    text = str(value).strip()
    # Accept numeric JSON values and strings like "1".
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text not in choices:
        raise ValueError(f"gold answer {value!r} is not one of {choices}")
    return text


# ---------------------------------------------------------------------------
# Candidate spans: structural proposal only, no role keywords
# ---------------------------------------------------------------------------


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        start, end = _trim_span(text, match.start(), match.end())
        if start < end:
            spans.append((start, end))
    return spans


def clause_spans(text: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for sent_start, sent_end in sentence_spans(text):
        sent = text[sent_start:sent_end]
        pieces = []
        cursor = 0
        for match in CLAUSE_BOUNDARY_RE.finditer(sent):
            a, b = cursor, match.start()
            a, b = _trim_span(sent, a, b)
            if a < b:
                pieces.append((sent_start + a, sent_start + b))
            cursor = match.end()
        a, b = _trim_span(sent, cursor, len(sent))
        if a < b:
            pieces.append((sent_start + a, sent_start + b))

        # Avoid creating tiny fragments. If clause splitting is poor, keep
        # the full sentence as one candidate.
        useful = [
            span for span in pieces
            if len(re.findall(r"\w+", text[span[0]:span[1]])) >= 3
        ]
        output.extend(useful if len(useful) >= 2 else [(sent_start, sent_end)])
    return output


def propose_spans(
    question: str,
    mode: str = "clause",
    include_question_span: bool = False,
    min_words: int = 3,
) -> list[CandidateSpan]:
    raw = clause_spans(question) if mode == "clause" else sentence_spans(question)
    candidates: list[CandidateSpan] = []

    for start, end in raw:
        text = question[start:end].strip()
        if len(re.findall(r"\w+", text)) < min_words:
            continue

        # Excluding the final interrogative is a structural safeguard. It does
        # not assign the remaining spans any semantic role.
        if not include_question_span and text.rstrip().endswith("?"):
            continue

        candidates.append(
            CandidateSpan(
                span_id=len(candidates),
                start=start,
                end=end,
                text=text,
            )
        )

    # If every candidate was excluded, fall back to all structural spans.
    if not candidates:
        for start, end in raw:
            text = question[start:end].strip()
            if len(re.findall(r"\w+", text)) >= min_words:
                candidates.append(
                    CandidateSpan(
                        span_id=len(candidates),
                        start=start,
                        end=end,
                        text=text,
                    )
                )

    return candidates


# ---------------------------------------------------------------------------
# Prompt rendering and interventions
# ---------------------------------------------------------------------------


class PromptAdapter:
    def __init__(
        self,
        tokenizer,
        question_field: str | None,
        prompt_field: str | None,
        gold_field: str | None,
        answer_instruction: str,
        apply_chat_template: bool,
    ):
        self.tok = tokenizer
        self.question_field = question_field
        self.prompt_field = prompt_field
        self.gold_field = gold_field
        self.answer_instruction = answer_instruction
        self.apply_chat_template = apply_chat_template

    def unpack(self, item: dict) -> tuple[str, str | None, Any]:
        q_key = pick_key(item, self.question_field, QUESTION_KEYS)
        g_key = pick_key(item, self.gold_field, GOLD_KEYS)
        question = str(item[q_key])

        prompt_key = None
        if self.prompt_field:
            prompt_key = pick_key(item, self.prompt_field, PROMPT_KEYS)
        else:
            for key in PROMPT_KEYS:
                if key in item:
                    prompt_key = key
                    break

        base_prompt = str(item[prompt_key]) if prompt_key else None
        return question, base_prompt, item[g_key]

    def user_text(
        self,
        original_question: str,
        modified_question: str,
        base_prompt: str | None,
    ) -> str:
        if base_prompt is None:
            text = modified_question
        elif "{question}" in base_prompt:
            text = base_prompt.replace("{question}", modified_question)
        elif original_question in base_prompt:
            text = base_prompt.replace(original_question, modified_question, 1)
        else:
            raise ValueError(
                "prompt field does not contain the exact question and has no "
                "'{question}' placeholder; safe intervention is impossible"
            )

        instruction = self.answer_instruction.strip()
        if instruction and instruction.lower() not in text.lower():
            text = text.rstrip() + "\n" + instruction
        return text

    def render(
        self,
        original_question: str,
        modified_question: str,
        base_prompt: str | None,
    ) -> str:
        user_text = self.user_text(original_question, modified_question, base_prompt)
        if not self.apply_chat_template:
            return user_text
        if not hasattr(self.tok, "apply_chat_template"):
            raise ValueError("tokenizer does not provide apply_chat_template")
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )


def normalize_edited_text(text: str) -> str:
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    return text.strip()


def intervene(question: str, span: CandidateSpan, kind: str) -> str:
    before, target, after = (
        question[:span.start],
        question[span.start:span.end],
        question[span.end:],
    )

    if kind == "delete":
        replacement = ""
    elif kind == "neutralize":
        replacement = "This detail is unavailable."
    elif kind == "mask":
        replacement = "[DETAIL OMITTED]"
    elif kind == "negate":
        stripped = target.strip()
        stripped = stripped[:-1] if stripped.endswith((".", "?", "!")) else stripped
        replacement = f"It is not true that {stripped}."
    else:
        raise ValueError(f"unknown intervention: {kind}")

    return normalize_edited_text(before + replacement + after)


# ---------------------------------------------------------------------------
# White-box extractor
# ---------------------------------------------------------------------------


def band_summary(values: np.ndarray, name: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    thirds = np.array_split(values, 3)
    out = {
        f"{name}_early": float(np.mean(thirds[0])),
        f"{name}_mid": float(np.mean(thirds[1])),
        f"{name}_late": float(np.mean(thirds[2])),
        f"{name}_mean": float(np.mean(values)),
        f"{name}_max": float(np.max(values)),
        f"{name}_std": float(np.std(values)),
    }
    return out


def safe_entropy_binary(log_a: float, log_b: float) -> float:
    vals = torch.tensor([log_a, log_b], dtype=torch.float64)
    p = torch.softmax(vals, dim=0)
    return float(-(p * torch.log(p + 1e-12)).sum().item())


class WeakWhiteboxExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lap_topk: int = 10,
    ):
        self.device = device
        self.dtype = dtype
        self.lap_topk = lap_topk

        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        if not getattr(self.tok, "is_fast", False):
            raise ValueError("a fast tokenizer is required for offset mappings")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        self._choice_id_cache: dict[str, list[int]] = {}

    def choice_token_ids(self, choice: str) -> list[int]:
        if choice in self._choice_id_cache:
            return self._choice_id_cache[choice]
        ids: set[int] = set()
        for variant in (choice, " " + choice):
            enc = self.tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            enc = self.tok.encode(choice, add_special_tokens=False)
            if not enc:
                raise ValueError(f"choice {choice!r} tokenized to nothing")
            warnings.warn(
                f"choice {choice!r} is not a single token; using first token only"
            )
            ids.add(enc[0])
        result = sorted(ids)
        self._choice_id_cache[choice] = result
        return result

    def _choice_logs(self, logits: torch.Tensor, choices: tuple[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids_a = self.choice_token_ids(choices[0])
        ids_b = self.choice_token_ids(choices[1])
        log_a = torch.logsumexp(logits[ids_a], dim=0)
        log_b = torch.logsumexp(logits[ids_b], dim=0)
        return log_a, log_b

    @torch.no_grad()
    def score_prompt(
        self,
        prompt: str,
        gold: str,
        choices: tuple[str, str],
    ) -> dict:
        enc = self.tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ids = enc["input_ids"].to(self.device)
        out = self.model(input_ids=ids, use_cache=False)
        logits = out.logits[0, -1].float()
        log_a, log_b = self._choice_logs(logits, choices)

        chosen = choices[0] if log_a >= log_b else choices[1]
        gold_log = log_a if gold == choices[0] else log_b
        other_log = log_b if gold == choices[0] else log_a
        chosen_log = torch.maximum(log_a, log_b)
        rejected_log = torch.minimum(log_a, log_b)

        return {
            "log_a": float(log_a.item()),
            "log_b": float(log_b.item()),
            "gold_margin": float((gold_log - other_log).item()),
            "chosen_margin": float((chosen_log - rejected_log).item()),
            "chosen": chosen,
            "choice_entropy": safe_entropy_binary(log_a.item(), log_b.item()),
        }

    def extract_original(
        self,
        prompt: str,
        question: str,
        spans: Sequence[CandidateSpan],
        choices: tuple[str, str],
    ) -> dict:
        """
        Extract features from the original prompt only.

        The decision-row attention is taken at the final prompt position,
        which is the causal position whose logits predict the first answer
        token. This avoids the one-token row shift in teacher-forced analyses.
        """
        enc = self.tok(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"][0]
        ids = enc["input_ids"].to(self.device)
        T = ids.shape[1]

        question_base = prompt.find(question)
        if question_base < 0:
            raise ValueError("rendered prompt does not contain the exact question")

        masks: list[torch.Tensor] = []
        for span in spans:
            absolute = (question_base + span.start, question_base + span.end)
            mask = self._char_mask(offsets, absolute, T)
            if not mask.any():
                raise ValueError(f"candidate span matched no tokens: {span.text!r}")
            masks.append(mask.to(self.device))

        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                output_attentions=True,
                use_cache=False,
            )
            logits = out.logits[0, -1].float()
            attns = out.attentions

        log_a, log_b = self._choice_logs(logits, choices)
        chosen = choices[0] if log_a >= log_b else choices[1]
        chosen_contrast = (log_a - log_b) if chosen == choices[0] else (log_b - log_a)

        global_features: dict[str, float] = {
            "choice_margin_abs": float(abs((log_a - log_b).item())),
            "choice_entropy": safe_entropy_binary(log_a.item(), log_b.item()),
            "chosen_is_a": float(chosen == choices[0]),
            "prompt_tokens": float(T),
            "num_candidate_spans": float(len(spans)),
        }

        # Gradient x input for the model's own chosen contrast.
        grad_norm, grad_signed = self._contrastive_input_gradient(
            ids=ids,
            choices=choices,
            chosen=chosen,
        )

        span_features: list[dict[str, float]] = []
        for span, mask in zip(spans, masks):
            idx = mask.nonzero().squeeze(-1)
            span_features.append(
                {
                    "span_words": float(len(re.findall(r"\w+", span.text))),
                    "span_tokens": float(idx.numel()),
                    "span_characters": float(span.end - span.start),
                    "span_relative_start": float(span.start / max(len(question), 1)),
                    "span_relative_end": float(span.end / max(len(question), 1)),
                    "span_relative_length": float((span.end - span.start) / max(len(question), 1)),
                    "grad_norm_sum": float(grad_norm[idx].sum().item()),
                    "grad_norm_density": float(grad_norm[idx].mean().item()),
                    "grad_signed_sum": float(grad_signed[idx].sum().item()),
                    "grad_signed_density": float(grad_signed[idx].mean().item()),
                }
            )

        layer_entropy = []
        layer_lapeigs = []
        # Temporary per-layer arrays for each span.
        per_span_attn_mass = [[] for _ in spans]
        per_span_attn_density = [[] for _ in spans]
        per_span_attn_headmax = [[] for _ in spans]
        per_span_lap = [[] for _ in spans]

        for A in attns:
            # [H, T, T]
            Ah = A[0].float()
            decision = Ah[:, T - 1, :T]  # [H, T], predicts the first answer token

            p = decision.mean(0)
            p = p / (p.sum() + EPS)
            layer_entropy.append(float((-(p * (p + 1e-12).log()).sum()).item()))

            # Head-averaged causal Laplacian diagonal. Token identity is kept
            # for span pooling, while sorted top-k is retained as a baseline.
            Amean = Ah.mean(0)
            col = torch.arange(T, device=Amean.device)
            denom = (T - 1 - col).clamp(min=1).float()
            received = Amean.tril(-1).sum(0) / denom
            lam = received - torch.diagonal(Amean)

            k = min(self.lap_topk, T)
            top = torch.sort(lam, descending=True).values[:k].cpu().numpy()
            layer_lapeigs.append(top)

            for j, mask in enumerate(masks):
                idx = mask.nonzero().squeeze(-1)
                head_mass = decision[:, idx].sum(-1)
                per_span_attn_mass[j].append(float(head_mass.mean().item()))
                per_span_attn_density[j].append(
                    float((decision[:, idx].mean(-1)).mean().item())
                )
                per_span_attn_headmax[j].append(float(head_mass.max().item()))
                per_span_lap[j].append(float(lam[idx].mean().item()))

        global_features.update(
            band_summary(np.asarray(layer_entropy), "decision_attn_entropy")
        )

        lap_arr = np.stack(layer_lapeigs) if layer_lapeigs else np.zeros((0, 0))
        if lap_arr.size:
            for rank in range(lap_arr.shape[1]):
                global_features.update(
                    band_summary(lap_arr[:, rank], f"lapeig{rank}")
                )

        for j in range(len(spans)):
            span_features[j].update(
                band_summary(np.asarray(per_span_attn_mass[j]), "attn_mass")
            )
            span_features[j].update(
                band_summary(np.asarray(per_span_attn_density[j]), "attn_density")
            )
            span_features[j].update(
                band_summary(np.asarray(per_span_attn_headmax[j]), "attn_headmax")
            )
            span_features[j].update(
                band_summary(np.asarray(per_span_lap[j]), "lap_token")
            )

        return {
            "chosen": chosen,
            "log_a": float(log_a.item()),
            "log_b": float(log_b.item()),
            "chosen_margin": float(abs((log_a - log_b).item())),
            "global_features": global_features,
            "span_features": span_features,
        }

    @staticmethod
    def _char_mask(offsets: torch.Tensor, span: tuple[int, int], n_tokens: int) -> torch.Tensor:
        start, end = span
        mask = torch.zeros(n_tokens, dtype=torch.bool)
        for i, (a, b) in enumerate(offsets[:n_tokens].tolist()):
            if a < end and b > start and b > a:
                mask[i] = True
        return mask

    def _contrastive_input_gradient(
        self,
        ids: torch.Tensor,
        choices: tuple[str, str],
        chosen: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.model.get_input_embeddings()
        inputs_embeds = embed(ids).detach().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
        logits = out.logits[0, -1].float()
        log_a, log_b = self._choice_logs(logits, choices)
        contrast = (log_a - log_b) if chosen == choices[0] else (log_b - log_a)
        contrast.backward()

        gx = inputs_embeds.grad * inputs_embeds
        grad_norm = gx.norm(dim=-1)[0].detach()
        grad_signed = gx.sum(dim=-1)[0].detach()
        self.model.zero_grad(set_to_none=True)
        return grad_norm, grad_signed


# ---------------------------------------------------------------------------
# Pseudo-role construction
# ---------------------------------------------------------------------------


def softmax_np(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / np.sum(exp)


def build_soft_role(
    deltas: Sequence[float],
    deadzone: float,
    temperature: float,
) -> dict:
    """
    Convert intervention contributions into a soft role distribution.

    contribution > 0: constraint-like
    contribution < 0: shortcut-like
    near zero: irrelevant-like
    """
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        raise ValueError("at least one intervention delta is required")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    logits = [
        median / max(temperature, EPS),
        -median / max(temperature, EPS),
        (deadzone - abs(median)) / max(temperature, EPS),
    ]
    probs = softmax_np(logits)

    if abs(median) >= deadzone:
        target_sign = 1.0 if median > 0 else -1.0
        informative = np.abs(values) >= deadzone * 0.25
        if informative.any():
            agreement = float(
                np.mean(np.sign(values[informative]) == target_sign)
            )
        else:
            agreement = 0.0
    else:
        agreement = float(np.mean(np.abs(values) <= deadzone))

    stability = math.exp(-mad / max(temperature, EPS))
    reliability = float(np.clip(agreement * stability, 0.0, 1.0))

    hard_id = int(np.argmax(probs))
    return {
        "median_contribution": median,
        "mad": mad,
        "agreement": agreement,
        "reliability": reliability,
        "role_probs": probs.tolist(),
        "hard_role": ROLE_NAMES[hard_id],
    }


# ---------------------------------------------------------------------------
# Feature matrix and classifier heads
# ---------------------------------------------------------------------------


class FeatureMatrix:
    def __init__(self, keys: Sequence[str] | None = None):
        self.keys = list(keys) if keys is not None else None

    def fit(self, dicts: Sequence[dict[str, float]]) -> "FeatureMatrix":
        self.keys = sorted(set().union(*(d.keys() for d in dicts)))
        return self

    def transform(self, dicts: Sequence[dict[str, float]]) -> np.ndarray:
        if self.keys is None:
            raise RuntimeError("FeatureMatrix is not fitted")
        X = np.asarray(
            [[float(d.get(k, 0.0)) for k in self.keys] for d in dicts],
            dtype=float,
        )
        # Keep missing/invalid numerical values from crashing sklearn. Their
        # occurrence should still be audited in output logs.
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


class SpanRoleHead:
    def __init__(self, C: float = 1.0):
        self.matrix = FeatureMatrix()
        self.C = C
        self.pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=4000,
                class_weight="balanced",
            ),
        )

    def fit(
        self,
        feat_dicts: Sequence[dict[str, float]],
        soft_labels: Sequence[Sequence[float]],
        reliabilities: Sequence[float],
    ) -> "SpanRoleHead":
        self.matrix.fit(feat_dicts)
        X_base = self.matrix.transform(feat_dicts)

        rows = []
        labels = []
        weights = []
        for x, probs, rel in zip(X_base, soft_labels, reliabilities):
            probs_arr = np.asarray(probs, dtype=float)
            for role_id in range(3):
                weight = float(max(rel, 1e-4) * max(probs_arr[role_id], 1e-6))
                rows.append(x)
                labels.append(role_id)
                weights.append(weight)

        X = np.asarray(rows)
        y = np.asarray(labels, dtype=int)
        w = np.asarray(weights, dtype=float)
        self.pipe.fit(X, y, logisticregression__sample_weight=w)
        return self

    def predict_proba(
        self,
        feat_dicts: Sequence[dict[str, float]],
    ) -> np.ndarray:
        X = self.matrix.transform(feat_dicts)
        probs = self.pipe.predict_proba(X)

        # Guarantee fixed role order 0,1,2 even if sklearn changes class order.
        classes = self.pipe.named_steps["logisticregression"].classes_
        out = np.zeros((len(feat_dicts), 3), dtype=float)
        for col, cls in enumerate(classes):
            out[:, int(cls)] = probs[:, col]
        return out

    def coefficient_report(self, top_n: int = 20) -> dict[str, list[tuple[str, float]]]:
        lr = self.pipe.named_steps["logisticregression"]
        report: dict[str, list[tuple[str, float]]] = {}
        for row, cls in zip(lr.coef_, lr.classes_):
            ranked = sorted(
                zip(self.matrix.keys or [], row),
                key=lambda pair: -abs(pair[1]),
            )[:top_n]
            report[ROLE_NAMES[int(cls)]] = [
                (name, float(coef)) for name, coef in ranked
            ]
        return report


class HallucinationHead:
    def __init__(self, cv: int = 5):
        self.matrix = FeatureMatrix()
        self.cv = cv
        self.pipe = None

    def _make_pipe(self, y: np.ndarray):
        pos = int(y.sum())
        neg = int(len(y) - pos)
        folds = min(self.cv, pos, neg)
        if folds >= 2:
            return make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=10,
                    cv=folds,
                    scoring="roc_auc",
                    solver="liblinear",
                    penalty="l1",
                    class_weight="balanced",
                    max_iter=5000,
                ),
            )
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                penalty="l1",
                class_weight="balanced",
                max_iter=5000,
            ),
        )

    def fit(
        self,
        feat_dicts: Sequence[dict[str, float]],
        labels: Sequence[int],
    ) -> "HallucinationHead":
        self.matrix.fit(feat_dicts)
        X = self.matrix.transform(feat_dicts)
        y = np.asarray(labels, dtype=int)
        self.pipe = self._make_pipe(y)
        self.pipe.fit(X, y)
        return self

    def predict_proba(
        self,
        feat_dicts: Sequence[dict[str, float]],
    ) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("HallucinationHead is not fitted")
        X = self.matrix.transform(feat_dicts)
        return self.pipe.predict_proba(X)[:, 1]

    def coefficient_report(self, top_n: int = 30) -> list[tuple[str, float]]:
        if self.pipe is None:
            return []
        final_name = list(self.pipe.named_steps.keys())[-1]
        lr = self.pipe.named_steps[final_name]
        row = lr.coef_[0]
        ranked = sorted(
            zip(self.matrix.keys or [], row),
            key=lambda pair: -abs(pair[1]),
        )[:top_n]
        return [(name, float(coef)) for name, coef in ranked]


def aggregate_role_predictions(
    span_features: Sequence[dict[str, float]],
    role_probs: np.ndarray,
) -> dict[str, float]:
    if len(span_features) == 0:
        return {
            "role_constraint_max": 0.0,
            "role_shortcut_max": 0.0,
            "role_irrelevant_mean": 1.0,
            "role_shortcut_minus_constraint": 0.0,
        }

    c = role_probs[:, ROLE_TO_ID["constraint"]]
    s = role_probs[:, ROLE_TO_ID["shortcut"]]
    i = role_probs[:, ROLE_TO_ID["irrelevant"]]

    attn = np.asarray(
        [f.get("attn_density_late", f.get("attn_density_mean", 0.0)) for f in span_features],
        dtype=float,
    )
    grad = np.asarray(
        [f.get("grad_norm_density", 0.0) for f in span_features],
        dtype=float,
    )

    return {
        "role_constraint_max": float(c.max()),
        "role_constraint_mean": float(c.mean()),
        "role_shortcut_max": float(s.max()),
        "role_shortcut_mean": float(s.mean()),
        "role_irrelevant_mean": float(i.mean()),
        "role_shortcut_minus_constraint": float(s.max() - c.max()),
        "role_shortcut_count_p50": float((s >= 0.5).sum()),
        "role_constraint_count_p50": float((c >= 0.5).sum()),
        "shortcut_weighted_attn_max": float(np.max(s * attn)),
        "constraint_weighted_attn_max": float(np.max(c * attn)),
        "shortcut_weighted_grad_max": float(np.max(s * grad)),
        "constraint_weighted_grad_max": float(np.max(c * grad)),
        "role_entropy_mean": float(
            np.mean(-(role_probs * np.log(role_probs + 1e-12)).sum(axis=1))
        ),
    }


def merge_item_features(
    global_features: dict[str, float],
    role_summary: dict[str, float],
) -> dict[str, float]:
    merged = dict(global_features)
    merged.update(role_summary)
    return merged


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def choose_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + EPS
    )
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_binary(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        average="binary",
        zero_division=0,
    )
    result = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, p))
        result["auprc"] = float(average_precision_score(y, p))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def rank_spans_for_intervention(record: dict, max_spans: int) -> list[int]:
    spans = record["spans"]
    if max_spans <= 0 or len(spans) <= max_spans:
        return list(range(len(spans)))

    # Unsupervised relevance ranking. It does not assign a role; it merely
    # limits expensive interventions to spans with high original influence.
    scored = []
    for j, span in enumerate(spans):
        feat = span["features"]
        score = (
            float(feat.get("attn_density_late", 0.0))
            + float(feat.get("grad_norm_density", 0.0))
        )
        scored.append((score, j))
    scored.sort(reverse=True)
    return sorted(j for _, j in scored[:max_spans])


def extract_base_records(
    items: list[dict],
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    args,
    cache_path: Path,
) -> list[dict]:
    cached = load_jsonl_index(cache_path) if args.resume else {}
    records: list[dict] = []

    for idx, item in enumerate(items):
        if idx in cached:
            records.append(cached[idx])
            continue

        question, base_prompt, gold_raw = adapter.unpack(item)
        gold = normalize_gold(gold_raw, choices)
        spans = propose_spans(
            question,
            mode=args.span_mode,
            include_question_span=args.include_question_span,
            min_words=args.min_span_words,
        )
        if not spans:
            warnings.warn(f"[{idx}] no candidate spans; skipped")
            continue

        prompt = adapter.render(question, question, base_prompt)
        try:
            extracted = extractor.extract_original(
                prompt=prompt,
                question=question,
                spans=spans,
                choices=choices,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            warnings.warn(f"[{idx}] CUDA OOM; skipped")
            continue
        except Exception as exc:
            warnings.warn(f"[{idx}] base extraction failed: {exc}")
            continue

        chosen = extracted["chosen"]
        record = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "chosen": chosen,
            "hallucinated": int(chosen != gold),
            "chosen_margin": extracted["chosen_margin"],
            "global_features": extracted["global_features"],
            "spans": [
                {
                    **asdict(span),
                    "features": features,
                }
                for span, features in zip(spans, extracted["span_features"])
            ],
        }
        append_jsonl(cache_path, record)
        records.append(record)

        if (idx + 1) % 10 == 0 or idx + 1 == len(items):
            print(f"base extraction: {idx + 1}/{len(items)}", flush=True)

    return records


def add_intervention_labels(
    records: list[dict],
    items: list[dict],
    train_indices: set[int],
    test_indices: set[int],
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    interventions: list[str],
    args,
    cache_path: Path,
) -> dict[int, dict]:
    cached = load_jsonl_index(cache_path) if args.resume else {}
    output = dict(cached)

    target_indices = set(train_indices)
    if args.intervene_test:
        target_indices |= set(test_indices)

    for count, record in enumerate(records, 1):
        idx = int(record["idx"])
        if idx not in target_indices or idx in output:
            continue

        item = items[idx]
        question, base_prompt, gold_raw = adapter.unpack(item)
        gold = normalize_gold(gold_raw, choices)
        original_prompt = adapter.render(question, question, base_prompt)
        original_score = extractor.score_prompt(original_prompt, gold, choices)
        original_gold_margin = original_score["gold_margin"]

        chosen_span_ids = rank_spans_for_intervention(
            record,
            args.max_intervention_spans,
        )

        span_labels = []
        for j, span_record in enumerate(record["spans"]):
            if j not in chosen_span_ids:
                span_labels.append(
                    {
                        "evaluated": False,
                        "role_probs": [0.0, 0.0, 1.0],
                        "hard_role": "irrelevant",
                        "reliability": 0.0,
                        "interventions": [],
                    }
                )
                continue

            span = CandidateSpan(
                span_id=int(span_record["span_id"]),
                start=int(span_record["start"]),
                end=int(span_record["end"]),
                text=str(span_record["text"]),
            )

            evidences = []
            deltas = []
            for kind in interventions:
                modified_question = intervene(question, span, kind)
                try:
                    variant_prompt = adapter.render(
                        question,
                        modified_question,
                        base_prompt,
                    )
                    score = extractor.score_prompt(variant_prompt, gold, choices)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    warnings.warn(f"[{idx}] OOM on {kind} span {j}")
                    continue
                except Exception as exc:
                    warnings.warn(f"[{idx}] intervention failed ({kind}, span {j}): {exc}")
                    continue

                delta = float(original_gold_margin - score["gold_margin"])
                deltas.append(delta)
                evidences.append(
                    {
                        "kind": kind,
                        "gold_margin": score["gold_margin"],
                        "chosen": score["chosen"],
                        "delta": delta,
                        "flip": bool(score["chosen"] != original_score["chosen"]),
                    }
                )

            if deltas:
                pseudo = build_soft_role(
                    deltas,
                    deadzone=args.role_deadzone,
                    temperature=args.role_temperature,
                )
                pseudo.update(
                    {
                        "evaluated": True,
                        "interventions": evidences,
                    }
                )
            else:
                pseudo = {
                    "evaluated": False,
                    "role_probs": [0.0, 0.0, 1.0],
                    "hard_role": "irrelevant",
                    "reliability": 0.0,
                    "interventions": evidences,
                }
            span_labels.append(pseudo)

        rec = {
            "idx": idx,
            "original_gold_margin": original_gold_margin,
            "original_chosen": original_score["chosen"],
            "span_pseudo_labels": span_labels,
        }
        append_jsonl(cache_path, rec)
        output[idx] = rec

        if count % 10 == 0 or count == len(records):
            print(
                f"interventions: processed base item {count}/{len(records)}",
                flush=True,
            )

    return output


def prepare_role_training(
    base_by_idx: dict[int, dict],
    intervention_by_idx: dict[int, dict],
    train_indices: Sequence[int],
    min_reliability: float,
) -> tuple[list[dict], list[list[float]], list[float], list[int], list[int]]:
    features = []
    soft_labels = []
    reliabilities = []
    groups = []
    span_local_ids = []

    for idx in train_indices:
        base = base_by_idx[idx]
        pseudo = intervention_by_idx.get(idx)
        if pseudo is None:
            continue
        for local_id, (span, label) in enumerate(
            zip(base["spans"], pseudo["span_pseudo_labels"])
        ):
            if not label.get("evaluated", False):
                continue
            reliability = float(label.get("reliability", 0.0))
            if reliability < min_reliability:
                continue
            features.append(span["features"])
            soft_labels.append(label["role_probs"])
            reliabilities.append(reliability)
            groups.append(idx)
            span_local_ids.append(local_id)

    if not features:
        raise RuntimeError(
            "no reliable pseudo-labeled spans; lower --min-role-reliability "
            "or inspect intervention validity"
        )
    return features, soft_labels, reliabilities, groups, span_local_ids


def make_role_predictions_by_item(
    base_by_idx: dict[int, dict],
    indices: Sequence[int],
    role_head: SpanRoleHead,
) -> dict[int, np.ndarray]:
    result = {}
    for idx in indices:
        features = [span["features"] for span in base_by_idx[idx]["spans"]]
        result[idx] = role_head.predict_proba(features)
    return result


def make_oof_role_predictions_by_item(
    base_by_idx: dict[int, dict],
    train_indices: Sequence[int],
    role_train_features: list[dict],
    role_train_soft: list[list[float]],
    role_train_rel: list[float],
    role_groups: list[int],
    role_local_ids: list[int],
    n_splits: int,
    full_role_head: SpanRoleHead,
) -> dict[int, np.ndarray]:
    """
    Predict every span of each training item with a role model that was fitted
    without any pseudo-labeled span from that item.

    role_local_ids is accepted for API compatibility and auditing; OOF
    prediction is deliberately performed for all spans, including spans whose
    pseudo-label was filtered for low reliability.
    """
    del role_local_ids
    indices = list(train_indices)
    splits = min(n_splits, len(indices))
    if splits < 2:
        return make_role_predictions_by_item(base_by_idx, indices, full_role_head)

    result: dict[int, np.ndarray] = {}
    splitter = KFold(n_splits=splits, shuffle=True, random_state=0)
    group_arr = np.asarray(role_groups, dtype=int)

    for train_pos, val_pos in splitter.split(indices):
        fit_items = {indices[i] for i in train_pos}
        val_items = [indices[i] for i in val_pos]
        fit_rows = [i for i, group in enumerate(group_arr) if int(group) in fit_items]

        if not fit_rows:
            fold_head = full_role_head
        else:
            fold_head = SpanRoleHead().fit(
                [role_train_features[i] for i in fit_rows],
                [role_train_soft[i] for i in fit_rows],
                [role_train_rel[i] for i in fit_rows],
            )

        for idx in val_items:
            all_span_features = [
                span["features"] for span in base_by_idx[idx]["spans"]
            ]
            result[idx] = fold_head.predict_proba(all_span_features)

    missing = [idx for idx in indices if idx not in result]
    if missing:
        result.update(
            make_role_predictions_by_item(base_by_idx, missing, full_role_head)
        )
    return result

def hall_oof_probabilities(
    feat_dicts: list[dict],
    labels: list[int],
    n_splits: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    matrix = FeatureMatrix().fit(feat_dicts)
    X = matrix.transform(feat_dicts)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    splits = min(n_splits, pos, neg)
    if splits < 2:
        head = HallucinationHead(cv=2).fit(feat_dicts, labels)
        return head.predict_proba(feat_dicts)

    splitter = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=0,
    )
    oof = np.zeros(len(y), dtype=float)
    for train_rows, val_rows in splitter.split(X, y):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                penalty="l1",
                class_weight="balanced",
                max_iter=5000,
            ),
        )
        pipe.fit(X[train_rows], y[train_rows])
        oof[val_rows] = pipe.predict_proba(X[val_rows])[:, 1]
    return oof


def explain_record(
    base: dict,
    role_probs: np.ndarray,
) -> dict:
    spans = base["spans"]
    c_idx = int(np.argmax(role_probs[:, ROLE_TO_ID["constraint"]]))
    s_idx = int(np.argmax(role_probs[:, ROLE_TO_ID["shortcut"]]))

    return {
        "predicted_constraint": {
            "span_id": int(spans[c_idx]["span_id"]),
            "text": spans[c_idx]["text"],
            "probability": float(role_probs[c_idx, ROLE_TO_ID["constraint"]]),
        },
        "predicted_shortcut": {
            "span_id": int(spans[s_idx]["span_id"]),
            "text": spans[s_idx]["text"],
            "probability": float(role_probs[s_idx, ROLE_TO_ID["shortcut"]]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weakly supervised latent-role white-box detector"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="question_and_result.json")
    parser.add_argument("--out-dir", default="weakbox_output")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--question-field")
    parser.add_argument("--prompt-field")
    parser.add_argument("--gold-field")
    parser.add_argument(
        "--answer-instruction",
        default="Reply with a single character: 1 or 2.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="treat the prompt field as the exact model input string",
    )

    parser.add_argument("--choice-a", default="1")
    parser.add_argument("--choice-b", default="2")
    parser.add_argument(
        "--span-mode",
        choices=["sentence", "clause"],
        default="clause",
    )
    parser.add_argument("--include-question-span", action="store_true")
    parser.add_argument("--min-span-words", type=int, default=3)

    parser.add_argument(
        "--interventions",
        default="delete,neutralize,mask",
        help="comma-separated subset of delete,neutralize,mask,negate",
    )
    parser.add_argument("--max-intervention-spans", type=int, default=6)
    parser.add_argument("--intervene-test", action="store_true")
    parser.add_argument("--role-deadzone", type=float, default=0.25)
    parser.add_argument("--role-temperature", type=float, default=0.75)
    parser.add_argument("--min-role-reliability", type=float, default=0.20)

    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role-oof-folds", type=int, default=5)
    parser.add_argument("--hall-oof-folds", type=int, default=5)
    parser.add_argument("--lap-topk", type=int, default=10)

    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cuda', 'cuda:0', or 'cpu'",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device == "cpu" and dtype != torch.float32:
        warnings.warn("CPU selected; forcing float32")
        dtype = torch.float32

    interventions = [
        x.strip() for x in args.interventions.split(",") if x.strip()
    ]
    allowed = {"delete", "neutralize", "mask", "negate"}
    invalid = set(interventions) - allowed
    if invalid:
        parser.error(f"invalid interventions: {sorted(invalid)}")
    if len(interventions) < 2:
        warnings.warn(
            "fewer than two interventions weakens agreement/reliability estimates"
        )

    choices = (str(args.choice_a), str(args.choice_b))
    if choices[0] == choices[1]:
        parser.error("choice-a and choice-b must differ")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cache = out_dir / "base_features.jsonl"
    intervention_cache = out_dir / "intervention_labels.jsonl"
    prediction_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"
    role_model_path = out_dir / "role_head.joblib"
    hall_model_path = out_dir / "hallucination_head.joblib"

    if not args.resume:
        for path in (base_cache, intervention_cache, prediction_path):
            if path.exists():
                path.unlink()

    items = read_records(args.data)
    if args.limit > 0:
        items = items[: args.limit]
    if len(items) < 10:
        warnings.warn("very small dataset; estimates will be unstable")

    print(f"loading model {args.model} on {device} ...", flush=True)
    extractor = WeakWhiteboxExtractor(
        args.model,
        device=device,
        dtype=dtype,
        lap_topk=args.lap_topk,
    )
    adapter = PromptAdapter(
        extractor.tok,
        question_field=args.question_field,
        prompt_field=args.prompt_field,
        gold_field=args.gold_field,
        answer_instruction=args.answer_instruction,
        apply_chat_template=not args.no_chat_template,
    )

    # Stage 1: original-prompt features for all examples.
    base_records = extract_base_records(
        items=items,
        extractor=extractor,
        adapter=adapter,
        choices=choices,
        args=args,
        cache_path=base_cache,
    )
    if len(base_records) < 4:
        raise RuntimeError("too few successfully extracted records")

    base_by_idx = {int(rec["idx"]): rec for rec in base_records}
    valid_indices = np.asarray(sorted(base_by_idx), dtype=int)
    hall_labels = np.asarray(
        [int(base_by_idx[i]["hallucinated"]) for i in valid_indices],
        dtype=int,
    )
    if len(np.unique(hall_labels)) < 2:
        raise RuntimeError("model produced only one hallucination class")

    train_idx, test_idx = train_test_split(
        valid_indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=hall_labels,
    )
    train_idx = sorted(int(x) for x in train_idx)
    test_idx = sorted(int(x) for x in test_idx)
    train_set, test_set = set(train_idx), set(test_idx)

    # Stage 2: interventions on training spans. Test interventions are optional
    # and are used only for behavioral evaluation, never as test inputs.
    intervention_by_idx = add_intervention_labels(
        records=base_records,
        items=items,
        train_indices=train_set,
        test_indices=test_set,
        extractor=extractor,
        adapter=adapter,
        choices=choices,
        interventions=interventions,
        args=args,
        cache_path=intervention_cache,
    )

    # Stage 3: weakly supervised span-role model.
    (
        role_features,
        role_soft,
        role_rel,
        role_groups,
        role_local_ids,
    ) = prepare_role_training(
        base_by_idx,
        intervention_by_idx,
        train_idx,
        min_reliability=args.min_role_reliability,
    )
    role_head = SpanRoleHead().fit(role_features, role_soft, role_rel)
    joblib.dump(role_head, role_model_path)

    train_role_probs = make_oof_role_predictions_by_item(
        base_by_idx=base_by_idx,
        train_indices=train_idx,
        role_train_features=role_features,
        role_train_soft=role_soft,
        role_train_rel=role_rel,
        role_groups=role_groups,
        role_local_ids=role_local_ids,
        n_splits=args.role_oof_folds,
        full_role_head=role_head,
    )
    test_role_probs = make_role_predictions_by_item(
        base_by_idx,
        test_idx,
        role_head,
    )

    # Stage 4: hallucination head.
    train_hall_features = []
    train_hall_labels = []
    for idx in train_idx:
        summary = aggregate_role_predictions(
            [span["features"] for span in base_by_idx[idx]["spans"]],
            train_role_probs[idx],
        )
        train_hall_features.append(
            merge_item_features(base_by_idx[idx]["global_features"], summary)
        )
        train_hall_labels.append(int(base_by_idx[idx]["hallucinated"]))

    test_hall_features = []
    test_hall_labels = []
    for idx in test_idx:
        summary = aggregate_role_predictions(
            [span["features"] for span in base_by_idx[idx]["spans"]],
            test_role_probs[idx],
        )
        test_hall_features.append(
            merge_item_features(base_by_idx[idx]["global_features"], summary)
        )
        test_hall_labels.append(int(base_by_idx[idx]["hallucinated"]))

    train_oof_prob = hall_oof_probabilities(
        train_hall_features,
        train_hall_labels,
        n_splits=args.hall_oof_folds,
    )
    threshold = choose_f1_threshold(
        np.asarray(train_hall_labels),
        train_oof_prob,
    )

    hall_head = HallucinationHead(cv=args.hall_oof_folds).fit(
        train_hall_features,
        train_hall_labels,
    )
    joblib.dump(hall_head, hall_model_path)

    test_prob = hall_head.predict_proba(test_hall_features)
    train_metrics = evaluate_binary(
        train_hall_labels,
        train_oof_prob,
        threshold,
    )
    test_metrics = evaluate_binary(
        test_hall_labels,
        test_prob,
        threshold,
    )

    # Final auditable output.
    if prediction_path.exists():
        prediction_path.unlink()

    test_prob_by_idx = {
        idx: float(prob) for idx, prob in zip(test_idx, test_prob)
    }
    for split_name, indices, role_map in (
        ("train", train_idx, train_role_probs),
        ("test", test_idx, test_role_probs),
    ):
        for idx in indices:
            base = base_by_idx[idx]
            probs = role_map[idx]
            explanation = explain_record(base, probs)
            record = {
                "idx": idx,
                "split": split_name,
                "question": base["question"],
                "gold": base["gold"],
                "chosen": base["chosen"],
                "hallucinated": bool(base["hallucinated"]),
                "chosen_margin": base["chosen_margin"],
                "hallucination_probability": (
                    test_prob_by_idx[idx] if split_name == "test" else None
                ),
                "predicted_roles": [
                    {
                        "span_id": int(span["span_id"]),
                        "text": span["text"],
                        "constraint_probability": float(
                            probs[j, ROLE_TO_ID["constraint"]]
                        ),
                        "shortcut_probability": float(
                            probs[j, ROLE_TO_ID["shortcut"]]
                        ),
                        "irrelevant_probability": float(
                            probs[j, ROLE_TO_ID["irrelevant"]]
                        ),
                    }
                    for j, span in enumerate(base["spans"])
                ],
                "explanation": explanation,
            }
            pseudo = intervention_by_idx.get(idx)
            if pseudo is not None:
                record["intervention_pseudo_labels"] = pseudo[
                    "span_pseudo_labels"
                ]
            append_jsonl(prediction_path, record)

    role_counts = {name: 0 for name in ROLE_NAMES}
    mean_reliability = []
    for idx in train_idx:
        pseudo = intervention_by_idx.get(idx)
        if pseudo is None:
            continue
        for label in pseudo["span_pseudo_labels"]:
            if label.get("evaluated", False):
                role_counts[label["hard_role"]] += 1
                mean_reliability.append(float(label["reliability"]))

    summary = {
        "model": args.model,
        "data": args.data,
        "device": device,
        "dtype": str(dtype),
        "choices": choices,
        "n_input": len(items),
        "n_extracted": len(base_records),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "interventions": interventions,
        "span_mode": args.span_mode,
        "role_pseudo_label_counts": role_counts,
        "mean_role_reliability": (
            float(np.mean(mean_reliability)) if mean_reliability else None
        ),
        "n_role_training_spans": len(role_features),
        "train_oof_metrics": train_metrics,
        "test_metrics": test_metrics,
        "selected_threshold_from_train_oof": threshold,
        "role_feature_coefficients": role_head.coefficient_report(),
        "hallucination_feature_coefficients": hall_head.coefficient_report(),
        "files": {
            "base_features": str(base_cache),
            "intervention_labels": str(intervention_cache),
            "predictions": str(prediction_path),
            "role_model": str(role_model_path),
            "hallucination_model": str(hall_model_path),
        },
        "method_notes": {
            "test_prediction_uses_interventions": False,
            "test_prediction_uses_gold_answer": False,
            "training_hallucination_labels_from_gold": True,
            "training_span_roles_from_intervention_soft_labels": True,
            "role_training_predictions_for_hall_head_are_group_oof": True,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    print("\n=== weakly supervised detector complete ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
