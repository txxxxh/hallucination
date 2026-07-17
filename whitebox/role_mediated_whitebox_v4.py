#!/usr/bin/env python3
"""
Role-mediated weakly supervised white-box hallucination detector (v4).

This version keeps the three latent span roles:
    CONSTRAINT / SHORTCUT / IRRELEVANT

and adds a strict post-prediction causal audit for the predicted SHORTCUT.

Prediction and audit are separated
----------------------------------
TEST-TIME PREDICTION uses only the original prompt:
    attention / gradient / spectral / logit features.

Only AFTER all test predictions and shortcut localizations are frozen, the
audit intervenes on:
    1. the predicted shortcut span;
    2. matched random non-shortcut spans;
    3. the highest-attention non-shortcut span;
    4. the highest-gradient non-shortcut span;
    5. the predicted constraint span, when available.

The audit never feeds back into the detector or changes its threshold.

Two kinds of shortcut-driving evidence are reported
---------------------------------------------------
A. Detector mediation:
   Remove the predicted shortcut contribution from the detector's role logit
   while keeping the residual channel fixed. Measure probability drops and
   threshold flips. A second ablation removes all shortcut evidence.

B. Behavioral causal evidence on the target model:
   Apply delete / neutralize / mask interventions to the frozen predicted
   shortcut. Measure:
       - increase in gold-answer logit margin;
       - answer flips;
       - wrong-to-right corrections;
       - agreement across intervention types;
       - improvement over random / max-attention / max-gradient baselines.

A case receives `causal_alignment = true` only when:
    (i) the predicted shortcut is detector-critical, and
    (ii) intervening on that span behaviorally improves the target model.

The summary also reports:
    - shortcut evidence in hallucinated vs correct answers;
    - shortcut detection prevalence and odds ratios;
    - hallucination rate by shortcut-evidence quartile;
    - shortcut-only AUROC;
    - paired bootstrap delta-AUROC against the residual-only channel;
    - correlations between predicted shortcut contribution and observed
      intervention effect.

No test intervention or gold answer is used for prediction.

Example
-------
python role_mediated_whitebox_v4.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data /home/tong56/question_and_result.json \
    --out-dir role_mediated_v4_output \
    --span-mode clause \
    --interventions delete,neutralize,mask \
    --audit-interventions delete,neutralize,mask \
    --audit-random-repeats 3 \
    --residual-cap 1.0 \
    --dtype bfloat16 \
    --resume

The causal audit is enabled by default. Use --skip-causal-audit only for a
fast predictive run.

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



class UsageNormalizer:
    """Convert original-prompt span features into a bounded usage score.

    The score is intentionally label-free. It combines standardized
    decision-row attention density, head-max attention, and contrastive
    gradient density. Higher usage means the model's chosen 1-vs-2 decision
    is more locally tied to the span.
    """

    DEFAULT_KEYS = (
        "attn_density_late",
        "attn_headmax_late",
        "grad_norm_density",
    )

    def __init__(self, keys: Sequence[str] | None = None, temperature: float = 1.0):
        self.keys = list(keys or self.DEFAULT_KEYS)
        self.temperature = float(temperature)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, span_feat_dicts: Sequence[dict[str, float]]) -> "UsageNormalizer":
        X = np.asarray(
            [[float(d.get(k, 0.0)) for k in self.keys] for d in span_feat_dicts],
            dtype=float,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, span_feat_dicts: Sequence[dict[str, float]]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("UsageNormalizer is not fitted")
        if not span_feat_dicts:
            return np.zeros(0, dtype=float)
        X = np.asarray(
            [[float(d.get(k, 0.0)) for k in self.keys] for d in span_feat_dicts],
            dtype=float,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        Z = self.scaler.transform(X)
        raw = Z.mean(axis=1) / max(self.temperature, EPS)
        raw = np.clip(raw, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-raw))

    def report(self) -> dict:
        return {
            "keys": list(self.keys),
            "temperature": self.temperature,
            "means": self.scaler.mean_.tolist() if self.fitted else None,
            "scales": self.scaler.scale_.tolist() if self.fitted else None,
        }


def build_role_evidence(
    span_features: Sequence[dict[str, float]],
    role_probs: np.ndarray,
    usage_normalizer: UsageNormalizer,
) -> dict:
    """Return additive, span-indexed evidence before mechanism weights."""
    n = len(span_features)
    if n == 0:
        return {
            "usage": np.zeros(0),
            "shortcut_base": np.zeros(0),
            "constraint_base": np.zeros(0),
            "shortcut_evidence": 0.0,
            "constraint_evidence": 0.0,
            "role_ambiguity": 0.0,
        }

    usage = usage_normalizer.transform(span_features)
    shortcut_p = role_probs[:, ROLE_TO_ID["shortcut"]]
    constraint_p = role_probs[:, ROLE_TO_ID["constraint"]]

    # Divide by n so evidence is comparable across questions with different
    # numbers of candidate spans and remains exactly additive.
    shortcut_base = shortcut_p * usage / n
    constraint_base = constraint_p * usage / n
    entropy = -(role_probs * np.log(role_probs + 1e-12)).sum(axis=1)

    return {
        "usage": usage,
        "shortcut_base": shortcut_base,
        "constraint_base": constraint_base,
        "shortcut_evidence": float(shortcut_base.sum()),
        "constraint_evidence": float(constraint_base.sum()),
        "role_ambiguity": float(entropy.mean()),
    }


class RoleMechanismHead:
    """Monotonic role channel with exact signed span contributions."""

    def __init__(
        self,
        epochs: int = 2500,
        lr: float = 0.03,
        l2: float = 1e-3,
        min_shortcut_weight: float = 0.05,
        seed: int = 0,
    ):
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.min_shortcut_weight = float(min_shortcut_weight)
        self.seed = int(seed)
        self.bias = 0.0
        self.beta_shortcut = 1.0
        self.beta_constraint = 1.0
        self.fitted = False

    def fit(
        self,
        evidences: Sequence[dict],
        labels: Sequence[int],
    ) -> "RoleMechanismHead":
        X = np.asarray(
            [
                [
                    float(e["shortcut_evidence"]),
                    float(e["constraint_evidence"]),
                ]
                for e in evidences
            ],
            dtype=np.float32,
        )
        y = np.asarray(labels, dtype=np.float32)
        if len(np.unique(y)) < 2:
            raise ValueError("RoleMechanismHead requires both classes")

        torch.manual_seed(self.seed)
        x_t = torch.tensor(X)
        y_t = torch.tensor(y)

        bias = torch.nn.Parameter(torch.tensor(0.0))
        raw_s = torch.nn.Parameter(torch.tensor(0.0))
        raw_c = torch.nn.Parameter(torch.tensor(0.0))
        params = [bias, raw_s, raw_c]
        opt = torch.optim.Adam(params, lr=self.lr)

        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        sample_w = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
        w_t = torch.tensor(sample_w, dtype=torch.float32)

        best = float("inf")
        best_values = None
        patience = 250
        stale = 0

        for _ in range(self.epochs):
            opt.zero_grad()
            beta_s = F.softplus(raw_s)
            beta_c = F.softplus(raw_c)
            logits = bias + beta_s * x_t[:, 0] - beta_c * x_t[:, 1]
            per_item = F.binary_cross_entropy_with_logits(
                logits, y_t, reduction="none"
            )
            loss = (per_item * w_t).mean()
            loss = loss + self.l2 * (beta_s.square() + beta_c.square())
            loss = loss + 5.0 * F.relu(
                torch.tensor(self.min_shortcut_weight) - beta_s
            ).square()
            loss.backward()
            opt.step()

            value = float(loss.detach().item())
            if value + 1e-7 < best:
                best = value
                best_values = (
                    float(bias.detach().item()),
                    float(F.softplus(raw_s).detach().item()),
                    float(F.softplus(raw_c).detach().item()),
                )
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_values is None:
            raise RuntimeError("role mechanism optimization failed")
        self.bias, self.beta_shortcut, self.beta_constraint = best_values
        self.fitted = True
        return self

    def decision_function(self, evidences: Sequence[dict]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RoleMechanismHead is not fitted")
        s = np.asarray([float(e["shortcut_evidence"]) for e in evidences])
        c = np.asarray([float(e["constraint_evidence"]) for e in evidences])
        return self.bias + self.beta_shortcut * s - self.beta_constraint * c

    def predict_proba(self, evidences: Sequence[dict]) -> np.ndarray:
        z = np.clip(self.decision_function(evidences), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def span_contributions(self, evidence: dict) -> tuple[np.ndarray, np.ndarray]:
        positive = self.beta_shortcut * np.asarray(evidence["shortcut_base"])
        negative = -self.beta_constraint * np.asarray(evidence["constraint_base"])
        return positive, negative

    def report(self) -> dict:
        return {
            "bias": self.bias,
            "beta_shortcut": self.beta_shortcut,
            "beta_constraint": self.beta_constraint,
            "formula": (
                "bias + beta_shortcut * shortcut_evidence "
                "- beta_constraint * constraint_evidence"
            ),
        }


class ResidualHead:
    """Global white-box channel that never receives role features."""

    def __init__(self, cv: int = 5):
        self.matrix = FeatureMatrix()
        self.cv = int(cv)
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
    ) -> "ResidualHead":
        self.matrix.fit(feat_dicts)
        X = self.matrix.transform(feat_dicts)
        y = np.asarray(labels, dtype=int)
        self.pipe = self._make_pipe(y)
        self.pipe.fit(X, y)
        return self

    def decision_function(
        self,
        feat_dicts: Sequence[dict[str, float]],
    ) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("ResidualHead is not fitted")
        X = self.matrix.transform(feat_dicts)
        return np.asarray(self.pipe.decision_function(X), dtype=float)

    def predict_proba(
        self,
        feat_dicts: Sequence[dict[str, float]],
    ) -> np.ndarray:
        z = np.clip(self.decision_function(feat_dicts), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

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


class FinalCalibrator:
    """Calibrate role + capped residual while preserving decomposition.

    The coefficient on role_logit is fixed to 1. The residual correction is
    bounded by residual_cap. Only a global bias and positive temperature are
    learned.
    """

    def __init__(
        self,
        residual_cap: float = 1.0,
        epochs: int = 2000,
        lr: float = 0.03,
        seed: int = 0,
    ):
        self.residual_cap = float(residual_cap)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.bias = 0.0
        self.temperature = 1.0
        self.fitted = False

    def residual_adjustment(self, residual_logits: Sequence[float]) -> np.ndarray:
        z = np.asarray(residual_logits, dtype=float)
        return self.residual_cap * np.tanh(z)

    def fit(
        self,
        role_logits: Sequence[float],
        residual_logits: Sequence[float],
        labels: Sequence[int],
    ) -> "FinalCalibrator":
        role = torch.tensor(np.asarray(role_logits), dtype=torch.float32)
        residual = torch.tensor(
            self.residual_adjustment(residual_logits), dtype=torch.float32
        )
        y_np = np.asarray(labels, dtype=np.float32)
        y = torch.tensor(y_np)

        torch.manual_seed(self.seed)
        bias = torch.nn.Parameter(torch.tensor(0.0))
        raw_temp = torch.nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.Adam([bias, raw_temp], lr=self.lr)

        pos = max(float(y_np.sum()), 1.0)
        neg = max(float(len(y_np) - y_np.sum()), 1.0)
        sample_w = np.where(
            y_np > 0.5,
            len(y_np) / (2.0 * pos),
            len(y_np) / (2.0 * neg),
        )
        w = torch.tensor(sample_w, dtype=torch.float32)

        best = float("inf")
        best_values = None
        stale = 0
        for _ in range(self.epochs):
            opt.zero_grad()
            temp = F.softplus(raw_temp) + 0.05
            logits = (role + residual + bias) / temp
            per_item = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = (per_item * w).mean()
            loss.backward()
            opt.step()

            value = float(loss.detach().item())
            if value + 1e-7 < best:
                best = value
                best_values = (
                    float(bias.detach().item()),
                    float(temp.detach().item()),
                )
                stale = 0
            else:
                stale += 1
                if stale >= 200:
                    break

        if best_values is None:
            raise RuntimeError("final calibration failed")
        self.bias, self.temperature = best_values
        self.fitted = True
        return self

    def decision_function(
        self,
        role_logits: Sequence[float],
        residual_logits: Sequence[float],
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FinalCalibrator is not fitted")
        role = np.asarray(role_logits, dtype=float)
        adjustment = self.residual_adjustment(residual_logits)
        return (role + adjustment + self.bias) / self.temperature

    def predict_proba(
        self,
        role_logits: Sequence[float],
        residual_logits: Sequence[float],
    ) -> np.ndarray:
        z = np.clip(
            self.decision_function(role_logits, residual_logits), -40.0, 40.0
        )
        return 1.0 / (1.0 + np.exp(-z))

    def report(self) -> dict:
        return {
            "residual_cap": self.residual_cap,
            "bias": self.bias,
            "temperature": self.temperature,
            "formula": (
                "(role_logit + residual_cap*tanh(residual_logit) + bias)"
                " / temperature"
            ),
        }


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

    # Weak role labels are produced on training items only.
    # Test interventions are performed later by a separate frozen audit.
    target_indices = set(train_indices)

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


def role_mechanism_oof_logits(
    evidences: list[dict],
    labels: list[int],
    n_splits: int,
    args,
) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    splits = min(n_splits, pos, neg)
    if splits < 2:
        head = RoleMechanismHead(
            epochs=args.mechanism_epochs,
            lr=args.mechanism_lr,
            l2=args.mechanism_l2,
            min_shortcut_weight=args.min_shortcut_weight,
            seed=args.seed,
        ).fit(evidences, labels)
        return head.decision_function(evidences)

    splitter = StratifiedKFold(
        n_splits=splits, shuffle=True, random_state=args.seed
    )
    oof = np.zeros(len(y), dtype=float)
    dummy = np.zeros((len(y), 1))

    for fold, (train_rows, val_rows) in enumerate(splitter.split(dummy, y)):
        head = RoleMechanismHead(
            epochs=args.mechanism_epochs,
            lr=args.mechanism_lr,
            l2=args.mechanism_l2,
            min_shortcut_weight=args.min_shortcut_weight,
            seed=args.seed + fold,
        ).fit(
            [evidences[i] for i in train_rows],
            [labels[i] for i in train_rows],
        )
        oof[val_rows] = head.decision_function(
            [evidences[i] for i in val_rows]
        )
    return oof


def compute_residual_oof_logits(
    feat_dicts: list[dict],
    labels: list[int],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    matrix = FeatureMatrix().fit(feat_dicts)
    X = matrix.transform(feat_dicts)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    splits = min(n_splits, pos, neg)
    if splits < 2:
        head = ResidualHead(cv=2).fit(feat_dicts, labels)
        return head.decision_function(feat_dicts)

    splitter = StratifiedKFold(
        n_splits=splits, shuffle=True, random_state=seed
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
        oof[val_rows] = pipe.decision_function(X[val_rows])
    return oof


def sigmoid_np(values: Sequence[float]) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def choose_distinct_explanation_spans(
    base: dict,
    role_probs: np.ndarray,
    evidence: dict,
    role_head: RoleMechanismHead,
    calibrator: FinalCalibrator,
    role_probability_threshold: float,
    usage_threshold: float,
    contribution_threshold: float,
) -> dict:
    """Choose distinct C/S spans and return exact signed contributions."""
    spans = base["spans"]
    usage = np.asarray(evidence["usage"], dtype=float)
    shortcut_pos, constraint_neg = role_head.span_contributions(evidence)

    # Convert exact role-logit contributions into final calibrated-logit units.
    shortcut_final = shortcut_pos / calibrator.temperature
    constraint_final = constraint_neg / calibrator.temperature
    total_final = shortcut_final + constraint_final

    s_prob = role_probs[:, ROLE_TO_ID["shortcut"]]
    c_prob = role_probs[:, ROLE_TO_ID["constraint"]]

    s_candidates = [
        i for i in range(len(spans))
        if s_prob[i] >= role_probability_threshold
        and usage[i] >= usage_threshold
        and shortcut_final[i] >= contribution_threshold
    ]
    c_candidates = [
        i for i in range(len(spans))
        if c_prob[i] >= role_probability_threshold
        and usage[i] >= usage_threshold
        and abs(constraint_final[i]) >= contribution_threshold
    ]

    best_pair = None
    best_score = -float("inf")
    for s_idx in s_candidates:
        for c_idx in c_candidates:
            if s_idx == c_idx:
                continue
            score = shortcut_final[s_idx] + abs(constraint_final[c_idx])
            if score > best_score:
                best_score = score
                best_pair = (s_idx, c_idx)

    shortcut_idx = None
    constraint_idx = None
    if best_pair is not None:
        shortcut_idx, constraint_idx = best_pair
    else:
        if s_candidates:
            shortcut_idx = max(s_candidates, key=lambda i: shortcut_final[i])
        remaining_c = [i for i in c_candidates if i != shortcut_idx]
        if remaining_c:
            constraint_idx = max(remaining_c, key=lambda i: abs(constraint_final[i]))

    per_span = []
    for i, span in enumerate(spans):
        per_span.append(
            {
                "span_id": int(span["span_id"]),
                "text": span["text"],
                "constraint_probability": float(c_prob[i]),
                "shortcut_probability": float(s_prob[i]),
                "irrelevant_probability": float(
                    role_probs[i, ROLE_TO_ID["irrelevant"]]
                ),
                "decision_usage": float(usage[i]),
                "shortcut_logit_contribution": float(shortcut_final[i]),
                "constraint_logit_contribution": float(constraint_final[i]),
                "net_role_logit_contribution": float(total_final[i]),
            }
        )

    def span_payload(idx: int | None, role: str):
        if idx is None:
            return {
                "resolved": False,
                "reason": (
                    f"no distinct {role} span passed probability, usage, "
                    "and contribution thresholds"
                ),
            }
        row = per_span[idx]
        return {"resolved": True, **row}

    return {
        "predicted_shortcut": span_payload(shortcut_idx, "shortcut"),
        "predicted_constraint": span_payload(constraint_idx, "constraint"),
        "distinct_pair_resolved": bool(
            shortcut_idx is not None and constraint_idx is not None
        ),
        "per_span_contributions": per_span,
    }


def explanation_status(
    combined_probability: float,
    role_probability: float,
    residual_probability: float,
    explanation: dict,
    decision_threshold: float,
) -> str:
    if combined_probability < decision_threshold:
        return "not_flagged"
    if (
        role_probability >= decision_threshold
        and explanation["predicted_shortcut"].get("resolved", False)
    ):
        if explanation["predicted_constraint"].get("resolved", False):
            return "role_supported_with_constraint_competition"
        return "role_supported_shortcut_only"
    if residual_probability >= decision_threshold:
        return "residual_only_no_faithful_role_explanation"
    return "insufficient_role_evidence"



def _safe_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _safe_median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> list[float] | None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        stats[b] = rng.choice(arr, size=arr.size, replace=True).mean()
    alpha = (1.0 - level) / 2.0
    return [
        float(np.quantile(stats, alpha)),
        float(np.quantile(stats, 1.0 - alpha)),
    ]


def bootstrap_group_difference_ci(
    positive: Sequence[float],
    negative: Sequence[float],
    n_boot: int,
    seed: int,
) -> list[float] | None:
    a = np.asarray(positive, dtype=float)
    b = np.asarray(negative, dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        aa = rng.choice(a, size=a.size, replace=True)
        bb = rng.choice(b, size=b.size, replace=True)
        stats[k] = aa.mean() - bb.mean()
    return [
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    ]


def paired_bootstrap_difference_ci(
    target: Sequence[float],
    baseline: Sequence[float],
    n_boot: int,
    seed: int,
) -> list[float] | None:
    a = np.asarray(target, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.size == 0 or a.size != b.size:
        return None
    diff = a - b
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        stats[k] = rng.choice(diff, size=diff.size, replace=True).mean()
    return [
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    ]


def paired_signflip_pvalue(
    target: Sequence[float],
    baseline: Sequence[float],
    n_perm: int,
    seed: int,
) -> float | None:
    a = np.asarray(target, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.size == 0 or a.size != b.size:
        return None
    diff = a - b
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=diff.size)
        if abs(float((diff * signs).mean())) >= observed - 1e-15:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def label_permutation_pvalue(
    values: Sequence[float],
    labels: Sequence[int],
    n_perm: int,
    seed: int,
) -> float | None:
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.size == 0 or len(np.unique(y)) < 2:
        return None
    observed = abs(float(x[y == 1].mean() - x[y == 0].mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        stat = abs(float(x[yp == 1].mean() - x[yp == 0].mean()))
        if stat >= observed - 1e-15:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        average_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def spearman_correlation(
    x: Sequence[float],
    y: Sequence[float],
) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 3 or a.size != b.size:
        return None
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    if np.std(ra) <= EPS or np.std(rb) <= EPS:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def paired_auc_bootstrap(
    labels: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    n_boot: int,
    seed: int,
) -> dict | None:
    y = np.asarray(labels, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if len(np.unique(y)) < 2:
        return None

    observed = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    diffs = []
    attempts = 0
    while len(diffs) < n_boot and attempts < n_boot * 10:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(
            float(
                roc_auc_score(y[idx], a[idx])
                - roc_auc_score(y[idx], b[idx])
            )
        )
    if not diffs:
        return None
    arr = np.asarray(diffs)
    return {
        "delta_auroc": observed,
        "bootstrap_95_ci": [
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
        ],
        "bootstrap_probability_delta_le_zero": float(np.mean(arr <= 0)),
        "n_bootstrap": int(len(arr)),
    }


def _cohens_d(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    a = np.asarray(positive, dtype=float)
    b = np.asarray(negative, dtype=float)
    if a.size < 2 or b.size < 2:
        return None
    pooled = math.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / max(a.size + b.size - 2, 1)
    )
    if pooled <= EPS:
        return None
    return float((a.mean() - b.mean()) / pooled)


def shortcut_explanatory_statistics(
    test_records: Sequence[dict],
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict:
    records = list(test_records)
    labels = np.asarray(
        [int(bool(r["hallucinated"])) for r in records], dtype=int
    )
    shortcut_evidence = np.asarray(
        [
            float(r["mechanism_decomposition"]["shortcut_evidence"])
            for r in records
        ],
        dtype=float,
    )
    detected = np.asarray(
        [
            bool(
                r["explanation"]["predicted_shortcut"].get(
                    "resolved", False
                )
            )
            for r in records
        ],
        dtype=bool,
    )
    max_probability = np.asarray(
        [
            max(
                (
                    float(x["shortcut_probability"])
                    for x in r["explanation"]["per_span_contributions"]
                ),
                default=0.0,
            )
            for r in records
        ],
        dtype=float,
    )
    max_contribution = np.asarray(
        [
            max(
                (
                    float(x["shortcut_logit_contribution"])
                    for x in r["explanation"]["per_span_contributions"]
                ),
                default=0.0,
            )
            for r in records
        ],
        dtype=float,
    )

    def group_summary(values: np.ndarray, group: int) -> dict:
        arr = values[labels == group]
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                arr, n_boot=n_boot, seed=seed + group
            ),
        }

    hall = shortcut_evidence[labels == 1]
    correct = shortcut_evidence[labels == 0]
    a = int(np.sum((labels == 1) & detected))
    b = int(np.sum((labels == 1) & ~detected))
    c = int(np.sum((labels == 0) & detected))
    d = int(np.sum((labels == 0) & ~detected))

    # Haldane-Anscombe correction only when needed.
    aa, bb, cc, dd = map(float, (a, b, c, d))
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5
    odds_ratio = (aa * dd) / (bb * cc)
    log_se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    odds_ci = [
        float(math.exp(math.log(odds_ratio) - 1.96 * log_se)),
        float(math.exp(math.log(odds_ratio) + 1.96 * log_se)),
    ]

    # Dose-response quartiles. qcut-like ranking avoids duplicate-boundary loss.
    order = np.argsort(shortcut_evidence, kind="mergesort")
    bins = np.empty(len(records), dtype=int)
    for rank, idx in enumerate(order):
        bins[idx] = min(3, (4 * rank) // max(len(records), 1))
    quartiles = []
    for q in range(4):
        mask = bins == q
        quartiles.append(
            {
                "quartile": q + 1,
                "n": int(mask.sum()),
                "shortcut_evidence_min": (
                    float(shortcut_evidence[mask].min()) if mask.any() else None
                ),
                "shortcut_evidence_max": (
                    float(shortcut_evidence[mask].max()) if mask.any() else None
                ),
                "hallucination_rate": (
                    float(labels[mask].mean()) if mask.any() else None
                ),
            }
        )

    detection_rate_difference = (
        a / max(a + b, 1) - c / max(c + d, 1)
    )
    detection_perm_p = label_permutation_pvalue(
        detected.astype(float), labels, n_perm=n_perm, seed=seed + 19
    )

    return {
        "shortcut_evidence_by_outcome": {
            "hallucination": group_summary(shortcut_evidence, 1),
            "correct": group_summary(shortcut_evidence, 0),
            "mean_difference_hallucination_minus_correct": float(
                hall.mean() - correct.mean()
            ),
            "difference_bootstrap_95_ci": bootstrap_group_difference_ci(
                hall, correct, n_boot=n_boot, seed=seed + 3
            ),
            "label_permutation_p_value": label_permutation_pvalue(
                shortcut_evidence, labels, n_perm=n_perm, seed=seed + 5
            ),
            "cohens_d": _cohens_d(hall, correct),
            "shortcut_evidence_auroc": float(
                roc_auc_score(labels, shortcut_evidence)
            ),
            "shortcut_evidence_auprc": float(
                average_precision_score(labels, shortcut_evidence)
            ),
        },
        "additional_shortcut_quantities": {
            "max_shortcut_probability": {
                "hallucination": group_summary(max_probability, 1),
                "correct": group_summary(max_probability, 0),
                "auroc": float(roc_auc_score(labels, max_probability)),
            },
            "max_shortcut_logit_contribution": {
                "hallucination": group_summary(max_contribution, 1),
                "correct": group_summary(max_contribution, 0),
                "auroc": float(roc_auc_score(labels, max_contribution)),
            },
        },
        "shortcut_detection_prevalence": {
            "contingency": {
                "hallucination_detected": a,
                "hallucination_not_detected": b,
                "correct_detected": c,
                "correct_not_detected": d,
            },
            "detected_rate_given_hallucination": a / max(a + b, 1),
            "detected_rate_given_correct": c / max(c + d, 1),
            "rate_difference": detection_rate_difference,
            "label_permutation_p_value": detection_perm_p,
            "hallucination_rate_given_detected": a / max(a + c, 1),
            "hallucination_rate_given_not_detected": b / max(b + d, 1),
            "odds_ratio": float(odds_ratio),
            "odds_ratio_95_ci": odds_ci,
        },
        "shortcut_evidence_dose_response": quartiles,
    }


def audit_one_span(
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    item: dict,
    span_record: dict,
    choices: tuple[str, str],
    interventions: Sequence[str],
    normalized_deadzone: float,
    consistency_threshold: float,
) -> dict:
    question, base_prompt, gold_raw = adapter.unpack(item)
    gold = normalize_gold(gold_raw, choices)
    original_prompt = adapter.render(question, question, base_prompt)
    original = extractor.score_prompt(original_prompt, gold, choices)

    span = CandidateSpan(
        span_id=int(span_record["span_id"]),
        start=int(span_record["start"]),
        end=int(span_record["end"]),
        text=str(span_record["text"]),
    )
    effects = []
    for kind in interventions:
        modified = intervene(question, span, kind)
        variant_prompt = adapter.render(question, modified, base_prompt)
        score = extractor.score_prompt(variant_prompt, gold, choices)
        improvement = float(score["gold_margin"] - original["gold_margin"])
        normalized = float(
            improvement
            / (
                abs(score["gold_margin"])
                + abs(original["gold_margin"])
                + EPS
            )
        )
        effects.append(
            {
                "kind": kind,
                "gold_margin_after": float(score["gold_margin"]),
                "gold_margin_improvement": improvement,
                "normalized_gold_margin_improvement": normalized,
                "chosen_after": score["chosen"],
                "answer_flip": bool(score["chosen"] != original["chosen"]),
                "wrong_to_right": bool(
                    original["chosen"] != gold and score["chosen"] == gold
                ),
                "right_to_wrong": bool(
                    original["chosen"] == gold and score["chosen"] != gold
                ),
            }
        )

    normalized_values = np.asarray(
        [e["normalized_gold_margin_improvement"] for e in effects],
        dtype=float,
    )
    raw_values = np.asarray(
        [e["gold_margin_improvement"] for e in effects], dtype=float
    )
    positive_fraction = float(
        np.mean(normalized_values > normalized_deadzone)
    )
    median_normalized = float(np.median(normalized_values))
    causal_success = bool(
        median_normalized > normalized_deadzone
        and positive_fraction >= consistency_threshold
    )

    return {
        "span_id": int(span.span_id),
        "text": span.text,
        "original_gold_margin": float(original["gold_margin"]),
        "original_chosen": original["chosen"],
        "gold": gold,
        "original_correct": bool(original["chosen"] == gold),
        "interventions": effects,
        "summary": {
            "mean_gold_margin_improvement": float(raw_values.mean()),
            "median_gold_margin_improvement": float(np.median(raw_values)),
            "mean_normalized_improvement": float(normalized_values.mean()),
            "median_normalized_improvement": median_normalized,
            "positive_consistency_fraction": positive_fraction,
            "answer_flip_any": bool(any(e["answer_flip"] for e in effects)),
            "wrong_to_right_any": bool(
                any(e["wrong_to_right"] for e in effects)
            ),
            "wrong_to_right_majority": bool(
                np.mean([e["wrong_to_right"] for e in effects]) >= 0.5
            ),
            "right_to_wrong_any": bool(
                any(e["right_to_wrong"] for e in effects)
            ),
            "causal_success": causal_success,
        },
    }


def _best_other_span(
    spans: Sequence[dict],
    predicted_shortcut_id: int,
    feature_key: str,
) -> int | None:
    candidates = [
        (
            float(span["features"].get(feature_key, 0.0)),
            int(span["span_id"]),
        )
        for span in spans
        if int(span["span_id"]) != predicted_shortcut_id
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def run_postprediction_causal_audit(
    test_indices: Sequence[int],
    items: list[dict],
    base_by_idx: dict[int, dict],
    explanations_by_idx: dict[int, dict],
    evidence_by_idx: dict[int, dict],
    role_logits_by_idx: dict[int, float],
    residual_logits_by_idx: dict[int, float],
    final_probabilities_by_idx: dict[int, float],
    role_mechanism: RoleMechanismHead,
    calibrator: FinalCalibrator,
    decision_threshold: float,
    extractor: WeakWhiteboxExtractor,
    adapter: PromptAdapter,
    choices: tuple[str, str],
    interventions: Sequence[str],
    random_repeats: int,
    normalized_deadzone: float,
    consistency_threshold: float,
    audit_max_items: int,
    seed: int,
    resume: bool,
    cache_path: Path,
) -> dict[int, dict]:
    cached = load_jsonl_index(cache_path) if resume else {}
    output: dict[int, dict] = {}
    signature = json.dumps(
        {
            "interventions": list(interventions),
            "random_repeats": random_repeats,
            "normalized_deadzone": normalized_deadzone,
            "consistency_threshold": consistency_threshold,
        },
        sort_keys=True,
    )

    eligible = [
        idx
        for idx in test_indices
        if explanations_by_idx[idx]["predicted_shortcut"].get(
            "resolved", False
        )
    ]
    if audit_max_items > 0:
        eligible = eligible[:audit_max_items]

    for count, idx in enumerate(eligible, 1):
        old = cached.get(idx)
        if old is not None and old.get("audit_signature") == signature:
            output[idx] = old
            continue

        base = base_by_idx[idx]
        explanation = explanations_by_idx[idx]
        shortcut = explanation["predicted_shortcut"]
        shortcut_id = int(shortcut["span_id"])
        spans = base["spans"]
        by_id = {int(s["span_id"]): s for s in spans}

        predicted_audit = audit_one_span(
            extractor,
            adapter,
            items[idx],
            by_id[shortcut_id],
            choices,
            interventions,
            normalized_deadzone,
            consistency_threshold,
        )

        rng = random.Random(seed + idx * 1009)
        other_ids = [
            int(s["span_id"])
            for s in spans
            if int(s["span_id"]) != shortcut_id
        ]
        rng.shuffle(other_ids)
        random_ids = other_ids[: min(random_repeats, len(other_ids))]
        random_audits = [
            audit_one_span(
                extractor,
                adapter,
                items[idx],
                by_id[sid],
                choices,
                interventions,
                normalized_deadzone,
                consistency_threshold,
            )
            for sid in random_ids
        ]

        attn_id = _best_other_span(
            spans, shortcut_id, "attn_density_late"
        )
        grad_id = _best_other_span(
            spans, shortcut_id, "grad_norm_density"
        )
        attn_audit = (
            audit_one_span(
                extractor,
                adapter,
                items[idx],
                by_id[attn_id],
                choices,
                interventions,
                normalized_deadzone,
                consistency_threshold,
            )
            if attn_id is not None
            else None
        )
        grad_audit = (
            audit_one_span(
                extractor,
                adapter,
                items[idx],
                by_id[grad_id],
                choices,
                interventions,
                normalized_deadzone,
                consistency_threshold,
            )
            if grad_id is not None
            else None
        )

        constraint_audit = None
        constraint = explanation["predicted_constraint"]
        if constraint.get("resolved", False):
            cid = int(constraint["span_id"])
            if cid in by_id and cid != shortcut_id:
                constraint_audit = audit_one_span(
                    extractor,
                    adapter,
                    items[idx],
                    by_id[cid],
                    choices,
                    interventions,
                    normalized_deadzone,
                    consistency_threshold,
                )

        evidence = evidence_by_idx[idx]
        shortcut_pos, _ = role_mechanism.span_contributions(evidence)
        role_logit = float(role_logits_by_idx[idx])
        residual_logit = float(residual_logits_by_idx[idx])
        original_probability = float(final_probabilities_by_idx[idx])

        predicted_contribution = float(shortcut_pos[shortcut_id])
        role_without_predicted = role_logit - predicted_contribution
        probability_without_predicted = float(
            calibrator.predict_proba(
                [role_without_predicted], [residual_logit]
            )[0]
        )
        role_without_all = role_logit - float(np.sum(shortcut_pos))
        probability_without_all = float(
            calibrator.predict_proba([role_without_all], [residual_logit])[0]
        )

        record = {
            "idx": idx,
            "audit_signature": signature,
            "hallucinated": bool(base["hallucinated"]),
            "predicted_shortcut": predicted_audit,
            "baselines": {
                "random_nonshortcut_spans": random_audits,
                "max_attention_nonshortcut": attn_audit,
                "max_gradient_nonshortcut": grad_audit,
            },
            "predicted_constraint": constraint_audit,
            "detector_mediation": {
                "original_probability": original_probability,
                "probability_without_predicted_shortcut": (
                    probability_without_predicted
                ),
                "probability_without_all_shortcut_evidence": (
                    probability_without_all
                ),
                "probability_drop_predicted_shortcut": float(
                    original_probability - probability_without_predicted
                ),
                "probability_drop_all_shortcuts": float(
                    original_probability - probability_without_all
                ),
                "original_flagged": bool(
                    original_probability >= decision_threshold
                ),
                "flagged_without_predicted_shortcut": bool(
                    probability_without_predicted >= decision_threshold
                ),
                "flagged_without_all_shortcuts": bool(
                    probability_without_all >= decision_threshold
                ),
                "predicted_shortcut_detector_critical": bool(
                    original_probability >= decision_threshold
                    and probability_without_predicted < decision_threshold
                ),
                "all_shortcuts_detector_critical": bool(
                    original_probability >= decision_threshold
                    and probability_without_all < decision_threshold
                ),
                "predicted_shortcut_role_logit_contribution": (
                    predicted_contribution
                ),
            },
        }
        record["causal_alignment"] = bool(
            record["detector_mediation"][
                "predicted_shortcut_detector_critical"
            ]
            and predicted_audit["summary"]["causal_success"]
        )
        append_jsonl(cache_path, record)
        output[idx] = record

        if count % 10 == 0 or count == len(eligible):
            print(
                f"post-prediction causal audit: {count}/{len(eligible)}",
                flush=True,
            )

    return output


def _mean_random_baseline(record: dict) -> float | None:
    audits = record["baselines"]["random_nonshortcut_spans"]
    if not audits:
        return None
    return float(
        np.mean(
            [
                x["summary"]["median_normalized_improvement"]
                for x in audits
            ]
        )
    )


def aggregate_postprediction_causal_audit(
    audit_by_idx: dict[int, dict],
    all_test_records: Sequence[dict],
    evidence_by_idx: dict[int, dict],
    role_logits_by_idx: dict[int, float],
    residual_logits_by_idx: dict[int, float],
    final_probabilities_by_idx: dict[int, float],
    decision_threshold: float,
    role_mechanism: RoleMechanismHead,
    calibrator: FinalCalibrator,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict | None:
    audits = list(audit_by_idx.values())
    if not audits:
        return None

    def method_value(rec: dict, method: str) -> float | None:
        if method == "predicted_shortcut":
            return float(
                rec["predicted_shortcut"]["summary"][
                    "median_normalized_improvement"
                ]
            )
        if method == "random_nonshortcut":
            return _mean_random_baseline(rec)
        audit = rec["baselines"].get(method)
        if audit is None:
            return None
        return float(audit["summary"]["median_normalized_improvement"])

    def summarize_method(method: str, label_filter: int | None) -> dict:
        selected = [
            rec
            for rec in audits
            if label_filter is None
            or int(bool(rec["hallucinated"])) == label_filter
        ]
        values = [
            method_value(rec, method)
            for rec in selected
            if method_value(rec, method) is not None
        ]
        causal_success = []
        wrong_to_right = []
        answer_flip = []
        for rec in selected:
            if method == "predicted_shortcut":
                audit = rec["predicted_shortcut"]
                causal_success.append(
                    bool(audit["summary"]["causal_success"])
                )
                wrong_to_right.append(
                    bool(audit["summary"]["wrong_to_right_any"])
                )
                answer_flip.append(bool(audit["summary"]["answer_flip_any"]))
            elif method == "random_nonshortcut":
                for audit in rec["baselines"]["random_nonshortcut_spans"]:
                    causal_success.append(
                        bool(audit["summary"]["causal_success"])
                    )
                    wrong_to_right.append(
                        bool(audit["summary"]["wrong_to_right_any"])
                    )
                    answer_flip.append(
                        bool(audit["summary"]["answer_flip_any"])
                    )
            elif method in (
                "max_attention_nonshortcut",
                "max_gradient_nonshortcut",
            ):
                audit = rec["baselines"].get(method)
                if audit is not None:
                    causal_success.append(
                        bool(audit["summary"]["causal_success"])
                    )
                    wrong_to_right.append(
                        bool(audit["summary"]["wrong_to_right_any"])
                    )
                    answer_flip.append(
                        bool(audit["summary"]["answer_flip_any"])
                    )
        return {
            "n": len(values),
            "mean_median_normalized_improvement": _safe_mean(values),
            "median_median_normalized_improvement": _safe_median(values),
            "bootstrap_mean_95_ci": bootstrap_mean_ci(
                values, n_boot=n_boot, seed=seed + {"predicted_shortcut": 11, "random_nonshortcut": 23, "max_attention_nonshortcut": 37, "max_gradient_nonshortcut": 53}[method]
            ),
            "causal_success_rate": (
                float(np.mean(causal_success)) if causal_success else None
            ),
            "answer_flip_any_rate": (
                float(np.mean(answer_flip)) if answer_flip else None
            ),
            "wrong_to_right_any_rate": (
                float(np.mean(wrong_to_right)) if wrong_to_right else None
            ),
        }

    methods = (
        "predicted_shortcut",
        "random_nonshortcut",
        "max_attention_nonshortcut",
        "max_gradient_nonshortcut",
    )
    method_summaries = {}
    for method in methods:
        method_summaries[method] = {
            "all": summarize_method(method, None),
            "hallucination": summarize_method(method, 1),
            "correct": summarize_method(method, 0),
        }

    paired_comparisons = {}
    for baseline in (
        "random_nonshortcut",
        "max_attention_nonshortcut",
        "max_gradient_nonshortcut",
    ):
        target_values = []
        baseline_values = []
        labels = []
        for rec in audits:
            target = method_value(rec, "predicted_shortcut")
            other = method_value(rec, baseline)
            if target is None or other is None:
                continue
            target_values.append(target)
            baseline_values.append(other)
            labels.append(int(bool(rec["hallucinated"])))
        paired_comparisons[baseline] = {
            "n": len(target_values),
            "mean_improvement_difference": (
                float(
                    np.mean(
                        np.asarray(target_values)
                        - np.asarray(baseline_values)
                    )
                )
                if target_values
                else None
            ),
            "paired_bootstrap_95_ci": paired_bootstrap_difference_ci(
                target_values,
                baseline_values,
                n_boot=n_boot,
                seed=seed + 101,
            ),
            "paired_signflip_p_value": paired_signflip_pvalue(
                target_values,
                baseline_values,
                n_perm=n_perm,
                seed=seed + 103,
            ),
            "hallucination_only": {},
        }
        hall_target = [
            t for t, y in zip(target_values, labels) if y == 1
        ]
        hall_base = [
            b for b, y in zip(baseline_values, labels) if y == 1
        ]
        paired_comparisons[baseline]["hallucination_only"] = {
            "n": len(hall_target),
            "mean_improvement_difference": (
                float(np.mean(np.asarray(hall_target) - np.asarray(hall_base)))
                if hall_target
                else None
            ),
            "paired_bootstrap_95_ci": paired_bootstrap_difference_ci(
                hall_target,
                hall_base,
                n_boot=n_boot,
                seed=seed + 107,
            ),
            "paired_signflip_p_value": paired_signflip_pvalue(
                hall_target,
                hall_base,
                n_perm=n_perm,
                seed=seed + 109,
            ),
        }

    # Detector mediation is computed for every test item directly from the
    # frozen role evidence. Behavioral interventions are not required here.
    labels = []
    original_prob = []
    without_predicted = []
    without_all = []
    for record in all_test_records:
        idx = int(record["idx"])
        labels.append(int(bool(record["hallucinated"])))
        p = float(final_probabilities_by_idx[idx])
        original_prob.append(p)

        evidence = evidence_by_idx[idx]
        shortcut_pos, _ = role_mechanism.span_contributions(evidence)
        role_logit = float(role_logits_by_idx[idx])
        residual_logit = float(residual_logits_by_idx[idx])

        role_without_all = role_logit - float(np.sum(shortcut_pos))
        without_all.append(
            float(
                calibrator.predict_proba(
                    [role_without_all], [residual_logit]
                )[0]
            )
        )

        predicted = record["explanation"]["predicted_shortcut"]
        if predicted.get("resolved", False):
            sid = int(predicted["span_id"])
            role_without_predicted = role_logit - float(shortcut_pos[sid])
            without_predicted.append(
                float(
                    calibrator.predict_proba(
                        [role_without_predicted], [residual_logit]
                    )[0]
                )
            )
        else:
            without_predicted.append(p)

    original_metrics = evaluate_binary(
        labels, original_prob, decision_threshold
    )
    targeted_metrics = evaluate_binary(
        labels, without_predicted, decision_threshold
    )
    all_zero_metrics = evaluate_binary(
        labels, without_all, decision_threshold
    )

    def lost_flags(counterfactual: Sequence[float]) -> dict:
        y = np.asarray(labels, dtype=int)
        p0 = np.asarray(original_prob)
        p1 = np.asarray(counterfactual)
        lost = (p0 >= decision_threshold) & (p1 < decision_threshold)
        return {
            "all_flags_lost": int(lost.sum()),
            "true_positive_flags_lost": int(np.sum(lost & (y == 1))),
            "false_positive_flags_lost": int(np.sum(lost & (y == 0))),
            "fraction_of_original_flags_lost": float(
                lost.sum() / max(np.sum(p0 >= decision_threshold), 1)
            ),
            "fraction_of_original_true_positives_lost": float(
                np.sum(lost & (y == 1))
                / max(np.sum((p0 >= decision_threshold) & (y == 1)), 1)
            ),
        }

    contribution = []
    improvement = []
    shortcut_probability = []
    usage = []
    hall_contribution = []
    hall_improvement = []
    for rec in audits:
        span = rec["predicted_shortcut"]
        contribution.append(
            float(
                rec["detector_mediation"][
                    "predicted_shortcut_role_logit_contribution"
                ]
            )
        )
        improvement.append(
            float(span["summary"]["median_normalized_improvement"])
        )
        # Probability and usage are supplied by the prediction file below.
        record = next(
            r for r in all_test_records if int(r["idx"]) == int(rec["idx"])
        )
        pred_span = record["explanation"]["predicted_shortcut"]
        shortcut_probability.append(
            float(pred_span["shortcut_probability"])
        )
        usage.append(float(pred_span["decision_usage"]))
        if rec["hallucinated"]:
            hall_contribution.append(contribution[-1])
            hall_improvement.append(improvement[-1])

    causal_alignment = [
        rec for rec in audits if bool(rec.get("causal_alignment", False))
    ]
    causal_alignment_hall = [
        rec
        for rec in causal_alignment
        if bool(rec["hallucinated"])
    ]

    return {
        "audit_population": {
            "n_test": len(all_test_records),
            "n_predicted_shortcut_resolved_and_audited": len(audits),
            "n_hallucination_audited": int(
                sum(bool(x["hallucinated"]) for x in audits)
            ),
            "n_correct_audited": int(
                sum(not bool(x["hallucinated"]) for x in audits)
            ),
        },
        "behavioral_effects": method_summaries,
        "predicted_shortcut_vs_baselines": paired_comparisons,
        "detector_mediation": {
            "structural_derivative_final_logit_per_unit_shortcut_evidence": (
                float(
                    role_mechanism.beta_shortcut
                    / calibrator.temperature
                )
            ),
            "original_metrics": original_metrics,
            "without_predicted_shortcut_metrics": targeted_metrics,
            "without_all_shortcut_evidence_metrics": all_zero_metrics,
            "predicted_shortcut_ablation": lost_flags(without_predicted),
            "all_shortcut_evidence_ablation": lost_flags(without_all),
            "mean_probability_drop_predicted_shortcut": float(
                np.mean(
                    np.asarray(original_prob)
                    - np.asarray(without_predicted)
                )
            ),
            "mean_probability_drop_all_shortcuts": float(
                np.mean(
                    np.asarray(original_prob) - np.asarray(without_all)
                )
            ),
        },
        "causal_alignment": {
            "definition": (
                "predicted shortcut is detector-critical AND its held-out "
                "interventions consistently improve the target model's "
                "gold-answer margin"
            ),
            "n": len(causal_alignment),
            "rate_among_audited": float(
                len(causal_alignment) / max(len(audits), 1)
            ),
            "n_hallucination": len(causal_alignment_hall),
            "rate_among_audited_hallucinations": float(
                len(causal_alignment_hall)
                / max(
                    sum(bool(x["hallucinated"]) for x in audits),
                    1,
                )
            ),
        },
        "predicted_strength_vs_observed_behavior": {
            "spearman_detector_contribution_vs_margin_improvement": (
                spearman_correlation(contribution, improvement)
            ),
            "spearman_shortcut_probability_vs_margin_improvement": (
                spearman_correlation(shortcut_probability, improvement)
            ),
            "spearman_decision_usage_vs_margin_improvement": (
                spearman_correlation(usage, improvement)
            ),
            "hallucination_only_spearman_contribution_vs_improvement": (
                spearman_correlation(hall_contribution, hall_improvement)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Role-mediated weakly supervised white-box detector v4"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Chat")
    parser.add_argument("--data", default="question_and_result.json")
    parser.add_argument("--out-dir", default="role_mediated_output_v4")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--question-field")
    parser.add_argument("--prompt-field")
    parser.add_argument("--gold-field")
    parser.add_argument(
        "--answer-instruction",
        default="Reply with a single character: 1 or 2.",
    )
    parser.add_argument("--no-chat-template", action="store_true")

    parser.add_argument("--choice-a", default="1")
    parser.add_argument("--choice-b", default="2")
    parser.add_argument(
        "--span-mode", choices=["sentence", "clause"], default="clause"
    )
    parser.add_argument("--include-question-span", action="store_true")
    parser.add_argument("--min-span-words", type=int, default=3)

    parser.add_argument(
        "--interventions", default="delete,neutralize,mask"
    )
    parser.add_argument("--max-intervention-spans", type=int, default=6)
    parser.set_defaults(causal_audit=True)
    parser.add_argument(
        "--causal-audit",
        dest="causal_audit",
        action="store_true",
        help="run frozen post-prediction shortcut interventions (default)",
    )
    parser.add_argument(
        "--skip-causal-audit",
        dest="causal_audit",
        action="store_false",
        help="skip the expensive held-out causal audit",
    )
    parser.add_argument(
        "--intervene-test",
        dest="causal_audit",
        action="store_true",
        help="deprecated alias for --causal-audit",
    )
    parser.add_argument(
        "--audit-interventions",
        default="delete,neutralize,mask",
        help="post-prediction intervention types",
    )
    parser.add_argument("--audit-random-repeats", type=int, default=3)
    parser.add_argument(
        "--audit-normalized-deadzone", type=float, default=0.05
    )
    parser.add_argument(
        "--audit-consistency-threshold", type=float, default=2.0 / 3.0
    )
    parser.add_argument(
        "--audit-max-items",
        type=int,
        default=0,
        help="0 audits every resolved test shortcut",
    )
    parser.add_argument("--statistics-bootstrap", type=int, default=2000)
    parser.add_argument("--statistics-permutations", type=int, default=5000)
    parser.add_argument("--role-deadzone", type=float, default=0.25)
    parser.add_argument("--role-temperature", type=float, default=0.75)
    parser.add_argument("--min-role-reliability", type=float, default=0.20)

    parser.add_argument("--usage-temperature", type=float, default=1.0)
    parser.add_argument("--mechanism-epochs", type=int, default=2500)
    parser.add_argument("--mechanism-lr", type=float, default=0.03)
    parser.add_argument("--mechanism-l2", type=float, default=1e-3)
    parser.add_argument("--min-shortcut-weight", type=float, default=0.05)
    parser.add_argument(
        "--residual-cap",
        type=float,
        default=1.0,
        help=(
            "maximum absolute global residual correction in logit units; "
            "smaller values force stronger role mediation"
        ),
    )

    parser.add_argument("--role-prob-threshold", type=float, default=0.45)
    parser.add_argument("--usage-threshold", type=float, default=0.35)
    parser.add_argument(
        "--span-contribution-threshold", type=float, default=0.005
    )

    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role-oof-folds", type=int, default=5)
    parser.add_argument("--hall-oof-folds", type=int, default=5)
    parser.add_argument("--lap-topk", type=int, default=10)

    parser.add_argument("--device", default="auto")
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
            "fewer than two interventions weakens role reliability estimates"
        )

    audit_interventions = [
        x.strip()
        for x in args.audit_interventions.split(",")
        if x.strip()
    ]
    invalid_audit = set(audit_interventions) - allowed
    if invalid_audit:
        parser.error(
            f"invalid audit interventions: {sorted(invalid_audit)}"
        )
    if args.causal_audit and len(audit_interventions) < 2:
        warnings.warn(
            "causal audit is stronger with at least two intervention types"
        )

    choices = (str(args.choice_a), str(args.choice_b))
    if choices[0] == choices[1]:
        parser.error("choice-a and choice-b must differ")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cache = out_dir / "base_features.jsonl"
    intervention_cache = out_dir / "intervention_labels.jsonl"
    prediction_path = out_dir / "predictions.jsonl"
    causal_audit_path = out_dir / "causal_audit.jsonl"
    summary_path = out_dir / "summary.json"
    bundle_path = out_dir / "role_mediated_bundle.joblib"

    if not args.resume:
        for path in (
            base_cache,
            intervention_cache,
            prediction_path,
            causal_audit_path,
        ):
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

    # 1. Original-prompt white-box features.
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
    all_labels = np.asarray(
        [int(base_by_idx[i]["hallucinated"]) for i in valid_indices],
        dtype=int,
    )
    if len(np.unique(all_labels)) < 2:
        raise RuntimeError("model produced only one hallucination class")

    train_idx, test_idx = train_test_split(
        valid_indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=all_labels,
    )
    train_idx = sorted(int(x) for x in train_idx)
    test_idx = sorted(int(x) for x in test_idx)
    train_set, test_set = set(train_idx), set(test_idx)

    # 2. Training-only weak role supervision. Optional test interventions are
    # stored only for later explanation audit.
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

    # OOF role probabilities for every training item prevent the mechanism
    # head from consuming same-item role pseudo-label fit.
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
        base_by_idx, test_idx, role_head
    )

    # Label-free usage scaling is fitted only on training spans.
    all_train_span_features = [
        span["features"]
        for idx in train_idx
        for span in base_by_idx[idx]["spans"]
    ]
    usage_normalizer = UsageNormalizer(
        temperature=args.usage_temperature
    ).fit(all_train_span_features)

    train_evidence = [
        build_role_evidence(
            [s["features"] for s in base_by_idx[idx]["spans"]],
            train_role_probs[idx],
            usage_normalizer,
        )
        for idx in train_idx
    ]
    test_evidence = [
        build_role_evidence(
            [s["features"] for s in base_by_idx[idx]["spans"]],
            test_role_probs[idx],
            usage_normalizer,
        )
        for idx in test_idx
    ]
    train_labels = [int(base_by_idx[idx]["hallucinated"]) for idx in train_idx]
    test_labels = [int(base_by_idx[idx]["hallucinated"]) for idx in test_idx]
    train_global = [base_by_idx[idx]["global_features"] for idx in train_idx]
    test_global = [base_by_idx[idx]["global_features"] for idx in test_idx]

    # 3. OOF role and residual channels for unbiased calibration/thresholding.
    role_oof_logits = role_mechanism_oof_logits(
        train_evidence,
        train_labels,
        n_splits=args.hall_oof_folds,
        args=args,
    )
    residual_oof_logits_fn = compute_residual_oof_logits(
        train_global,
        train_labels,
        n_splits=args.hall_oof_folds,
        seed=args.seed,
    )

    calibrator = FinalCalibrator(
        residual_cap=args.residual_cap,
        seed=args.seed,
    ).fit(role_oof_logits, residual_oof_logits_fn, train_labels)

    train_final_oof_prob = calibrator.predict_proba(
        role_oof_logits, residual_oof_logits_fn
    )
    threshold = choose_f1_threshold(
        np.asarray(train_labels), train_final_oof_prob
    )

    # 4. Full-data channel heads for held-out prediction.
    role_mechanism = RoleMechanismHead(
        epochs=args.mechanism_epochs,
        lr=args.mechanism_lr,
        l2=args.mechanism_l2,
        min_shortcut_weight=args.min_shortcut_weight,
        seed=args.seed,
    ).fit(train_evidence, train_labels)

    residual_head = ResidualHead(cv=args.hall_oof_folds).fit(
        train_global, train_labels
    )

    train_role_full_logits = role_mechanism.decision_function(train_evidence)
    test_role_logits = role_mechanism.decision_function(test_evidence)
    train_residual_full_logits = residual_head.decision_function(train_global)
    test_residual_logits = residual_head.decision_function(test_global)

    train_role_oof_prob = sigmoid_np(role_oof_logits)
    train_residual_oof_prob = sigmoid_np(residual_oof_logits_fn)
    test_role_prob = sigmoid_np(test_role_logits)
    test_residual_prob = sigmoid_np(test_residual_logits)
    test_final_prob = calibrator.predict_proba(
        test_role_logits, test_residual_logits
    )

    role_threshold = choose_f1_threshold(
        np.asarray(train_labels), train_role_oof_prob
    )
    residual_threshold = choose_f1_threshold(
        np.asarray(train_labels), train_residual_oof_prob
    )

    channel_metrics = {
        "train_oof": {
            "role_only": evaluate_binary(
                train_labels, train_role_oof_prob, role_threshold
            ),
            "residual_only": evaluate_binary(
                train_labels, train_residual_oof_prob, residual_threshold
            ),
            "combined": evaluate_binary(
                train_labels, train_final_oof_prob, threshold
            ),
        },
        "test": {
            "role_only": evaluate_binary(
                test_labels, test_role_prob, role_threshold
            ),
            "residual_only": evaluate_binary(
                test_labels, test_residual_prob, residual_threshold
            ),
            "combined": evaluate_binary(
                test_labels, test_final_prob, threshold
            ),
        },
    }

    # 5. Freeze predictions and explanations before any test intervention.
    test_position = {idx: j for j, idx in enumerate(test_idx)}
    train_position = {idx: j for j, idx in enumerate(train_idx)}
    explanations_by_idx: dict[int, dict] = {}
    evidence_by_idx: dict[int, dict] = {}
    role_logits_by_idx: dict[int, float] = {}
    residual_logits_by_idx: dict[int, float] = {}
    final_probabilities_by_idx: dict[int, float] = {}
    prediction_records_by_idx: dict[int, dict] = {}

    for split_name, indices, role_map, evidence_list in (
        ("train", train_idx, train_role_probs, train_evidence),
        ("test", test_idx, test_role_probs, test_evidence),
    ):
        for local_pos, idx in enumerate(indices):
            base = base_by_idx[idx]
            probs = role_map[idx]
            evidence = evidence_list[local_pos]

            if split_name == "test":
                role_logit = float(test_role_logits[test_position[idx]])
                residual_logit = float(
                    test_residual_logits[test_position[idx]]
                )
                role_probability = float(test_role_prob[test_position[idx]])
                residual_probability = float(
                    test_residual_prob[test_position[idx]]
                )
                final_probability = float(test_final_prob[test_position[idx]])
            else:
                role_logit = float(
                    train_role_full_logits[train_position[idx]]
                )
                residual_logit = float(
                    train_residual_full_logits[train_position[idx]]
                )
                role_probability = float(sigmoid_np([role_logit])[0])
                residual_probability = float(
                    sigmoid_np([residual_logit])[0]
                )
                final_probability = float(
                    calibrator.predict_proba(
                        [role_logit], [residual_logit]
                    )[0]
                )

            explanation = choose_distinct_explanation_spans(
                base=base,
                role_probs=probs,
                evidence=evidence,
                role_head=role_mechanism,
                calibrator=calibrator,
                role_probability_threshold=args.role_prob_threshold,
                usage_threshold=args.usage_threshold,
                contribution_threshold=args.span_contribution_threshold,
            )
            status = explanation_status(
                combined_probability=final_probability,
                role_probability=role_probability,
                residual_probability=residual_probability,
                explanation=explanation,
                decision_threshold=threshold,
            )

            residual_adjustment = float(
                calibrator.residual_adjustment([residual_logit])[0]
            )
            final_logit = float(
                calibrator.decision_function(
                    [role_logit], [residual_logit]
                )[0]
            )

            record = {
                "idx": idx,
                "split": split_name,
                "question": base["question"],
                "gold": base["gold"],
                "chosen": base["chosen"],
                "hallucinated": bool(base["hallucinated"]),
                "chosen_margin": base["chosen_margin"],
                "prediction": {
                    "combined_probability": final_probability,
                    "role_only_probability": role_probability,
                    "residual_only_probability": residual_probability,
                    "combined_threshold": threshold,
                    "flagged": bool(final_probability >= threshold),
                    "explanation_status": status,
                },
                "mechanism_decomposition": {
                    "shortcut_evidence": float(
                        evidence["shortcut_evidence"]
                    ),
                    "constraint_evidence": float(
                        evidence["constraint_evidence"]
                    ),
                    "role_intercept": role_mechanism.bias,
                    "role_logit": role_logit,
                    "residual_raw_logit": residual_logit,
                    "residual_capped_adjustment": residual_adjustment,
                    "calibration_bias": calibrator.bias,
                    "calibration_temperature": calibrator.temperature,
                    "final_logit": final_logit,
                },
                "explanation": explanation,
            }

            pseudo = intervention_by_idx.get(idx)
            if pseudo is not None:
                record["intervention_pseudo_labels"] = pseudo[
                    "span_pseudo_labels"
                ]

            prediction_records_by_idx[idx] = record
            explanations_by_idx[idx] = explanation
            evidence_by_idx[idx] = evidence
            role_logits_by_idx[idx] = role_logit
            residual_logits_by_idx[idx] = residual_logit
            final_probabilities_by_idx[idx] = final_probability

    # Descriptive shortcut statistics use frozen, intervention-free test output.
    test_prediction_records = [
        prediction_records_by_idx[idx] for idx in test_idx
    ]
    shortcut_statistics = shortcut_explanatory_statistics(
        test_prediction_records,
        n_boot=args.statistics_bootstrap,
        n_perm=args.statistics_permutations,
        seed=args.seed,
    )
    shortcut_statistics["combined_vs_residual_paired_auc"] = (
        paired_auc_bootstrap(
            test_labels,
            test_final_prob,
            test_residual_prob,
            n_boot=args.statistics_bootstrap,
            seed=args.seed + 211,
        )
    )

    # Strictly post-prediction causal audit.
    causal_audit_by_idx: dict[int, dict] = {}
    causal_audit_summary = None
    if args.causal_audit:
        causal_audit_by_idx = run_postprediction_causal_audit(
            test_indices=test_idx,
            items=items,
            base_by_idx=base_by_idx,
            explanations_by_idx=explanations_by_idx,
            evidence_by_idx=evidence_by_idx,
            role_logits_by_idx=role_logits_by_idx,
            residual_logits_by_idx=residual_logits_by_idx,
            final_probabilities_by_idx=final_probabilities_by_idx,
            role_mechanism=role_mechanism,
            calibrator=calibrator,
            decision_threshold=threshold,
            extractor=extractor,
            adapter=adapter,
            choices=choices,
            interventions=audit_interventions,
            random_repeats=args.audit_random_repeats,
            normalized_deadzone=args.audit_normalized_deadzone,
            consistency_threshold=args.audit_consistency_threshold,
            audit_max_items=args.audit_max_items,
            seed=args.seed,
            resume=args.resume,
            cache_path=causal_audit_path,
        )
        causal_audit_summary = aggregate_postprediction_causal_audit(
            audit_by_idx=causal_audit_by_idx,
            all_test_records=test_prediction_records,
            evidence_by_idx=evidence_by_idx,
            role_logits_by_idx=role_logits_by_idx,
            residual_logits_by_idx=residual_logits_by_idx,
            final_probabilities_by_idx=final_probabilities_by_idx,
            decision_threshold=threshold,
            role_mechanism=role_mechanism,
            calibrator=calibrator,
            n_boot=args.statistics_bootstrap,
            n_perm=args.statistics_permutations,
            seed=args.seed,
        )

    # Add audit results only after every prediction has been frozen.
    for idx, audit in causal_audit_by_idx.items():
        if idx in prediction_records_by_idx:
            prediction_records_by_idx[idx][
                "postprediction_causal_audit"
            ] = audit

    if prediction_path.exists():
        prediction_path.unlink()
    for split_name, indices in (("train", train_idx), ("test", test_idx)):
        for idx in indices:
            append_jsonl(prediction_path, prediction_records_by_idx[idx])

    role_counts = {name: 0 for name in ROLE_NAMES}
    reliabilities = []
    for idx in train_idx:
        pseudo = intervention_by_idx.get(idx)
        if pseudo is None:
            continue
        for label in pseudo["span_pseudo_labels"]:
            if label.get("evaluated", False):
                role_counts[label["hard_role"]] += 1
                reliabilities.append(float(label["reliability"]))

    resolved_test = [
        explanations_by_idx[idx] for idx in test_idx
        if idx in explanations_by_idx
    ]
    explanation_coverage = {
        "shortcut_resolved_rate": float(
            np.mean([
                x["predicted_shortcut"].get("resolved", False)
                for x in resolved_test
            ])
        ),
        "constraint_resolved_rate": float(
            np.mean([
                x["predicted_constraint"].get("resolved", False)
                for x in resolved_test
            ])
        ),
        "distinct_pair_resolved_rate": float(
            np.mean([x["distinct_pair_resolved"] for x in resolved_test])
        ),
    }

    # Store only standard sklearn objects and plain state. This avoids
    # pickling custom classes under __main__, which is fragile across runs.
    bundle = {
        "role_head": {
            "feature_keys": role_head.matrix.keys,
            "pipeline": role_head.pipe,
        },
        "usage_normalizer": {
            "keys": usage_normalizer.keys,
            "temperature": usage_normalizer.temperature,
            "scaler": usage_normalizer.scaler,
        },
        "role_mechanism": role_mechanism.report(),
        "residual_head": {
            "feature_keys": residual_head.matrix.keys,
            "pipeline": residual_head.pipe,
        },
        "calibrator": calibrator.report(),
        "thresholds": {
            "combined": threshold,
            "role": role_threshold,
            "residual": residual_threshold,
            "role_probability": args.role_prob_threshold,
            "usage": args.usage_threshold,
            "span_contribution": args.span_contribution_threshold,
        },
        "choices": choices,
        "model_name": args.model,
    }
    joblib.dump(bundle, bundle_path)

    summary = {
        "method": "role-mediated weakly supervised white-box detector v4",
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
            float(np.mean(reliabilities)) if reliabilities else None
        ),
        "n_role_training_spans": len(role_features),
        "channel_metrics": channel_metrics,
        "selected_thresholds_from_train_oof": {
            "combined": threshold,
            "role_only": role_threshold,
            "residual_only": residual_threshold,
        },
        "role_mechanism": role_mechanism.report(),
        "usage_normalizer": usage_normalizer.report(),
        "final_calibrator": calibrator.report(),
        "role_feature_coefficients": role_head.coefficient_report(),
        "residual_feature_coefficients": residual_head.coefficient_report(),
        "explanation_coverage_test": explanation_coverage,
        "shortcut_explanatory_statistics": shortcut_statistics,
        "postprediction_causal_audit": causal_audit_summary,
        "files": {
            "base_features": str(base_cache),
            "intervention_labels": str(intervention_cache),
            "predictions": str(prediction_path),
            "causal_audit": (
                str(causal_audit_path) if args.causal_audit else None
            ),
            "model_bundle": str(bundle_path),
        },
        "method_notes": {
            "test_prediction_uses_interventions": False,
            "test_prediction_uses_gold_answer": False,
            "test_interventions_are_strictly_postprediction_audit_only": True,
            "causal_audit_enabled": args.causal_audit,
            "causal_alignment_requires_detector_and_behavioral_evidence": True,
            "audit_compares_predicted_shortcut_to_random_attention_gradient_baselines": True,
            "behavioral_interventions_support_causal_claims_but_may_induce_distribution_shift": True,
            "all_shortcut_zero_ablation_covers_every_test_item": True,
            "span_roles_are_part_of_role_logit": True,
            "span_contributions_sum_exactly_to_role_logit_minus_intercept": True,
            "shortcut_weight_is_nonnegative": True,
            "constraint_weight_is_nonnegative_and_subtracted": True,
            "global_residual_is_capped": args.residual_cap,
            "role_coefficient_in_final_raw_logit_is_fixed_to_one": True,
            "distinct_constraint_shortcut_required_for_pair_explanation": True,
            "explanation_can_abstain": True,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary, f, ensure_ascii=False, indent=2, default=json_default
        )

    print("\n=== role-mediated detector complete ===")
    print(json.dumps(channel_metrics["test"], indent=2))
    print("\n=== shortcut explanatory statistics ===")
    print(
        json.dumps(
            shortcut_statistics["shortcut_evidence_by_outcome"],
            indent=2,
        )
    )
    if causal_audit_summary is not None:
        print("\n=== post-prediction causal audit ===")
        print(json.dumps(causal_audit_summary, indent=2))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
