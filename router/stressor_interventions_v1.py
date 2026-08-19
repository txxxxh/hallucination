#!/usr/bin/env python3
"""
Intervention teacher with robust evaluation and hidden-state recording.

Implemented stressors
---------------------
S1: Knowledge scarcity / internally inaccessible knowledge
S2: Prior or shortcut conflict
S3: Context and structural complexity
S4: Compute-budget insufficiency or misallocation
S6: Epistemic-control deficiency

S5 (external-system availability/quality) is intentionally omitted because this
version targets a local non-tool-using Llama-3.1-8B-Instruct model.

The script:
1. Loads JSON or JSONL benchmark records.
2. Generates a base answer and evaluates it.
3. Runs stressor-specific targeted interventions and matched controls.
4. Computes behavioral recovery and specificity scores.
5. Writes detailed JSONL traces plus a summary JSON.

This version can save compact hidden-state traces for the base answer and every
intervention variant. These traces can later supervise one-vs-rest stressor routers.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


LOGGER = logging.getLogger("stressor_interventions")

STRESSOR_NAMES = {
    "S1": "knowledge_scarcity",
    "S2": "prior_shortcut_conflict",
    "S3": "context_structural_complexity",
    "S4": "compute_budget_misallocation",
    "S6": "epistemic_control_deficiency",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's question accurately. "
    "Do not invent facts. If a short answer is sufficient, keep it concise."
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json_or_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object")
                rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise ValueError("JSON list must contain objects")
        return list(obj)

    if isinstance(obj, dict):
        # Common wrappers used by datasets.
        for key in ("data", "examples", "records", "items"):
            if key in obj and isinstance(obj[key], list):
                return list(obj[key])
        # A dict keyed by sample id.
        if all(isinstance(v, dict) for v in obj.values()):
            rows = []
            for key, value in obj.items():
                row = dict(value)
                row.setdefault("_dict_key", key)
                rows.append(row)
            return rows

    raise ValueError("Unsupported JSON structure; expected list, wrapped list, or dict of objects")


def write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path: Union[str, Path], obj: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_completed_ids(path: Union[str, Path]) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    completed.add(str(obj["id"]))
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed final JSONL line while resuming")
    return completed


def get_nested(obj: Mapping[str, Any], dotted_key: Optional[str], default: Any = None) -> Any:
    if not dotted_key:
        return default
    cur: Any = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(as_text(x) for x in value if as_text(x))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def answer_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [as_text(x) for x in value if as_text(x)]
    if isinstance(value, dict):
        # Common aliases: {"text": [...]}, {"answer": "..."}
        for key in ("text", "answers", "answer", "value"):
            if key in value:
                return answer_list(value[key])
    return [as_text(value)] if as_text(value) else []


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[`*_#]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_choice(text: str) -> Optional[str]:
    patterns = [
        r"(?:final\s*answer|answer|option|choice)\s*[:：]?\s*\(?([A-F1-6])\)?\b",
        r"^\s*\(?([A-F1-6])\)?[\s\.:：-]",
        r"\b\(?([A-F])\)?\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|python|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_first_json(text: str) -> Optional[Any]:
    text = strip_code_fences(text)
    # Try the whole output first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def clean_model_text(text: str) -> str:
    return text.strip().replace("<|eot_id|>", "").strip()


def is_abstention_text(text: str) -> bool:
    normalized = normalize_text(text)
    patterns = [
        "i do not have enough information",
        "i don t have enough information",
        "cannot answer reliably",
        "can t answer reliably",
        "insufficient information",
        "unable to determine",
        "cannot determine",
        "i don t know",
        "i do not know",
    ]
    return any(pattern in normalized for pattern in patterns)


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.mean(clean) if clean else None


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    id: str
    question: str
    context: str
    references: List[str]
    raw_prompt: str = ""
    oracle_facts: str = ""
    symptom: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    token_count: int
    mean_token_logprob: Optional[float]
    finish_reason: str
    seed: int
    elapsed_seconds: float


@dataclass
class EvaluationResult:
    status: str
    correct: Optional[bool]
    score: Optional[float]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatedRun:
    text: str
    correct: Optional[bool]
    evaluation_status: str
    is_abstention: bool
    evaluation_score: Optional[float]
    token_count: int
    mean_token_logprob: Optional[float]
    reference_logprob: Optional[float]
    seed: int
    elapsed_seconds: float
    hidden_state_path: Optional[str] = None
    hidden_state_error: Optional[str] = None
    evaluation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Variant:
    variant_id: str
    stressor: str
    kind: str
    user_prompt: str
    is_control: bool = False
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class LocalCausalLM:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        device_map: Optional[str] = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        max_input_tokens: int = 4096,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.system_prompt = system_prompt

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        LOGGER.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        LOGGER.info("Loading model: %s", model_name)
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if device_map:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not device_map:
            self.model.to(device)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        LOGGER.info("Model input device: %s", self.input_device)

    def format_chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("apply_chat_template failed; using fallback format: %s", exc)
            return (
                f"System: {messages[0]['content']}\n\n"
                f"User: {messages[1]['content']}\n\nAssistant:"
            )

    def encode_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, torch.Tensor]:
        formatted = self.format_chat(user_prompt, system_prompt)
        encoded = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        return {k: v.to(self.input_device) for k, v in encoded.items()}

    @torch.inference_mode()
    def generate(
        self,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        seed_everything(seed)
        encoded = self.encode_prompt(user_prompt, system_prompt)
        input_len = encoded["input_ids"].shape[1]
        do_sample = temperature > 0

        start = time.perf_counter()
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
            generation_kwargs["top_p"] = top_p
        outputs = self.model.generate(**encoded, **generation_kwargs)
        elapsed = time.perf_counter() - start

        sequences = outputs.sequences
        generated_ids = sequences[:, input_len:]
        text = clean_model_text(
            self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        )

        mean_logprob: Optional[float] = None
        try:
            transition_scores = self.model.compute_transition_scores(
                sequences,
                outputs.scores,
                normalize_logits=True,
            )
            if transition_scores.numel() > 0:
                # Exclude trailing pad positions when present.
                gen_tokens = generated_ids[0]
                valid = gen_tokens.ne(self.tokenizer.pad_token_id)
                vals = transition_scores[0][: valid.shape[0]][valid]
                if vals.numel() > 0:
                    mean_logprob = float(vals.float().mean().item())
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Could not compute transition scores: %s", exc)

        finish_reason = "eos" if (
            generated_ids.numel() > 0
            and int(generated_ids[0, -1].item()) == self.tokenizer.eos_token_id
        ) else "length_or_other"

        return GenerationResult(
            text=text,
            token_count=int(generated_ids.shape[1]),
            mean_token_logprob=mean_logprob,
            finish_reason=finish_reason,
            seed=seed,
            elapsed_seconds=elapsed,
        )

    @torch.inference_mode()
    def score_continuation(
        self,
        user_prompt: str,
        continuation: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[float]:
        if not continuation.strip():
            return None

        prompt_text = self.format_chat(user_prompt, system_prompt)
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )["input_ids"]
        full_text = prompt_text + continuation
        full = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens + 512,
            add_special_tokens=False,
        )
        input_ids = full["input_ids"].to(self.input_device)
        attention_mask = full["attention_mask"].to(self.input_device)
        prompt_len = min(prompt_ids.shape[1], input_ids.shape[1] - 1)
        if prompt_len >= input_ids.shape[1]:
            return None

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        start_idx = max(prompt_len - 1, 0)
        continuation_lp = token_lp[:, start_idx:]
        if continuation_lp.numel() == 0:
            return None
        return float(continuation_lp.mean().item())


    def resolve_hidden_layers(self, layer_spec: str) -> List[int]:
        """Resolve comma-separated absolute layers or percentages.

        Hugging Face returns hidden_states[0] for embeddings and hidden_states[i]
        for the output of transformer layer i. Therefore valid transformer layer
        indices are 1..num_hidden_layers.
        """
        n_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        if n_layers <= 0:
            raise RuntimeError("Model config does not expose num_hidden_layers")
        values: List[int] = []
        for raw in layer_spec.split(","):
            token = raw.strip()
            if not token:
                continue
            if token.endswith("%"):
                fraction = float(token[:-1]) / 100.0
                index = int(round(fraction * n_layers))
            else:
                index = int(token)
                if index < 0:
                    index = n_layers + 1 + index
            index = max(1, min(n_layers, index))
            if index not in values:
                values.append(index)
        if not values:
            values = [
                max(1, int(round(n_layers * p)))
                for p in (0.25, 0.50, 0.65, 0.80, 1.00)
            ]
        return sorted(set(values))

    @torch.inference_mode()
    def save_hidden_state_trace(
        self,
        user_prompt: str,
        answer: str,
        output_path: Union[str, Path],
        layer_spec: str,
    ) -> Dict[str, Any]:
        """Run one post-hoc causal forward pass and save selected states.

        Only selected layers and a few prompt/answer positions are retained, so
        storage is roughly a few hundred KB per run instead of hundreds of MB.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        encoded = self.encode_prompt(user_prompt)
        prompt_ids = encoded["input_ids"]
        answer_ids = self.tokenizer(
            answer,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.input_device)

        max_positions = int(getattr(self.model.config, "max_position_embeddings", 8192))
        max_full = min(max_positions, self.max_input_tokens + max(1, answer_ids.shape[1]))
        if answer_ids.shape[1] >= max_full:
            answer_ids = answer_ids[:, -max_full:]
            prompt_ids = prompt_ids[:, :0]
        elif prompt_ids.shape[1] + answer_ids.shape[1] > max_full:
            keep_prompt = max_full - answer_ids.shape[1]
            prompt_ids = prompt_ids[:, -keep_prompt:] if keep_prompt > 0 else prompt_ids[:, :0]

        full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        if full_ids.shape[1] == 0:
            raise RuntimeError("Cannot extract hidden states from an empty sequence")
        attention_mask = torch.ones_like(full_ids, device=self.input_device)

        outputs = self.model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        layer_indices = self.resolve_hidden_layers(layer_spec)

        prompt_len = int(prompt_ids.shape[1])
        answer_len = int(answer_ids.shape[1])
        positions: Dict[str, int] = {}
        if prompt_len > 0:
            positions["prompt_last"] = prompt_len - 1
        if answer_len > 0:
            positions["answer_first"] = prompt_len
            positions["answer_q25"] = prompt_len + int(round((answer_len - 1) * 0.25))
            positions["answer_q50"] = prompt_len + int(round((answer_len - 1) * 0.50))
            positions["answer_q75"] = prompt_len + int(round((answer_len - 1) * 0.75))
            positions["answer_last"] = prompt_len + answer_len - 1
        if not positions:
            positions["sequence_last"] = full_ids.shape[1] - 1

        position_names = list(positions)
        position_indices = [positions[name] for name in position_names]
        layer_tensors: List[torch.Tensor] = []
        for layer_index in layer_indices:
            layer = hidden_states[layer_index][0]
            selected = layer[position_indices].detach().to(dtype=torch.float16, device="cpu")
            layer_tensors.append(selected)
        hidden = torch.stack(layer_tensors, dim=0)

        payload = {
            "model": self.model_name,
            "layer_indices": layer_indices,
            "position_names": position_names,
            "position_indices": position_indices,
            "prompt_token_count": prompt_len,
            "answer_token_count": answer_len,
            "sequence_token_count": int(full_ids.shape[1]),
            "hidden": hidden,
        }
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(output_path)
        return {
            "path": str(output_path),
            "shape": list(hidden.shape),
            "layer_indices": layer_indices,
            "position_names": position_names,
            "prompt_token_count": prompt_len,
            "answer_token_count": answer_len,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Evaluator:
    LABELS = ("INVALID_REFERENCE", "UNANSWERABLE", "INCORRECT", "CORRECT")

    def __init__(
        self,
        mode: str,
        engine: LocalCausalLM,
        f1_threshold: float = 0.65,
        judge_max_new_tokens: int = 8,
        judge_retries: int = 2,
        fast_reference_match: bool = True,
    ) -> None:
        self.mode = mode
        self.engine = engine
        self.f1_threshold = f1_threshold
        self.judge_max_new_tokens = judge_max_new_tokens
        self.judge_retries = judge_retries
        self.fast_reference_match = fast_reference_match

    @staticmethod
    def _result(
        status: str,
        correct: Optional[bool],
        score: Optional[float],
        **details: Any,
    ) -> EvaluationResult:
        return EvaluationResult(status=status, correct=correct, score=score, details=details)

    def _fast_match(
        self,
        prediction: str,
        references: Sequence[str],
    ) -> Optional[EvaluationResult]:
        """High-precision shortcuts only; uncertain cases still go to the judge."""
        pred = normalize_text(prediction)
        if not pred:
            return None
        ref_norms = [normalize_text(x) for x in references if normalize_text(x)]
        if not ref_norms:
            return None
        if pred in ref_norms:
            return self._result("correct", True, 1.0, evaluator="fast_exact")

        first = pred.split()[0] if pred.split() else ""
        for ref in ref_norms:
            if ref in {"yes", "no"} and first == ref:
                return self._result("correct", True, 1.0, evaluator="fast_yes_no")

        # Containment is accepted only for concise answers. Long answers can
        # contain the reference and then contradict it, so they are judged.
        pred_tokens = pred.split()
        for ref in ref_norms:
            ref_tokens = ref.split()
            concise_limit = max(12, 3 * len(ref_tokens) + 6)
            if ref and ref in pred and len(pred_tokens) <= concise_limit:
                before = pred.split(ref, 1)[0].split()[-3:]
                if not any(tok in {"not", "never", "no"} for tok in before):
                    return self._result("correct", True, 1.0, evaluator="fast_concise_contains")
        return None

    @classmethod
    def _parse_label(cls, text: str) -> Optional[str]:
        upper = strip_code_fences(text).strip().upper()
        for label in cls.LABELS:
            if re.search(rf"\b{re.escape(label)}\b", upper):
                return label
        return None

    def evaluate(
        self,
        question: str,
        context: str,
        prediction: str,
        references: Sequence[str],
        seed: int,
    ) -> EvaluationResult:
        if not references:
            return self._result("unresolved", None, None, error="missing_reference")

        if self.mode == "exact":
            pred = normalize_text(prediction)
            scores = [float(pred == normalize_text(ref)) for ref in references]
            score = max(scores)
            return self._result("correct" if score else "incorrect", bool(score), score)

        if self.mode == "contains":
            pred = normalize_text(prediction)
            scores = [float(normalize_text(ref) in pred) for ref in references if normalize_text(ref)]
            score = max(scores) if scores else 0.0
            return self._result("correct" if score else "incorrect", bool(score), score)

        if self.mode == "choice":
            pred_choice = extract_choice(prediction)
            ref_choices = [extract_choice(ref) or normalize_text(ref).upper() for ref in references]
            score = float(pred_choice is not None and pred_choice in ref_choices)
            return self._result(
                "correct" if score else "incorrect",
                bool(score),
                score,
                prediction_choice=pred_choice,
                reference_choices=ref_choices,
            )

        if self.mode == "token_f1":
            scores = [token_f1(prediction, ref) for ref in references]
            score = max(scores)
            correct = score >= self.f1_threshold
            return self._result("correct" if correct else "incorrect", correct, score)

        if self.mode != "llm_judge":
            raise ValueError(f"Unsupported evaluation mode: {self.mode}")

        if self.fast_reference_match:
            fast = self._fast_match(prediction, references)
            if fast is not None:
                return fast

        reference_text = "\n".join(f"- {ref}" for ref in references)
        context_text = context.strip() or "[No context supplied; use the question and reference.]"
        attempts: List[Dict[str, Any]] = []
        for retry in range(self.judge_retries + 1):
            judge_prompt = f"""
Evaluate the candidate answer using the CONTEXT as the main source of truth.
The accepted reference may occasionally be wrong or incompatible with the context.

QUESTION:
{question}

CONTEXT:
{context_text}

ACCEPTED REFERENCE ANSWER(S):
{reference_text}

CANDIDATE ANSWER:
{prediction}

Output exactly one label and nothing else:
CORRECT          - candidate answers the question and adds no material false claim
INCORRECT        - candidate is wrong, contradictory, incomplete, or adds a material false claim
INVALID_REFERENCE - context supports a different answer and the accepted reference is invalid
UNANSWERABLE     - the supplied context/question does not contain enough information to decide
""".strip()
            result = self.engine.generate(
                judge_prompt,
                max_new_tokens=self.judge_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=seed + 7919 + retry * 101,
                system_prompt="Return exactly one evaluation label. Do not explain.",
            )
            label = self._parse_label(result.text)
            attempts.append({"output": result.text, "parsed_label": label, "retry": retry})
            if label == "CORRECT":
                return self._result("correct", True, 1.0, judge_attempts=attempts)
            if label == "INCORRECT":
                return self._result("incorrect", False, 0.0, judge_attempts=attempts)
            if label == "INVALID_REFERENCE":
                return self._result("invalid_reference", None, None, judge_attempts=attempts)
            if label == "UNANSWERABLE":
                return self._result("unanswerable", None, None, judge_attempts=attempts)

        return self._result("unresolved", None, None, judge_attempts=attempts, parse_failed=True)


# ---------------------------------------------------------------------------
# Prompt helpers and intervention proposal
# ---------------------------------------------------------------------------


class PromptBuilder:
    @staticmethod
    def base_user_prompt(sample: Sample) -> str:
        if sample.raw_prompt:
            return sample.raw_prompt.strip()
        parts: List[str] = []
        if sample.context:
            parts.append(f"Context:\n{sample.context}")
        parts.append(f"Question:\n{sample.question}")
        parts.append("Give the final answer clearly.")
        return "\n\n".join(parts)

    @staticmethod
    def rebuild(sample: Sample, question: Optional[str] = None, context: Optional[str] = None) -> str:
        tmp = copy.copy(sample)
        tmp.question = sample.question if question is None else question
        tmp.context = sample.context if context is None else context
        # raw_prompt cannot be safely edited; rebuild from fields.
        tmp.raw_prompt = ""
        return PromptBuilder.base_user_prompt(tmp)


class ProposalGenerator:
    def __init__(self, engine: LocalCausalLM, seed: int) -> None:
        self.engine = engine
        self.seed = seed

    def generate_oracle_facts(self, sample: Sample) -> Tuple[str, Dict[str, Any]]:
        if sample.oracle_facts:
            return sample.oracle_facts, {"source": "oracle_facts_field", "reference_guided": False}

        reference_text = "\n".join(f"- {x}" for x in sample.references)
        prompt = f"""
Construct a minimal set of verified facts that would help solve the question.

Question:
{sample.question}

Original context, if any:
{sample.context or '[none]'}

Reference answer used only for supervision:
{reference_text}

Requirements:
- Return 1 to 4 short factual bullet points.
- State supporting facts, not a multiple-choice label.
- Do not say "the answer is ...".
- Do not include explanations or JSON.
""".strip()
        result = self.engine.generate(
            prompt,
            max_new_tokens=160,
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + 101,
            system_prompt="You extract minimal verified evidence. Follow the output constraints exactly.",
        )
        facts = clean_model_text(result.text)
        if not facts:
            facts = reference_text
        return facts, {
            "source": "reference_guided_generation",
            "reference_guided": True,
            "generator_output": result.text,
        }

    def generate_knowledge_probes(
        self,
        sample: Sample,
        max_probes: int,
    ) -> List[Dict[str, str]]:
        reference_text = "\n".join(f"- {x}" for x in sample.references)
        prompt = f"""
Create up to {max_probes} independent factual probes for the minimum knowledge needed to answer
the original question. The probes must test prerequisite facts, not repeat the original composite
question. Each answer should be short and objectively checkable.

ORIGINAL QUESTION:
{sample.question}

ORIGINAL CONTEXT, if any:
{sample.context or '[none]'}

REFERENCE ANSWER(S), used only for supervision:
{reference_text}

Return a JSON array only:
[{{"question": "standalone factual question", "answer": "short expected answer"}}]
Do not include explanations.
""".strip()
        result = self.engine.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + 151,
            system_prompt="You create atomic prerequisite knowledge probes. Return valid JSON only.",
        )
        parsed = parse_first_json(result.text)
        if not isinstance(parsed, list):
            return []
        probes: List[Dict[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue
            question = as_text(item.get("question"))
            answer = as_text(item.get("answer"))
            key = (question, answer)
            if not question or not answer or key in seen:
                continue
            seen.add(key)
            probes.append({
                "question": question,
                "answer": answer,
                "proposal_output": result.text,
            })
            if len(probes) >= max_probes:
                break
        return probes

    def propose_shortcut_spans(
        self,
        sample: Sample,
        base_answer: str,
        max_spans: int,
    ) -> List[Dict[str, Any]]:
        source = "\n".join(x for x in [sample.context, sample.question] if x)
        if not source.strip():
            return []
        refs = "\n".join(f"- {x}" for x in sample.references)
        prompt = f"""
Identify up to {max_spans} exact contiguous substrings in the ORIGINAL SOURCE that may have
biased the model toward its incorrect answer through a prior, salient association, famous entity,
lexical shortcut, or distractor. Do not select text that is itself the decisive logical constraint.

ORIGINAL SOURCE:
{source}

MODEL ANSWER:
{base_answer}

REFERENCE ANSWER(S):
{refs}

Return a JSON array only. Each item must be:
{{"span": "exact substring copied from ORIGINAL SOURCE", "reason": "brief explanation"}}
If none are plausible, return [].
""".strip()
        result = self.engine.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + 211,
            system_prompt="You identify candidate causal shortcut spans. Return valid JSON only.",
        )
        parsed = parse_first_json(result.text)
        if not isinstance(parsed, list):
            return []

        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue
            span = as_text(item.get("span"))
            if not span or span in seen or span not in source:
                continue
            seen.add(span)
            candidates.append({
                "span": span,
                "reason": as_text(item.get("reason")),
                "proposal_output": result.text,
            })
            if len(candidates) >= max_spans:
                break
        return candidates

    def rewrite_counterfactual(self, span: str) -> str:
        prompt = f"""
Rewrite the following text into a minimally changed counterfactual or negated version.
Preserve names and writing style where possible. Return only the rewritten text.

TEXT:
{span}
""".strip()
        result = self.engine.generate(
            prompt,
            max_new_tokens=96,
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + 307,
            system_prompt="Return only a concise counterfactual rewrite.",
        )
        rewritten = clean_model_text(result.text).strip('"')
        return rewritten if rewritten and rewritten != span else f"It is not the case that {span}"

    def neutralize_span(self, span: str) -> str:
        # Lightweight deterministic neutralization for names / salient phrases.
        if len(span.split()) <= 5 and re.search(r"[A-Z]", span):
            return "the relevant entity"
        return "[neutralized detail]"

    def compress_context(self, sample: Sample) -> Tuple[str, Dict[str, Any]]:
        if not sample.context:
            return "", {"error": "no_context"}
        refs = "\n".join(f"- {x}" for x in sample.references)
        prompt = f"""
Compress the context for answering the question.

QUESTION:
{sample.question}

CONTEXT:
{sample.context}

REFERENCE ANSWER(S), used only to ensure no necessary evidence is deleted:
{refs}

Requirements:
- Keep every fact and constraint necessary to determine the answer.
- Remove background, repetition, and distractors.
- Do not state the final answer or add new facts.
- Return only the compressed context.
""".strip()
        result = self.engine.generate(
            prompt,
            max_new_tokens=min(384, max(96, len(sample.context.split()))),
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + 401,
            system_prompt="You perform lossless task-focused context compression.",
        )
        return clean_model_text(result.text), {
            "reference_guided": True,
            "generator_output": result.text,
        }

    def structure_context(self, sample: Sample) -> Tuple[str, Dict[str, Any]]:
        if not sample.context:
            return "", {"error": "no_context"}
        refs = "\n".join(f"- {x}" for x in sample.references)
        prompt = f"""
Reorganize the context into a clear structured representation for the question.

QUESTION:
{sample.question}

CONTEXT:
{sample.context}

REFERENCE ANSWER(S), used only to preserve necessary evidence:
{refs}

Requirements:
- Group facts by entity or object.
- List explicit constraints separately.
- Preserve all answer-relevant facts.
- Remove only clearly irrelevant material.
- Do not reveal the final answer.
- Return only the reorganized context.
""".strip()
        result = self.engine.generate(
            prompt,
            max_new_tokens=min(448, max(128, len(sample.context.split()) + 64)),
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + 409,
            system_prompt="You reorganize evidence without solving the problem.",
        )
        return clean_model_text(result.text), {
            "reference_guided": True,
            "generator_output": result.text,
        }

    def format_only_control(self, sample: Sample) -> Tuple[str, Dict[str, Any]]:
        if not sample.context:
            return "", {"error": "no_context"}
        prompt = f"""
Paraphrase the context while preserving all information, ordering, approximate length, and emphasis.
Do not simplify the task, remove distractors, group entities, or add facts.
Return only the paraphrased context.

CONTEXT:
{sample.context}
""".strip()
        result = self.engine.generate(
            prompt,
            max_new_tokens=min(448, max(128, len(sample.context.split()) + 64)),
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + 419,
            system_prompt="You make a surface paraphrase only; do not simplify content.",
        )
        return clean_model_text(result.text), {
            "reference_guided": False,
            "generator_output": result.text,
        }


# ---------------------------------------------------------------------------
# Intervention construction
# ---------------------------------------------------------------------------


def replace_exact_once(text: str, span: str, replacement: str) -> Tuple[str, bool]:
    if span not in text:
        return text, False
    return text.replace(span, replacement, 1), True


def edit_sample_span(sample: Sample, span: str, replacement: str) -> Tuple[str, Dict[str, Any]]:
    if span in sample.context:
        context, changed = replace_exact_once(sample.context, span, replacement)
        return PromptBuilder.rebuild(sample, context=context), {
            "edited_field": "context",
            "changed": changed,
        }
    if span in sample.question:
        question, changed = replace_exact_once(sample.question, span, replacement)
        return PromptBuilder.rebuild(sample, question=question), {
            "edited_field": "question",
            "changed": changed,
        }
    return PromptBuilder.base_user_prompt(sample), {"edited_field": None, "changed": False}


def random_matched_span(sample: Sample, target_span: str, rng: random.Random) -> Optional[str]:
    candidates_source = sample.context or sample.question
    words = list(re.finditer(r"\S+", candidates_source))
    target_n = max(1, len(target_span.split()))
    if len(words) < target_n:
        return None
    starts = list(range(0, len(words) - target_n + 1))
    rng.shuffle(starts)
    for start in starts:
        end = start + target_n - 1
        candidate = candidates_source[words[start].start(): words[end].end()]
        if candidate != target_span and target_span not in candidate and candidate not in target_span:
            return candidate
    return None


class InterventionFactory:
    def __init__(
        self,
        engine: LocalCausalLM,
        proposal: ProposalGenerator,
        rng: random.Random,
        base_max_new_tokens: int,
        high_budget_tokens: int,
        low_budget_tokens: int,
        temperature: float,
        max_shortcut_spans: int,
    ) -> None:
        self.engine = engine
        self.proposal = proposal
        self.rng = rng
        self.base_max_new_tokens = base_max_new_tokens
        self.high_budget_tokens = high_budget_tokens
        self.low_budget_tokens = low_budget_tokens
        self.temperature = temperature
        self.max_shortcut_spans = max_shortcut_spans

    def s1_variants(
        self,
        sample: Sample,
        unrelated_facts: str,
    ) -> List[Variant]:
        oracle_facts, fact_meta = self.proposal.generate_oracle_facts(sample)
        base = PromptBuilder.base_user_prompt(sample)
        targeted_prompt = f"""
{base}

Additional verified evidence:
{oracle_facts}

Use the verified evidence when answering. Do not simply repeat it; solve the original question.
""".strip()
        variants = [
            Variant(
                variant_id="S1_oracle_fact",
                stressor="S1",
                kind="oracle_fact_injection",
                user_prompt=targeted_prompt,
                is_control=False,
                metadata={"oracle_facts": oracle_facts, **fact_meta},
            )
        ]
        if unrelated_facts:
            control_prompt = f"""
{base}

Additional verified but unrelated information:
{unrelated_facts}

Answer the original question accurately.
""".strip()
            variants.append(
                Variant(
                    variant_id="S1_unrelated_fact_control",
                    stressor="S1",
                    kind="unrelated_fact_control",
                    user_prompt=control_prompt,
                    is_control=True,
                    metadata={"unrelated_facts": unrelated_facts},
                )
            )
        return variants

    def s2_variants(self, sample: Sample, base_answer: str) -> List[Variant]:
        proposals = self.proposal.propose_shortcut_spans(
            sample,
            base_answer=base_answer,
            max_spans=self.max_shortcut_spans,
        )
        variants: List[Variant] = []
        for index, item in enumerate(proposals):
            span = item["span"]
            deletion_prompt, edit_meta = edit_sample_span(sample, span, "")
            variants.append(
                Variant(
                    variant_id=f"S2_delete_{index}",
                    stressor="S2",
                    kind="shortcut_deletion",
                    user_prompt=deletion_prompt,
                    is_control=False,
                    metadata={"span": span, **item, **edit_meta},
                )
            )

            neutral = self.proposal.neutralize_span(span)
            neutral_prompt, edit_meta = edit_sample_span(sample, span, neutral)
            variants.append(
                Variant(
                    variant_id=f"S2_neutralize_{index}",
                    stressor="S2",
                    kind="shortcut_neutralization",
                    user_prompt=neutral_prompt,
                    is_control=False,
                    metadata={"span": span, "replacement": neutral, **item, **edit_meta},
                )
            )

            counterfactual = self.proposal.rewrite_counterfactual(span)
            cf_prompt, edit_meta = edit_sample_span(sample, span, counterfactual)
            variants.append(
                Variant(
                    variant_id=f"S2_counterfactual_{index}",
                    stressor="S2",
                    kind="shortcut_counterfactual",
                    user_prompt=cf_prompt,
                    is_control=False,
                    metadata={"span": span, "replacement": counterfactual, **item, **edit_meta},
                )
            )

            random_span = random_matched_span(sample, span, self.rng)
            if random_span:
                control_prompt, edit_meta = edit_sample_span(sample, random_span, "")
                variants.append(
                    Variant(
                        variant_id=f"S2_random_delete_control_{index}",
                        stressor="S2",
                        kind="matched_random_deletion",
                        user_prompt=control_prompt,
                        is_control=True,
                        metadata={
                            "target_span": span,
                            "control_span": random_span,
                            **edit_meta,
                        },
                    )
                )
        return variants

    def s3_variants(self, sample: Sample) -> List[Variant]:
        if not sample.context:
            return []
        compressed, compress_meta = self.proposal.compress_context(sample)
        structured, structure_meta = self.proposal.structure_context(sample)
        format_control, control_meta = self.proposal.format_only_control(sample)
        variants: List[Variant] = []
        if compressed:
            variants.append(
                Variant(
                    variant_id="S3_compress",
                    stressor="S3",
                    kind="gold_preserving_compression",
                    user_prompt=PromptBuilder.rebuild(sample, context=compressed),
                    is_control=False,
                    metadata={
                        "new_context": compressed,
                        "original_words": len(sample.context.split()),
                        "new_words": len(compressed.split()),
                        **compress_meta,
                    },
                )
            )
        if structured:
            variants.append(
                Variant(
                    variant_id="S3_structure",
                    stressor="S3",
                    kind="entity_constraint_reorganization",
                    user_prompt=PromptBuilder.rebuild(sample, context=structured),
                    is_control=False,
                    metadata={
                        "new_context": structured,
                        "original_words": len(sample.context.split()),
                        "new_words": len(structured.split()),
                        **structure_meta,
                    },
                )
            )
        if format_control:
            variants.append(
                Variant(
                    variant_id="S3_format_control",
                    stressor="S3",
                    kind="surface_paraphrase_control",
                    user_prompt=PromptBuilder.rebuild(sample, context=format_control),
                    is_control=True,
                    metadata={
                        "new_context": format_control,
                        "original_words": len(sample.context.split()),
                        "new_words": len(format_control.split()),
                        **control_meta,
                    },
                )
            )
        return variants

    def s4_variants(self, sample: Sample, base_answer: str) -> List[Variant]:
        base = PromptBuilder.base_user_prompt(sample)
        high_prompt = f"""
{base}

Reason through every necessary step before giving the final answer. Check the result once.
End with a line beginning exactly with "FINAL:".
""".strip()
        verify_prompt = f"""
Original task:
{base}

Initial draft answer:
{base_answer}

Independently verify the draft using only the original task information. Correct any logical,
numerical, or factual execution error. Return the final answer clearly.
""".strip()
        verbosity_control = f"""
{base}

Give a detailed and well-written response. Expand the explanation, but do not perform a separate
verification pass or introduce outside information.
""".strip()
        return [
            Variant(
                variant_id="S4_low_budget",
                stressor="S4",
                kind="low_reasoning_budget",
                user_prompt=base,
                is_control=True,
                max_new_tokens=self.low_budget_tokens,
                temperature=self.temperature,
            ),
            Variant(
                variant_id="S4_high_budget",
                stressor="S4",
                kind="high_reasoning_budget",
                user_prompt=high_prompt,
                is_control=False,
                max_new_tokens=self.high_budget_tokens,
                temperature=self.temperature,
            ),
            Variant(
                variant_id="S4_verification",
                stressor="S4",
                kind="dedicated_verification_pass",
                user_prompt=verify_prompt,
                is_control=False,
                max_new_tokens=self.high_budget_tokens,
                temperature=0.0,
                metadata={"initial_draft": base_answer},
            ),
            Variant(
                variant_id="S4_verbosity_control",
                stressor="S4",
                kind="long_output_without_verification_control",
                user_prompt=verbosity_control,
                is_control=True,
                max_new_tokens=self.high_budget_tokens,
                temperature=self.temperature,
            ),
        ]

    def s6_gate_decision(self, sample: Sample, seed: int) -> Tuple[Dict[str, Any], str]:
        base = PromptBuilder.base_user_prompt(sample)
        gate_prompt = f"""
Choose the epistemically appropriate action before answering the task.

TASK:
{base}

Output exactly one token:
ANSWER  - available information is sufficient
VERIFY  - a draft should be checked before finalizing
ABSTAIN - available information is insufficient
""".strip()
        result = self.engine.generate(
            gate_prompt,
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            system_prompt="Return exactly ANSWER, VERIFY, or ABSTAIN.",
        )
        match = re.search(r"\b(ANSWER|VERIFY|ABSTAIN)\b", result.text.upper())
        action = match.group(1) if match else "UNRESOLVED"
        parsed = {
            "action": action,
            "parse_failed": match is None,
        }
        return parsed, result.text

    def s6_variants(
        self,
        sample: Sample,
        base_answer: str,
        seed: int,
    ) -> Tuple[List[Variant], Optional[Dict[str, Any]]]:
        decision, raw_decision = self.s6_gate_decision(sample, seed)
        action = decision["action"]
        base = PromptBuilder.base_user_prompt(sample)
        metadata = {
            "gate_decision": decision,
            "gate_raw_output": raw_decision,
        }

        if action == "ANSWER":
            gate_prompt = f"""
{base}

Epistemic decision: ANSWER. The available information is judged sufficient.
Give the answer directly and avoid unsupported additions.
""".strip()
        elif action == "VERIFY":
            gate_prompt = f"""
Original task:
{base}

Candidate answer:
{base_answer}

Epistemic decision: VERIFY. Check whether the candidate follows from the available information.
Correct it if necessary, then return the final answer clearly. Do not add outside information.
""".strip()
            metadata["initial_draft"] = base_answer
        elif action == "ABSTAIN":
            # The variant is represented as a fixed abstention, not another model call.
            metadata["fixed_answer"] = "I do not have enough information to answer reliably."
            gate_prompt = base
        else:
            # Do not turn a parse failure into an artificial abstention. The
            # prerequisite will fail and S6 will be marked unavailable/negative.
            metadata["gate_unresolved"] = True
            gate_prompt = base

        generic_control = f"""
{base}

Think carefully before answering.
""".strip()

        variants = [
            Variant(
                variant_id="S6_epistemic_gate",
                stressor="S6",
                kind=f"epistemic_gate_{action.lower()}",
                user_prompt=gate_prompt if action != "ABSTAIN" else base,
                is_control=False,
                max_new_tokens=self.high_budget_tokens if action == "VERIFY" else self.base_max_new_tokens,
                temperature=0.0,
                metadata=metadata,
            ),
            Variant(
                variant_id="S6_generic_careful_control",
                stressor="S6",
                kind="generic_careful_control",
                user_prompt=generic_control,
                is_control=True,
                max_new_tokens=self.base_max_new_tokens,
                temperature=0.0,
            ),
        ]
        return variants, decision


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    def __init__(
        self,
        engine: LocalCausalLM,
        evaluator: Evaluator,
        intervention_factory: InterventionFactory,
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        base_seed: int,
        score_reference: bool,
        strong_positive_recovery: float,
        specificity_threshold: float,
        negative_recovery_threshold: float,
        max_knowledge_probes: int,
        knowledge_known_threshold: float,
        knowledge_scarce_threshold: float,
        output_dir: Path,
        save_hidden_states: bool,
        hidden_layers: str,
        save_probe_hidden_states: bool,
        abstention_policy: str,
    ) -> None:
        self.engine = engine
        self.evaluator = evaluator
        self.factory = intervention_factory
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_seed = base_seed
        self.score_reference = score_reference
        self.strong_positive_recovery = strong_positive_recovery
        self.specificity_threshold = specificity_threshold
        self.negative_recovery_threshold = negative_recovery_threshold
        self.max_knowledge_probes = max_knowledge_probes
        self.knowledge_known_threshold = knowledge_known_threshold
        self.knowledge_scarce_threshold = knowledge_scarce_threshold
        self.output_dir = output_dir
        self.hidden_dir = output_dir / "hidden_states"
        self.save_hidden_states = save_hidden_states
        self.hidden_layers = hidden_layers
        self.save_probe_hidden_states = save_probe_hidden_states
        self.abstention_policy = abstention_policy

    @staticmethod
    def _safe_component(text: str, max_len: int = 80) -> str:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
        return (clean or "trace")[:max_len]

    def hidden_path(self, sample_id: str, trace_id: str, run_index: int) -> Path:
        sample_dir = self.hidden_dir / self._safe_component(sample_id)
        filename = f"{self._safe_component(trace_id)}__run_{run_index:02d}.pt"
        return sample_dir / filename

    @staticmethod
    def determine_base_status(base: Mapping[str, Any]) -> str:
        valid = int(base.get("n_valid_evaluations", 0))
        counts = base.get("evaluation_status_counts", {})
        if valid > 0:
            rate = base.get("correct_rate")
            return "correct" if rate is not None and float(rate) >= 0.5 else "hallucination"
        if counts.get("invalid_reference", 0) > 0:
            return "invalid_reference"
        if counts.get("unanswerable", 0) > 0:
            return "unanswerable"
        return "unresolved"

    def run_prompt(
        self,
        sample: Sample,
        user_prompt: str,
        seed_offset: int,
        trace_id: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        fixed_answer: Optional[str] = None,
        save_hidden_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        runs: List[EvaluatedRun] = []
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = self.temperature if temperature is None else temperature
        should_save_hidden = self.save_hidden_states if save_hidden_override is None else save_hidden_override

        for index in range(self.n_samples):
            seed = self.base_seed + seed_offset * 1009 + index * 17
            if fixed_answer is not None:
                generation = GenerationResult(
                    text=fixed_answer,
                    token_count=0,
                    mean_token_logprob=None,
                    finish_reason="fixed",
                    seed=seed,
                    elapsed_seconds=0.0,
                )
            else:
                generation = self.engine.generate(
                    user_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=self.top_p,
                    seed=seed,
                )

            evaluation = self.evaluator.evaluate(
                sample.question,
                sample.context,
                generation.text,
                sample.references,
                seed=seed,
            )
            reference_lp = None
            if self.score_reference and sample.references:
                reference_lp = self.engine.score_continuation(user_prompt, sample.references[0])

            hidden_state_path: Optional[str] = None
            hidden_state_error: Optional[str] = None
            if should_save_hidden:
                try:
                    path = self.hidden_path(sample.id, trace_id, index)
                    info = self.engine.save_hidden_state_trace(
                        user_prompt=user_prompt,
                        answer=generation.text,
                        output_path=path,
                        layer_spec=self.hidden_layers,
                    )
                    hidden_state_path = info["path"]
                except Exception as exc:  # noqa: BLE001
                    hidden_state_error = repr(exc)
                    LOGGER.exception("Hidden-state extraction failed for %s/%s", sample.id, trace_id)

            runs.append(
                EvaluatedRun(
                    text=generation.text,
                    correct=evaluation.correct,
                    evaluation_status=evaluation.status,
                    is_abstention=is_abstention_text(generation.text),
                    evaluation_score=evaluation.score,
                    token_count=generation.token_count,
                    mean_token_logprob=generation.mean_token_logprob,
                    reference_logprob=reference_lp,
                    seed=seed,
                    elapsed_seconds=generation.elapsed_seconds,
                    hidden_state_path=hidden_state_path,
                    hidden_state_error=hidden_state_error,
                    evaluation_details=evaluation.details,
                )
            )

        valid_runs = [x for x in runs if x.correct is not None]
        correct_rate = (
            statistics.mean(float(x.correct) for x in valid_runs)
            if valid_runs else None
        )
        abstention_rate = statistics.mean(float(x.is_abstention) for x in runs) if runs else None
        safe_values = [float(bool(x.correct) or x.is_abstention) for x in valid_runs]
        safe_rate = statistics.mean(safe_values) if safe_values else None
        eval_scores = [x.evaluation_score for x in runs if x.evaluation_score is not None]
        eval_score_mean = statistics.mean(eval_scores) if eval_scores else None
        answer_counts = Counter(normalize_text(x.text) for x in runs)
        consistency = max(answer_counts.values()) / len(runs) if runs else 0.0
        status_counts = Counter(x.evaluation_status for x in runs)
        return {
            "prompt": user_prompt,
            "runs": [dataclasses.asdict(x) for x in runs],
            "correct_rate": correct_rate,
            "n_valid_evaluations": len(valid_runs),
            "evaluation_status_counts": dict(status_counts),
            "abstention_rate": abstention_rate,
            "safe_rate": safe_rate,
            "evaluation_score_mean": eval_score_mean,
            "consistency": consistency,
            "mean_token_count": statistics.mean(x.token_count for x in runs) if runs else None,
            "mean_token_logprob": mean_or_none([x.mean_token_logprob for x in runs]),
            "mean_reference_logprob": mean_or_none([x.reference_logprob for x in runs]),
            "representative_answer": runs[0].text if runs else "",
        }

    def run_knowledge_probes(self, sample: Sample) -> Dict[str, Any]:
        probes = self.factory.proposal.generate_knowledge_probes(
            sample,
            max_probes=self.max_knowledge_probes,
        )
        details: List[Dict[str, Any]] = []
        for index, probe in enumerate(probes):
            probe_sample = Sample(
                id=f"{sample.id}__knowledge_probe_{index}",
                question=probe["question"],
                context="",
                references=[probe["answer"]],
            )
            prompt = f"Question:\n{probe['question']}\n\nGive a concise factual answer."
            result = self.run_prompt(
                probe_sample,
                prompt,
                seed_offset=40 + index,
                trace_id=f"knowledge_probe_{index}",
                max_new_tokens=min(64, self.max_new_tokens),
                temperature=0.0,
                save_hidden_override=self.save_hidden_states and self.save_probe_hidden_states,
            )
            details.append({**probe, "result": result})
        valid_scores = [
            float(x["result"]["correct_rate"])
            for x in details
            if x["result"].get("correct_rate") is not None
        ]
        score = statistics.mean(valid_scores) if valid_scores else None
        return {
            "score": score,
            "n_probes": len(details),
            "n_valid_probes": len(valid_scores),
            "probes": details,
        }

    def run_variant(
        self,
        sample: Sample,
        variant: Variant,
        variant_index: int,
    ) -> Dict[str, Any]:
        fixed_answer = variant.metadata.get("fixed_answer")
        aggregate = self.run_prompt(
            sample,
            variant.user_prompt,
            seed_offset=100 + variant_index,
            trace_id=variant.variant_id,
            max_new_tokens=variant.max_new_tokens,
            temperature=variant.temperature,
            fixed_answer=fixed_answer,
        )
        return {
            "variant_id": variant.variant_id,
            "stressor": variant.stressor,
            "kind": variant.kind,
            "is_control": variant.is_control,
            "metadata": variant.metadata,
            **aggregate,
        }

    def score_stressors(
        self,
        base: Dict[str, Any],
        variants: Sequence[Dict[str, Any]],
        sample: Sample,
        knowledge_probe_score: Optional[float],
        gate_decision: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base_rate = base.get("correct_rate")
        if base_rate is None:
            return {
                "per_stressor": {},
                "primary_stressor": None,
                "secondary_stressors": [],
                "unidentified": True,
                "note": "base_evaluation_not_valid",
            }
        base_rate = float(base_rate)
        by_stressor: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for variant in variants:
            by_stressor[variant["stressor"]].append(variant)

        results: Dict[str, Any] = {}
        for stressor in STRESSOR_NAMES:
            group = by_stressor.get(stressor, [])
            targeted = [x for x in group if not x["is_control"] and x.get("correct_rate") is not None]
            controls = [x for x in group if x["is_control"] and x.get("correct_rate") is not None]

            rate_key = "safe_rate" if (stressor == "S6" and self.abstention_policy == "safe") else "correct_rate"
            base_effective = base.get(rate_key)
            if base_effective is None:
                base_effective = base_rate
            base_effective = float(base_effective)
            target_recoveries = [
                float(x.get(rate_key) if x.get(rate_key) is not None else x["correct_rate"]) - base_effective
                for x in targeted
            ]
            control_recoveries = [
                float(x.get(rate_key) if x.get(rate_key) is not None else x["correct_rate"]) - base_effective
                for x in controls
            ]
            max_target = max(target_recoveries) if target_recoveries else None
            max_control = max(control_recoveries) if control_recoveries else 0.0
            specificity = (max_target - max_control) if max_target is not None else None

            prerequisite = True
            prerequisite_notes: List[str] = []
            if stressor == "S1":
                if knowledge_probe_score is None:
                    prerequisite = False
                    prerequisite_notes.append("knowledge_probes_unavailable")
                elif knowledge_probe_score > self.knowledge_scarce_threshold:
                    prerequisite = False
                    prerequisite_notes.append("necessary_knowledge_is_accessible")
            if stressor == "S2":
                if not targeted:
                    prerequisite = False
                    prerequisite_notes.append("no_valid_shortcut_candidate_or_evaluation")
                if knowledge_probe_score is None:
                    prerequisite = False
                    prerequisite_notes.append("knowledge_probes_unavailable")
                elif knowledge_probe_score < self.knowledge_known_threshold:
                    prerequisite = False
                    prerequisite_notes.append("necessary_knowledge_not_stably_accessible")
            if stressor == "S3" and not sample.context:
                prerequisite = False
                prerequisite_notes.append("no_context")
            if stressor == "S6":
                action = as_text((gate_decision or {}).get("action")).upper()
                if action not in {"VERIFY", "ABSTAIN"}:
                    prerequisite = False
                    prerequisite_notes.append("gate_did_not_detect_need_for_control_action")
                if action == "ABSTAIN" and self.abstention_policy == "incorrect":
                    prerequisite_notes.append("abstention_not_rewarded_on_answerable_qa")

            if max_target is None:
                label = "unavailable"
                soft_label = 0.0
            else:
                positive = (
                    prerequisite
                    and max_target >= self.strong_positive_recovery
                    and specificity is not None
                    and specificity >= self.specificity_threshold
                )
                negative = max_target <= self.negative_recovery_threshold
                if positive:
                    label = "positive"
                elif negative:
                    label = "negative"
                else:
                    label = "uncertain"
                soft_label = clip(
                    ((specificity or 0.0) - self.negative_recovery_threshold)
                    / max(self.specificity_threshold + self.strong_positive_recovery, 1e-6)
                )
                if not prerequisite:
                    soft_label = 0.0

            best_target = None
            if targeted:
                best_target = max(
                    targeted,
                    key=lambda x: float(x.get(rate_key) if x.get(rate_key) is not None else x["correct_rate"]) - base_effective,
                )["variant_id"]
            best_control = None
            if controls:
                best_control = max(
                    controls,
                    key=lambda x: float(x.get(rate_key) if x.get(rate_key) is not None else x["correct_rate"]) - base_effective,
                )["variant_id"]

            results[stressor] = {
                "name": STRESSOR_NAMES[stressor],
                "prerequisite_met": prerequisite,
                "prerequisite_notes": prerequisite_notes,
                "recovery_metric": rate_key,
                "max_target_recovery": max_target,
                "max_control_recovery": max_control if controls else None,
                "specificity": specificity,
                "soft_label": soft_label,
                "label": label,
                "best_target_variant": best_target,
                "best_control_variant": best_control,
                "n_targeted": len(targeted),
                "n_controls": len(controls),
            }

        ranked = sorted(
            results.items(),
            key=lambda kv: (kv[1]["soft_label"], kv[1].get("specificity") or -999.0),
            reverse=True,
        )
        primary = ranked[0][0] if ranked and ranked[0][1]["label"] == "positive" else None
        secondary = [
            stressor for stressor, info in ranked[1:]
            if info["label"] == "positive"
        ][:2]
        return {
            "per_stressor": results,
            "primary_stressor": primary,
            "secondary_stressors": secondary,
            "unidentified": primary is None,
        }

    def process_sample(
        self,
        sample: Sample,
        unrelated_facts: str,
        enabled_stressors: Sequence[str],
        precomputed_base: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base_prompt = PromptBuilder.base_user_prompt(sample)
        base = precomputed_base or self.run_prompt(
            sample,
            base_prompt,
            seed_offset=0,
            trace_id="base",
        )
        base_status = self.determine_base_status(base)
        if base_status != "hallucination":
            return {
                "id": sample.id,
                "question": sample.question,
                "context": sample.context,
                "references": sample.references,
                "symptom": sample.symptom,
                "metadata": sample.metadata,
                "base": base,
                "base_status": base_status,
                "base_is_hallucination": False if base_status == "correct" else None,
                "interventions_run": False,
                "variants": [],
                "diagnosis": {
                    "per_stressor": {},
                    "primary_stressor": None,
                    "secondary_stressors": [],
                    "unidentified": True,
                    "note": f"interventions_not_run_for_base_status_{base_status}",
                },
            }

        base_answer = base["representative_answer"]
        knowledge_probes = self.run_knowledge_probes(sample) if ("S1" in enabled_stressors or "S2" in enabled_stressors) else {
            "score": None, "n_probes": 0, "n_valid_probes": 0, "probes": []
        }

        variants: List[Variant] = []
        if "S1" in enabled_stressors:
            variants.extend(self.factory.s1_variants(sample, unrelated_facts))
        if "S2" in enabled_stressors:
            variants.extend(self.factory.s2_variants(sample, base_answer))
        if "S3" in enabled_stressors:
            variants.extend(self.factory.s3_variants(sample))
        if "S4" in enabled_stressors:
            variants.extend(self.factory.s4_variants(sample, base_answer))
        gate_decision: Optional[Dict[str, Any]] = None
        if "S6" in enabled_stressors:
            s6_variants, gate_decision = self.factory.s6_variants(
                sample,
                base_answer,
                seed=self.base_seed + 601,
            )
            variants.extend(s6_variants)

        variant_results: List[Dict[str, Any]] = []
        for index, variant in enumerate(variants):
            LOGGER.debug("Running %s / %s", sample.id, variant.variant_id)
            try:
                variant_results.append(self.run_variant(sample, variant, index))
            except torch.cuda.OutOfMemoryError:
                LOGGER.exception("CUDA OOM in variant %s; clearing cache", variant.variant_id)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                variant_results.append({
                    "variant_id": variant.variant_id,
                    "stressor": variant.stressor,
                    "kind": variant.kind,
                    "is_control": variant.is_control,
                    "metadata": variant.metadata,
                    "error": "cuda_out_of_memory",
                    "correct_rate": None,
                })
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Variant failed: %s", variant.variant_id)
                variant_results.append({
                    "variant_id": variant.variant_id,
                    "stressor": variant.stressor,
                    "kind": variant.kind,
                    "is_control": variant.is_control,
                    "metadata": variant.metadata,
                    "error": repr(exc),
                    "correct_rate": None,
                })

        diagnosis = self.score_stressors(
            base,
            variant_results,
            sample,
            knowledge_probe_score=knowledge_probes.get("score"),
            gate_decision=gate_decision,
        )
        return {
            "id": sample.id,
            "question": sample.question,
            "context": sample.context,
            "references": sample.references,
            "symptom": sample.symptom,
            "metadata": sample.metadata,
            "base": base,
            "base_status": base_status,
            "base_is_hallucination": True,
            "interventions_run": True,
            "knowledge_probes": knowledge_probes,
            "epistemic_gate_decision": gate_decision,
            "variants": variant_results,
            "diagnosis": diagnosis,
        }


# ---------------------------------------------------------------------------
# Dataset conversion and summary
# ---------------------------------------------------------------------------


def convert_rows_to_samples(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Sample]:
    samples: List[Sample] = []
    for index, row in enumerate(rows):
        question = as_text(get_nested(row, args.question_field, ""))
        context = as_text(get_nested(row, args.context_field, "")) if args.context_field else ""
        raw_prompt = as_text(get_nested(row, args.prompt_field, "")) if args.prompt_field else ""
        refs = answer_list(get_nested(row, args.answer_field, None))
        oracle_facts = as_text(get_nested(row, args.oracle_facts_field, "")) if args.oracle_facts_field else ""
        symptom = as_text(get_nested(row, args.symptom_field, "")) if args.symptom_field else ""

        raw_id = get_nested(row, args.id_field, None) if args.id_field else None
        if raw_id is None:
            raw_id = row.get("_dict_key") if isinstance(row, Mapping) else None
        if raw_id is None:
            raw_id = f"sample_{index:06d}_{stable_hash(question + context)}"
        sample_id = str(raw_id)

        if not question and not raw_prompt:
            LOGGER.warning("Skipping sample %s: missing question/prompt", sample_id)
            continue
        if not refs:
            LOGGER.warning("Skipping sample %s: missing reference answer", sample_id)
            continue

        metadata = {
            "source_index": index,
        }
        for key in args.metadata_fields:
            value = get_nested(row, key, None)
            if value is not None:
                metadata[key] = value

        samples.append(
            Sample(
                id=sample_id,
                question=question or raw_prompt,
                context=context,
                references=refs,
                raw_prompt=raw_prompt,
                oracle_facts=oracle_facts,
                symptom=symptom,
                metadata=metadata,
            )
        )
    return samples


def summarize(results_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(results_path)
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if not rows:
        return {"n_samples": 0}

    base_status_counts: Counter[str] = Counter()
    evaluable_rates: List[float] = []
    label_counts: Dict[str, Counter[str]] = {s: Counter() for s in STRESSOR_NAMES}
    recoveries: Dict[str, List[float]] = defaultdict(list)
    specificities: Dict[str, List[float]] = defaultdict(list)
    primary_counts: Counter[str] = Counter()
    n_intervention_samples = 0
    hidden_files = 0
    hidden_errors = 0

    for row in rows:
        if "base" not in row:
            base_status_counts["sample_error"] += 1
            continue
        status = row.get("base_status")
        if not status:
            status = ExperimentRunner.determine_base_status(row["base"])
        base_status_counts[status] += 1
        if status in {"correct", "hallucination"} and row["base"].get("correct_rate") is not None:
            evaluable_rates.append(float(row["base"]["correct_rate"]))

        for run in row.get("base", {}).get("runs", []):
            hidden_files += int(bool(run.get("hidden_state_path")))
            hidden_errors += int(bool(run.get("hidden_state_error")))
        for variant in row.get("variants", []):
            for run in variant.get("runs", []):
                hidden_files += int(bool(run.get("hidden_state_path")))
                hidden_errors += int(bool(run.get("hidden_state_error")))
        for probe in row.get("knowledge_probes", {}).get("probes", []):
            for run in probe.get("result", {}).get("runs", []):
                hidden_files += int(bool(run.get("hidden_state_path")))
                hidden_errors += int(bool(run.get("hidden_state_error")))

        if not row.get("interventions_run"):
            continue
        n_intervention_samples += 1
        diag = row.get("diagnosis", {}).get("per_stressor", {})
        primary = row.get("diagnosis", {}).get("primary_stressor")
        primary_counts[primary or "unidentified"] += 1
        for stressor in STRESSOR_NAMES:
            info = diag.get(stressor)
            if not info:
                continue
            label_counts[stressor][info.get("label", "missing")] += 1
            if info.get("max_target_recovery") is not None:
                recoveries[stressor].append(float(info["max_target_recovery"]))
            if info.get("specificity") is not None:
                specificities[stressor].append(float(info["specificity"]))

    per_stressor = {}
    for stressor, name in STRESSOR_NAMES.items():
        per_stressor[stressor] = {
            "name": name,
            "label_counts": dict(label_counts[stressor]),
            "mean_max_target_recovery": statistics.mean(recoveries[stressor]) if recoveries[stressor] else None,
            "mean_specificity": statistics.mean(specificities[stressor]) if specificities[stressor] else None,
        }

    return {
        "n_samples": len(rows),
        "base_status_counts": dict(base_status_counts),
        "n_base_evaluable": len(evaluable_rates),
        "n_base_hallucinations": base_status_counts.get("hallucination", 0),
        "n_invalid_reference": base_status_counts.get("invalid_reference", 0),
        "n_unanswerable": base_status_counts.get("unanswerable", 0),
        "n_unresolved": base_status_counts.get("unresolved", 0),
        "base_accuracy": statistics.mean(evaluable_rates) if evaluable_rates else None,
        "n_intervention_samples": n_intervention_samples,
        "primary_stressor_counts": dict(primary_counts),
        "per_stressor": per_stressor,
        "hidden_state_files": hidden_files,
        "hidden_state_errors": hidden_errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stressor interventions with robust evaluation and hidden-state logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory for results and hidden states")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="For example: auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--question-field", default="question")
    parser.add_argument("--context-field", default="knowledge")
    parser.add_argument("--answer-field", default="right_answer")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--oracle-facts-field", default=None)
    parser.add_argument("--symptom-field", default=None)
    parser.add_argument("--metadata-fields", nargs="*", default=[])

    parser.add_argument("--eval-mode", choices=["exact", "contains", "choice", "token_f1", "llm_judge"], default="llm_judge")
    parser.add_argument("--f1-threshold", type=float, default=0.65)
    parser.add_argument("--judge-max-new-tokens", type=int, default=8)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--fast-reference-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-reference", action="store_true")

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1, help="Generation samples per prompt/variant")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--low-budget-tokens", type=int, default=32)
    parser.add_argument("--high-budget-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shortcut-spans", type=int, default=3)
    parser.add_argument("--max-knowledge-probes", type=int, default=3)
    parser.add_argument("--knowledge-known-threshold", type=float, default=0.67)
    parser.add_argument("--knowledge-scarce-threshold", type=float, default=0.34)

    parser.add_argument(
        "--stressors",
        nargs="+",
        choices=list(STRESSOR_NAMES),
        default=list(STRESSOR_NAMES),
        help="Enabled families; S5 is intentionally omitted for the local non-tool model",
    )
    parser.add_argument(
        "--base-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only robust base/reference audit and no interventions",
    )
    parser.add_argument(
        "--only-base-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run interventions only for cleanly evaluated base hallucinations",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--save-hidden-states", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--hidden-layers",
        default="25%,50%,65%,80%,100%",
        help="Comma-separated absolute layers or relative percentages",
    )
    parser.add_argument(
        "--save-probe-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save hidden states for independent S1 knowledge probes",
    )

    parser.add_argument(
        "--abstention-policy",
        choices=["incorrect", "safe"],
        default="incorrect",
        help="On answerable QA, use incorrect; safe is for risk-sensitive/open-world settings",
    )
    parser.add_argument("--strong-positive-recovery", type=float, default=0.5)
    parser.add_argument("--specificity-threshold", type=float, default=0.25)
    parser.add_argument("--negative-recovery-threshold", type=float, default=0.1)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def make_base_only_result(sample: Sample, base: Dict[str, Any], note: str) -> Dict[str, Any]:
    status = ExperimentRunner.determine_base_status(base)
    return {
        "id": sample.id,
        "question": sample.question,
        "context": sample.context,
        "references": sample.references,
        "symptom": sample.symptom,
        "metadata": sample.metadata,
        "base": base,
        "base_status": status,
        "base_is_hallucination": True if status == "hallucination" else (False if status == "correct" else None),
        "interventions_run": False,
        "variants": [],
        "diagnosis": {
            "per_stressor": {},
            "primary_stressor": None,
            "secondary_stressors": [],
            "unidentified": True,
            "note": note,
        },
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "intervention_results.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"
    write_json(config_path, vars(args))

    rows = read_json_or_jsonl(args.input)
    samples = convert_rows_to_samples(rows, args)
    samples = samples[args.start_index:]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No valid samples loaded")

    completed = load_completed_ids(results_path) if args.resume else set()
    if not args.resume and results_path.exists():
        results_path.unlink()

    engine = LocalCausalLM(
        model_name=args.model,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_input_tokens=args.max_input_tokens,
    )
    evaluator = Evaluator(
        mode=args.eval_mode,
        engine=engine,
        f1_threshold=args.f1_threshold,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_retries=args.judge_retries,
        fast_reference_match=args.fast_reference_match,
    )
    proposal = ProposalGenerator(engine=engine, seed=args.seed)
    rng = random.Random(args.seed)
    factory = InterventionFactory(
        engine=engine,
        proposal=proposal,
        rng=rng,
        base_max_new_tokens=args.max_new_tokens,
        high_budget_tokens=args.high_budget_tokens,
        low_budget_tokens=args.low_budget_tokens,
        temperature=args.temperature,
        max_shortcut_spans=args.max_shortcut_spans,
    )
    runner = ExperimentRunner(
        engine=engine,
        evaluator=evaluator,
        intervention_factory=factory,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        base_seed=args.seed,
        score_reference=args.score_reference,
        strong_positive_recovery=args.strong_positive_recovery,
        specificity_threshold=args.specificity_threshold,
        negative_recovery_threshold=args.negative_recovery_threshold,
        max_knowledge_probes=args.max_knowledge_probes,
        knowledge_known_threshold=args.knowledge_known_threshold,
        knowledge_scarce_threshold=args.knowledge_scarce_threshold,
        output_dir=output_dir,
        save_hidden_states=args.save_hidden_states,
        hidden_layers=args.hidden_layers,
        save_probe_hidden_states=args.save_probe_hidden_states,
        abstention_policy=args.abstention_policy,
    )

    unrelated_pool: List[str] = []
    for sample in samples:
        material = sample.context.strip() or "\n".join(sample.references)
        unrelated_pool.append(material[:1500])

    processed_now = 0
    for idx, sample in enumerate(tqdm(samples, desc="Audit/interventions")):
        if sample.id in completed:
            continue
        unrelated_facts = unrelated_pool[(idx + 1) % len(unrelated_pool)] if len(unrelated_pool) > 1 else ""
        try:
            base_prompt = PromptBuilder.base_user_prompt(sample)
            base = runner.run_prompt(
                sample,
                base_prompt,
                seed_offset=0,
                trace_id="base",
            )
            base_status = runner.determine_base_status(base)

            if args.base_only:
                result = make_base_only_result(sample, base, "base_only_reference_audit_v2")
            elif args.only_base_errors and base_status != "hallucination":
                result = make_base_only_result(
                    sample,
                    base,
                    f"interventions_skipped_for_base_status_{base_status}",
                )
            else:
                result = runner.process_sample(
                    sample,
                    unrelated_facts,
                    args.stressors,
                    precomputed_base=base,
                )

            append_jsonl(results_path, result)
            processed_now += 1
            if processed_now % 5 == 0:
                partial = summarize(results_path)
                partial["config"] = vars(args)
                write_json(summary_path, partial)
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted; partial results are preserved")
            break
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Sample failed: %s", sample.id)
            append_jsonl(results_path, {
                "id": sample.id,
                "question": sample.question,
                "error": repr(exc),
            })
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_summary = summarize(results_path)
    final_summary["config"] = vars(args)
    write_json(summary_path, final_summary)
    LOGGER.info("Results: %s", results_path)
    LOGGER.info("Summary: %s", summary_path)
    if args.save_hidden_states:
        LOGGER.info("Hidden states: %s", output_dir / "hidden_states")


if __name__ == "__main__":
    main()