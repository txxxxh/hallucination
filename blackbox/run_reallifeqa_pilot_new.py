#!/usr/bin/env python3
"""Minimal RealLifeQA cue-intervention pilot.

Runs an OpenAI-compatible target model on original and edited binary-choice
questions, then measures whether removing shortcut cues predicts mistakes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CONDITIONS = ("original", "shortcut_removed", "constraint_removed", "control")
VARIANT_KEYS = ("shortcut_removed", "constraint_removed", "control")
EXPECTED_ROLE_BY_CONDITION = {
    "original": "correct",
    "shortcut_removed": "correct",
    "constraint_removed": "shortcut",
    "control": "correct",
}
EXPECTED_PATTERN = "C/C/S/C"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4"
TARGET_MAX_TOKENS = 32
REQUIRED_ITEM_KEYS = (
    "question",
    "options",
    "answer",
    "correct_option",
    "benchmark_prompt",
    "mistake_models",
    "short_justification",
)
EPSILON = 1e-12


class JsonCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                backup = path.with_suffix(path.suffix + ".broken")
                path.replace(backup)
                self.data = {}

    def make_key(self, namespace: str, payload: Dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def _resolve_input_path(input_path: str) -> Path:
    candidates = [Path(input_path)]
    if not Path(input_path).is_absolute():
        candidates.append(Path("real_life_constrained_qa") / input_path)

    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        joined = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find input file. Tried: {joined}")
    return path


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return data


def _has_keys(item: Dict[str, Any], keys: Iterable[str]) -> bool:
    return all(key in item for key in keys)


def _merge_variant_only_input(
    variant_items: List[Dict[str, Any]], source_path: Path
) -> Optional[List[Dict[str, Any]]]:
    """Merge variant-only question_remove JSON with full-schema base items."""
    if not variant_items or not _has_keys(variant_items[0], VARIANT_KEYS):
        return None

    sibling = source_path.parent / "question_and_result.json"
    if not sibling.exists():
        return None

    base_items = _read_json_list(sibling)
    by_id = {item.get("id"): item for item in base_items if isinstance(item, dict)}
    merged: List[Dict[str, Any]] = []
    for index, variant_item in enumerate(variant_items):
        base = by_id.get(variant_item.get("id"))
        if base is None and index < len(base_items):
            base = base_items[index]
        if not isinstance(base, dict):
            continue
        combined = dict(base)
        if variant_item.get("id") is not None:
            combined["id"] = variant_item["id"]
        combined["_input_variants"] = {
            key: variant_item.get(key) for key in VARIANT_KEYS if key in variant_item
        }
        if isinstance(variant_item.get("expected_roles"), dict):
            combined["_expected_roles"] = dict(variant_item["expected_roles"])
        if isinstance(variant_item.get("variant_design"), dict):
            combined["_variant_design"] = dict(variant_item["variant_design"])
        if isinstance(variant_item.get("expected_options"), dict):
            combined["_expected_options"] = dict(variant_item["expected_options"])
        merged.append(combined)
    return merged


def load_data(input_path: str) -> List[Dict[str, Any]]:
    """Load a JSON list of RealLifeQA items.

    Relative paths are resolved first from the working directory, then from the
    repo's real_life_constrained_qa directory for convenient CLI usage.
    """
    path = _resolve_input_path(input_path)
    data = _read_json_list(path)
    if data and isinstance(data[0], dict) and not _has_keys(data[0], REQUIRED_ITEM_KEYS):
        merged = _merge_variant_only_input(data, path)
        if merged is not None:
            return merged
    return data


def make_editor_prompt(item: Dict[str, Any]) -> str:
    """Create the editor prompt for three cue-intervention variants."""
    answer = int(item["answer"])
    shortcut = 1 if answer == 2 else 2
    options = item["options"]
    option1 = options[0]
    option2 = options[1]

    return (
        "Edit one binary-choice RealLifeQA prompt into three variants.\n"
        "Return only valid JSON with exactly these string keys: "
        "shortcut_removed, constraint_removed, control.\n"
        "Do not include explanations or markdown.\n\n"
        "Rules for every variant:\n"
        "- Keep the option lines exactly unchanged:\n"
        f"  Option1: {option1}\n"
        f"  Option2: {option2}\n"
        "- Keep the final answer format instruction asking for 1 or 2.\n"
        "- Keep the prompt concise and natural.\n"
        "- Do not leave the intended answer ambiguous.\n\n"
        "Variant definitions:\n"
        "- shortcut_removed: remove or neutralize the misleading shortcut cue, "
        "while preserving the original correct option as unambiguously correct.\n"
        "- constraint_removed: construct a minimal counterfactual in which the key "
        "physical/procedural constraint no longer holds and the original shortcut "
        "option becomes unambiguously correct. Do not merely delete information and "
        "leave a 50/50 question; explicitly state the fact that makes the shortcut "
        "option appropriate.\n"
        "- control: paraphrase without changing cues, constraints, or the original "
        "correct option.\n\n"
        f"Original correct option number: {answer}\n"
        f"Original correct option text: {item['correct_option']}\n"
        f"Original shortcut option number: {shortcut}\n"
        f"Original shortcut option text: {options[shortcut - 1]}\n"
        f"Factual note: {item.get('short_justification', '')}\n\n"
        "Expected answer roles after editing:\n"
        "- shortcut_removed -> original correct option\n"
        "- constraint_removed -> original shortcut option\n"
        "- control -> original correct option\n\n"
        "Original prompt:\n"
        f"{item['benchmark_prompt']}"
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Editor output was not a JSON object.")
    return parsed


def _validate_variants(variants: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, str]:
    missing = [key for key in VARIANT_KEYS if key not in variants]
    if missing:
        raise ValueError(f"Editor output missing keys: {missing}")

    options = item["options"]
    expected_lines = [f"Option1: {options[0]}", f"Option2: {options[1]}"]
    validated: Dict[str, str] = {}

    for key in VARIANT_KEYS:
        value = variants[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Variant {key!r} is not a non-empty string.")
        prompt = value.strip()
        for line in expected_lines:
            if line not in prompt:
                raise ValueError(f"Variant {key!r} does not preserve {line!r}.")
        if "Answer 1" not in prompt or "2" not in prompt:
            raise ValueError(f"Variant {key!r} lacks the final 1/2 answer instruction.")
        validated[key] = prompt

    return validated


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _chat_create(client: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for max_tokens vs max_completion_tokens."""
    try:
        return client.chat.completions.create(**kwargs)
    except TypeError as exc:
        if "max_tokens" not in kwargs:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["max_completion_tokens"] = retry_kwargs.pop("max_tokens")
        try:
            return client.chat.completions.create(**retry_kwargs)
        except Exception:
            raise exc
    except Exception as exc:
        message = str(exc)
        if "max_tokens" not in kwargs or "max_completion_tokens" not in message:
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["max_completion_tokens"] = retry_kwargs.pop("max_tokens")
        return client.chat.completions.create(**retry_kwargs)


def _choice_content(choice: Any) -> str:
    message = _get_attr(choice, "message", {})
    content = _get_attr(message, "content", "")
    return content or ""


def _response_choices(response: Any) -> List[Any]:
    return list(_get_attr(response, "choices", []) or [])


def _call_chat_text(
    client: Any,
    cache: JsonCache,
    namespace: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": extra or {},
    }
    key = cache.make_key(namespace, payload)
    cached = cache.get(key)
    if cached is not None:
        cached = dict(cached)
        cached["cached"] = True
        return cached

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra:
        kwargs.update(extra)

    response = _chat_create(client, **kwargs)
    choices = _response_choices(response)
    content = _choice_content(choices[0]) if choices else ""
    result = {"content": content, "cached": False}
    cache.set(key, result)
    return result


def generate_variants(
    client: Any,
    item: Dict[str, Any],
    editor_model: str,
    cache: JsonCache,
    max_retries: int = 3,
) -> Dict[str, str]:
    """Generate and validate the three editor variants for one item."""
    prompt = make_editor_prompt(item)
    cache_payload = {
        "model": editor_model,
        "prompt": prompt,
        "max_retries": max_retries,
        "version": 2,
    }
    high_level_key = cache.make_key("editor_variants", cache_payload)
    cached = cache.get(high_level_key)
    if cached is not None:
        return _validate_variants(cached, item)

    messages = [
        {
            "role": "system",
            "content": "You edit prompts and return only valid JSON. Do not explain.",
        },
        {"role": "user", "content": prompt},
    ]

    errors: List[str] = []
    for attempt in range(1, max_retries + 1):
        temperature = 0.2 if attempt == 1 else 0.0
        result = _call_chat_text(
            client=client,
            cache=cache,
            namespace=f"editor_raw_attempt_{attempt}",
            model=editor_model,
            messages=messages,
            temperature=temperature,
            max_tokens=1200,
        )
        try:
            parsed = _extract_json_object(result["content"])
            validated = _validate_variants(parsed, item)
            cache.set(high_level_key, validated)
            return validated
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")

    raise ValueError("Could not get valid editor JSON; " + " | ".join(errors))


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return -math.inf
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _extract_top_logprobs(choice: Any) -> Dict[str, float]:
    logprobs = _get_attr(choice, "logprobs", None)
    content_logprobs = _get_attr(logprobs, "content", None)
    if not content_logprobs:
        raise ValueError("No token logprobs returned.")

    first_token = content_logprobs[0]
    top_logprobs = _get_attr(first_token, "top_logprobs", None)
    if not top_logprobs:
        raise ValueError("No top_logprobs returned.")

    by_label: Dict[str, List[float]] = {"1": [], "2": []}
    for entry in top_logprobs:
        token = str(_get_attr(entry, "token", ""))
        label = token.strip()
        if label in by_label:
            by_label[label].append(float(_get_attr(entry, "logprob")))

    if not by_label["1"] or not by_label["2"]:
        raise ValueError("Top logprobs did not include both '1' and '2'.")

    return {label: _logsumexp(values) for label, values in by_label.items()}


def _extract_option(text: str) -> Optional[str]:
    match = re.search(r"(?<!\d)[12](?!\d)", (text or "").strip())
    return match.group(0) if match else None


def _target_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Answer with exactly one character: 1 or 2. Do not explain.",
        },
        {"role": "user", "content": prompt},
    ]


def _ask_with_logprobs(
    client: Any,
    model: str,
    prompt: str,
    cache: JsonCache,
) -> Dict[str, Any]:
    messages = _target_messages(prompt)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": TARGET_MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": 5,
        "version": 2,
    }
    key = cache.make_key("target_logprobs", payload)
    cached = cache.get(key)
    if cached is not None:
        if isinstance(cached, dict) and cached.get("error"):
            raise ValueError(f"cached logprobs failure: {cached['error']}")
        cached = dict(cached)
        cached["cached"] = True
        return cached

    try:
        response = _chat_create(
            client,
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=TARGET_MAX_TOKENS,
            logprobs=True,
            top_logprobs=5,
        )
        choices = _response_choices(response)
        if not choices:
            raise ValueError("No target choices returned.")

        choice = choices[0]
        content = _choice_content(choice)
        logprobs = _extract_top_logprobs(choice)
        prediction = _extract_option(content) or (
            "1" if logprobs["1"] >= logprobs["2"] else "2"
        )
        probs = {label: math.exp(logprobs[label]) for label in ("1", "2")}
        result = {
            "method": "logprobs",
            "prediction": prediction,
            "raw_output": content,
            "logprob_1": logprobs["1"],
            "logprob_2": logprobs["2"],
            "prob_1": probs["1"],
            "prob_2": probs["2"],
            "cached": False,
        }
        cache.set(key, result)
        return result
    except Exception as exc:
        cache.set(key, {"error": str(exc)})
        raise


def _ask_by_sampling(
    client: Any,
    model: str,
    prompt: str,
    samples: int,
    cache: JsonCache,
) -> Dict[str, Any]:
    messages = _target_messages(prompt)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": TARGET_MAX_TOKENS,
        "samples": samples,
        "version": 3,
    }
    key = cache.make_key("target_sampling", payload)
    cached = cache.get(key)
    if cached is not None:
        cached = dict(cached)
        cached["cached"] = True
        return cached

    outputs: List[str] = []
    try:
        response = _chat_create(
            client,
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=TARGET_MAX_TOKENS,
            n=samples,
        )
        outputs.extend(
            _choice_content(choice) for choice in _response_choices(response)[:samples]
        )
    except Exception:
        pass

    # Some OpenAI-compatible APIs accept n but still return only one choice.
    # Top up with individual calls so --samples K means K actual model outputs.
    while len(outputs) < samples:
        response = _chat_create(
            client,
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=TARGET_MAX_TOKENS,
        )
        choices = _response_choices(response)
        outputs.append(_choice_content(choices[0]) if choices else "")

    counts = {"1": 0, "2": 0}
    for output in outputs:
        label = _extract_option(output)
        if label in counts:
            counts[label] += 1

    valid = counts["1"] + counts["2"]
    invalid = len(outputs) - valid
    denominator = valid + 2
    prob_1 = (counts["1"] + 1) / denominator
    prob_2 = (counts["2"] + 1) / denominator
    prediction = "1" if prob_1 >= prob_2 else "2"
    result = {
        "method": "sampling",
        "prediction": prediction,
        "raw_outputs": outputs,
        "counts": counts,
        "invalid_count": invalid,
        "valid_count": valid,
        "logprob_1": math.log(max(prob_1, EPSILON)),
        "logprob_2": math.log(max(prob_2, EPSILON)),
        "prob_1": prob_1,
        "prob_2": prob_2,
        "cached": False,
    }
    cache.set(key, result)
    return result


def ask_model_probs(
    client: Any,
    prompt: str,
    target_model: str,
    cache: JsonCache,
    samples: int = 30,
) -> Dict[str, Any]:
    """Return prediction plus probabilities/log-probabilities for labels 1 and 2."""
    try:
        return _ask_with_logprobs(client, target_model, prompt, cache)
    except Exception as exc:
        sampled = _ask_by_sampling(client, target_model, prompt, samples, cache)
        sampled["logprobs_error"] = str(exc)
        return sampled


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def compute_metrics(
    item: Dict[str, Any],
    predictions: Dict[str, Dict[str, Any]],
    include_probabilities: bool = False,
    allow_deltas: bool = False,
) -> Dict[str, Any]:
    """Compute answer roles and test the designed C/C/S/C intervention pattern."""
    correct = str(int(item["answer"]))
    shortcut = "1" if correct == "2" else "2"
    row: Dict[str, Any] = {
        "item_id": item.get("item_id", item.get("id", item.get("_item_index"))),
        "correct_option": correct,
        "shortcut_option": shortcut,
        "expected_pattern": EXPECTED_PATTERN,
    }

    observed_codes: List[str] = []
    all_expected = True
    for condition in CONDITIONS:
        pred = predictions.get(condition, {})
        prediction = pred.get("prediction")
        if prediction not in {"1", "2"}:
            prediction = None

        expected_role = EXPECTED_ROLE_BY_CONDITION[condition]
        expected_option = correct if expected_role == "correct" else shortcut
        if prediction == correct:
            answer_role = "correct"
            role_code = "C"
        elif prediction == shortcut:
            answer_role = "shortcut"
            role_code = "S"
        else:
            answer_role = "invalid"
            role_code = "?"

        matches_expected = prediction == expected_option if prediction else None
        if matches_expected is not True:
            all_expected = False
        observed_codes.append(role_code)

        row[f"prediction_{condition}"] = prediction
        row[f"answer_role_{condition}"] = answer_role
        row[f"expected_role_{condition}"] = expected_role
        row[f"expected_option_{condition}"] = expected_option
        row[f"matches_expected_{condition}"] = matches_expected
        row[f"predicts_original_correct_{condition}"] = (
            prediction == correct if prediction else None
        )
        row[f"predicts_shortcut_{condition}"] = (
            prediction == shortcut if prediction else None
        )
        # Backward-compatible field: this always means the original benchmark answer.
        row[f"correct_{condition}"] = (
            prediction == correct if prediction else None
        )
        row[f"method_{condition}"] = pred.get("method")

        if include_probabilities:
            row[f"prob_correct_{condition}"] = _safe_float(pred.get(f"prob_{correct}"))
            row[f"prob_shortcut_{condition}"] = _safe_float(pred.get(f"prob_{shortcut}"))
            row[f"prob_expected_{condition}"] = _safe_float(pred.get(f"prob_{expected_option}"))
            log_correct = _safe_float(pred.get(f"logprob_{correct}"))
            log_shortcut = _safe_float(pred.get(f"logprob_{shortcut}"))
            row[f"L_{condition}"] = (
                None if log_correct is None or log_shortcut is None
                else log_shortcut - log_correct
            )

    row["observed_pattern"] = "/".join(observed_codes)
    row["matches_expected_pattern"] = all_expected
    row["original_is_correct"] = row.get("matches_expected_original")
    row["shortcut_removed_preserves_correct"] = row.get("matches_expected_shortcut_removed")
    row["constraint_removed_activates_shortcut"] = row.get("matches_expected_constraint_removed")
    row["control_preserves_correct"] = row.get("matches_expected_control")

    for condition in ("shortcut_removed", "constraint_removed", "control"):
        original_prediction = row.get("prediction_original")
        condition_prediction = row.get(f"prediction_{condition}")
        flip = (
            original_prediction in {"1", "2"}
            and condition_prediction in {"1", "2"}
            and original_prediction != condition_prediction
        )
        row[f"flip_{condition}"] = flip
        if not flip:
            direction = ""
        elif original_prediction == shortcut and condition_prediction == correct:
            direction = "shortcut_to_correct"
        elif original_prediction == correct and condition_prediction == shortcut:
            direction = "correct_to_shortcut"
        else:
            direction = f"{original_prediction}_to_{condition_prediction}"
        row[f"flip_direction_{condition}"] = direction

    if include_probabilities:
        original_l = row.get("L_original")
        shortcut_removed_l = row.get("L_shortcut_removed")
        constraint_removed_l = row.get("L_constraint_removed")
        control_l = row.get("L_control")
        if not allow_deltas or original_l is None:
            row["shortcut_removal_effect"] = None
            row["constraint_removal_effect"] = None
            row["control_drift"] = None
        else:
            row["shortcut_removal_effect"] = (
                None if shortcut_removed_l is None else original_l - shortcut_removed_l
            )
            row["constraint_removal_effect"] = (
                None if constraint_removed_l is None else constraint_removed_l - original_l
            )
            row["control_drift"] = (
                None if control_l is None else control_l - original_l
            )

    return row


def _format_optional_float(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_raw_prediction_record(
    item_id: Any,
    variant_type: str,
    prompt: Optional[str],
    correct_option: str,
    shortcut_option: str,
    prediction_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a raw row with expected answer role for each intervention."""
    prediction_data = prediction_data or {}
    prediction = prediction_data.get("prediction")
    if prediction not in {"1", "2"}:
        prediction = None
    counts = prediction_data.get("counts")
    if not isinstance(counts, dict):
        counts = None

    expected_role = EXPECTED_ROLE_BY_CONDITION[variant_type]
    expected_option = correct_option if expected_role == "correct" else shortcut_option
    row = dict(prediction_data)
    row.update(
        {
            "item_id": item_id,
            "variant_type": variant_type,
            "prompt": prompt,
            "correct_option": correct_option,
            "shortcut_option": shortcut_option,
            "expected_role": expected_role,
            "expected_option": expected_option,
            "prediction": prediction,
            "answer_role": (
                "correct" if prediction == correct_option
                else "shortcut" if prediction == shortcut_option
                else "invalid"
            ),
            "matches_expected": prediction == expected_option if prediction else None,
            "is_original_correct": prediction == correct_option if prediction else None,
            "is_shortcut": prediction == shortcut_option if prediction else None,
            "method": prediction_data.get("method"),
            "counts": counts,
            "valid_count": prediction_data.get("valid_count"),
            "invalid_count": prediction_data.get("invalid_count"),
        }
    )
    if error:
        row["error"] = error
    return row


def _build_prediction_results(
    raw_rows: List[Dict[str, Any]],
    include_probabilities: bool,
    allow_deltas: bool,
) -> List[Dict[str, Any]]:
    """Pair variants by explicit item_id and variant_type, never cache ordering."""
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    item_ids: Dict[str, Any] = {}
    ordered_keys: List[str] = []

    for raw in raw_rows:
        item_id = raw.get("item_id")
        item_key = json.dumps(item_id, sort_keys=True, ensure_ascii=False)
        if item_key not in grouped:
            grouped[item_key] = {}
            item_ids[item_key] = item_id
            ordered_keys.append(item_key)
        variant_type = raw.get("variant_type")
        if variant_type in CONDITIONS:
            grouped[item_key][variant_type] = raw

    rows: List[Dict[str, Any]] = []
    for item_key in ordered_keys:
        predictions = grouped[item_key]
        first = next(iter(predictions.values()), {})
        correct = first.get("correct_option")
        if correct not in {"1", "2"}:
            rows.append({"item_id": item_ids[item_key], "error": "missing correct option"})
            continue
        item = {"item_id": item_ids[item_key], "answer": int(correct)}
        row = compute_metrics(
            item,
            predictions,
            include_probabilities=include_probabilities,
            allow_deltas=allow_deltas,
        )
        rows.append(row)
    return rows


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _item_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _predicted_span_index(item: Dict[str, Any], field: str) -> Optional[int]:
    text_value = item.get(field)
    if text_value is None:
        return None
    spans = item.get("spans")
    if not isinstance(spans, list):
        return None
    matches = [
        span.get("span_index")
        for span in spans
        if isinstance(span, dict) and span.get("span_text") == text_value
    ]
    return int(matches[0]) if len(matches) == 1 and matches[0] is not None else None


def _evaluate_cue_files(
    prediction_path: Path, gold_path: Path
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    predictions = _read_jsonl_records(prediction_path)
    gold_rows = _read_jsonl_records(gold_path)
    gold = {_item_key(row.get("id")): row for row in gold_rows}

    by_item: Dict[str, Dict[str, Any]] = {}
    summary = {
        "items_evaluated": 0,
        "shortcut_high_n": 0,
        "shortcut_high_hit": 0,
        "shortcut_explicit_n": 0,
        "shortcut_explicit_hit": 0,
        "constraint_n": 0,
        "constraint_strict_hit": 0,
        "constraint_proposition_hit": 0,
    }

    for pred in predictions:
        key = _item_key(pred.get("id"))
        g = gold.get(key)
        if g is None:
            continue
        sidx = _predicted_span_index(pred, "pred_shortcut_span")
        cidx = _predicted_span_index(pred, "pred_constraint_span")
        shortcut_gold = set(g.get("shortcut_candidate_indices") or [])
        constraint_gold = set(g.get("constraint_candidate_indices") or [])
        proposition_gold = set(g.get("constraint_proposition_indices") or [])
        confidence = str(g.get("shortcut_confidence", "none"))

        shortcut_eligible = confidence in {"high", "medium"}
        shortcut_hit = (sidx in shortcut_gold) if shortcut_eligible else None
        constraint_hit = cidx in constraint_gold
        proposition_hit = cidx in proposition_gold
        if shortcut_eligible:
            if shortcut_hit and constraint_hit:
                status = "both_strict"
            elif shortcut_hit:
                status = "shortcut_only"
            elif constraint_hit:
                status = "constraint_only"
            else:
                status = "neither"
        else:
            status = "no_explicit_shortcut"

        record = {
            "pred_shortcut_span": pred.get("pred_shortcut_span"),
            "pred_constraint_span": pred.get("pred_constraint_span"),
            "shortcut_cue_confidence": confidence,
            "shortcut_cue_eligible": shortcut_eligible,
            "shortcut_cue_strict_hit": shortcut_hit,
            "constraint_cue_strict_hit": constraint_hit,
            "constraint_cue_proposition_hit": proposition_hit,
            "cue_identification_status": status,
        }
        by_item[key] = record

        summary["items_evaluated"] += 1
        if confidence == "high":
            summary["shortcut_high_n"] += 1
            summary["shortcut_high_hit"] += int(bool(shortcut_hit))
        if shortcut_eligible:
            summary["shortcut_explicit_n"] += 1
            summary["shortcut_explicit_hit"] += int(bool(shortcut_hit))
        summary["constraint_n"] += 1
        summary["constraint_strict_hit"] += int(constraint_hit)
        summary["constraint_proposition_hit"] += int(proposition_hit)

    return by_item, summary


def _merge_cue_results(
    result_rows: List[Dict[str, Any]], cue_by_item: Dict[str, Dict[str, Any]]
) -> None:
    for row in result_rows:
        cue = cue_by_item.get(_item_key(row.get("item_id")))
        if cue:
            row.update(cue)


def _prediction_cell(row: Dict[str, Any], condition: str) -> str:
    prediction = row.get(f"prediction_{condition}") or "?"
    role = row.get(f"answer_role_{condition}", "invalid")
    code = {"correct": "C", "shortcut": "S", "invalid": "?"}.get(role, "?")
    mark = "✓" if row.get(f"matches_expected_{condition}") is True else "✗"
    return f"{prediction} ({code}{mark})"


def _format_console_result(row: Dict[str, Any]) -> str:
    cells = [f"{condition}={_prediction_cell(row, condition)}" for condition in CONDITIONS]
    pattern_mark = "PASS" if row.get("matches_expected_pattern") else "FAIL"
    return (
        f"item {row.get('item_id')} | " + " | ".join(cells)
        + f" | pattern={row.get('observed_pattern')} expected={EXPECTED_PATTERN} {pattern_mark}"
    )


def _write_summary(
    path: Path,
    rows: List[Dict[str, Any]],
    raw_count: int,
    methods: set[str],
    cue_summary: Optional[Dict[str, Any]] = None,
) -> None:
    valid_rows = [row for row in rows if not row.get("error")]
    exact_hits = sum(row.get("matches_expected_pattern") is True for row in valid_rows)
    lines = [
        "# RealLifeQA Cue-Intervention Pilot Summary",
        "",
        "## Intended intervention pattern",
        "",
        "`original = correct`, `shortcut_removed = correct`, "
        "`constraint_removed = shortcut`, `control = correct`",
        "",
        f"Expected role pattern: **{EXPECTED_PATTERN}**",
        "",
        f"- Items with result rows: {len(rows)}",
        f"- Valid item rows: {len(valid_rows)}",
        f"- Raw prediction rows: {raw_count}",
        f"- Exact C/C/S/C pattern: {exact_hits}/{len(valid_rows)} "
        f"({_format_optional_float(exact_hits / len(valid_rows) if valid_rows else None)})",
        "",
        "## Per-condition behavior",
        "",
        "| Condition | Expected answer role | Expected match | Chose original correct | Chose shortcut | Invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for condition in CONDITIONS:
        valid = [
            row for row in valid_rows
            if row.get(f"prediction_{condition}") in {"1", "2"}
        ]
        expected_hits = sum(row.get(f"matches_expected_{condition}") is True for row in valid)
        correct_hits = sum(row.get(f"predicts_original_correct_{condition}") is True for row in valid)
        shortcut_hits = sum(row.get(f"predicts_shortcut_{condition}") is True for row in valid)
        invalid = len(valid_rows) - len(valid)
        expected_role = EXPECTED_ROLE_BY_CONDITION[condition]
        lines.append(
            f"| {condition} | {expected_role} | {expected_hits}/{len(valid)} | "
            f"{correct_hits}/{len(valid)} | {shortcut_hits}/{len(valid)} | {invalid} |"
        )

    lines.extend(["", "## Original-to-variant flips", ""])
    for variant_type in VARIANT_KEYS:
        directions = [
            row.get(f"flip_direction_{variant_type}")
            for row in valid_rows
            if row.get(f"flip_{variant_type}") is True
        ]
        shortcut_to_correct = sum(d == "shortcut_to_correct" for d in directions)
        correct_to_shortcut = sum(d == "correct_to_shortcut" for d in directions)
        lines.append(
            f"- original -> {variant_type}: {len(directions)} flips "
            f"(shortcut_to_correct={shortcut_to_correct}, "
            f"correct_to_shortcut={correct_to_shortcut})"
        )

    if cue_summary:
        lines.extend([
            "",
            "## Gold-cue identification",
            "",
            f"- Items evaluated: {cue_summary.get('items_evaluated', 0)}",
            "- Shortcut strict recall (high confidence): "
            f"{cue_summary.get('shortcut_high_hit', 0)}/"
            f"{cue_summary.get('shortcut_high_n', 0)}",
            "- Shortcut strict recall (all explicit high+medium): "
            f"{cue_summary.get('shortcut_explicit_hit', 0)}/"
            f"{cue_summary.get('shortcut_explicit_n', 0)}",
            "- Constraint strict recall: "
            f"{cue_summary.get('constraint_strict_hit', 0)}/"
            f"{cue_summary.get('constraint_n', 0)}",
            "- Constraint proposition-overlap recall: "
            f"{cue_summary.get('constraint_proposition_hit', 0)}/"
            f"{cue_summary.get('constraint_n', 0)}",
        ])

    lines.extend([
        "",
        "## Per-item answers",
        "",
        "`C` means the original correct option; `S` means the original shortcut option. "
        "The check mark compares the answer with that condition's expected role.",
        "",
        "| ID | Original | Shortcut removed | Constraint removed | Control | Pattern | Expected pattern? | Cue status |",
        "|---:|---|---|---|---|---|---:|---|",
    ])
    for row in valid_rows:
        cue_status = row.get("cue_identification_status", "n/a")
        lines.append(
            f"| {row.get('item_id')} | {_prediction_cell(row, 'original')} | "
            f"{_prediction_cell(row, 'shortcut_removed')} | "
            f"{_prediction_cell(row, 'constraint_removed')} | "
            f"{_prediction_cell(row, 'control')} | {row.get('observed_pattern')} | "
            f"{'yes' if row.get('matches_expected_pattern') else 'no'} | {cue_status} |"
        )

    failed = [row for row in valid_rows if row.get("matches_expected_pattern") is not True]
    if failed:
        lines.extend(["", "## Pattern failures", ""])
        for row in failed:
            lines.append(
                f"- Item {row.get('item_id')}: observed {row.get('observed_pattern')}; "
                f"failed conditions: "
                + ", ".join(
                    condition for condition in CONDITIONS
                    if row.get(f"matches_expected_{condition}") is not True
                )
            )

    method_text = ", ".join(sorted(methods)) if methods else "none"
    lines.extend(["", f"- Prediction methods present: {method_text}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    missing = [key for key in REQUIRED_ITEM_KEYS if key not in item]
    if missing:
        raise ValueError(f"missing required keys: {missing}")
    if not isinstance(item["options"], list) or len(item["options"]) != 2:
        raise ValueError("options must be a list of exactly two strings")
    answer = int(item["answer"])
    if answer not in (1, 2):
        raise ValueError("answer must be 1 or 2")
    expected_options = item.get("_expected_options")
    if isinstance(expected_options, dict):
        correct = str(answer)
        shortcut = "1" if correct == "2" else "2"
        derived = {
            "original": correct,
            "shortcut_removed": correct,
            "constraint_removed": shortcut,
            "control": correct,
        }
        normalized_expected = {key: str(expected_options.get(key)) for key in CONDITIONS}
        if normalized_expected != derived:
            raise ValueError(
                f"expected_options metadata {normalized_expected} does not match derived {derived}"
            )
    normalized = dict(item)
    normalized["_item_index"] = index
    normalized["answer"] = answer
    return normalized


def _mistake_model_count(item: Any) -> int:
    if not isinstance(item, dict):
        return 0
    value = item.get("mistake_models")
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, str):
        return 1 if value.strip() else 0
    return 0


def _make_client(base_url: str = DEFAULT_BASE_URL) -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI Python package first: pip install openai") from exc

    kwargs: Dict[str, Any] = {"api_key": api_key}
    kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="question_remove.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--target-model", default=DEFAULT_MODEL)
    parser.add_argument("--editor-model", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--outdir", default="outputs_chatgpt_5.4/reallifeqa_pilot")
    parser.add_argument(
        "--analysis-mode",
        choices=("prediction", "probability"),
        default="prediction",
        help="prediction reports answer roles/patterns; probability also writes L/effect fields",
    )
    parser.add_argument(
        "--cue-predictions",
        default=None,
        help="optional cue-extraction JSONL to evaluate against gold cues",
    )
    parser.add_argument(
        "--gold-cues",
        default=None,
        help="optional gold-cue JSONL; use together with --cue-predictions",
    )
    parser.add_argument(
        "--filter-mistake-models-min",
        type=int,
        default=None,
        metavar="N",
        help="only run items with at least N entries in mistake_models (applied before --limit)",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.filter_mistake_models_min is not None and args.filter_mistake_models_min < 0:
        parser.error("--filter-mistake-models-min must be non-negative")
    if bool(args.cue_predictions) != bool(args.gold_cues):
        parser.error("--cue-predictions and --gold-cues must be supplied together")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = JsonCache(outdir / "cache.json")

    data = load_data(args.input)
    indexed_items = list(enumerate(data))
    if args.filter_mistake_models_min is not None:
        indexed_items = [
            pair for pair in indexed_items
            if _mistake_model_count(pair[1]) >= args.filter_mistake_models_min
        ]
    if args.limit is not None and args.limit >= 0:
        indexed_items = indexed_items[: args.limit]
    editor_model = args.editor_model or args.target_model
    client = _make_client(args.base_url)

    variant_rows: List[Dict[str, Any]] = []
    raw_prediction_rows: List[Dict[str, Any]] = []
    invalid_result_rows: List[Dict[str, Any]] = []

    for run_index, (source_index, raw_item) in enumerate(indexed_items):
        item_id = raw_item.get("id", source_index) if isinstance(raw_item, dict) else source_index
        try:
            if not isinstance(raw_item, dict):
                raise ValueError("item is not a JSON object")
            item = _validate_item(raw_item, source_index)
        except Exception as exc:
            error_row = {"item_id": item_id, "error": f"invalid item: {exc}"}
            variant_rows.append(error_row)
            invalid_result_rows.append(error_row)
            continue

        correct = str(int(item["answer"]))
        shortcut = "1" if correct == "2" else "2"
        prompts: Dict[str, str] = {"original": item["benchmark_prompt"]}
        variant_error = ""

        try:
            if item.get("_input_variants"):
                variants = _validate_variants(item["_input_variants"], item)
                variant_source = "input"
            else:
                variants = generate_variants(client, item, editor_model, cache)
                variant_source = "editor"
            prompts.update(variants)
            variant_rows.append({
                "item_index": source_index,
                "id": item.get("id", source_index),
                "correct_option": correct,
                "shortcut_option": shortcut,
                "expected_roles": dict(EXPECTED_ROLE_BY_CONDITION),
                "variant_source": variant_source,
                **variants,
            })
        except Exception as exc:
            variant_error = str(exc)
            variant_rows.append({
                "item_index": source_index,
                "id": item.get("id", source_index),
                "correct_option": correct,
                "shortcut_option": shortcut,
                "expected_roles": dict(EXPECTED_ROLE_BY_CONDITION),
                "error": variant_error,
            })

        item_raw_rows: List[Dict[str, Any]] = []
        for condition in CONDITIONS:
            prompt = prompts.get(condition)
            if prompt is None:
                raw_row = _make_raw_prediction_record(
                    item_id=item.get("id", source_index),
                    variant_type=condition,
                    prompt=None,
                    correct_option=correct,
                    shortcut_option=shortcut,
                    error=variant_error or "missing prompt",
                )
            else:
                try:
                    prediction = ask_model_probs(
                        client=client,
                        prompt=prompt,
                        target_model=args.target_model,
                        cache=cache,
                        samples=args.samples,
                    )
                    raw_row = _make_raw_prediction_record(
                        item_id=item.get("id", source_index),
                        variant_type=condition,
                        prompt=prompt,
                        correct_option=correct,
                        shortcut_option=shortcut,
                        prediction_data=prediction,
                    )
                except Exception as exc:
                    raw_row = _make_raw_prediction_record(
                        item_id=item.get("id", source_index),
                        variant_type=condition,
                        prompt=prompt,
                        correct_option=correct,
                        shortcut_option=shortcut,
                        error=str(exc),
                    )
            raw_prediction_rows.append(raw_row)
            item_raw_rows.append(raw_row)

        preview_rows = _build_prediction_results(
            item_raw_rows, include_probabilities=False, allow_deltas=False
        )
        if preview_rows:
            print(
                f"[{run_index + 1}/{len(indexed_items)}] " + _format_console_result(preview_rows[0]),
                file=sys.stderr,
                flush=True,
            )
        time.sleep(0.05)

    methods = {
        str(row["method"])
        for row in raw_prediction_rows
        if row.get("method") in {"logprobs", "sampling"}
    }
    mixed_methods = len(methods) > 1
    include_probabilities = args.analysis_mode == "probability"
    result_rows = _build_prediction_results(
        raw_prediction_rows,
        include_probabilities=include_probabilities,
        allow_deltas=include_probabilities and not mixed_methods,
    )
    result_rows.extend(invalid_result_rows)

    cue_summary: Optional[Dict[str, Any]] = None
    if args.cue_predictions and args.gold_cues:
        cue_by_item, cue_summary = _evaluate_cue_files(
            Path(args.cue_predictions), Path(args.gold_cues)
        )
        _merge_cue_results(result_rows, cue_by_item)

    _write_jsonl(outdir / "variants.jsonl", variant_rows)
    _write_jsonl(outdir / "raw_predictions.jsonl", raw_prediction_rows)
    _write_results_csv(outdir / "results.csv", result_rows)
    failures = [row for row in result_rows if row.get("matches_expected_pattern") is False]
    _write_jsonl(outdir / "pattern_failures.jsonl", failures)
    _write_summary(
        outdir / "summary.md",
        result_rows,
        len(raw_prediction_rows),
        methods,
        cue_summary=cue_summary,
    )

    valid_rows = [row for row in result_rows if not row.get("error")]
    exact_hits = sum(row.get("matches_expected_pattern") is True for row in valid_rows)
    print(
        f"Overall expected {EXPECTED_PATTERN}: {exact_hits}/{len(valid_rows)}; "
        f"constraint_removed -> shortcut: "
        f"{sum(row.get('constraint_removed_activates_shortcut') is True for row in valid_rows)}/"
        f"{len(valid_rows)}",
        file=sys.stderr,
    )
    if cue_summary:
        print(
            "Cue strict hits: shortcut "
            f"{cue_summary['shortcut_explicit_hit']}/{cue_summary['shortcut_explicit_n']}, "
            "constraint "
            f"{cue_summary['constraint_strict_hit']}/{cue_summary['constraint_n']}",
            file=sys.stderr,
        )
    print(f"Wrote outputs to {outdir}", file=sys.stderr)
    if include_probabilities and mixed_methods:
        print(
            "Skipped probability effects because predictions mix logprobs and sampling.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
