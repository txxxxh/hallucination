#!/usr/bin/env python3
"""Forward-only, unsupervised perturbation-response study for two-profile QA.

The expensive stage never calls ``generate``.  For every prompt condition it
records selected pre-answer hidden states, next-token distribution summaries,
and teacher-forced likelihoods of the two candidate names.  The analysis stage
clusters paired response features without using rgt_ans/wrg_ans; answer labels
are used only for optional cohort selection and post-hoc interpretation.

Examples
--------
Smoke-test prompt construction without loading a model::

    python profile_perturbation_unsupervised.py prepare --limit 5

Extract (downloads/cache files under /tmp)::

    python profile_perturbation_unsupervised.py extract --limit 100

Analyze completed item files::

    python profile_perturbation_unsupervised.py analyze \
      --selection base_wrong_full_correct
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DELIMITER = (
    "Choose exactly one profile from the two, and output the name of the "
    "person as the answer to the following question:"
)
DEFAULT_DATA = Path(__file__).with_name("shuffled_prepend_profiles_question.json")
DEFAULT_OUTPUT = Path(__file__).with_name("profile_perturbation_forward_output")
DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CACHE = "/tmp/hf_profile_perturbation_cache"


@dataclass(frozen=True)
class Profile:
    name: str
    fields: tuple[tuple[str, tuple[str, ...]], ...]

    def values(self) -> list[str]:
        return [v for _, values in self.fields for v in values]


@dataclass(frozen=True)
class ParsedItem:
    key: str
    header: str
    profiles: tuple[Profile, Profile]
    question: str
    right_answer: str
    wrong_answer: str


@dataclass(frozen=True)
class Condition:
    name: str
    prompt: str
    changed: bool = True
    note: str = ""


def stable_seed(text: str, seed: int = 0) -> int:
    raw = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "little")


def split_values(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(";") if x.strip())


def parse_profile_lines(lines: Sequence[str]) -> Profile:
    if not lines or not lines[0].startswith("name:"):
        raise ValueError("profile block must start with 'name:'")
    name = lines[0].split(":", 1)[1].strip()
    fields: list[tuple[str, tuple[str, ...]]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        if ":" not in line:
            key, value = "unparsed", line.strip()
        else:
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
        fields.append((key, split_values(value)))
    return Profile(name=name, fields=tuple(fields))


def parse_item(row: dict[str, Any]) -> ParsedItem:
    prompt = str(row["prompt"])
    marker = "\n" + DELIMITER + "\n"
    if marker not in prompt:
        raise ValueError(f"{row.get('key')}: delimiter not found exactly once")
    context, question = prompt.split(marker, 1)
    lines = context.splitlines()
    name_indices = [i for i, line in enumerate(lines) if line.startswith("name:")]
    if len(name_indices) != 2:
        raise ValueError(f"{row.get('key')}: expected two profiles, got {len(name_indices)}")
    header = "\n".join(lines[: name_indices[0]]).strip()
    p1 = parse_profile_lines(lines[name_indices[0] : name_indices[1]])
    p2 = parse_profile_lines(lines[name_indices[1] :])
    return ParsedItem(
        key=str(row["key"]),
        header=header or "Given two profiles of two persons:",
        profiles=(p1, p2),
        question=question.strip(),
        right_answer=str(row["rgt_ans"]),
        wrong_answer=str(row["wrg_ans"]),
    )


def render_profile(profile: Profile, fields: Sequence[tuple[str, Sequence[str]]] | None = None) -> str:
    rows = [f"name: {profile.name}"]
    for key, values in (profile.fields if fields is None else fields):
        rows.append(f"{key}: {'; '.join(values)}")
    return "\n".join(rows)


def render_prompt(
    item: ParsedItem,
    profiles: Sequence[tuple[Profile, Sequence[tuple[str, Sequence[str]]] | None]] | None,
    question: str | None = None,
    cue: str | None = None,
) -> str:
    pieces: list[str] = []
    if profiles is not None:
        pieces.append(item.header)
        pieces.extend(render_profile(profile, fields) for profile, fields in profiles)
        pieces.append(DELIMITER)
    else:
        pieces.append("Answer the following question and output only the person's name:")
    if cue:
        pieces.append(cue.strip())
    pieces.append((question or item.question).strip())
    return "\n".join(pieces)


def normalized_contains(question: str, value: str) -> bool:
    # Exact phrase matching is deliberately conservative.  Short/common values
    # otherwise turn almost every occupation into alleged decisive evidence.
    q = re.sub(r"\s+", " ", question.casefold())
    v = re.sub(r"\s+", " ", value.casefold()).strip(" .,;:")
    return len(v) >= 5 and v in q


def evidence_values(item: ParsedItem) -> set[str]:
    return {
        value
        for profile in item.profiles
        for value in profile.values()
        if normalized_contains(item.question, value)
    }


def filtered_fields(
    profile: Profile, evidence: set[str], keep_evidence: bool
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    result: list[tuple[str, tuple[str, ...]]] = []
    for key, values in profile.fields:
        selected = tuple(v for v in values if ((v in evidence) == keep_evidence))
        if selected:
            result.append((key, selected))
    return tuple(result)


def structure_only_fields(profile: Profile) -> tuple[tuple[str, tuple[str, ...]], ...]:
    result = []
    for key, values in profile.fields:
        # Preserve field and list cardinality without retaining factual content.
        masked = tuple(f"[withheld {key} {j + 1}]" for j in range(len(values)))
        result.append((key, masked))
    return tuple(result)


def shuffled_fields(profile: Profile, seed: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows = list(profile.fields)
    random.Random(seed).shuffle(rows)
    return tuple(rows)


NEGATION_PARAPHRASES: tuple[tuple[str, str], ...] = (
    (r"\bnever been awarded\b", "not at any point been a recipient of"),
    (r"\bnever received\b", "not at any point received"),
    (r"\bnever served\b", "did not at any time serve"),
    (r"\bdid not receive\b", "was never a recipient of"),
    (r"\bdid not pursue\b", "never pursued"),
    (r"\bwas not awarded\b", "never received"),
    (r"\bhas not been awarded\b", "has never received"),
)

NEGATION_FLIPS: tuple[tuple[str, str], ...] = (
    (r"\bhas never been awarded\b", "has been awarded"),
    (r"\bhave never been awarded\b", "have been awarded"),
    (r"\bwas never awarded\b", "was awarded"),
    (r"\bwere never awarded\b", "were awarded"),
    (r"\bnever been awarded\b", "been awarded"),
    (r"\bnever received\b", "received"),
    (r"\bnever served\b", "served"),
    (r"\bdid not receive\b", "received"),
    (r"\bdid not pursue\b", "pursued"),
    (r"\bwas not awarded\b", "was awarded"),
    (r"\bhas not been awarded\b", "has been awarded"),
    # Broad lexical fallbacks are intentionally last: use a grammatical
    # phrase-level rewrite when available, otherwise flip the first explicit
    # negator instead of dropping the condition from the experiment.
    (r"\b(?:entirely\s+)?unrelated to\b", "directly related to"),
    (r"\bnever\b\s*", ""),
    (r"\bnot\b\s*", ""),
    (r"\bnor\b", "and"),
)


def replace_first_pattern(text: str, replacements: Sequence[tuple[str, str]]) -> tuple[str, bool]:
    for pattern, replacement in replacements:
        changed, n = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if n:
            return changed, True
    return text, False


def paraphrase_entity(question: str, evidence: set[str]) -> tuple[str, bool, str]:
    def entity_priority(value: str) -> tuple[int, int]:
        lower = value.casefold()
        if lower.startswith("nobel prize"):
            rank = 4
        elif re.search(r"\b(prize|award|medal|order)\b", lower):
            rank = 3
        elif re.search(r"\b(minister|chancellor|president|position)\b", lower):
            rank = 2
        elif "university" in lower or "institute" in lower:
            rank = 1
        else:
            rank = 0
        return rank, len(value)

    candidates = sorted(
        (v for v in evidence if v.casefold() in question.casefold()),
        key=entity_priority,
        reverse=True,
    )
    for value in candidates:
        m = re.fullmatch(r"Nobel Prize in (.+)", value, flags=re.IGNORECASE)
        if m:
            replacement = f"the Nobel distinction recognizing work in {m.group(1)}"
        elif re.search(r"\b(prize|award|medal|order)\b", value, flags=re.IGNORECASE):
            # Remove the exact surface form while retaining a controlled
            # description.  The original is logged for audit.
            kind = re.search(r"\b(prize|award|medal|order)\b", value, flags=re.IGNORECASE)
            replacement = f"the specific {kind.group(1).lower()} described by this distinction"
        elif "university" in value.casefold() or "institute" in value.casefold():
            replacement = "that named higher-education institution"
        else:
            continue
        changed, n = re.subn(re.escape(value), replacement, question, count=1, flags=re.IGNORECASE)
        if n:
            return changed, True, f"{value} -> {replacement}"
    return question, False, "no supported entity paraphrase rule matched"


def build_conditions(item: ParsedItem, seed: int = 0) -> list[Condition]:
    ev = evidence_values(item)
    full = [(p, None) for p in item.profiles]
    without = [(p, filtered_fields(p, ev, keep_evidence=False)) for p in item.profiles]
    minimal = [(p, filtered_fields(p, ev, keep_evidence=True)) for p in item.profiles]
    swapped = [(item.profiles[1], None), (item.profiles[0], None)]
    shuffled = [
        (p, shuffled_fields(p, stable_seed(f"{item.key}:{j}", seed)))
        for j, p in enumerate(item.profiles)
    ]
    structure = [(p, structure_only_fields(p)) for p in item.profiles]
    neg_para, neg_para_changed = replace_first_pattern(item.question, NEGATION_PARAPHRASES)
    neg_flip, neg_flip_changed = replace_first_pattern(item.question, NEGATION_FLIPS)
    entity_q, entity_changed, entity_note = paraphrase_entity(item.question, ev)
    cue = (
        "Compare the two profiles constraint by constraint internally. Pay special "
        "attention to negation and distinguishing attributes. Output only the name."
    )
    return [
        Condition("question_only", render_prompt(item, None)),
        Condition("full_context", render_prompt(item, full)),
        Condition(
            "without_question_evidence",
            render_prompt(item, without),
            changed=bool(ev),
            note=f"removed {len(ev)} exact question-mentioned values from both profiles",
        ),
        Condition(
            "minimal_decisive_evidence",
            render_prompt(item, minimal),
            changed=bool(ev),
            note=f"retained {len(ev)} exact question-mentioned values",
        ),
        Condition("profile_order_swap", render_prompt(item, swapped)),
        Condition("attribute_order_shuffle", render_prompt(item, shuffled)),
        Condition("structure_only_context", render_prompt(item, structure)),
        Condition(
            "negation_paraphrase",
            render_prompt(item, full, question=neg_para),
            changed=neg_para_changed,
            note="meaning-preserving negation rewrite" if neg_para_changed else "no rule matched",
        ),
        Condition(
            "negation_flip",
            render_prompt(item, full, question=neg_flip),
            changed=neg_flip_changed,
            note="semantic counterfactual; original gold is not used for evaluation",
        ),
        Condition(
            "entity_paraphrase",
            render_prompt(item, full, question=entity_q),
            changed=entity_changed,
            note=entity_note,
        ),
        Condition("structured_comparison_cue", render_prompt(item, full, cue=cue)),
    ]


def load_rows(path: Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("benchmark must be a JSON list")
    end = None if limit is None else offset + limit
    return rows[offset:end]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


class ForwardExtractor:
    def __init__(self, args: argparse.Namespace):
        # Set cache locations before importing transformers/huggingface_hub;
        # both libraries read some environment configuration at import time.
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.cache_dir) / "hub")
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = args.device
        self.max_input_tokens = args.max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, cache_dir=args.cache_dir, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dtype = getattr(torch, args.dtype)
        kwargs: dict[str, Any] = {
            "cache_dir": args.cache_dir,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if args.device == "cuda":
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        if args.device != "cuda":
            self.model.to(args.device)
        self.model.eval()
        n_layers = int(self.model.config.num_hidden_layers)
        if args.layers:
            requested = [int(x) for x in args.layers.split(",")]
            # output.hidden_states uses 0=embedding, 1..L=block outputs.
            self.layers = [x if x >= 0 else n_layers + 1 + x for x in requested]
        else:
            self.layers = sorted({max(1, n_layers // 4), n_layers // 2, 3 * n_layers // 4, n_layers})
        if any(x < 0 or x > n_layers for x in self.layers):
            raise ValueError(f"layers must index hidden_states in [0,{n_layers}]: {self.layers}")

    def render_chat(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt.rstrip() + "\nAnswer:"

    def tokenize_prompt(self, text: str) -> dict[str, Any]:
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        n = int(enc["input_ids"].shape[1])
        if n > self.max_input_tokens:
            raise ValueError(f"prompt has {n} tokens, exceeds --max-input-tokens={self.max_input_tokens}")
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def prompt_forward(self, prompt: str) -> tuple[np.ndarray, np.ndarray, float, int, list[int]]:
        text = self.render_chat(prompt)
        enc = self.tokenize_prompt(text)
        with self.torch.inference_mode():
            out = self.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = np.stack(
            [out.hidden_states[layer][0, -1].float().cpu().numpy() for layer in self.layers]
        ).astype(np.float32)
        logits_t = out.logits[0, -1].float()
        probs = self.torch.softmax(logits_t, dim=-1)
        entropy = float(-(probs * self.torch.log(probs.clamp_min(1e-30))).sum().cpu())
        logits = logits_t.cpu().numpy().astype(np.float32)
        return hidden, logits, entropy, int(enc["input_ids"].shape[1]), enc["input_ids"][0].tolist()

    def candidate_scores(self, prefix_ids: Sequence[int], names: Sequence[str]) -> np.ndarray:
        rows: list[list[int]] = []
        starts: list[int] = []
        for name in names:
            suffix = self.tokenizer(" " + name, add_special_tokens=False)["input_ids"]
            if not suffix:
                raise ValueError(f"candidate tokenized empty: {name!r}")
            rows.append(list(prefix_ids) + list(suffix))
            starts.append(len(prefix_ids))
        max_len = max(map(len, rows))
        pad = int(self.tokenizer.pad_token_id)
        input_ids = self.torch.full((len(rows), max_len), pad, dtype=self.torch.long)
        attention = self.torch.zeros_like(input_ids)
        for j, row in enumerate(rows):
            input_ids[j, : len(row)] = self.torch.tensor(row)
            attention[j, : len(row)] = 1
        input_ids, attention = input_ids.to(self.model.device), attention.to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True
            ).logits.float()
        scores = []
        for j, row in enumerate(rows):
            start = starts[j]
            target = input_ids[j, start : len(row)]
            pred = logits[j, start - 1 : len(row) - 1]
            logp = self.torch.log_softmax(pred, dim=-1)
            token_scores = logp.gather(1, target[:, None]).squeeze(1)
            scores.append(float(token_scores.mean().cpu()))
        return np.asarray(scores, dtype=np.float32)


def symmetric_kl(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    def log_softmax(x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float64) - float(np.max(x))
        return y - math.log(float(np.exp(y).sum()))

    la, lb = log_softmax(logits_a), log_softmax(logits_b)
    pa, pb = np.exp(la), np.exp(lb)
    return float(0.5 * (np.sum(pa * (la - lb)) + np.sum(pb * (lb - la))))


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def extract_one(
    item: ParsedItem,
    extractor: ForwardExtractor,
    item_path: Path,
    seed: int,
) -> dict[str, Any]:
    conditions = build_conditions(item, seed)
    hidden_rows, logits_rows, entropy, token_counts, scores = [], [], [], [], []
    for condition in conditions:
        hidden, logits, ent, n_tokens, prefix_ids = extractor.prompt_forward(condition.prompt)
        hidden_rows.append(hidden)
        logits_rows.append(logits)
        entropy.append(ent)
        token_counts.append(n_tokens)
        scores.append(extractor.candidate_scores(prefix_ids, [p.name for p in item.profiles]))
    logits_array = np.stack(logits_rows)
    base_idx = next(i for i, c in enumerate(conditions) if c.name == "question_only")
    full_idx = next(i for i, c in enumerate(conditions) if c.name == "full_context")
    skl_base = np.asarray([symmetric_kl(x, logits_array[base_idx]) for x in logits_array], np.float32)
    skl_full = np.asarray([symmetric_kl(x, logits_array[full_idx]) for x in logits_array], np.float32)
    metadata = {
        "key": item.key,
        "profile_names": [p.name for p in item.profiles],
        "right_answer": item.right_answer,
        "wrong_answer": item.wrong_answer,
        "right_index": [p.name for p in item.profiles].index(item.right_answer),
        "condition_names": [c.name for c in conditions],
        "condition_changed": [c.changed for c in conditions],
        "condition_notes": [c.note for c in conditions],
        "evidence_values": sorted(evidence_values(item)),
        "layers": extractor.layers,
    }
    item_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = item_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        hidden=np.stack(hidden_rows).astype(np.float32),
        candidate_scores=np.stack(scores).astype(np.float32),
        entropy=np.asarray(entropy, np.float32),
        token_counts=np.asarray(token_counts, np.int32),
        skl_to_question_only=skl_base,
        skl_to_full_context=skl_full,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    tmp.replace(item_path)
    return metadata


def command_prepare(args: argparse.Namespace) -> int:
    rows = load_rows(args.data, args.limit, args.offset)
    out = []
    failures = []
    for row in rows:
        try:
            item = parse_item(row)
            conditions = build_conditions(item, args.seed)
            out.append(
                {
                    "key": item.key,
                    "names": [p.name for p in item.profiles],
                    "right_answer": item.right_answer,
                    "evidence_values": sorted(evidence_values(item)),
                    "conditions": [
                        {
                            "name": c.name,
                            "changed": c.changed,
                            "note": c.note,
                            "prompt": c.prompt,
                        }
                        for c in conditions
                    ],
                }
            )
        except Exception as exc:  # audit malformed benchmark rows rather than hiding them
            failures.append({"key": row.get("key"), "error": repr(exc)})
    path = args.output / "prepared_conditions.json"
    json_dump(path, {"items": out, "failures": failures})
    print(f"prepared {len(out)} items, {len(failures)} failures -> {path}")
    return 0 if not failures else 1


def command_extract(args: argparse.Namespace) -> int:
    rows = load_rows(args.data, args.limit, args.offset)
    items_dir = args.output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    extractor = ForwardExtractor(args)
    config = {
        "model": args.model,
        "cache_dir": args.cache_dir,
        "data": str(args.data.resolve()),
        "offset": args.offset,
        "limit": args.limit,
        "layers": extractor.layers,
        "max_input_tokens": args.max_input_tokens,
        "forward_only": True,
        "conditions": [c.name for c in build_conditions(parse_item(rows[0]), args.seed)] if rows else [],
    }
    json_dump(args.output / "run_config.json", config)
    errors_path = args.output / "errors.jsonl"
    done = skipped = errors = 0
    for n, row in enumerate(rows, 1):
        key = str(row.get("key", f"row_{args.offset + n - 1}"))
        path = items_dir / f"{safe_name(key)}.npz"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            item = parse_item(row)
            extract_one(item, extractor, path, args.seed)
            done += 1
        except Exception as exc:
            errors += 1
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "error": repr(exc)}, ensure_ascii=False) + "\n")
        if n % args.progress_every == 0 or n == len(rows):
            print(f"[{n}/{len(rows)}] extracted={done} resumed={skipped} errors={errors}", flush=True)
    return 0 if errors == 0 else 1


def load_item_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "hidden": z["hidden"],
            "candidate_scores": z["candidate_scores"],
            "entropy": z["entropy"],
            "token_counts": z["token_counts"],
            "skl_to_question_only": z["skl_to_question_only"],
            "skl_to_full_context": z["skl_to_full_context"],
            "metadata": json.loads(str(z["metadata"].item())),
        }


def cosine_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1.0 - num / np.maximum(den, 1e-12)


def response_features(record: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = record["metadata"]["condition_names"]
    index = {name: i for i, name in enumerate(names)}
    hidden = record["hidden"].astype(np.float64)
    scores = record["candidate_scores"].astype(np.float64)
    margins = scores[:, 0] - scores[:, 1]
    entropy = record["entropy"].astype(np.float64)
    token_counts = record["token_counts"].astype(np.float64)
    full = index["full_context"]
    base = index["question_only"]
    # Most conditions are controlled modifications of full_context.  The
    # question-only condition is retained as the information-restoration axis.
    controls = {
        "full_context": base,
        "without_question_evidence": full,
        "minimal_decisive_evidence": full,
        "profile_order_swap": full,
        "attribute_order_shuffle": full,
        "structure_only_context": full,
        "negation_paraphrase": full,
        "negation_flip": full,
        "entity_paraphrase": full,
        "structured_comparison_cue": full,
    }
    values: list[float] = []
    labels: list[str] = []
    changed = dict(zip(names, record["metadata"]["condition_changed"]))
    for condition, control_name_idx in controls.items():
        if condition not in index:
            continue
        j, c = index[condition], control_name_idx
        condition_start = len(values)
        for layer_pos, layer_id in enumerate(record["metadata"]["layers"]):
            dh = hidden[j, layer_pos] - hidden[c, layer_pos]
            denom = max(float(np.linalg.norm(hidden[c, layer_pos])), 1e-12)
            values.extend(
                [
                    float(np.linalg.norm(dh) / denom),
                    float(cosine_distance_rows(hidden[j, layer_pos], hidden[c, layer_pos])),
                ]
            )
            labels.extend(
                [
                    f"{condition}|layer{layer_id}|relative_delta_norm",
                    f"{condition}|layer{layer_id}|cosine_distance",
                ]
            )
        scalar = {
            "candidate_margin_delta": margins[j] - margins[c],
            "candidate_margin_abs_delta": abs(margins[j] - margins[c]),
            "entropy_delta": entropy[j] - entropy[c],
            "token_count_delta": token_counts[j] - token_counts[c],
            "symmetric_kl": symmetric_kl_proxy(record, j, c, base, full),
            "condition_changed": float(bool(changed.get(condition, True))),
        }
        for metric, value in scalar.items():
            values.append(float(value))
            labels.append(f"{condition}|scalar|{metric}")
        if not bool(changed.get(condition, True)):
            # A rule that did not apply is missing, not evidence of zero model
            # response. Keep only the audit flag; analysis removes that flag
            # and median-imputes these response features.
            count = len(values) - 1 - condition_start
            values[condition_start : len(values) - 1] = [float("nan")] * count
    meta = record["metadata"]
    base_pred = int(np.argmax(scores[base]))
    full_pred = int(np.argmax(scores[full]))
    audit = {
        "key": meta["key"],
        "right_index": int(meta["right_index"]),
        "base_pred": base_pred,
        "full_pred": full_pred,
        "base_correct": base_pred == int(meta["right_index"]),
        "full_correct": full_pred == int(meta["right_index"]),
        "base_margin_for_profile0": float(margins[base]),
        "full_margin_for_profile0": float(margins[full]),
        "n_evidence_values": len(meta.get("evidence_values", [])),
    }
    return np.asarray(values, np.float64), labels, audit


def symmetric_kl_proxy(record: dict[str, Any], j: int, c: int, base: int, full: int) -> float:
    # Exact pairwise KLs to base/full were computed while logits were resident;
    # no vocabulary-sized arrays are persisted.  All declared controls are one
    # of these two, so this remains exact for the comparisons used here.
    if c == base:
        return float(record["skl_to_question_only"][j])
    if c == full:
        return float(record["skl_to_full_context"][j])
    raise ValueError("KL control must be question_only or full_context")


def select_audits(audits: Sequence[dict[str, Any]], selection: str) -> np.ndarray:
    if selection == "all":
        return np.ones(len(audits), dtype=bool)
    if selection == "base_wrong":
        return np.asarray([not x["base_correct"] for x in audits], bool)
    if selection == "base_wrong_full_correct":
        return np.asarray([(not x["base_correct"]) and x["full_correct"] for x in audits], bool)
    if selection == "any_failure":
        return np.asarray([not (x["base_correct"] and x["full_correct"]) for x in audits], bool)
    if selection == "full_wrong":
        return np.asarray([not x["full_correct"] for x in audits], bool)
    raise ValueError(selection)


def command_analyze(args: argparse.Namespace) -> int:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    paths = sorted((args.output / "items").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no extracted items under {args.output / 'items'}")
    rows, audits, labels = [], [], None
    for path in paths:
        values, current_labels, audit = response_features(load_item_npz(path))
        if labels is None:
            labels = current_labels
        elif labels != current_labels:
            raise ValueError(f"feature schema mismatch at {path}")
        rows.append(values)
        audits.append(audit)
    X_all = np.stack(rows)
    mask = select_audits(audits, args.selection)
    X, selected_audits = X_all[mask], [a for a, keep in zip(audits, mask) if keep]
    if len(X) < max(10, args.min_clusters * 3):
        raise ValueError(f"selection {args.selection!r} leaves only {len(X)} items")
    # Applicability and prompt length are nuisance variables, not responses.
    feature_keep = np.asarray([
        not x.endswith("|condition_changed") and not x.endswith("|token_count_delta")
        for x in labels
    ], bool)
    cluster_labels = [x for x, keep in zip(labels, feature_keep) if keep]
    X = X[:, feature_keep]
    pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = pre.fit_transform(X)
    max_pc = min(args.pca_components, Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=max_pc, random_state=args.seed)
    P = pca.fit_transform(Z)
    # Keep enough PCs for requested explained variance, with at least two.
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    use_pc = min(max(2, int(np.searchsorted(cumulative, args.pca_variance) + 1)), max_pc)
    Y = P[:, :use_pc]
    candidates = []
    upper = min(args.max_clusters, len(Y) // 3)
    for k in range(args.min_clusters, upper + 1):
        gm = GaussianMixture(n_components=k, covariance_type=args.gmm_covariance, reg_covar=1e-5,
                             n_init=10, random_state=args.seed)
        labels_k = gm.fit_predict(Y)
        sil = float(silhouette_score(Y, labels_k)) if len(set(labels_k)) > 1 else float("nan")
        candidates.append({"k": k, "bic": float(gm.bic(Y)), "silhouette": sil, "model": gm})
    chosen = min(candidates, key=lambda x: x["bic"])
    gm = chosen.pop("model")
    for row in candidates:
        row.pop("model", None)
    assignments = gm.predict(Y)
    probabilities = gm.predict_proba(Y).max(axis=1)
    rng = np.random.default_rng(args.seed)
    bootstrap_ari = []
    for b in range(args.bootstrap):
        sample = rng.integers(0, len(Y), len(Y))
        boot = GaussianMixture(
            n_components=int(chosen["k"]), covariance_type=args.gmm_covariance, reg_covar=1e-5,
            n_init=3, random_state=args.seed + b + 1,
        ).fit(Y[sample])
        bootstrap_ari.append(float(adjusted_rand_score(assignments, boot.predict(Y))))
    cluster_summaries = []
    for cluster in sorted(set(assignments.tolist())):
        idx = assignments == cluster
        # condition_changed was removed before preprocessing, so Z and
        # cluster_labels already share the same feature dimension.
        means = Z[idx].mean(axis=0)
        order = np.argsort(np.abs(means))[::-1][: args.top_features]
        cluster_summaries.append(
            {
                "cluster": int(cluster),
                "n": int(idx.sum()),
                "mean_membership_probability": float(probabilities[idx].mean()),
                "top_response_features": [
                    {"feature": cluster_labels[j], "mean_z": float(means[j])} for j in order
                ],
                "posthoc_base_correct_rate": float(np.mean([selected_audits[j]["base_correct"] for j in np.where(idx)[0]])),
                "posthoc_full_correct_rate": float(np.mean([selected_audits[j]["full_correct"] for j in np.where(idx)[0]])),
                "posthoc_mean_evidence_matches": float(np.mean([selected_audits[j]["n_evidence_values"] for j in np.where(idx)[0]])),
            }
        )
    assignments_path = args.output / "cluster_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "cluster", "membership_probability",
                                                "base_correct", "full_correct", "n_evidence_values"])
        writer.writeheader()
        for audit, cluster, prob in zip(selected_audits, assignments, probabilities):
            writer.writerow({
                "key": audit["key"], "cluster": int(cluster),
                "membership_probability": float(prob), "base_correct": audit["base_correct"],
                "full_correct": audit["full_correct"],
                "n_evidence_values": audit["n_evidence_values"],
            })
    summary = {
        "unsupervised_fit_uses_answer_labels": False,
        "selection": args.selection,
        "n_extracted": len(paths),
        "n_selected": len(X),
        "n_features": len(cluster_labels),
        "pca_components_fit": max_pc,
        "pca_components_clustered": use_pc,
        "pca_variance_clustered": float(cumulative[use_pc - 1]),
        "gmm_candidates": candidates,
        "chosen_clusters_by_bic": int(chosen["k"]),
        "bootstrap_ari_mean": float(np.mean(bootstrap_ari)),
        "bootstrap_ari_q10": float(np.quantile(bootstrap_ari, 0.1)),
        "bootstrap_ari_values": bootstrap_ari,
        "cluster_summaries": cluster_summaries,
        "interpretation_warning": (
            "Clusters are operational perturbation-response types, not proven causal failure modes. "
            "Negation flip changes semantics; entity paraphrases require audit; exact-match evidence "
            "extraction can miss paraphrased evidence."
        ),
    }
    json_dump(args.output / "analysis_summary.json", summary)
    np.savez_compressed(
        args.output / "analysis_arrays.npz", standardized_features=Z, pca_scores=P,
        assignments=assignments, membership_probability=probabilities,
        selected_keys=np.asarray([x["key"] for x in selected_audits]),
        feature_names=np.asarray(cluster_labels),
    )
    print(
        f"analyzed {len(X)}/{len(paths)} items; k={chosen['k']}, "
        f"PCs={use_pc}, bootstrap ARI mean={np.mean(bootstrap_ari):.3f}"
    )
    print(f"summary -> {args.output / 'analysis_summary.json'}")
    print(f"assignments -> {assignments_path}")
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="materialize/audit perturbation prompts; no model")
    add_common(prepare)
    prepare.set_defaults(func=command_prepare)

    extract = sub.add_parser("extract", help="run forward-only feature extraction")
    add_common(extract)
    extract.add_argument("--model", default=DEFAULT_MODEL)
    extract.add_argument("--cache-dir", default=DEFAULT_CACHE,
                         help="HF model/tokenizer cache; defaults under /tmp")
    extract.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    extract.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    extract.add_argument("--layers", default=None,
                         help="comma-separated output.hidden_states indices; default quartiles + final")
    extract.add_argument("--max-input-tokens", type=int, default=8192)
    extract.add_argument("--progress-every", type=int, default=10)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(func=command_extract)

    analyze = sub.add_parser("analyze", help="unsupervised PCA + GMM analysis")
    add_common(analyze)
    analyze.add_argument("--selection", choices=["all", "base_wrong", "base_wrong_full_correct", "any_failure", "full_wrong"],
                         default="base_wrong")
    analyze.add_argument("--pca-components", type=int, default=10)
    analyze.add_argument("--pca-variance", type=float, default=0.90)
    analyze.add_argument("--min-clusters", type=int, default=2)
    analyze.add_argument("--max-clusters", type=int, default=6)
    analyze.add_argument("--gmm-covariance", choices=["diag", "tied", "full"], default="diag")
    analyze.add_argument("--bootstrap", type=int, default=100)
    analyze.add_argument("--top-features", type=int, default=12)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("--offset must be >=0 and --limit must be positive")
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
