#!/usr/bin/env python3
"""
Span-centered ScientistQA hallucination router (v3).

This version keeps v2's knowledge-probe teacher and intervention labels, but
changes the learned representation:

* the primary cause head uses only base-answer and isolated-probe hidden
  states; probe accuracy and intervention-presence features are excluded;
* every candidate span is represented by mean/max/first/last token states;
* option-1/option-2 and model-chosen answer relations are computed without
  using which option is gold;
* an optional full-context span readout asks the model which option the
  specified phrase supports;
* mask/neutralize receive aligned span-state deltas, while deletion is used
  only for answer/behavior transitions because the deleted span has no aligned
  post-state;
* answer transitions are retained separately for each intervention operator.

Gold correctness and intervention recovery create teacher labels only.  They
are never classifier inputs.  The aligned profile file remains teacher
supervision for constructing and scoring isolated knowledge probes.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import stressor_interventions_v2 as v2


LOGGER = logging.getLogger("stressor_interventions_v3")
SCHEMA_VERSION = "scientist_stressor_interventions_v3_span_centered"
POOL_NAMES = ("mean", "max", "first", "last")
ALIGNED_OPERATORS = ("neutralize", "mask")


class SpanCenteredEngine(v2.ScientistEngine):
    @torch.inference_mode()
    def extract_rich_trace(
        self,
        user_prompt: str,
        answer: str,
        spans: Sequence[tuple[int, int]],
        layer_spec: str,
    ) -> dict[str, Any]:
        """Extract answer-last and mean/max/first/last span states."""
        formatted = self.format_chat(user_prompt)
        answer = answer.strip() or "[NO ANSWER]"
        encoded = self.tokenizer(
            formatted + answer,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        ids = encoded["input_ids"].to(self.input_device)
        mask = encoded["attention_mask"].to(self.input_device)
        outputs = self.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        layers = self.trace_layers(layer_spec)
        answer_start = len(formatted)
        answer_tokens = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and end > answer_start
        ] or [ids.shape[1] - 1]
        answer_hidden = torch.stack(
            [
                outputs.hidden_states[layer][0, answer_tokens[-1]]
                .detach()
                .to("cpu", dtype=torch.float16)
                for layer in layers
            ]
        )
        user_start = formatted.find(user_prompt)
        if user_start < 0:
            raise RuntimeError("Could not locate user prompt inside chat template")

        pooled_spans: list[torch.Tensor] = []
        token_counts: list[int] = []
        for start, end in spans:
            absolute = (user_start + start, user_start + end)
            indices = [
                index
                for index, (left, right) in enumerate(offsets)
                if right > left and max(left, absolute[0]) < min(right, absolute[1])
            ]
            token_counts.append(len(indices))
            if not indices:
                pooled_spans.append(
                    torch.zeros(
                        (len(layers), len(POOL_NAMES), answer_hidden.shape[-1]),
                        dtype=torch.float16,
                    )
                )
                continue
            per_layer: list[torch.Tensor] = []
            for layer in layers:
                states = outputs.hidden_states[layer][0, indices].float()
                pools = torch.stack(
                    [states.mean(0), states.amax(0), states[0], states[-1]]
                )
                per_layer.append(pools.detach().to("cpu", dtype=torch.float16))
            pooled_spans.append(torch.stack(per_layer))
        width = answer_hidden.shape[-1]
        return {
            "layer_indices": layers,
            "answer_hidden": answer_hidden,
            "span_pools": (
                torch.stack(pooled_spans)
                if pooled_spans
                else torch.empty(
                    (0, len(layers), len(POOL_NAMES), width), dtype=torch.float16
                )
            ),
            "span_token_counts": token_counts,
            "answer_token_count": len(answer_tokens),
            "sequence_token_count": int(ids.shape[1]),
        }


def option_from_generation(text: str, example: v2.ScientistExample) -> Optional[str]:
    chosen = v2.parse_chosen_name(text, example)
    for option, name in example.option_map.items():
        if chosen == name:
            return option
    match = re.search(r"\b(?:option|answer|choice)?\s*([12])\b", text, re.I)
    return match.group(1) if match else None


def relation_features(
    span_pools: torch.Tensor,
    answer_hidden: torch.Tensor,
    option1_hidden: torch.Tensor,
    option2_hidden: torch.Tensor,
    chosen_option: Optional[str],
) -> torch.Tensor:
    """Layerwise scalar relations, with no gold-oriented answer direction."""
    pools = span_pools.float()
    vectors = {
        "answer": answer_hidden.float(),
        "option1": option1_hidden.float(),
        "option2": option2_hidden.float(),
    }
    chosen = (
        option1_hidden.float()
        if chosen_option == "1"
        else option2_hidden.float()
        if chosen_option == "2"
        else answer_hidden.float()
    )
    other = (
        option2_hidden.float()
        if chosen_option == "1"
        else option1_hidden.float()
        if chosen_option == "2"
        else (option1_hidden.float() + option2_hidden.float()) / 2
    )
    vectors["chosen_direction"] = chosen - other
    features = []
    for vector in vectors.values():
        expanded = vector[:, None, :].expand_as(pools)
        features.append(F.cosine_similarity(pools, expanded, dim=-1))
    return torch.cat(features, dim=-1).to(torch.float16)


def readout_prompt(example: v2.ScientistExample, span: v2.Span) -> str:
    return (
        f"{example.prompt}\n\n"
        "Analyze only the following phrase as evidence in the complete question:\n"
        f'PHRASE: "{span.text}"\n'
        "Which of the two numbered options does this phrase support more strongly? "
        "Answer exactly Option 1, Option 2, or Neither."
    )


class SpanCenteredCollector(v2.Collector):
    engine: SpanCenteredEngine

    def run_probes_rich(
        self, example: v2.ScientistExample
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run probes and preserve option-1/option-2 identity in hidden stats."""
        probes = v2.make_fact_probes(
            example, self.args.probes_per_person
        )
        rows: list[dict[str, Any]] = []
        hidden: list[torch.Tensor] = []
        hidden_by_person: dict[str, list[torch.Tensor]] = {
            name: [] for name in example.option_map.values()
        }
        for index, probe in enumerate(probes):
            prompt = v2.probe_prompt(probe)
            generation = self.generate(
                prompt, self.args.seed + 1000 + index, 8
            )
            parsed = v2.parse_yes_no(generation.text)
            trace = self.engine.extract_trace(
                prompt, generation.text, [], self.args.hidden_layers
            )
            state = trace["answer_hidden"]
            hidden.append(state)
            hidden_by_person.setdefault(probe.person, []).append(state)
            rows.append(
                {
                    **dataclasses.asdict(probe),
                    "prompt": prompt,
                    "answer": generation.text,
                    "parsed_yes": parsed,
                    "correct": (
                        parsed is not None and parsed == probe.expected_yes
                    ),
                    "parse_valid": parsed is not None,
                    "mean_token_logprob": generation.mean_token_logprob,
                }
            )
        layer_count = len(
            self.engine.trace_layers(self.args.hidden_layers)
        )
        width = int(self.engine.model.config.hidden_size)
        stack = (
            torch.stack(hidden)
            if hidden
            else torch.zeros(
                (0, layer_count, width), dtype=torch.float16
            )
        )
        person_scores: dict[str, Optional[float]] = {}
        for person in example.option_map.values():
            valid = [
                row
                for row in rows
                if row["person"] == person and row["parse_valid"]
            ]
            person_scores[person] = (
                float(np.mean([row["correct"] for row in valid]))
                if valid
                else None
            )
        valid_scores = [
            score for score in person_scores.values() if score is not None
        ]
        score = min(valid_scores) if len(valid_scores) == 2 else None
        if score is None:
            knowledge_state = "ambiguous"
        elif score >= self.args.knowledge_known_threshold:
            knowledge_state = "known"
        elif score <= self.args.knowledge_gap_threshold:
            knowledge_state = "unknown"
        else:
            knowledge_state = "ambiguous"

        # Store by visible option order, never by gold right/wrong order.
        person_stats: list[torch.Tensor] = []
        for option in ("1", "2"):
            person = example.option_map[option]
            person_stack = torch.stack(hidden_by_person.get(person, []))
            person_mean = person_stack.mean(0)
            person_std = (
                person_stack.float().std(0).to(torch.float16)
                if len(person_stack) > 1
                else torch.zeros_like(person_mean)
            )
            person_stats.append(
                torch.stack([person_mean, person_std])
            )
        return (
            {
                "state": knowledge_state,
                "score": score,
                "person_scores": person_scores,
                "n_probes": len(rows),
                "n_valid": sum(row["parse_valid"] for row in rows),
                "probes": rows,
                "hidden_person_order": [
                    example.option_map["1"],
                    example.option_map["2"],
                ],
            },
            (
                stack.mean(0)
                if len(stack)
                else torch.zeros(
                    (layer_count, width), dtype=torch.float16
                )
            ),
            (
                stack.float().std(0).to(torch.float16)
                if len(stack) > 1
                else torch.zeros(
                    (layer_count, width), dtype=torch.float16
                )
            ),
            torch.stack(person_stats),
        )

    def collect_one(self, example: v2.ScientistExample) -> dict[str, Any]:
        seed = self.args.seed + example.source_index * 1009
        generation = self.generate(example.prompt, seed)
        chosen = v2.parse_chosen_name(generation.text, example)
        chosen_option = option_from_generation(generation.text, example)
        base_correct = chosen == example.right_name
        all_spans = v2.propose_atomic_spans(
            example.prompt, self.args.min_span_words
        )
        base_trace = self.engine.extract_rich_trace(
            example.prompt,
            generation.text,
            [(span.start, span.end) for span in all_spans],
            self.args.hidden_layers,
        )
        (
            probe_info,
            probe_mean,
            probe_std,
            probe_person_stats,
        ) = self.run_probes_rich(example)

        option1_trace = self.engine.extract_trace(
            example.prompt,
            example.option_map["1"],
            [],
            self.args.hidden_layers,
        )
        option2_trace = self.engine.extract_trace(
            example.prompt,
            example.option_map["2"],
            [],
            self.args.hidden_layers,
        )
        option1_hidden = option1_trace["answer_hidden"]
        option2_hidden = option2_trace["answer_hidden"]

        rank_slot = max(
            0,
            min(len(base_trace["layer_indices"]) - 1, self.args.rank_layer_slot),
        )
        span_similarity: list[tuple[float, v2.Span]] = []
        for index, span in enumerate(all_spans):
            mean_state = base_trace["span_pools"][index, rank_slot, 0].float()
            similarity = float(
                F.cosine_similarity(
                    mean_state,
                    base_trace["answer_hidden"][rank_slot].float(),
                    dim=0,
                ).item()
            )
            span_similarity.append((similarity, span))
        ranked = sorted(span_similarity, key=lambda row: (-row[0], row[1].start))
        mandatory = [
            span
            for _, span in ranked
            if span.span_type == "negation" or v2.NEGATION_RE.search(span.text)
        ]
        budget = (
            len(all_spans)
            if self.args.max_intervention_spans <= 0
            else self.args.max_intervention_spans
        )
        selected: list[v2.Span] = []
        selected_indices: set[int] = set()
        for span in mandatory + [row[1] for row in ranked]:
            if span.index not in selected_indices:
                selected.append(span)
                selected_indices.add(span.index)
            if len(selected) >= budget:
                break
        controls = [
            row[1]
            for row in sorted(span_similarity, key=lambda row: (row[0], row[1].start))
            if row[1].index not in selected_indices
        ]
        control_span = controls[0] if controls else None
        should_intervene = (
            not base_correct and probe_info["state"] == "known" and bool(selected)
        )
        control_recovery = 0.0
        if should_intervene and control_span is not None:
            control_prompt, _ = v2.apply_intervention(
                example.prompt, control_span, "delete"
            )
            control_generation = self.generate(control_prompt, seed + 7000)
            control_recovery = float(
                v2.parse_chosen_name(control_generation.text, example)
                == example.right_name
            )

        span_lookup = {span.index: index for index, span in enumerate(all_spans)}
        span_traces: list[dict[str, Any]] = []
        if should_intervene:
            for span_rank, span in enumerate(selected):
                base_pools = base_trace["span_pools"][
                    span_lookup[span.index]
                ]
                relations = relation_features(
                    base_pools,
                    base_trace["answer_hidden"],
                    option1_hidden,
                    option2_hidden,
                    chosen_option,
                )
                readout_hidden = torch.zeros_like(base_trace["answer_hidden"])
                readout_row: dict[str, Any] = {
                    "enabled": bool(self.args.span_readout),
                    "answer": None,
                    "chosen_option": None,
                }
                if self.args.span_readout:
                    local_prompt = readout_prompt(example, span)
                    local_generation = self.generate(
                        local_prompt,
                        seed + 7500 + span_rank,
                        self.args.span_readout_max_tokens,
                    )
                    local_trace = self.engine.extract_trace(
                        local_prompt,
                        local_generation.text,
                        [],
                        self.args.hidden_layers,
                    )
                    readout_hidden = local_trace["answer_hidden"]
                    readout_row.update(
                        {
                            "answer": local_generation.text,
                            "chosen_option": option_from_generation(
                                local_generation.text, example
                            ),
                            "mean_token_logprob": local_generation.mean_token_logprob,
                        }
                    )

                fixed_deltas: list[torch.Tensor] = []
                regenerated_deltas: list[torch.Tensor] = []
                aligned_deltas: list[torch.Tensor] = []
                operator_rows: list[dict[str, Any]] = []
                for operator_index, operator in enumerate(v2.OPERATORS):
                    modified, replacement_bounds = v2.apply_intervention(
                        example.prompt, span, operator
                    )
                    variant_generation = self.generate(
                        modified,
                        seed + 8000 + span_rank * 101 + operator_index,
                    )
                    if operator in ALIGNED_OPERATORS:
                        fixed_trace = self.engine.extract_rich_trace(
                            modified,
                            generation.text,
                            [replacement_bounds],
                            self.args.hidden_layers,
                        )
                        replacement_pools = fixed_trace["span_pools"][0]
                        aligned_deltas.append(replacement_pools - base_pools)
                        fixed_answer_hidden = fixed_trace["answer_hidden"]
                    else:
                        fixed_trace = self.engine.extract_trace(
                            modified,
                            generation.text,
                            [],
                            self.args.hidden_layers,
                        )
                        fixed_answer_hidden = fixed_trace["answer_hidden"]
                    regenerated_trace = self.engine.extract_trace(
                        modified,
                        variant_generation.text,
                        [],
                        self.args.hidden_layers,
                    )
                    fixed_deltas.append(
                        fixed_answer_hidden - base_trace["answer_hidden"]
                    )
                    regenerated_deltas.append(
                        regenerated_trace["answer_hidden"]
                        - base_trace["answer_hidden"]
                    )
                    new_choice = v2.parse_chosen_name(
                        variant_generation.text, example
                    )
                    operator_rows.append(
                        {
                            "operator": operator,
                            "answer": variant_generation.text,
                            "chosen_name": new_choice,
                            "answer_changed": (
                                v2.canonical(variant_generation.text)
                                != v2.canonical(generation.text)
                            ),
                            "recovered_correct": new_choice == example.right_name,
                            "mean_token_logprob": (
                                variant_generation.mean_token_logprob
                            ),
                        }
                    )
                recovery_rate = float(
                    np.mean([row["recovered_correct"] for row in operator_rows])
                )
                specificity = recovery_rate - control_recovery
                culprit = (
                    recovery_rate >= self.args.span_recovery_threshold
                    and specificity >= self.args.span_specificity_threshold
                )
                aligned_stack = torch.stack(aligned_deltas)
                span_traces.append(
                    {
                        "span": dataclasses.asdict(span),
                        "similarity_rank": span_rank + 1,
                        "operators": operator_rows,
                        "readout": readout_row,
                        "base_span_pools": base_pools,
                        "span_answer_relations": relations,
                        "span_readout_hidden": readout_hidden,
                        "fixed_answer_delta_by_operator": torch.stack(fixed_deltas),
                        "regen_answer_delta_by_operator": torch.stack(
                            regenerated_deltas
                        ),
                        "aligned_span_delta_mean": aligned_stack.mean(0),
                        "aligned_span_delta_maxabs": aligned_stack.abs().amax(0),
                        "teacher": {
                            "recovery_rate": recovery_rate,
                            "control_recovery": control_recovery,
                            "specificity": specificity,
                            "culprit": culprit,
                        },
                    }
                )

        if base_correct:
            teacher_cause = "correct"
        elif probe_info["state"] == "unknown":
            teacher_cause = "knowledge_gap"
        elif probe_info["state"] == "ambiguous":
            teacher_cause = "ambiguous"
        elif any(row["teacher"]["culprit"] for row in span_traces):
            teacher_cause = "contextual_interference"
        else:
            teacher_cause = "known_but_unlocalized"

        trace_path = self.trace_dir / f"{example.item_id}.pt"
        tensor_payload = {
            "schema_version": SCHEMA_VERSION,
            "item_id": example.item_id,
            "layer_indices": base_trace["layer_indices"],
            "base_answer_hidden": base_trace["answer_hidden"],
            "probe_hidden_mean": probe_mean,
            "probe_hidden_std": probe_std,
            "probe_person_hidden_stats": probe_person_stats,
            "option1_answer_hidden": option1_hidden,
            "option2_answer_hidden": option2_hidden,
            "span_traces": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"operators", "teacher", "readout"}
                }
                for row in span_traces
            ],
        }
        v2.atomic_torch(trace_path, tensor_payload)
        return {
            "schema_version": SCHEMA_VERSION,
            "id": example.item_id,
            "source_index": example.source_index,
            "base": {
                "answer": generation.text,
                "chosen_name": chosen,
                "chosen_option": chosen_option,
                "correct": base_correct,
                "parse_valid": chosen is not None,
                "mean_token_logprob": generation.mean_token_logprob,
            },
            "knowledge_probe": probe_info,
            "interventions_run": should_intervene,
            "n_candidate_spans": len(all_spans),
            "selected_spans": [
                {
                    "span": row["span"],
                    "similarity_rank": row["similarity_rank"],
                    "operators": row["operators"],
                    "readout": row["readout"],
                    "teacher": row["teacher"],
                }
                for row in span_traces
            ],
            "control_span": (
                dataclasses.asdict(control_span) if control_span else None
            ),
            "control_recovery": control_recovery if should_intervene else None,
            "teacher_cause": teacher_cause,
            "trace_path": str(trace_path),
            "supervision": {
                "right_name": example.right_name,
                "wrong_name": example.wrong_name,
                "right_qid": example.right_qid,
                "wrong_qid": example.wrong_qid,
            },
        }


def collect(
    args: argparse.Namespace,
    examples: Sequence[v2.ScientistExample],
    out: Path,
) -> None:
    records_path = out / "teacher_records.jsonl"
    if args.force and records_path.exists():
        records_path.unlink()
    done = (
        v2.completed_ids(records_path)
        if args.resume and not args.force
        else set()
    )
    engine = SpanCenteredEngine(
        args.model,
        args.device,
        args.dtype,
        args.max_input_tokens,
        args.quantize_4bit,
        args.trust_remote_code,
    )
    collector = SpanCenteredCollector(engine, args, out)
    for example in tqdm(examples, desc="Scientist span-centered collection"):
        if example.item_id in done:
            continue
        try:
            v2.append_jsonl(records_path, collector.collect_one(example))
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted; completed records are preserved")
            break
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Failed %s", example.item_id)
            v2.append_jsonl(
                out / "errors.jsonl",
                {
                    "id": example.item_id,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del collector, engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def item_vector(
    record: Mapping[str, Any],
    trace: Mapping[str, Any],
    layer: int,
    mode: str,
) -> np.ndarray:
    base = trace["base_answer_hidden"][layer].float().numpy()
    if mode == "hallucination":
        confidence = float(record["base"].get("mean_token_logprob") or 0.0)
        return np.concatenate([base, np.asarray([confidence], np.float32)])
    return np.concatenate(
        [
            base,
            trace["probe_person_hidden_stats"][:, :, layer]
            .float()
            .numpy()
            .reshape(-1),
        ]
    )


def span_vector(row: Mapping[str, Any], layer: int) -> np.ndarray:
    parts = [
        row["base_span_pools"][layer].float().numpy().reshape(-1),
        row["span_answer_relations"][layer].float().numpy().reshape(-1),
        row["span_readout_hidden"][layer].float().numpy(),
        row["fixed_answer_delta_by_operator"][:, layer]
        .float()
        .numpy()
        .reshape(-1),
        row["regen_answer_delta_by_operator"][:, layer]
        .float()
        .numpy()
        .reshape(-1),
        row["aligned_span_delta_mean"][layer].float().numpy().reshape(-1),
        row["aligned_span_delta_maxabs"][layer]
        .float()
        .numpy()
        .reshape(-1),
    ]
    return np.concatenate(parts)


def fit_span_head(
    records: Sequence[Mapping[str, Any]],
    select_k: int,
    seed: int,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    item_ids: list[str] = []
    layer_indices: Optional[list[int]] = None
    for record in records:
        trace = v2.load_trace(record)
        layer_indices = trace["layer_indices"]
        teachers = {
            int(row["span"]["index"]): bool(row["teacher"]["culprit"])
            for row in record.get("selected_spans", [])
        }
        for row in trace.get("span_traces", []):
            index = int(row["span"]["index"])
            if index in teachers:
                rows.append(row)
                labels.append(int(teachers[index]))
                item_ids.append(str(record["id"]))
    if not rows or len(set(labels)) < 2 or layer_indices is None:
        return None, None

    unique_items = np.asarray(sorted(set(item_ids)), dtype=object)
    train_items, test_items = train_test_split(
        unique_items, test_size=0.2, random_state=seed
    )
    train_items, validation_items = train_test_split(
        train_items, test_size=0.125, random_state=seed
    )
    item_array = np.asarray(item_ids, dtype=object)
    train_indices = np.flatnonzero(np.isin(item_array, train_items))
    validation_indices = np.flatnonzero(
        np.isin(item_array, validation_items)
    )
    test_indices = np.flatnonzero(np.isin(item_array, test_items))
    labels_array = np.asarray(labels, dtype=np.int64)
    best_slot, best_score = 0, -np.inf
    layer_selection: dict[str, Any] = {}
    for slot, model_layer in enumerate(layer_indices):
        features = np.stack([span_vector(row, slot) for row in rows])
        pipeline = v2.make_pipeline(features.shape[1], select_k, seed)
        pipeline.fit(features[train_indices], labels_array[train_indices])
        predictions = pipeline.predict(features[validation_indices])
        probabilities = pipeline.predict_proba(features[validation_indices])
        score = float(
            f1_score(
                labels_array[validation_indices],
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        layer_selection[str(model_layer)] = v2.metrics(
            labels_array[validation_indices],
            predictions,
            probabilities,
            pipeline.classes_,
        )
        if score > best_score:
            best_slot, best_score = slot, score
    features = np.stack([span_vector(row, best_slot) for row in rows])
    final = v2.make_pipeline(features.shape[1], select_k, seed)
    final.fit(
        features[np.concatenate([train_indices, validation_indices])],
        labels_array[np.concatenate([train_indices, validation_indices])],
    )
    predictions = final.predict(features[test_indices])
    probabilities = final.predict_proba(features[test_indices])
    result = {
        "n_items": len(unique_items),
        "n_spans": len(rows),
        "class_counts": dict(Counter(labels)),
        "selected_layer": int(layer_indices[best_slot]),
        "selected_slot": best_slot,
        "layer_selection": layer_selection,
        "test": v2.metrics(
            labels_array[test_indices],
            predictions,
            probabilities,
            final.classes_,
        ),
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "mode": "span_culprit",
        "model": final,
        "selected_layer": int(layer_indices[best_slot]),
        "selected_slot": best_slot,
        "pool_names": POOL_NAMES,
        "operator_order": v2.OPERATORS,
        "aligned_operators": ALIGNED_OPERATORS,
        "feature_contract": (
            "mean/max/first/last span states + non-gold option relations + "
            "full-context span readout + operator-specific answer transitions + "
            "aligned neutralize/mask span transitions"
        ),
    }
    return result, bundle


def train(args: argparse.Namespace, out: Path) -> None:
    records = [
        row
        for row in v2.read_jsonl(out / "teacher_records.jsonl")
        if row.get("trace_path") and Path(row["trace_path"]).exists()
    ]
    if not records:
        raise RuntimeError("No collected records with traces")
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_records": len(records),
        "teacher_cause_counts": dict(
            Counter(row["teacher_cause"] for row in records)
        ),
        "cause_feature_contract": (
            "base answer hidden + option-ordered per-person probe hidden "
            "mean/std only; no probe scores, intervention-presence indicators, "
            "or correctness/recovery labels; probe prompts remain "
            "profile-supervised"
        ),
    }

    original_item_vector = v2.item_vector
    v2.item_vector = item_vector
    try:
        hallucination_records = [
            row for row in records if row["base"].get("parse_valid")
        ]
        hallucination_labels = [
            "hallucination" if not row["base"]["correct"] else "correct"
            for row in hallucination_records
        ]
        if (
            len(set(hallucination_labels)) >= 2
            and min(Counter(hallucination_labels).values()) >= 3
        ):
            result, bundle = v2.fit_layer_scanned_head(
                hallucination_records,
                hallucination_labels,
                "hallucination",
                args.select_k,
                args.seed,
            )
            bundle["schema_version"] = SCHEMA_VERSION
            results["hallucination_head"] = result
            joblib.dump(bundle, model_dir / "hallucination_head.joblib")
        else:
            results["hallucination_head"] = {
                "skipped": "insufficient class coverage"
            }

        cause_records = [
            row
            for row in records
            if row["teacher_cause"] in v2.CAUSE_LABELS
        ]
        cause_labels = [row["teacher_cause"] for row in cause_records]
        if (
            len(set(cause_labels)) >= 2
            and min(Counter(cause_labels).values()) >= 3
        ):
            result, bundle = v2.fit_layer_scanned_head(
                cause_records,
                cause_labels,
                "cause",
                args.select_k,
                args.seed,
            )
            bundle.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "feature_contract": (
                        "base answer hidden + option-ordered per-person probe "
                        "hidden mean/std; "
                        "no explicit probe scores or intervention features"
                    ),
                }
            )
            results["cause_head"] = result
            joblib.dump(bundle, model_dir / "cause_head.joblib")
        else:
            results["cause_head"] = {
                "skipped": (
                    "need >=3 knowledge_gap and >=3 "
                    "contextual_interference labels"
                ),
                "class_counts": dict(Counter(cause_labels)),
            }
    finally:
        v2.item_vector = original_item_vector

    span_result, span_bundle = fit_span_head(
        records, args.span_select_k, args.seed
    )
    results["span_culprit_head"] = (
        span_result
        or {"skipped": "insufficient positive/negative localized spans"}
    )
    if span_bundle is not None:
        joblib.dump(span_bundle, model_dir / "span_culprit_head.joblib")
    v2.atomic_json(out / "training_summary.json", results)


def summarize(out: Path) -> None:
    rows = v2.read_jsonl(out / "teacher_records.jsonl")
    v2.atomic_json(
        out / "collection_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "n_records": len(rows),
            "base_correct": sum(row["base"]["correct"] for row in rows),
            "base_unparsed": sum(
                not row["base"]["parse_valid"] for row in rows
            ),
            "teacher_cause_counts": dict(
                Counter(row["teacher_cause"] for row in rows)
            ),
            "interventions_run": sum(
                row["interventions_run"] for row in rows
            ),
            "localized_culprit_spans": sum(
                span["teacher"]["culprit"]
                for row in rows
                for span in row.get("selected_spans", [])
            ),
            "span_representation": {
                "pooling": POOL_NAMES,
                "readout": "full-context option-support readout",
                "aligned_span_operators": ALIGNED_OPERATORS,
                "delete_policy": (
                    "answer/behavior transition only; no fake span post-state"
                ),
                "gold_oriented_option_features": False,
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = v2.build_parser()
    parser.description = (
        "ScientistQA span-centered hidden-state hallucination router v3"
    )
    parser.set_defaults(select_k=128, max_intervention_spans=5)
    parser.add_argument(
        "--span-readout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a full-context, phrase-specific option-support readout",
    )
    parser.add_argument(
        "--span-readout-max-tokens",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--span-select-k",
        type=int,
        default=256,
        help="Selected coordinates for the high-dimensional span head",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    v2.seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    v2.atomic_json(out / "config.json", vars(args))
    examples = v2.load_examples(
        Path(args.input),
        Path(args.profiles_data),
        args.max_samples,
        args.start_index,
    )
    if not examples:
        raise RuntimeError("No aligned ScientistQA examples")
    if args.stage in {"collect", "all"}:
        collect(args, examples, out)
        summarize(out)
    if args.stage in {"train", "all"}:
        train(args, out)
    LOGGER.info("Output: %s", out)


if __name__ == "__main__":
    main()
