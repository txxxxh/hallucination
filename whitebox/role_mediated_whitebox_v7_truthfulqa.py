#!/usr/bin/env python3
"""TruthfulQA adapter for the v7 interventional multimodal detector.

The original TruthfulQA validation parquet contains one correct answer and
several incorrect answers per question, while v7 expects a binary choice task.
This adapter deterministically samples one incorrect answer and balances the
position of the correct answer, then delegates the full experiment to v7.

Example:
    CUDA_VISIBLE_DEVICES=0 python role_mediated_whitebox_v7_truthfulqa.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --data ../other_bench/truthfal_qa/validation-00000-of-00001.parquet \
      --out-dir ../truthfulqa_v7_output \
      --span-mode atomic --interventions delete,neutralize,mask \
      --max-intervention-spans 0 --dtype bfloat16

Install a parquet reader if needed:
    pip install pyarrow
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import role_mediated_whitebox_v7_interventional_multimodal as v7


def _read_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Read parquet without requiring pandas when pyarrow is available."""
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ImportError:
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
        except ImportError as exc:
            raise RuntimeError(
                "Reading TruthfulQA parquet requires pyarrow (recommended) or "
                "pandas plus a parquet engine. Install it with: pip install pyarrow"
            ) from exc


def _stable_digest(question: str) -> bytes:
    return hashlib.sha256(question.encode("utf-8")).digest()


def _adapt_record(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError(f"row {row_index}: missing question")

    best_answer = str(row.get("best_answer", "")).strip()
    correct_answers = [
        str(value).strip() for value in (row.get("correct_answers") or [])
        if str(value).strip()
    ]
    if not best_answer and correct_answers:
        best_answer = correct_answers[0]
    if not best_answer:
        raise ValueError(f"row {row_index}: missing best_answer/correct_answers")

    incorrect_answers = [
        str(value).strip() for value in (row.get("incorrect_answers") or [])
        if str(value).strip()
    ]
    if not incorrect_answers:
        raise ValueError(f"row {row_index}: no incorrect_answers")

    digest = _stable_digest(question)
    incorrect = incorrect_answers[int.from_bytes(digest[:8], "big") % len(incorrect_answers)]
    correct_first = digest[8] % 2 == 0
    options = (best_answer, incorrect) if correct_first else (incorrect, best_answer)
    gold = "1" if correct_first else "2"

    # Keep the raw question separate: interventions alter only the question,
    # while PromptAdapter safely substitutes it inside this binary-choice prompt.
    prompt = (
        f"Question: {question}\n\n"
        f"Choose the truthful answer:\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}"
    )
    return {
        "question": question,
        "prompt": prompt,
        "answer": gold,
        "truthfulqa_type": row.get("type"),
        "truthfulqa_category": row.get("category"),
        "truthfulqa_source": row.get("source"),
        "truthfulqa_best_answer": best_answer,
        "truthfulqa_sampled_incorrect_answer": incorrect,
        "truthfulqa_row_index": row_index,
    }


def read_truthfulqa_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        # Preserve v7's JSON/JSONL support for converted or smoke-test inputs.
        return v7.read_records_original(path)

    output = []
    for index, row in enumerate(_read_parquet(path)):
        try:
            output.append(_adapt_record(row, index))
        except ValueError as exc:
            raise ValueError(f"invalid TruthfulQA parquet: {exc}") from exc
    return output


v7.read_records_original = v7.read_records
v7.read_records = read_truthfulqa_records


if __name__ == "__main__":
    v7.main()
