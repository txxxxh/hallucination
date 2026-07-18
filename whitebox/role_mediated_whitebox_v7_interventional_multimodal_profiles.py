#!/usr/bin/env python3
"""Profile-pair adapter for the v7 interventional multimodal pipeline."""

from __future__ import annotations

import re

import role_mediated_whitebox_v7_interventional_multimodal as v7


PROFILE_NAME_RE = re.compile(r"(?im)^\s*name:\s*(?P<name>.+?)\s*$")
PROFILE_QUESTION_RE = re.compile(
    r"(?is)Choose\s+exactly\s+one\s+profile\s+from\s+the\s+two,.*?"
    r"following\s+question:\s*"
)
PROFILE_MAPPING_RE = re.compile(r"(?im)^\s*Profile number mapping:\s*$")


def profile_names(prompt: str) -> tuple[str, str]:
    names: list[str] = []
    for match in PROFILE_NAME_RE.finditer(prompt):
        name = match.group("name").strip()
        if name not in names:
            names.append(name)
        if len(names) == 2:
            break
    if len(names) != 2:
        raise ValueError(f"expected two profile names, found {names!r}")
    return names[0], names[1]


def add_profile_number_mapping(prompt: str) -> str:
    if PROFILE_MAPPING_RE.search(prompt):
        return prompt
    marker = PROFILE_QUESTION_RE.search(prompt)
    if marker is None:
        raise ValueError("profile question introduction was not found")
    first, second = profile_names(prompt)
    mapping = f"Profile number mapping:\n1. {first}\n2. {second}\n"
    return prompt[: marker.start()] + mapping + prompt[marker.start() :]


def read_profile_records(path: str) -> list[dict]:
    records = v7.read_records_original(path)
    output: list[dict] = []
    for item in records:
        adapted = dict(item)
        adapted["prompt"] = add_profile_number_mapping(str(adapted["prompt"]))
        output.append(adapted)
    return output


def profile_question_body_bounds(text: str) -> tuple[int, int] | None:
    marker = PROFILE_QUESTION_RE.search(text)
    if marker is None:
        return v7.scientistqa_body_bounds_original(text)
    body_start = marker.end()
    final_match = v7.FINAL_PERSON_QUESTION_RE.search(text, body_start)
    body_end = final_match.start() if final_match is not None else len(text)
    body_start, body_end = v7._trim_span(text, body_start, body_end)
    return (body_start, body_end) if body_start < body_end else None


v7.read_records_original = v7.read_records
v7.scientistqa_body_bounds_original = v7.scientistqa_body_bounds
v7.read_records = read_profile_records
v7.scientistqa_body_bounds = profile_question_body_bounds


if __name__ == "__main__":
    v7.main()
