#!/usr/bin/env python3
"""ScientistQA profile-pair adapter for the atomic role-mediated v5 pipeline.

The profile dataset names two people with repeated ``name:`` fields but does
not number them.  This entry point adds an explicit 1/2 scoring map in profile
order, maps ``rgt_ans`` through that map, and restricts candidate spans to the
final question description.  All feature extraction, weak supervision,
training, held-out prediction, and causal audit are inherited unchanged from
``role_mediated_whitebox_v5_scientistqa_atomic``.
"""

from __future__ import annotations

import re
import json
import sys
from pathlib import Path

import role_mediated_whitebox_v5_scientistqa_atomic as v5


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
    """Make the binary label semantics explicit without altering profiles."""
    if PROFILE_MAPPING_RE.search(prompt):
        return prompt
    marker = PROFILE_QUESTION_RE.search(prompt)
    if marker is None:
        raise ValueError("profile question introduction was not found")
    first, second = profile_names(prompt)
    mapping = (
        "Profile number mapping:\n"
        f"1. {first}\n"
        f"2. {second}\n"
    )
    return prompt[: marker.start()] + mapping + prompt[marker.start() :]


def read_profile_records(path: str | Path) -> list[dict]:
    records = v5.read_records_original(path)
    output: list[dict] = []
    for item in records:
        adapted = dict(item)
        if "prompt" not in adapted:
            raise KeyError("profiles input requires a 'prompt' field")
        adapted["prompt"] = add_profile_number_mapping(str(adapted["prompt"]))
        output.append(adapted)
    return output


def profile_question_body_bounds(text: str) -> tuple[int, int] | None:
    """Return only the final descriptive question, excluding both profiles."""
    marker = PROFILE_QUESTION_RE.search(text)
    if marker is None:
        return v5.scientistqa_body_bounds_original(text)
    body_start = marker.end()
    final_match = v5.FINAL_PERSON_QUESTION_RE.search(text, body_start)
    body_end = final_match.start() if final_match is not None else len(text)
    body_start, body_end = v5._trim_span(text, body_start, body_end)
    return (body_start, body_end) if body_start < body_end else None


# Retain explicit handles so the adapters can safely delegate on other input.
v5.read_records_original = v5.read_records
v5.scientistqa_body_bounds_original = v5.scientistqa_body_bounds
v5.read_records = read_profile_records
v5.scientistqa_body_bounds = profile_question_body_bounds


def update_profiles_summary_metadata() -> None:
    out_dir = "role_mediated_output"
    if "--out-dir" in sys.argv:
        pos = sys.argv.index("--out-dir")
        if pos + 1 < len(sys.argv):
            out_dir = sys.argv[pos + 1]
    path = Path(out_dir) / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    notes = summary.setdefault("method_notes", {})
    notes["scientistqa_input_variant"] = "two full structured profiles"
    notes["scientistqa_candidate_region"] = (
        "after 'following question:' and before final 'Who is this person?'"
    )
    notes["profile_number_mapping"] = (
        "1=first name field; 2=second name field; mapping inserted before question"
    )
    notes["profile_fields_are_intervened"] = False
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=v5.json_default),
        encoding="utf-8",
    )


if __name__ == "__main__":
    v5.main()
    update_profiles_summary_metadata()
