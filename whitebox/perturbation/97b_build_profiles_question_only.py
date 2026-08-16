#!/usr/bin/env python3
"""Represent profiles as fixed prefix and expose only the question to perturbation."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "shuffled_prepend_profiles_question.json"
OUT = HERE / "runs/97_profiles_question_only_data.json"
MARKER = ("\nChoose exactly one profile from the two, and output the name of the person "
          "as the answer to the following question:\n")

rows = []
for row in json.load(open(SOURCE)):
    head, sep, question = row["prompt"].partition(MARKER)
    if not sep:
        raise ValueError(f"missing profiles/question marker for {row['key']}")
    rows.append(dict(row, context_prefix=head + MARKER, context=question,
                     question="Which of the two profiles answers the question?"))
json.dump(rows, open(OUT, "w"), ensure_ascii=False)
print(f"wrote {len(rows)} rows to {OUT}")
