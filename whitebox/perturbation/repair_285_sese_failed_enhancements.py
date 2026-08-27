#!/usr/bin/env python3
"""Remove fallback SeSE scores so the scorer can resume only failed enhancements."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


LOG = Path("/tmp/sese285_score.log")
SCORES = Path("/tmp/sese_official_original_benchmarks/scientist/scores.jsonl")
BACKUP = SCORES.with_name("scores.mixed_before_gpt4o_repair.jsonl")


def main() -> None:
    current = 0
    failed: set[int] = set()

    for line in LOG.read_text(errors="replace").splitlines():
        if "Enhancing answers" in line:
            if current == 1084:
                break
            current += 1
        elif "Enhancing failed" in line:
            if current == 0:
                raise RuntimeError("failure appeared before the first enhancement")
            failed.add(current)

    rows = [json.loads(line) for line in SCORES.read_text().splitlines() if line.strip()]
    if current != 1084 or len(rows) != 1084 or len(failed) != 114:
        raise RuntimeError(
            f"refusing repair: enhancements={current}, scores={len(rows)}, failures={len(failed)}"
        )

    if BACKUP.exists():
        raise FileExistsError(f"backup already exists: {BACKUP}")
    shutil.copy2(SCORES, BACKUP)

    kept = [row for index, row in enumerate(rows, 1) if index not in failed]
    temporary = SCORES.with_suffix(".jsonl.repairing")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    temporary.replace(SCORES)
    print(f"kept={len(kept)} removed_for_retry={len(failed)} backup={BACKUP}")


if __name__ == "__main__":
    main()
