"""Merge freshly rerun Z1 matrix rows with the preserved Z2/Z4/Z6 rows."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(current, preserved, expected_z1, expected_treatments):
    current = Path(current)
    preserved = Path(preserved)
    fresh = read_jsonl(current)
    old = read_jsonl(preserved)

    z1 = [row for row in fresh if row["stressor"] == "Z1"]
    counts = Counter(row["treatment"] for row in z1)
    if len(z1) != expected_z1 * expected_treatments:
        raise ValueError(
            f"incomplete fresh Z1 matrix: {len(z1)} rows, "
            f"expected {expected_z1 * expected_treatments}; counts={dict(counts)}"
        )
    if len(counts) != expected_treatments or set(counts.values()) != {expected_z1}:
        raise ValueError(f"incomplete Z1 treatment coverage: {dict(counts)}")

    retained = [row for row in old if row["stressor"] != "Z1"]
    merged = z1 + retained
    tmp = current.with_suffix(current.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(current)
    print(f"[merge] fresh Z1={len(z1)}, retained non-Z1={len(retained)}, total={len(merged)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--preserved", required=True)
    ap.add_argument("--expected-z1", type=int, required=True)
    ap.add_argument("--expected-treatments", type=int, default=9)
    args = ap.parse_args()
    main(args.current, args.preserved, args.expected_z1, args.expected_treatments)
