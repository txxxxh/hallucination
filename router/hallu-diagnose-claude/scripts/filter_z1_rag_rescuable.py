"""Create an explicitly outcome-conditioned Z1 pool from a completed matrix run.

This is a construction/screening utility, not an unbiased treatment evaluation.
It keeps every strict T-RAG success and a stratified deterministic sample of
failures, while backing up the original pool and writing an audit manifest.
"""
import argparse
import collections
import hashlib
import json
import math
import shutil
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main(zpath, result_path, target_rate, backup, manifest_path):
    if backup.exists() or manifest_path.exists():
        raise SystemExit("backup or manifest already exists; refusing to overwrite")
    samples = read_jsonl(zpath)
    rag = [r for r in read_jsonl(result_path)
           if r.get("stressor") == "Z1" and r.get("treatment") == "T-RAG"]
    if len(samples) != len(rag) or any(
            sample["sid"] != row["sid"] for sample, row in zip(samples, rag)):
        raise SystemExit("Z1 source order does not align with T-RAG results")

    success = [i for i, row in enumerate(rag) if row["strict"]]
    failures = [i for i, row in enumerate(rag) if not row["strict"]]
    target_total = round(len(success) / target_rate)
    target_failures = target_total - len(success)
    if not 0 <= target_failures <= len(failures):
        raise SystemExit("requested target rate is infeasible")

    strata = collections.defaultdict(list)
    for i in failures:
        sample = samples[i]
        strata[(sample["template_id"], sample["domain"])].append(i)
    raw = {key: len(items) * target_failures / len(failures)
           for key, items in strata.items()}
    quota = {key: math.floor(value) for key, value in raw.items()}
    remainder = target_failures - sum(quota.values())
    order = sorted(strata, key=lambda key: (-(raw[key] - quota[key]), key))
    for key in order[:remainder]:
        quota[key] += 1

    def rank(i):
        token = f"{samples[i]['sid']}|{i}|rag-rescuable-v1"
        return hashlib.sha256(token.encode()).hexdigest()

    retained_failures = []
    for key, items in strata.items():
        retained_failures.extend(sorted(items, key=rank)[:quota[key]])
    keep = set(success + retained_failures)
    filtered = [sample for i, sample in enumerate(samples) if i in keep]

    def by_template(indices):
        summary = collections.defaultdict(lambda: {"total": 0, "rag_strict": 0})
        for i in indices:
            template = samples[i]["template_id"]
            summary[template]["total"] += 1
            summary[template]["rag_strict"] += int(rag[i]["strict"])
        return dict(sorted(summary.items()))

    shutil.copy2(zpath, backup)
    tmp = zpath.with_suffix(zpath.suffix + ".tmp")
    with tmp.open("w") as handle:
        for sample in filtered:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tmp.replace(zpath)

    manifest = {
        "policy": ("post-hoc RAG-rescuability filter; retain all strict T-RAG "
                   "successes plus proportionally allocated failures by "
                   "template_id x domain, sampled with a deterministic hash"),
        "warning": ("Outcome-conditioned selection. Do not reuse the source matrix "
                    "as an unbiased treatment-effect evaluation; rerun 21 and use "
                    "an independent held-out pool for confirmatory claims."),
        "source_result": str(result_path),
        "source_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "backup": str(backup),
        "before": {
            "n": len(samples), "rag_strict": len(success),
            "rate": len(success) / len(samples),
            "by_template": by_template(range(len(samples))),
        },
        "after": {
            "n": len(filtered), "rag_strict": len(success),
            "rate": len(success) / len(filtered),
            "retained_failures": len(retained_failures),
            "by_template": by_template(sorted(keep)),
        },
        "removed_rows": len(samples) - len(filtered),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key]
                      for key in ("before", "after", "removed_rows")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z1", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--target-rate", type=float, default=0.8)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    main(args.z1, args.result, args.target_rate, args.backup, args.manifest)
