"""Build an isolated 1,000-instance BRIGHT/TheoremQA math screening pool.

BRIGHT is a retrieval benchmark and its AoPS split deliberately has no final
answer labels.  Therefore TheoremQA supplies the answer-verified Z1 examples,
while BRIGHT-AoPS is retained only for concrete-answer/Z6-risk auditing.
"""
import argparse
import json
from pathlib import Path

from common import Sample, DATA, sid_of, write_jsonl


OUT = DATA / "processed/bright_math_1000"


def theoremqa_rows():
    from datasets import load_dataset

    rows = load_dataset("TIGER-Lab/TheoremQA", split="test")
    out = []
    for i, row in enumerate(rows):
        if row.get("Picture") is not None:
            continue
        question = row["Question"].strip()
        answer = str(row["Answer"]).strip()
        answer_type = str(row.get("Answer_type", ""))
        out.append(Sample(
            sid=sid_of(f"theoremqa:{i}:{question}", "bm1tq"),
            stressor="Z1", domain="math", template_id="bright-theoremqa",
            q_trig=question, q_clean=question, answer=answer,
            meta={
                "source": "TIGER-Lab/TheoremQA",
                "source_index": i,
                "answer_type": answer_type,
                "numeric": answer_type in ("integer", "float"),
                "answer_verified": True,
                "unique_question_id": f"theoremqa:{i}",
                "instance_variant": "original",
            },
        ))
    return out


def bright_aops_rows():
    from datasets import load_dataset

    rows = load_dataset("xlangai/BRIGHT", "examples", split="aops")
    out = []
    for row in rows:
        question = row["query"].strip()
        source_id = str(row["id"])
        # UNVERIFIED is never passed through the Z1 correctness evaluator.
        out.append(Sample(
            sid=sid_of(f"bright-aops:{source_id}:{question}", "bm1ao"),
            stressor="Z1", domain="math", template_id="bright-aops",
            q_trig=question, q_clean=question, answer="UNVERIFIED",
            meta={
                "source": "xlangai/BRIGHT:aops",
                "source_id": source_id,
                "answer_verified": False,
                "unique_question_id": f"bright-aops:{source_id}",
                "instance_variant": "original",
                "retrieval_reasoning": row.get("reasoning", ""),
                "gold_ids": row.get("gold_ids", []),
                "gold_ids_long": row.get("gold_ids_long", []),
            },
        ))
    return out


def main(target):
    theorem = theoremqa_rows()
    aops = bright_aops_rows()
    originals = theorem + aops
    if target < len(originals):
        raise ValueError(f"target {target} is below {len(originals)} unique originals")

    variants = []
    need = target - len(originals)
    for base in theorem[:need]:
        row = json.loads(base.dump())
        row["sid"] = sid_of(row["sid"] + ":answer-only", "bm1tv")
        row["q_trig"] = row["q_trig"] + "\n\nGive only the final answer, without explanation."
        row["q_clean"] = row["q_trig"]
        row["template_id"] = "bright-theoremqa-answer-only"
        row["meta"]["instance_variant"] = "answer_only"
        variants.append(row)

    pool = originals + variants
    write_jsonl(pool, OUT / "candidate_manifest.jsonl")
    write_jsonl([x for x in pool if (x.meta if isinstance(x, Sample) else x["meta"])["answer_verified"]],
                OUT / "z1_pool.jsonl")
    manifest = {
        "instances": len(pool),
        "unique_questions": len(originals),
        "theoremqa_text_originals": len(theorem),
        "bright_aops_originals": len(aops),
        "answer_only_variants": len(variants),
        "note": "AoPS has no final-answer labels in BRIGHT and is audit-only.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "build_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000)
    main(ap.parse_args().target)
