"""Build 700 balanced math-Z6 candidates from official TreeCut samples."""
import argparse, json, random
from pathlib import Path
from common import DATA, Sample, sid_of, write_jsonl

LICENSE = "If the question is unanswerable or underdetermined, say so explicitly. "

def main(treecut_root, per_level=175):
    root = Path(treecut_root)
    files = sorted((root / "sample_data").glob("*hallu-True*.jsonl"))
    if len(files) != 4:
        raise ValueError(f"expected 4 TreeCut difficulty files, found {len(files)}")
    rng = random.Random(42)
    out = []
    for path in files:
        rows = [json.loads(x) for x in path.open() if x.strip()]
        rng.shuffle(rows)
        level = path.stem.split("numVars-")[1].split("_")[0]
        for row in rows[:per_level]:
            q = row["problem"].strip()
            out.append(Sample(
                sid=sid_of(f"treecut:{level}:{q}", "z6tc"), stressor="Z6",
                domain="math", template_id=f"treecut-nvars{level}",
                intensity=float(level), q_trig=q, q_clean=LICENSE + q,
                answer="UNANSWERABLE",
                meta={"source": "TreeCut", "num_vars": int(level),
                      "proof": row.get("proof", ""), "license": "Apache-2.0"},
            ))
    target = DATA / "processed/treecut_z6_700"
    write_jsonl(out, target / "z6_pool.jsonl")
    print(f"unique_sid={len({x.sid for x in out})}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--treecut-root", default="/home/tong56/router/treecut-math")
    ap.add_argument("--per-level", type=int, default=175)
    a = ap.parse_args(); main(a.treecut_root, a.per_level)
