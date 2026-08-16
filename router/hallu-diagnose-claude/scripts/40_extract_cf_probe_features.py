"""40: generate four deployable treatment probes and extract response deltas."""
import argparse, json
import numpy as np
from pathlib import Path
from common import DATA, LM, read_jsonl
from cf_probe_common import (FEATURE_VERSION, TREATMENTS, TfidfRetriever,
                             extract_record, feature_dir)

def main(model, stressors, rag_mode, corpus, limit, probe_tokens, budget_think):
    outdir = feature_dir(model); outdir.mkdir(parents=True, exist_ok=True)
    retriever = TfidfRetriever(corpus) if corpus else None
    samples = []
    for z in stressors:
        rows = read_jsonl(DATA / f"processed/{z}_final.jsonl")
        samples += rows[:limit] if limit else rows
    lm = LM(model)
    index, audit_path = [], outdir / "probe_responses.jsonl"
    audit_tmp = audit_path.with_suffix(".jsonl.tmp")
    with open(audit_tmp, "w") as audit:
        for i, sample in enumerate(samples, 1):
            path = outdir / f"{sample['sid']}.npz"
            valid = False
            if path.exists():
                try:
                    with np.load(path) as d:
                        valid = int(d["feature_version"]) == FEATURE_VERSION
                except Exception:
                    pass
            if valid:
                record = None
            else:
                record = extract_record(lm, sample, rag_mode, retriever,
                                        probe_tokens, budget_think)
                np.savez_compressed(path, feature_version=record["feature_version"],
                                    conditions=record["conditions"], states=record["states"],
                                    scalars=record["scalars"])
            if record:
                audit.write(json.dumps({"sid": sample["sid"],
                    "responses": record["responses"], "metadata": record["metadata"]},
                    ensure_ascii=False) + "\n")
            index.append({"sid": sample["sid"], "label": sample["stressor"],
                          "domain": sample["domain"], "template_id": sample["template_id"]})
            print(f"[cf40] {i}/{len(samples)} {sample['sid']} {'resume' if valid else 'extract'}")
    audit_tmp.replace(audit_path)
    tmp = outdir / "index.jsonl.tmp"
    with open(tmp, "w") as f:
        for row in index: f.write(json.dumps(row) + "\n")
    tmp.replace(outdir / "index.jsonl")
    json.dump({"feature_version": FEATURE_VERSION, "rag_mode": rag_mode,
               "corpus": corpus, "treatments": list(TREATMENTS),
               "probe_max_tokens": probe_tokens, "budget_max_think": budget_think},
              open(outdir / "config.json", "w"), indent=2)
    print(f"[cf40 save] {outdir} n={len(index)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--stressors", nargs="+", default=["z1", "z2", "z4", "z6"])
    ap.add_argument("--rag-mode", choices=["gold", "corpus"], default="gold")
    ap.add_argument("--retrieval-corpus")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--probe-max-tokens", type=int, default=512)
    ap.add_argument("--budget-max-think", type=int, default=1024)
    a = ap.parse_args()
    if a.rag_mode == "corpus" and not a.retrieval_corpus: ap.error("corpus mode requires --retrieval-corpus")
    main(a.model, a.stressors, a.rag_mode, a.retrieval_corpus,
         a.limit, a.probe_max_tokens, a.budget_max_think)
