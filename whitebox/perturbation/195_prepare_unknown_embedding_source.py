#!/usr/bin/env python3
"""Create the missing unknown-item source list for run-120 feature parity."""
import importlib, json
from pathlib import Path

B = importlib.import_module("184_sparse_fullcache_confirmation_fixed").B
OUT = Path("runs/195_unknown_embedding_source.jsonl")
rows, *_ = B.load_rows()
unknown = [r for r in rows if int(r["known"]) == 0]
with OUT.open("w") as f:
    for r in unknown:
        f.write(json.dumps({"key": r["key"]}) + "\n")
print({"output": str(OUT), "unknown": len(unknown)})
