#!/usr/bin/env python3
"""Generate a balanced, model-specific DROP manifest for the paper matrix."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--target-per-class", type=int, default=500)
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    cli = ap.parse_args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = cli.out_dir / "drop_balanced_raw.jsonl"
    final_manifest = cli.out_dir / "drop.jsonl"
    complete = cli.out_dir / "manifest.done"
    if complete.exists():
        print(f"already complete: {final_manifest}")
        return
    base = importlib.import_module("166_prepare_drop1000")
    args = SimpleNamespace(
        model=cli.model, items=cli.items, generations=cli.out_dir / "generations.jsonl",
        manifest=raw_manifest, out_dir=cli.out_dir, batch=cli.batch,
        target_per_class=cli.target_per_class, max_input_tokens=cli.max_input_tokens,
        max_new_tokens=cli.max_new_tokens, seed=cli.seed, resume=cli.resume)
    base.generate(args)
    base.balance(args)
    rows = [json.loads(line) for line in raw_manifest.open() if line.strip()]
    with final_manifest.open("w") as output:
        for row in rows:
            record = {
                "key": row["key"], "group": row["group"],
                "correct": int(row["correct"]), "context": row["context"],
                "question": row["question"], "pred": row["generation"],
                "other": row["other_answer"], "prompt_mode": False,
                "model": cli.model,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    complete.write_text("ok\n")
    print(f"wrote {len(rows)} rows to {final_manifest}")

if __name__ == "__main__":
    main()
