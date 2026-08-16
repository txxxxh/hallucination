#!/usr/bin/env python3
"""Collect current127 features for natural-error GSM8K in Scientist MC format."""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np
from spanattr.core import Item, SpanAttributor, set_seed

HERE = Path(__file__).resolve().parent; RUNS = HERE / "runs"
base = importlib.import_module("141_collect_gsm8k_natural_current127")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=RUNS / "143_gsm8k_natural_mc.jsonl")
    p.add_argument("--out-dir", type=Path, default=RUNS / "144_gsm8k_natural_mc_current127")
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch", type=int, default=24); p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0); a = p.parse_args(); set_seed(42)
    rows = [json.loads(x) for x in a.manifest.open() if x.strip()]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = importlib.import_module("61_grad_span_proposal").load_model(a.model, "bfloat16", "cuda")
    att = SpanAttributor(model, tok, device="cuda", baseline="mean", length_norm=True, max_rows=a.batch)
    for num, r in enumerate(rows[:a.limit or None], 1):
        fp = a.out_dir / f"{r['key']}.npz"
        if fp.exists() and a.resume: continue
        item = Item.from_dict({"key":r["key"], "prompt":r["prepend_names_prompt"],
                               "pred":r["generation"], "gold":r["other_answer"]})
        prep = att.prepare(item); ss, cc = base.spans(att, prep); p1, o1 = base.scan(att, prep, ss)
        u = (p1[0]-p1[1:])-(o1[0]-o1[1:]); top = int(np.argmax(np.abs(u)))
        ids = np.argsort(-np.abs(u))[:min(5,len(u))]; ph,oh,h14 = base.selected_hidden(att,prep,ids)
        ca,cb=cc[top]; deleted=re.sub(r"[ \t]+"," ",item.context[:ca]+item.context[cb:])
        deleted=re.sub(r"\s+([,.;:!?])",r"\1",deleted).strip()
        item2=Item(r["key"]+"_d",deleted,item.question,r["other_answer"],r["generation"],
                   context_prefix=item.context_prefix)
        prep2=att.prepare(item2);ss2,_=base.spans(att,prep2);p2,o2=base.scan(att,prep2,ss2)
        u2=(p2[0]-p2[1:])-(o2[0]-o2[1:]);ids2=np.argsort(-np.abs(u2))[:min(5,len(u2))]
        np.savez_compressed(fp,key=np.asarray(r["key"]),group=np.asarray(r["group"]),
          correct=np.asarray(int(r["correct"])),choice=np.asarray(r["choice"]),
          correct_position=np.asarray(r["correct_position"]),p_choice1=np.asarray(r["p_choice1"]),
          p_choice2=np.asarray(r["p_choice2"]),deleted_text=np.asarray(ss[top].text),
          stage1_pred=np.r_[p1[0],p1[1:][ids]],stage1_other=np.r_[o1[0],o1[1:][ids]],
          stage2_pred=np.r_[p2[0],p2[1:][ids2]],stage2_other=np.r_[o2[0],o2[1:][ids2]],
          pred_hidden=ph.astype(np.float16),other_hidden=oh.astype(np.float16),
          layer14=h14.astype(np.float16));print(f"[{num}/{len(rows)}] {r['key']} correct={r['correct']}",flush=True)

if __name__ == "__main__": main()
