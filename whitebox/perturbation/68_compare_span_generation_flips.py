# -*- coding: utf-8 -*-
"""Compare per-span embedding-mask response for correct vs incorrect answers."""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from spanattr.core import Item, SpanAttributor, set_seed


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../shuffled_prepend_names_question.json")
    ap.add_argument("--records", default="../tool_gate_correctness_names_llama31_8b/records.jsonl")
    ap.add_argument("--keys", nargs="+", default=["question_0007", "question_0001"])
    ap.add_argument("--out", default="runs/68_correct_wrong_span_flips.json")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); set_seed(args.seed)

    data = {str(x["key"]): x for x in json.load(open(args.data))}
    records = {x["key"]: x for x in map(json.loads, open(args.records))}
    load_model = importlib.import_module("61_grad_span_proposal").load_model
    parse_choice = importlib.import_module("tool_gate_correctness_stratification").parse_choice
    model, tok = load_model(args.model, args.dtype, args.device)
    att = SpanAttributor(model, tok, device=args.device, baseline="mean",
                         length_norm=True, max_rows=args.batch)
    results = []
    for key in args.keys:
        raw, rr = data[key], records[key]
        original = str(rr["parsed_answer"])
        right, wrong = str(raw["rgt_ans"]), str(raw["wrg_ans"])
        other = wrong if original == right else right
        item = Item.from_dict(dict(raw, pred=original, gold=other))
        item.pred, item.gold = original, other
        prep = att.prepare(item)
        spans = att.build_word_spans(prep, widths=(2, 3), stride=1)
        prep.spans = spans
        S0 = att.S0(prep)
        u, _ = att.u_of_sets(prep, [[i] for i in range(len(spans))], S0=S0)

        generations = []
        for start in range(0, len(spans), args.batch):
            ids = list(range(start, min(start + args.batch, len(spans))))
            A = torch.stack([att.alpha_from_spans(prep, [i]) for i in ids])
            pe = att._embeds(prep, A)
            mask = torch.ones(pe.shape[:2], device=args.device, dtype=torch.long)
            with torch.no_grad():
                g = model.generate(inputs_embeds=pe, attention_mask=mask,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=getattr(tok, "pad_token_id", 0) or 0)
            generations.extend(tok.batch_decode(g, skip_special_tokens=True))
            print(f"{key}: generated {len(generations)}/{len(spans)}", flush=True)

        rows = []
        for i, (sp, gen) in enumerate(zip(spans, generations)):
            parsed, correct = parse_choice(gen, right, wrong)
            rows.append({
                "idx": i, "text": sp.text, "start": sp.start, "end": sp.end,
                "delta_margin": float(u[i]), "generation": gen.strip(),
                "parsed": parsed, "correct_after": bool(correct) if parsed else None,
                "kept_original": parsed == original,
                "flipped_option": parsed is not None and parsed != original,
            })
        valid = [x for x in rows if x["parsed"] is not None]
        summary = {
            "n_spans": len(rows), "parse_rate": len(valid) / len(rows),
            "original_correct": bool(rr["correct"]),
            "kept_original_rate": float(np.mean([x["kept_original"] for x in valid])) if valid else None,
            "option_flip_rate": float(np.mean([x["flipped_option"] for x in valid])) if valid else None,
            "correct_after_rate": float(np.mean([x["correct_after"] for x in valid])) if valid else None,
            "delta_abs_mean": float(np.mean(np.abs(u))),
            "delta_abs_max": float(np.max(np.abs(u))),
        }
        results.append({"key": key, "right": right, "wrong": wrong,
            "original_generation": rr["generation"], "original_parsed": original,
            "S0_generated_vs_other": S0, "summary": summary, "spans": rows})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps([{r["key"]: r["summary"]} for r in results], indent=2))
    print("wrote", args.out)


if __name__ == "__main__": main()
