#!/usr/bin/env python3
"""Stage 81a: learn and diagnose a shared span-gradient active subspace.

This is a calibration-only white-box diagnostic.  The saved basis must be
frozen before Stage 82 evaluates held-out questions with forward-only ZO.
"""
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spanattr.core import Item, SpanAttributor, build_toy, set_seed

SMOKE = [Item("cal", "alpha beta gamma delta epsilon zeta eta theta",
              "which letter", "delta", "theta")]

def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument("--items"); p.add_argument("--item_id", nargs="+")
    p.add_argument("--basis_out",default="runs/81_active_basis.pt")
    p.add_argument("--report",default="runs/81_active_subspace.json")
    p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype",default="bfloat16"); p.add_argument("--device",default=None)
    p.add_argument("--ranks",type=int,nargs="+",default=[4,8,16,32,64])
    p.add_argument("--widths",type=int,nargs="+",default=[2,3])
    p.add_argument("--seed",type=int,default=42); p.add_argument("--smoke",action="store_true")
    a=p.parse_args(); set_seed(a.seed)
    a.device=a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    extra={}
    if a.smoke:
        model,tok=build_toy(); a.device="cpu"; items=SMOKE
        a.basis_out="runs/81_active_smoke.pt"; a.report="runs/81_active_smoke.json"
        extra=dict(prefix="ctx: ",middle=" q: {question} a: ")
    else:
        if not a.items: raise SystemExit("--items is required")
        loader=importlib.import_module("61_grad_span_proposal").load_model
        model,tok=loader(a.model,a.dtype,a.device)
        items=[Item.from_dict(x) for x in json.load(open(a.items))]
        if a.item_id: items=[x for x in items if x.item_id in set(a.item_id)]
    att=SpanAttributor(model,tok,device=a.device,baseline="mean",length_norm=True,
                       max_rows=16,**extra)
    rows=[]; provenance=[]
    for item in items:
        prep=att.prepare(item); spans=att.build_word_spans(prep,widths=a.widths,stride=1)
        g=att.grad_embed(prep)
        for sp in spans:
            rows.append(g[sp.start:sp.end].sum(0))
            provenance.append({"item_id":item.item_id,"text":sp.text})
        print(f"{item.item_id}: {len(spans)} gradient rows",flush=True)
    if not rows: raise SystemExit("no calibration spans")
    G=np.asarray(rows,dtype=np.float32)
    _,singular,vh=np.linalg.svd(G,full_matrices=False)
    energy=singular**2; cumulative=np.cumsum(energy)/(energy.sum()+1e-30)
    ranks=sorted(set(min(r,len(singular)) for r in a.ranks))
    payload={"basis":torch.from_numpy(vh[:max(ranks)].T.copy()),
             "ranks":ranks,"embedding_dim":G.shape[1],
             "calibration_item_ids":[x.item_id for x in items]}
    Path(a.basis_out).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,a.basis_out)
    report={"n_items":len(items),"n_span_gradients":len(G),"embedding_dim":G.shape[1],
            "ranks":ranks,"explained_energy":{str(r):float(cumulative[r-1]) for r in ranks},
            "effective_rank_90":int(np.searchsorted(cumulative,.90)+1),
            "effective_rank_95":int(np.searchsorted(cumulative,.95)+1),
            "singular_values":singular.tolist(),"basis_out":a.basis_out,
            "calibration_item_ids":[x.item_id for x in items]}
    Path(a.report).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("n_span_gradients","explained_energy","effective_rank_90","effective_rank_95")},indent=2))

if __name__=="__main__": main()
