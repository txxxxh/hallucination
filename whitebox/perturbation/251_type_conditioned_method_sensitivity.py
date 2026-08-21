#!/usr/bin/env python3
"""Type-conditioned UEPR sensitivity at benchmark-frozen 10% FPR."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SRC = RUNS / "249_unified_hallucination_types" / "items.jsonl"
OUT = RUNS / "251_type_conditioned_method_sensitivity"
METHODS = {"Perturbation": "p_score", "Uncertainty": "u_score",
           "Representation": "r_score", "Evidence": "e_score"}
TYPES = ["knowledge_missing", "unsupported_confident_fabrication",
         "context_induced", "reasoning_unstable", "stable_self_consistent"]


def read(path): return [json.loads(x) for x in path.open() if x.strip()]


def wilson(k, n, z=1.959963984540054):
    if not n: return None
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]

def tie_aware_boundary(correct, target=.10):
    """Return boundary and fractional inclusion of ties for exact expected FPR."""
    v=np.asarray(correct,float); n=len(v)
    for t in sorted(set(v),reverse=True):
        above=float(np.sum(v>t)); tied=float(np.sum(v==t)); need=target*n-above
        if need>=-1e-12 and need<=tied+1e-12:
            return float(t),float(np.clip(need/tied,0,1))
    return float(np.max(v)),0.0

def expected_rate(values,t,alpha):
    v=np.asarray(values,float)
    return float(np.mean((v>t).astype(float)+alpha*(v==t)))


def main():
    OUT.mkdir(parents=True, exist_ok=True); rows=read(SRC)
    benches=sorted({x["benchmark"] for x in rows})
    thresholds={}; cells={}; overall_error_tpr={}
    for b in benches:
        br=[x for x in rows if x["benchmark"]==b]; correct=[x for x in br if not x["error"]]
        thresholds[b]={}
        for method,field in METHODS.items():
            vals=[x[field] for x in correct if field in x and x[field] is not None]
            thresholds[b][method]=None if not vals else dict(zip(("boundary","tie_fraction"),tie_aware_boundary(vals)))
        overall_error_tpr[b]={}
        all_errors=[x for x in br if x["error"]]
        for method,field in METHODS.items():
            th=thresholds[b][method]; vals=[x[field]for x in all_errors if field in x and x[field]is not None]
            overall_error_tpr[b][method]=None if th is None or not vals else expected_rate(vals,th["boundary"],th["tie_fraction"])
        cells[b]={}
        for typ in TYPES:
            er=[x for x in br if x["error"] and typ in x["hallucination_types"]]
            cells[b][typ]={}
            for method,field in METHODS.items():
                th=thresholds[b][method]; valid=[x for x in er if field in x and x[field] is not None]
                if th is None or not valid: cells[b][typ][method]={"n":0,"tpr":None,"ci95":None}; continue
                tpr=expected_rate([x[field]for x in valid],th["boundary"],th["tie_fraction"])
                # AUROC compares this error type against all correct baseline items.
                auc=float(roc_auc_score([0]*len(correct)+[1]*len(valid),
                     [x[field]for x in correct if field in x and x[field]is not None]+[x[field]for x in valid]))
                cells[b][typ][method]={"n":len(valid),"expected_tpr":tpr,"tpr":tpr,
                    "enrichment_vs_all_errors":tpr-overall_error_tpr[b][method],"auroc_vs_correct":auc}
    macro={}
    for typ in TYPES:
        macro[typ]={}
        for method in METHODS:
            zs=[cells[b][typ][method]["tpr"] for b in benches if cells[b][typ][method]["tpr"] is not None]
            macro[typ][method]={"benchmark_cells":len(zs),"macro_tpr":None if not zs else float(np.mean(zs)),
                                "macro_enrichment":None if not zs else float(np.mean([cells[b][typ][method]["enrichment_vs_all_errors"]for b in benches if cells[b][typ][method]["tpr"]is not None])),
                                "values":{b:cells[b][typ][method]["tpr"] for b in benches if cells[b][typ][method]["tpr"] is not None}}
    report={"protocol":"threshold per benchmark/method = correct-sample 90th percentile; strict score > threshold; type TPR; macro-average across applicable benchmark cells",
            "target_fpr":.10,"tie_handling":"fractional inclusion at boundary gives exact expected FPR","thresholds":thresholds,"overall_error_tpr":overall_error_tpr,"cells":cells,"macro":macro,
            "circularity_warning":"types are defined from UEPR axes; this is operational alignment, not independent mechanism validation"}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    labels={"knowledge_missing":"知识缺失","unsupported_confident_fabrication":"无依据但自信的编造",
            "context_induced":"上下文诱导/局部错误依赖","reasoning_unstable":"推理不稳定",
            "stable_self_consistent":"稳定自洽错误"}
    lines=[r"\begin{tabular}{lcccc}",r"\toprule",
           r"Hallucination type & Perturbation & Uncertainty & Representation & Evidence \\",r"\midrule"]
    for typ in TYPES:
        cs=[]
        for m in METHODS:
            v=macro[typ][m]["macro_tpr"];d=macro[typ][m]["macro_enrichment"];cs.append("--"if v is None else f"{100*v:.1f}\\% ({100*d:+.1f})")
        lines.append(labels[typ]+" & "+" & ".join(cs)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (OUT/"table.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps({"thresholds":thresholds,"macro":macro},ensure_ascii=False,indent=2)); print("\n".join(lines))

if __name__=="__main__": main()
