"""50: at test time run all four probes, route, and reuse the selected response."""
import argparse, json
from pathlib import Path
import joblib, numpy as np
from common import LM, outcome, read_jsonl, write_jsonl
from cf_probe_common import (BASE, TREATMENTS, TfidfRetriever, extract_record,
                             vector_at_layer)

POLICY={"Z1":"T-RAG","Z2":"T-Clean","Z4":"T-Budget","Z6":"T-Abstain"}

def main(model,input_path,router_path,rag_mode,corpus,limit,probe_tokens,budget_think,output):
    bundle=joblib.load(router_path); rows=read_jsonl(Path(input_path)); rows=rows[:limit] if limit else rows
    retriever=TfidfRetriever(corpus) if corpus else None; lm=LM(model); results=[]
    for i,s in enumerate(rows,1):
        rec=extract_record(lm,s,rag_mode,retriever,probe_tokens,budget_think)
        class D:
            def __getitem__(self,k): return rec[k]
        x=vector_at_layer(D(),bundle["layer"])[None]
        pred=str(bundle["classifier"].predict(bundle["scaler"].transform(bundle["imputer"].transform(x)))[0])
        treatment=POLICY.get(pred,BASE); response=rec["responses"][treatment]
        gold=s["answer"] if s["answer"]!="UNKNOWN_ENTITY" else "UNANSWERABLE"
        score=outcome(response,gold,s.get("answer_aliases",[]),bool(s.get("meta",{}).get("numeric")))
        results.append({"sid":s["sid"],"gold_stressor":s.get("stressor"),"predicted_stressor":pred,
                        "selected_treatment":treatment,"response":response,
                        "probe_metadata":rec["metadata"],**score})
        print(f"[cf50] {i}/{len(rows)} {s['sid']} pred={pred} treatment={treatment}")
    write_jsonl(results,Path(output));
    print(f"strict={np.mean([r['strict'] for r in results]):.1%} honest={np.mean([r['honest'] for r in results]):.1%}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit")
    ap.add_argument("--input",required=True); ap.add_argument("--router",required=True)
    ap.add_argument("--rag-mode",choices=["gold","corpus"],default="corpus"); ap.add_argument("--retrieval-corpus")
    ap.add_argument("--limit",type=int); ap.add_argument("--probe-max-tokens",type=int,default=512)
    ap.add_argument("--budget-max-think",type=int,default=1024)
    ap.add_argument("--output",default="data/results/cf_routed_mitigation.jsonl")
    a=ap.parse_args()
    if a.rag_mode=="corpus" and not a.retrieval_corpus: ap.error("corpus mode requires --retrieval-corpus")
    main(a.model,a.input,a.router,a.rag_mode,a.retrieval_corpus,a.limit,
         a.probe_max_tokens,a.budget_max_think,a.output)
