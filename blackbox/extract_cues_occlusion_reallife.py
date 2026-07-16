#!/usr/bin/env python3
"""Scientist-style deletion→negation hallucination detector for RealLifeQA.

Original and negated prompts are binary (1/2). Only deletion prompts allow 3,
meaning that the remaining scenario does not determine either option. A span
enters negation when deletion produces 3 or flips between 1 and 2. At least one
valid non-flipping negation is hallucination evidence. Gold is joined only after
all predictions finish.
"""
from __future__ import annotations

import argparse, csv, json, statistics, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cue_spans import Span, delete_span, segment_scenario  # noqa: E402
import run_reallifeqa_pilot as pilot  # noqa: E402
import extract_cues_occlusion_scientist as scientist  # noqa: E402

GOLD_FIELDS = frozenset({"answer", "correct_option", "shortcut_option", "gold_label"})
NEGATOR_SYSTEM = "You minimally negate one specified proposition and return only valid JSON."


def _with_thinking_disabled(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Add DashScope's non-standard Qwen thinking switch to a request."""
    configured = dict(kwargs)
    extra_body = dict(configured.get("extra_body") or {})
    extra_body["enable_thinking"] = False
    configured["extra_body"] = extra_body
    return configured


def disable_qwen_thinking() -> None:
    """Disable thinking for both sampling and negator OpenAI-compatible calls.

    DashScope requires ``n=1`` while thinking is enabled.  This experiment
    deliberately uses ``n>1`` for majority-vote sampling, so Qwen must run in
    non-thinking mode on both call paths.
    """
    original_completion_create = scientist._completion_create
    original_chat_create = pilot._chat_create

    def qwen_completion_create(client: Any, **kwargs: Any) -> Any:
        return original_completion_create(
            client, **_with_thinking_disabled(kwargs))

    def qwen_chat_create(client: Any, **kwargs: Any) -> Any:
        return original_chat_create(client, **_with_thinking_disabled(kwargs))

    scientist._completion_create = qwen_completion_create
    pilot._chat_create = qwen_chat_create


def option(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text if text in {"1", "2", "3"} else None


def evaluation_prompt(prompt: str, allow_uncertain: bool) -> str:
    protocol = (
        "\n\nEvaluation answer protocol (this overrides earlier answer wording):\n"
        "1 = Option1\n2 = Option2\n"
    )
    if allow_uncertain:
        protocol += (
            "3 = after the deletion, the remaining scenario does not determine "
            "either Option1 or Option2\nOutput only 1, 2, or 3."
        )
    else:
        protocol += "You must choose Option1 or Option2. Output only 1 or 2; never output 3."
    return prompt.rstrip() + protocol


def containing_sentence(prompt: str, span: Span) -> tuple[int, int, str]:
    start = max(prompt.rfind(".", 0, span.start), prompt.rfind("?", 0, span.start),
                prompt.rfind("!", 0, span.start), prompt.rfind("\n", 0, span.start)) + 1
    ends = [p for p in (prompt.find(".", span.end), prompt.find("?", span.end),
                        prompt.find("!", span.end), prompt.find("\n", span.end)) if p >= 0]
    end = min(ends) + 1 if ends else len(prompt)
    while start < end and prompt[start].isspace(): start += 1
    return start, end, prompt[start:end]


def negate_sentence(client: Any, model: str, prompt: str, span: Span,
                    cache: pilot.JsonCache) -> Dict[str, Any]:
    start, end, sentence = containing_sentence(prompt, span)
    request = (
        "Negate only the proposition expressed by the target span. Preserve entities, "
        "nouns, descriptions, numbers, options, and every unrelated fact. Use the "
        "smallest grammatical rewrite. Do not answer the question. Return only JSON "
        "with keys negated_sentence, rewrite_valid, notes.\n\n"
        f"Full prompt:\n{prompt}\n\nTarget span index: {span.index}\n"
        f"Target span: {span.text}\nFull containing sentence: {sentence}"
    )
    key = cache.make_key("reallife_minimal_negation_v1", {
        "model": model, "prompt": prompt, "span_index": span.index,
        "start": span.start, "end": span.end, "version": 1})
    cached = cache.get(key)
    if cached is not None: return cached
    raw = pilot._call_chat_text(client=client, cache=cache,
        namespace="reallife_minimal_negation_raw_v1", model=model,
        messages=[{"role":"system","content":NEGATOR_SYSTEM},
                  {"role":"user","content":request}],
        temperature=1, max_tokens=2000)
    try:
        obj = pilot._extract_json_object(raw["content"])
        sentence_new = obj.get("negated_sentence")
        valid = isinstance(sentence_new, str) and bool(sentence_new.strip())
        out = {"negated_sentence": sentence_new,
               "negated_prompt": prompt[:start] + sentence_new + prompt[end:] if valid else None,
               "rewrite_valid": bool(valid), "negation_notes": str(obj.get("notes", ""))}
    except Exception as exc:
        out = {"negated_sentence":None,"negated_prompt":None,"rewrite_valid":False,
               "negation_notes":f"invalid JSON/rewrite: {exc}"}
    cache.set(key, out); return out


def measure(client: Any, model: str, prompt: str, cache: pilot.JsonCache,
            samples: int, method: str, allow_uncertain: bool,
            max_batch: int) -> Dict[str, Any]:
    return scientist.measure(client, model, prompt, cache, samples, method,
                             allow_uncertain, max_batch)


def choose_method(client: Any, model: str, prompt: str, cache: pilot.JsonCache,
                  samples: int, max_batch: int) -> tuple[str, Dict[str, Any]]:
    return scientist.choose_method(client, model, prompt, cache, samples,
                                   allow_uncertain=False,
                                   max_batch_size=max_batch)


def effect(base: Dict[str, Any], changed: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    base_pred, changed_pred = option(base.get("prediction")), option(changed.get("prediction"))
    person_flip = base_pred in {"1","2"} and changed_pred in {"1","2"} and base_pred != changed_pred
    return {
        f"{prefix}_prediction": changed_pred,
        f"{prefix}_logprob_1": changed.get("logprob_1"),
        f"{prefix}_logprob_2": changed.get("logprob_2"),
        f"{prefix}_logprob_3": changed.get("logprob_3"),
        f"{prefix}_prob_1": changed.get("prob_1"),
        f"{prefix}_prob_2": changed.get("prob_2"),
        f"{prefix}_prob_3": changed.get("prob_3"),
        f"{prefix}_choice_margin": changed.get("choice_margin"),
        f"{prefix}_uncertain": changed_pred == "3",
        f"{prefix}_person_flip": person_flip,
        f"{prefix}_same_as_original": changed_pred == base_pred,
        f"{prefix}_flip": None if changed_pred is None or base_pred is None else changed_pred != base_pred,
    }


def empty_negation(reason: str) -> Dict[str, Any]:
    return {"selected_for_negation":False,"negated_sentence":None,"negated_prompt":None,
        "rewrite_valid":None,"negation_notes":reason,"negation_prediction":None,
        "negation_logprob_1":None,"negation_logprob_2":None,"negation_logprob_3":None,
        "negation_prob_1":None,"negation_prob_2":None,"negation_prob_3":None,
        "negation_choice_margin":None,"negation_uncertain":None,"negation_person_flip":None,
        "negation_same_as_original":None,"negation_flip":None,"method_consistent":None}


def run_item_once(client: Any, item: Dict[str, Any], args: argparse.Namespace,
                  cache: pilot.JsonCache, forced_method: Optional[str]=None) -> Dict[str, Any]:
    if GOLD_FIELDS.intersection(item): raise AssertionError("gold fields entered detection")
    source, item_id = item["benchmark_prompt"], item["id"]
    binary_prompt = evaluation_prompt(source, False)
    deletion_prompt = evaluation_prompt(source, True)
    spans = segment_scenario(source,args.min_clause_words,args.min_span_words,args.max_span_words)
    if not spans: raise ValueError("no scenario spans")
    if forced_method is None:
        method,base=choose_method(client,args.target_model,binary_prompt,cache,args.samples,args.max_sampling_batch)
    else:
        method=forced_method
        base=measure(client,args.target_model,binary_prompt,cache,args.samples,method,False,args.max_sampling_batch)
    base_pred=option(base.get("prediction"))
    if base_pred not in {"1","2"}: raise RuntimeError("binary original prediction was not 1 or 2")
    rows=[]
    for span in spans:
        deleted_prompt=delete_span(deletion_prompt,span)
        deleted=measure(client,args.target_model,deleted_prompt,cache,args.samples,method,True,args.max_sampling_batch)
        row={"candidate_index":span.index,"candidate_text":span.text,"span_start":span.start,
             "span_end":span.end,"deleted_prompt":deleted_prompt}
        row.update(effect(base,deleted,"delete"))
        row["deletion_stage_pass"]=bool(row["delete_uncertain"] or row["delete_person_flip"])
        row["deletion_stage_reason"]=("delete_uncertain" if row["delete_uncertain"] else
            "delete_person_flip" if row["delete_person_flip"] else "deletion neither uncertain nor option flip")
        row.update(empty_negation(row["deletion_stage_reason"])); rows.append(row)
    span_map={s.index:s for s in spans}
    informative=[r for r in rows if r["deletion_stage_pass"]]
    for row in informative:
        span=span_map[row["candidate_index"]]
        neg=negate_sentence(client,args.negator_model or args.target_model,binary_prompt,span,cache)
        row.update(neg); row["selected_for_negation"]=True
        if neg["rewrite_valid"]:
            changed=measure(client,args.target_model,neg["negated_prompt"],cache,args.samples,method,False,args.max_sampling_batch)
            row.update(effect(base,changed,"negation")); row["method_consistent"]=changed["method"]==method
        else:
            keep={"selected_for_negation":True,**neg}; row.update(empty_negation("invalid negation rewrite")); row.update(keep); row["method_consistent"]=True
    valid=[r for r in informative if r.get("rewrite_valid") is True and r.get("method_consistent") is True and r.get("negation_flip") in {True,False}]
    if any(r["negation_flip"] is False for r in valid): pred,decision=True,"negation_nonflip"
    elif not informative: pred,decision=False,"no_informative_deletion"
    elif len(valid)==len(informative) and all(r["negation_flip"] is True for r in valid): pred,decision=False,"all_negations_flip"
    else: pred,decision=None,"ambiguous"
    return {"id":item_id,"prediction_original":base_pred,"original_logprob_1":base["logprob_1"],
        "original_logprob_2":base["logprob_2"],"original_prob_1":base["prob_1"],
        "original_prob_2":base["prob_2"],"probability_method":method,"n_spans":len(spans),
        "n_delete_uncertain":sum(r["delete_uncertain"] for r in rows),
        "n_delete_option_flips":sum(r["delete_person_flip"] for r in rows),
        "n_informative_deletions":len(informative),"n_valid_negations":len(valid),
        "predicted_hallucination":pred,"decision":decision,"candidates":rows}


def run_item(client: Any,item:Dict[str,Any],args:argparse.Namespace,cache:pilot.JsonCache)->Dict[str,Any]:
    try:return run_item_once(client,item,args,cache)
    except Exception:return run_item_once(client,item,args,cache,"sampling")


def metrics(rows:List[Dict[str,Any]])->Dict[str,Any]:
    x=[r for r in rows if r.get("predicted_hallucination") is not None and r.get("true_hallucination") is not None]
    tp=sum(r["predicted_hallucination"] is True and r["true_hallucination"] is True for r in x); fp=sum(r["predicted_hallucination"] is True and r["true_hallucination"] is False for r in x)
    tn=sum(r["predicted_hallucination"] is False and r["true_hallucination"] is False for r in x); fn=sum(r["predicted_hallucination"] is False and r["true_hallucination"] is True for r in x)
    p=tp/(tp+fp) if tp+fp else None; rc=tp/(tp+fn) if tp+fn else None
    return {"n":len(x),"tp":tp,"fp":fp,"tn":tn,"fn":fn,"precision":p,"recall":rc,"f1":2*p*rc/(p+rc) if p is not None and rc is not None and p+rc else None}


def write_outputs(outdir:Path,records:List[Dict[str,Any]])->None:
    outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"cue_extraction.jsonl").open("w",encoding="utf-8") as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    flat=[]
    for r in records:
        base={k:v for k,v in r.items() if k!="candidates"}
        if r.get("candidates"): flat.extend(base|c for c in r["candidates"])
        else: flat.append(base)
    fields=[]
    for r in flat:
        for k in r:
            if k not in fields:fields.append(k)
    with (outdir/"cue_extraction.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    ok=[r for r in records if not r.get("error")]; m=metrics(ok); fmt=lambda x:"n/a" if x is None else f"{x:.3f}"
    candidates=[c for r in ok for c in r.get("candidates",[])]
    lines=["# RealLifeQA deletion-uncertain → negation-nonflip summary","",f"Items processed: {len(ok)}",
        f"Items entering negation: {sum(r.get('n_informative_deletions',0)>0 for r in ok)}/{len(ok)}",
        f"Deletion uncertain spans: {sum(c.get('delete_uncertain') is True for c in candidates)}",
        f"Deletion option-flip spans: {sum(c.get('delete_person_flip') is True for c in candidates)}",
        f"Predicted hallucination: {sum(r.get('predicted_hallucination') is True for r in ok)}/{len(ok)}","",
        "## Gold evaluation","",f"Precision: {fmt(m['precision'])}",f"Recall: {fmt(m['recall'])}",f"F1: {fmt(m['f1'])}",
        f"Confusion matrix: TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}"]
    (outdir/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",default="question_and_result.json");p.add_argument("--limit",type=int,default=50)
    p.add_argument("--target-model",default=pilot.DEFAULT_MODEL);p.add_argument("--negator-model",default=None);p.add_argument("--base-url",default=pilot.DEFAULT_BASE_URL)
    p.add_argument("--samples",type=int,default=20);p.add_argument("--max-sampling-batch",type=int,default=8)
    p.add_argument("--min-clause-words",type=int,default=10);p.add_argument("--min-span-words",type=int,default=2);p.add_argument("--max-span-words",type=int,default=8)
    p.add_argument("--outdir",default="outputs_reallifeqa/only_deletion_uncertain_5mini");args=p.parse_args()
    if args.samples<1 or args.max_sampling_batch<1:p.error("sampling values must be positive")
    models = (args.target_model, args.negator_model or args.target_model)
    if any(str(model).lower().startswith("qwen") for model in models):
        disable_qwen_thinking()
    data=pilot.load_data(args.input);data=data[:args.limit] if args.limit>=0 else data
    outdir=Path(args.outdir);outdir.mkdir(parents=True,exist_ok=True);cache=pilot.JsonCache(outdir/"cache.json");client=pilot._make_client(args.base_url)
    records=[];gold={}
    for i,raw in enumerate(data):
        item_id=raw.get("id",i+1) if isinstance(raw,dict) else i+1
        try:
            item=pilot._validate_item(raw,i);gold[json.dumps(item_id,sort_keys=True)]=str(int(item["answer"]))
            records.append(run_item(client,{"id":item_id,"benchmark_prompt":item["benchmark_prompt"]},args,cache))
        except Exception as exc:records.append({"id":item_id,"error":str(exc)})
        print(f"[{i+1}/{len(data)}] item {item_id} done",file=sys.stderr,flush=True);time.sleep(.05)
    for r in records:
        correct=gold.get(json.dumps(r.get("id"),sort_keys=True))
        if correct is not None and not r.get("error"):
            r["correct_option_eval_only"]=correct;r["true_hallucination"]=r["prediction_original"]!=correct
    write_outputs(outdir,records);print(f"Wrote outputs to {outdir}",file=sys.stderr);return 0


if __name__=="__main__":raise SystemExit(main())
