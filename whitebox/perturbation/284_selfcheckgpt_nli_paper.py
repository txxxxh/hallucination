#!/usr/bin/env python3
"""SelfCheckGPT-NLI on the frozen Scientist/TriviaQA/GSM8K/DROP matrix.

The benchmark rows, primary responses, and prompts are frozen from experiment 261.
The method follows SelfCheckGPT (Manakul et al., EMNLP 2023): N=20 stochastic
responses at temperature 1.0 and sentence-level contradiction probability from
potsawee/deberta-v3-large-mnli, averaged over samples (then sentences per item).
"""
from __future__ import annotations
import argparse, importlib, json, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
GEN_MODEL = "/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
NLI_MODEL = "potsawee/deberta-v3-large-mnli"
N_SAMPLES = 20

base = importlib.import_module("261_paper_baseline_matrix")

def read(path):
    p = Path(path)
    return [json.loads(x) for x in p.open() if x.strip()] if p.exists() else []

def sample(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rs = base.rows(args.dataset)
    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.jsonl"
    done = {x["key"] for x in read(path)} if args.resume else set()
    tok = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True,
    ).eval()
    with path.open("a" if args.resume else "w") as fh:
        for st in range(0, len(rs), args.batch):
            part = [r for r in rs[st:st + args.batch] if r["key"] not in done]
            if part:
                prompts = [tok.apply_chat_template(
                    [{"role": "user", "content": base.user_text(args.dataset, r)}],
                    tokenize=False, add_generation_prompt=True) for r in part]
                z = tok(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
                torch.manual_seed(args.seed + st)
                with torch.inference_mode():
                    g = model.generate(
                        **z, do_sample=True, temperature=1.0,
                        num_return_sequences=N_SAMPLES,
                        max_new_tokens=192 if args.dataset == "gsm8k" else 32,
                        pad_token_id=tok.pad_token_id,
                    )
                texts = tok.batch_decode(g[:, z.input_ids.shape[1]:], skip_special_tokens=True)
                for i, r in enumerate(part):
                    rec = {"key": r["key"], "correct": int(r["correct"]),
                           "primary_response": str(r["pred"]),
                           "sampled_passages": texts[i*N_SAMPLES:(i+1)*N_SAMPLES]}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done.add(r["key"])
            print(args.dataset, min(st + args.batch, len(rs)), "/", len(rs), flush=True)

def sentences(text):
    """spaCy-compatible sentence boundary fallback without linguistic rewriting."""
    text = str(text).strip()
    if not text:
        return [""]
    # en_core_web_sm's relevant role here is sentence boundaries. Use it when
    # installed; the deterministic fallback preserves every non-empty span.
    try:
        import spacy
        nlp = sentences._nlp
    except AttributeError:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner", "lemmatizer"])
        except Exception:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        sentences._nlp = nlp
    except Exception:
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()] or [text]
    vals = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    return vals or [text]

def score(args):
    import torch
    from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = args.out / args.dataset
    records = read(out / "samples.jsonl")
    scored_path = out / "scores.jsonl"
    done = {x["key"] for x in read(scored_path)} if args.resume else set()
    tok = DebertaV2Tokenizer.from_pretrained(NLI_MODEL, cache_dir=args.cache)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        NLI_MODEL, cache_dir=args.cache, torch_dtype=torch.float16).to("cuda").eval()
    mode = "a" if args.resume else "w"
    with scored_path.open(mode) as fh:
        for ix, r in enumerate(records):
            if r["key"] in done:
                continue
            ss = sentences(r["primary_response"])
            pairs = [(sent, passage) for sent in ss for passage in r["sampled_passages"]]
            probs = []
            for st in range(0, len(pairs), args.nli_batch):
                enc = tok(pairs[st:st+args.nli_batch], padding=True, truncation=True,
                          return_tensors="pt", return_token_type_ids=True,
                          return_attention_mask=True).to("cuda")
                with torch.inference_mode():
                    # Official SelfCheckNLI: softmax index 1 is contradiction;
                    # this checkpoint already omits the neutral class.
                    p = model(**enc).logits.softmax(-1)[:, 1]
                probs.extend(p.float().cpu().tolist())
            matrix = np.asarray(probs).reshape(len(ss), N_SAMPLES)
            sent_scores = matrix.mean(axis=1)
            rec = {"key": r["key"], "correct": int(r["correct"]),
                   "sentences": ss, "sentence_scores": sent_scores.tolist(),
                   "score": float(sent_scores.mean())}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(r["key"])
            if (ix + 1) % 25 == 0:
                print(args.dataset, ix + 1, "/", len(records), flush=True)
    vals = read(scored_path)
    y = np.asarray([1-int(x["correct"]) for x in vals])
    u = np.asarray([x["score"] for x in vals])
    report = {
        "dataset": args.dataset, "n": len(y), "errors": int(y.sum()),
        "method": "SelfCheckGPT SelfCheck-NLI (Manakul et al., EMNLP 2023)",
        "adaptation": "original frozen benchmark/prompts/primary responses and Llama-3.1-8B target model",
        "sampling": {"N": N_SAMPLES, "temperature": 1.0, "top_p": 1.0,
                     "seed": args.seed},
        "nli_model": NLI_MODEL,
        "aggregation": "mean contradiction probability over 20 sampled passages, then mean over primary-response sentences",
        "auroc": float(roc_auc_score(y, u)),
        "auprc": float(average_precision_score(y, u)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sample", "score"])
    p.add_argument("dataset", choices=["scientist", "trivia", "gsm8k", "drop"])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--nli-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache", type=Path, default=Path("/tmp/selfcheckgpt_hf"))
    p.add_argument("--out", type=Path, default=Path("/tmp/selfcheckgpt_original_benchmarks"))
    a = p.parse_args()
    (sample if a.stage == "sample" else score)(a)

if __name__ == "__main__":
    main()
