"""Shared runtime for deployable counterfactual treatment probes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from common import DATA, LM, extract_final, is_abstain, is_truncated

FEATURE_VERSION = 1
BASE = "none"
TREATMENTS = ("T-RAG", "T-Clean", "T-Budget", "T-Abstain")
CLEAN_ANSWER = (
    "First remove or ignore statements irrelevant to solving the question. "
    "Then answer the cleaned question. Do not discuss the cleaning step.\n\n"
)
ABSTAIN = (
    "Before answering, check whether the premises are valid and the information "
    "is sufficient for a unique supported answer. If not, explicitly say that "
    "the question is unanswerable or underdetermined instead of guessing.\n\n"
)


class TfidfRetriever:
    """Small local retriever for deployment; corpus JSONL needs id/text fields."""
    def __init__(self, path):
        from sklearn.feature_extraction.text import TfidfVectorizer
        rows = [json.loads(x) for x in open(path) if x.strip()]
        self.ids = [str(r.get("id", i)) for i, r in enumerate(rows)]
        self.texts = [str(r.get("text", r.get("content", ""))) for r in rows]
        self.vec = TfidfVectorizer(stop_words="english", max_features=100000)
        self.matrix = self.vec.fit_transform(self.texts)

    def retrieve(self, query):
        scores = (self.matrix @ self.vec.transform([query]).T).toarray()[:, 0]
        i = int(scores.argmax())
        return self.texts[i], self.ids[i], float(scores[i])


def rag_context(sample, rag_mode, retriever=None):
    if rag_mode == "gold":
        text = sample.get("meta", {}).get("gold_passage", "")
        return text, "gold", 1.0 if text else 0.0
    if retriever is None:
        return "", "missing-corpus", 0.0
    return retriever.retrieve(sample["q_trig"])


def prompt_for(treatment, sample, rag_mode="gold", retriever=None):
    q = sample["q_trig"]
    if treatment == BASE:
        return q, {}
    if treatment == "T-RAG":
        passage, doc_id, score = rag_context(sample, rag_mode, retriever)
        prompt = f"Reference material:\n{passage}\n\nQuestion:\n{q}" if passage else q
        return prompt, {"rag_doc_id": doc_id, "rag_score": score, "rag_found": bool(passage)}
    if treatment == "T-Clean":
        return CLEAN_ANSWER + q, {}
    if treatment == "T-Abstain":
        return ABSTAIN + q, {}
    return q, {}


def generate_one(lm, treatment, sample, rag_mode="gold", retriever=None,
                 probe_max_tokens=512, budget_max_think=1024):
    prompt, meta = prompt_for(treatment, sample, rag_mode, retriever)
    kwargs = dict(temperature=0.0, n=1, max_tokens=probe_max_tokens)
    if treatment == "T-Budget" and lm.is_reasoner:
        kwargs["max_think"] = max(
            budget_max_think,
            int(sample.get("meta", {}).get("avg_think_tokens", 0) * 1.5),
        )
    response = lm.chat([prompt], **kwargs)[0][0]
    cap = probe_max_tokens
    return prompt, response, meta, cap


def endpoint_features(lm, prompt, response):
    """Teacher-force prompt+probe response and extract endpoint states."""
    import torch
    rendered = lm._render([prompt])[0] + response
    old_side = lm.tok.truncation_side
    lm.tok.truncation_side = "left"
    enc = lm.tok(rendered, return_tensors="pt", truncation=True,
                 max_length=4096).to(lm.device)
    lm.tok.truncation_side = old_side
    with torch.inference_mode():
        out = lm.llm(**enc, output_hidden_states=True, use_cache=False)
    states = torch.stack(out.hidden_states)[:, 0, -1].float().cpu().numpy()
    logits = out.logits[0, -1].float()
    p = torch.softmax(logits, -1)
    top2 = p.topk(2).values
    scalars = np.array([
        -(p * (p + 1e-9).log()).sum().item(),
        (top2[0] - top2[1]).item(),
        logits.max().item(),
        len(lm.tok.encode(response, add_special_tokens=False)),
        len(extract_final(response)),
        float(is_abstain(response)),
    ], dtype=np.float32)
    return states.astype(np.float16), scalars


def extract_record(lm, sample, rag_mode="gold", retriever=None,
                   probe_max_tokens=512, budget_max_think=1024):
    conditions = (BASE,) + TREATMENTS
    states, scalars, responses, prompts, metadata = [], [], {}, {}, {}
    for treatment in conditions:
        prompt, response, meta, cap = generate_one(
            lm, treatment, sample, rag_mode, retriever,
            probe_max_tokens, budget_max_think,
        )
        h, s = endpoint_features(lm, prompt, response)
        s = np.concatenate([s, np.array([float(is_truncated(response, lm, cap))], np.float32)])
        states.append(h); scalars.append(s)
        responses[treatment] = response; prompts[treatment] = prompt; metadata[treatment] = meta
    return {
        "feature_version": np.array(FEATURE_VERSION, dtype=np.int64),
        "conditions": np.array(conditions),
        "states": np.stack(states),
        "scalars": np.stack(scalars),
        "responses": responses,
        "prompts": prompts,
        "metadata": metadata,
    }


def vector_at_layer(npz, layer):
    states = npz["states"].astype(np.float32)
    scalars = npz["scalars"].astype(np.float32)
    delta_h = states[1:, layer] - states[0, layer]
    delta_s = scalars[1:] - scalars[0]
    # Include treatment endpoints and changes; baseline itself is deliberately omitted.
    return np.concatenate([delta_h.reshape(-1), states[1:, layer].reshape(-1),
                           delta_s.reshape(-1), scalars[1:].reshape(-1)])


def feature_dir(model_name):
    return DATA / "features" / f"{model_name.split('/')[-1]}__cf_probe"
