#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKID — Shortcut-Key Intervention Detector
=========================================

Black-box (API-only, no GPU) detection of shortcut-key hallucinations of the
kind formalized in TRAPQA ("Understanding Why Language Models Hallucinate:
Testing Reasoning Against Priors", arXiv:2607.00447).

Core idea
---------
The paper models an answer as a mixture over two latent inference paths:

    P(y|z) ~= P(k*,t*|z) P(y|z;k*,t*)  +  P(ks,ts|z) P(y|z;ks,ts)

where (k*,t*) is the constraint-sensitive path and (ks,ts) is a
pretraining-frequent shortcut path. A hallucination occurs when the shortcut
posterior dominates (Thm 3.4), which by Thm 3.6 implies positive inference
loss. Crucially, the decisive constraint C enters the computation only
through the constraint path; under shortcut dominance (Assumption 3.1(ii) +
3.3) the answer is causally *inert* to C.

SKID therefore performs do()-style interventions on the prompt and checks the
answer's *sensitivity profile*:

    A faithful answer is COVARIANT with the constraint and INVARIANT to
    salience / surface form. A shortcut answer is INVARIANT to the constraint
    and (often) COVARIANT with salience / surface form.

Signals (all obtained via chat-completions API calls to the *subject* model):

  K  probe_violation : closed-book eliminative probe on the *chosen* answer
                       contradicts the decisive constraint (mirrors the
                       paper's supplementary probes; catches "known-fact
                       hallucinations", i.e. Table 2's `Hall. | both`).
  N  neg_invariant   : answer unchanged when the constraint's polarity is
                       flipped  -> constraint causally inert.
  A  abl_invariant   : answer unchanged when the constraint is removed
                       -> answer equals the zero-constraint prior default.
  E  emph_rescue     : answer changes when the constraint is made maximally
                       salient -> original answer was salience-limited.
  P  para_flip       : answer changes under a meaning-preserving paraphrase
                       -> answer keyed to surface statistics.
  S  swap_flip       : the chosen *entity/action* changes when option order
                       is swapped -> positional shortcut.

Detection rule in this copy (both benchmarks):

    (probe_two_sided AND (neg_invariant OR abl_invariant))
        OR (neg_invariant AND emph_rescue)

Branch 1 is a knowledge contradiction: the subject's own closed-book answers
discriminate against its multiple-choice pick (scientist: per-candidate fact
probes; reallife: per-option action-feasibility probes). Branch 2 is the
salience-limited-shortcut signature: the constraint is causally inert under a
polarity flip yet restating it with emphasis changes the answer. The original
weighted score is retained as `weighted_score` for auditing, but does not
determine the flag. Every item gets a full evidence trace for auditing.

Usage
-----
  # Real run (subject = model being audited; generator builds perturbations)
  export DASHSCOPE_API_KEY=...
  python skid.py --benchmark scientist --data shuffled_prepend_names_question.json \
      --subject qwen:qwen3.5-flash --generator qwen:qwen3.5-flash \
      --limit 200 --out sci_results.jsonl

  python skid.py --benchmark reallife --data question_and_result.json \
      --subject qwen:qwen3.5-flash --generator qwen:qwen3.5-flash \
      --out rl_results.jsonl

  # Offline pipeline validation with a simulated subject (no network needed)
  python skid.py --benchmark scientist --data shuffled_prepend_names_question.json \
      --subject mock:subject,shortcut=0.30,seed=7 --generator mock:generator --limit 300

Provider specs:  qwen:MODEL | anthropic:MODEL | openai:MODEL | deepseek:MODEL | gemini:MODEL
                 | openai-compat:MODEL@BASE_URL#ENV_VAR | mock:...
API keys read from DASHSCOPE_API_KEY (or QWEN_API_KEY) / ANTHROPIC_API_KEY /
OPENAI_API_KEY / DEEPSEEK_API_KEY /
GEMINI_API_KEY (or the env var named in an openai-compat spec).

Only Python stdlib is required.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def norm_text(s: str) -> str:
    """Accent-insensitive, case-insensitive, whitespace-collapsed form."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

def stable_unit(*parts: str) -> float:
    """Deterministic pseudo-random float in [0,1) from strings (for mocks)."""
    h = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00c0-\u00dc])")

def split_sentences(text: str):
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]

def extract_json_block(text: str):
    """Parse the first JSON object in a model response (fences tolerated)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None

# --------------------------------------------------------------------------
# Providers (uniform .chat interface). `tag`/`meta` are ignored by real
# providers; they exist so the Mock subject can simulate regime-dependent
# behavior for offline pipeline validation.
# --------------------------------------------------------------------------

class ProviderError(RuntimeError):
    pass

def _post_json(url: str, headers: dict, payload: dict,
               timeout: int = 180, retries: int = 5) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
            if e.code in (429, 500, 502, 503, 504, 529):
                retry_after = e.headers.get("retry-after")
                delay = float(retry_after) if retry_after else min(2 ** attempt * 1.5, 30)
                time.sleep(delay + random.random())
                continue
            raise ProviderError(last)
        except Exception as e:  # timeouts, connection resets
            last = repr(e)
            time.sleep(min(2 ** attempt * 1.5, 30) + random.random())
    raise ProviderError(f"exhausted retries: {last}")

class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise ProviderError("ANTHROPIC_API_KEY not set")
        self.spec = f"anthropic:{model}"

    def chat(self, system: str, user: str, temperature: float = 0.0,
             max_tokens: int = 1024, tag=None, meta=None) -> str:
        out = _post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            {"model": self.model, "max_tokens": max_tokens,
             "temperature": temperature, "system": system,
             "messages": [{"role": "user", "content": user}]},
        )
        return "".join(b.get("text", "") for b in out.get("content", [])
                       if b.get("type") == "text").strip()

class OpenAICompatProvider:
    """Covers OpenAI, DeepSeek, Gemini (OpenAI-compat endpoint), or custom."""
    def __init__(self, model: str, base_url: str, env_var: str, label: str,
                 extra_body: dict | None = None, fallback_env_var: str | None = None):
        self.model, self.base = model, base_url.rstrip("/")
        self.key = os.environ.get(env_var)
        if not self.key and fallback_env_var:
            self.key = os.environ.get(fallback_env_var)
        if not self.key:
            names = f"{env_var} or {fallback_env_var}" if fallback_env_var else env_var
            raise ProviderError(f"{names} not set")
        self.spec = f"{label}:{model}"
        self.extra_body = dict(extra_body or {})

    def chat(self, system: str, user: str, temperature: float = 0.0,
             max_tokens: int = 1024, tag=None, meta=None) -> str:
        payload = {"model": self.model, "temperature": temperature,
                   "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        payload.update(self.extra_body)
        out = _post_json(
            f"{self.base}/chat/completions",
            {"Authorization": f"Bearer {self.key}"},
            payload,
        )
        msg = out["choices"][0]["message"]
        return (msg.get("content") or "").strip()

def make_provider(spec: str):
    kind, _, rest = spec.partition(":")
    kind = kind.lower()
    if kind == "qwen":
        return OpenAICompatProvider(
            rest, os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "DASHSCOPE_API_KEY", "qwen", {"enable_thinking": False},
            fallback_env_var="QWEN_API_KEY")
    if kind == "anthropic":
        return AnthropicProvider(rest)
    if kind == "openai":
        return OpenAICompatProvider(rest, "https://api.openai.com/v1",
                                    "OPENAI_API_KEY", "openai")
    if kind == "deepseek":
        return OpenAICompatProvider(rest, "https://api.deepseek.com",
                                    "DEEPSEEK_API_KEY", "deepseek")
    if kind == "gemini":
        return OpenAICompatProvider(
            rest, "https://generativelanguage.googleapis.com/v1beta/openai",
            "GEMINI_API_KEY", "gemini")
    if kind == "openai-compat":
        m = re.match(r"([^@]+)@([^#]+)#(\w+)$", rest)
        if not m:
            raise ProviderError("openai-compat spec: MODEL@BASE_URL#ENV_VAR")
        return OpenAICompatProvider(m.group(1), m.group(2), m.group(3),
                                    "openai-compat")
    if kind == "mock":
        return make_mock(rest)
    raise ProviderError(f"unknown provider spec: {spec}")

# --------------------------------------------------------------------------
# Mock subject + mock generator (offline pipeline validation only).
#
# MockSubject simulates the paper's latent key–task regimes per item:
#   SHORTCUT items answer the trap option regardless of the constraint
#   (invariant under negate/ablate; sometimes rescued by emphasis) and
#   answer isolated probes correctly with prob `probe_acc` — i.e. the
#   "known-fact hallucination" phenomenon of Table 2.
#   FAITHFUL items follow the constraint (flip under negation, revert to the
#   prior under ablation, stable under emphasis/paraphrase).
# Small noise terms create ignorance-driven errors (detector should partly
# miss these) and correct-for-wrong-reason items (false-positive pressure),
# so mock metrics are informative rather than trivially perfect.
# --------------------------------------------------------------------------

class MockSubject:
    def __init__(self, shortcut=0.30, rescue=0.40, probe_acc=0.85,
                 ignorance=0.05, lucky_prior=0.10, seed=0):
        self.p, self.rescue, self.probe_acc = shortcut, rescue, probe_acc
        self.ignorance, self.lucky = ignorance, lucky_prior
        self.seed = str(seed)
        self.spec = (f"mock:subject,shortcut={shortcut},seed={seed}")

    # -- regime assignment ------------------------------------------------
    def _regime(self, key: str) -> str:
        r = stable_unit(self.seed, key, "regime")
        if r < self.p:
            # a shortcut item; a slice of them have prior pointing at the
            # *correct* option (correct-for-wrong-reasons -> FP pressure)
            return ("shortcut_lucky"
                    if stable_unit(self.seed, key, "lucky") < self.lucky
                    else "shortcut")
        if stable_unit(self.seed, key, "ign") < self.ignorance:
            return "ignorant"       # wrong for non-shortcut reasons
        return "faithful"

    def _answer_entity(self, meta, tag) -> str:
        """Return the *entity/action text* the mock intends to choose."""
        key, gold, wrong = meta["key"], meta["gold_text"], meta["wrong_text"]
        reg = self._regime(key)
        prior = wrong  # the benchmark is built so the salient prior = trap
        if reg == "shortcut_lucky":
            prior = gold
        if tag in (None, "original", "paraphrase"):
            if reg == "faithful":
                return gold
            if reg == "ignorant":
                return wrong
            return prior
        if tag == "swap":
            return self._answer_entity(dict(meta), None)  # order-invariant
        if tag == "ablate":
            # constraint gone -> everyone reverts to the prior default
            return prior if reg.startswith("shortcut") else (
                wrong if stable_unit(self.seed, key, "abl") < 0.8 else gold)
        if tag == "negate":
            if reg == "faithful":
                return wrong          # flips with the constraint
            if reg == "ignorant":
                return gold           # constraint-sensitive, wrong knowledge
            return prior              # inert
        if tag == "emphasize":
            if reg.startswith("shortcut"):
                return (gold if stable_unit(self.seed, key, "resc") < self.rescue
                        else prior)
            return gold if reg == "faithful" else wrong
        return prior

    def chat(self, system, user, temperature=0.0, max_tokens=1024,
             tag=None, meta=None):
        if meta is None:
            return "1"
        if tag == "probe":
            # closed-book fact probe about meta["probe_name"]
            key = meta["key"]
            truth = meta["probe_truth"]          # "yes"/"no"
            ok = stable_unit(self.seed, key, meta["probe_name"], "p") < self.probe_acc
            if self._regime(key) == "ignorant":
                ok = stable_unit(self.seed, key, "ignp") < 0.35
            return truth if ok else ("no" if truth == "yes" else "yes")
        ent = self._answer_entity(meta, tag)
        if meta["kind"] == "scientist":
            return ent
        # reallife: reply with the digit of `ent` in the options *as shown*
        shown = meta["shown_options"]
        return str(shown.index(ent) + 1) if ent in shown else "1"

class MockGenerator:
    """Deterministic perturbations so mock runs need zero network.
    NOTE: uses gold labels only to tag which Real-Life option satisfies the
    constraint — acceptable because this generator exists purely to validate
    plumbing, never for real detection."""
    spec = "mock:generator"

    def chat(self, system, user, temperature=0.0, max_tokens=2048,
             tag=None, meta=None):
        item, kind = meta["item"], meta["kind"]
        if kind == "scientist":
            return json.dumps(heuristic_scientist_perturbations(
                item["question_body"]))
        sents = split_sentences(item["scenario"])
        cons = sents[1] if len(sents) > 1 else sents[0]
        keep = [s for s in sents if s != cons]
        return json.dumps({
            "constraint": cons,
            "required_condition": ("The physically constrained item/medium "
                                   "must be present for this step."),
            "ablated_scenario": " ".join(keep) + " They are waiting there.",
            "negated_scenario": item["scenario"] +
                " Update: the physical item is explicitly NOT needed for "
                "this step and must stay where it is.",
            "emphasized_scenario": item["scenario"] + " Note: " + cons,
            "paraphrased_scenario": "Situation: " + item["scenario"],
            "feasibility_questions": [
                f"If someone takes the action \"{item['options'][0]}\", "
                "can the constrained step in the scenario be completed?",
                f"If someone takes the action \"{item['options'][1]}\", "
                "can the constrained step in the scenario be completed?",
            ],
            "option_satisfies": [i + 1 == item["gold_idx"] for i in range(2)],
        })

def make_mock(rest: str):
    parts = [p for p in rest.split(",") if p]
    if parts and parts[0] == "generator":
        return MockGenerator()
    kw = {}
    for p in parts[1:] if parts and parts[0] == "subject" else parts:
        k, _, v = p.partition("=")
        kw[k] = float(v) if "." in v else int(v) if v.isdigit() else v
    return MockSubject(**{k: v for k, v in kw.items()
                          if k in ("shortcut", "rescue", "probe_acc",
                                   "ignorance", "lucky_prior", "seed")})

# --------------------------------------------------------------------------
# Benchmark loading -> common item schema
# --------------------------------------------------------------------------

SCI_HEADER = re.compile(
    r"Choose one of the following two options.*?\n1\.\s*(.+?)\n2\.\s*(.+?)\n"
    r"Question:\s*\n?(.*)", re.S)

def load_items(kind: str, path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    if kind == "scientist":
        for it in raw:
            m = SCI_HEADER.match(it["prompt"])
            if not m:
                continue
            c1, c2, q = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            items.append({
                "kind": kind, "key": it["key"],
                "cands": [c1, c2],
                "question_body": q,
                "gold_text": it["rgt_ans"], "wrong_text": it["wrg_ans"],
            })
    elif kind == "reallife":
        for i, it in enumerate(raw):
            gold_idx = int(it["answer"])
            items.append({
                "kind": kind, "key": f"rl_{i:04d}",
                "scenario": it["question"].strip(),
                "options": [o.strip() for o in it["options"]],
                "gold_idx": gold_idx,
                "gold_text": it["options"][gold_idx - 1].strip(),
                "wrong_text": it["options"][2 - gold_idx].strip(),
                "mistake_models": it.get("mistake_models", []),
            })
    else:
        raise ValueError(kind)
    return items

# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------

SUBJ_SYS_SCI = ("Answer with exactly one of the two candidate names, "
                "and nothing else.")
SUBJ_SYS_RL = "Answer with a single digit, 1 or 2, and nothing else."
PROBE_SYS = "Answer with exactly one word: Yes or No."

def sci_prompt(c1: str, c2: str, question: str) -> str:
    return ("Choose one of the following two options as the answer to the "
            f"question below:\n1. {c1}\n2. {c2}\nQuestion:\n{question}")

def rl_prompt(scenario: str, o1: str, o2: str) -> str:
    return (f"Scenario: {scenario}\nOption1: {o1}\nOption2: {o2}\n"
            "Question: Which one should I choose? "
            "Answer 1 for Option1 and 2 for Option2.")

# --------------------------------------------------------------------------
# Perturbation generation
# --------------------------------------------------------------------------

GEN_SYS = ("You are a meticulous text-transformation engine for a "
           "hallucination-detection experiment. Output strict JSON only — "
           "no markdown fences, no commentary.")

GEN_USER_SCI = """Below is a two-candidate disambiguation question. It contains a biographical description broadly compatible with both candidates, plus one DECISIVE CONSTRAINT (often a negated statement such as "However, ... never ...") that rules out exactly one candidate.

Candidates: 1. {c1}   2. {c2}

QUESTION:
{question}

Return strict JSON with exactly these keys:
- "constraint": the decisive constraint sentence(s), copied verbatim from the question.
- "ablated_question": the question with the decisive constraint completely removed (fix grammar; change nothing else; keep the final "Who is this person?").
- "negated_question": the question with the polarity of every decisive clause flipped ("never received X" -> "received X"; "did not do Y" -> "did Y"; an affirmative decisive clause becomes negated). Change nothing else.
- "emphasized_question": the same question, but with the decisive constraint ALSO stated at the very front as "Decisive fact: <constraint>". Keep all original content.
- "paraphrased_question": a full paraphrase preserving every fact and the exact meaning of the decisive constraint, ending with "Who is this person?".
- "probe_question_template": a standalone closed-book Yes/No question testing the decisive attribute for a single person, using the placeholder {{NAME}} (e.g., "Was {{NAME}} awarded the Nobel Prize in Physiology or Medicine?"). If the constraint has multiple clauses, probe the single most decisive one.
- "violating_answer": "yes" or "no" — the probe answer that means the person VIOLATES the decisive constraint as stated in the original question."""

GEN_USER_RL = """Below is a two-option everyday scenario. One option is a tempting shortcut; the other is required by a physical / spatial / procedural / medium-specific constraint stated in the scenario.

SCENARIO:
{scenario}

Option1: {o1}
Option2: {o2}

Return strict JSON with exactly these keys:
- "constraint": the clause in the scenario that states the requirement, copied verbatim.
- "required_condition": one sentence stating what must be physically true or present for the step to succeed (e.g., "The motorcycle must be physically present so the frame VIN can be inspected.").
- "ablated_scenario": the scenario with the requirement removed or neutralized (e.g., the person is simply waiting to meet you), keeping the setting and both options natural.
- "negated_scenario": the scenario rewritten so the requirement now points the OTHER way — explicitly state that the previously required object/medium/action is NOT needed and/or must not be used for this step (e.g., "the paperwork alone is checked; the motorcycle must remain parked"). Keep the same setting; both options must remain grammatical.
- "emphasized_scenario": the original scenario with one added final sentence: "Note: <required_condition>".
- "paraphrased_scenario": a faithful paraphrase preserving every fact including the requirement.
- "feasibility_questions": a JSON array of two standalone closed-book Yes/No questions, one per option, in option order. Question i must (a) describe the concrete situation that results from taking Option{{i+1}} (restate the action; do NOT use the words "Option1"/"Option2"), and (b) ask whether the constrained step stated in the scenario can then be completed. Each question must be answerable from everyday physical/procedural knowledge alone, with "Yes" meaning the step CAN be completed. Example: "If someone walks to the inspection lane without their motorcycle, can a clerk compare the motorcycle's frame VIN with a title document there?"
- "option_satisfies": a JSON array of two booleans; element i is true iff choosing Option{{i+1}} satisfies the required condition (brings/keeps the required object, medium, or original item)."""

REQ_KEYS = {
    "scientist": ["constraint", "ablated_question", "negated_question",
                  "emphasized_question", "paraphrased_question",
                  "probe_question_template", "violating_answer"],
    "reallife": ["constraint", "required_condition", "ablated_scenario",
                 "negated_scenario", "emphasized_scenario",
                 "paraphrased_scenario", "feasibility_questions",
                 "option_satisfies"],
}

NEG_TOKENS = re.compile(r"\b(never|not|nor|no\b|without|didn.t|wasn.t)\b", re.I)

def heuristic_scientist_perturbations(question: str) -> dict:
    """Structural fallback: the decisive constraint is the final sentence(s)
    before 'Who is this person?' (per the benchmark's construction)."""
    q = re.sub(r"\s*Who is this person\?\s*$", "", question.strip())
    sents = split_sentences(q)
    cut = len(sents) - 1
    if cut > 0 and not NEG_TOKENS.search(sents[-1]):
        cut -= 1
    constraint = " ".join(sents[cut:])
    body = " ".join(sents[:cut]).strip()
    neg = constraint
    neg = re.sub(r"\bnever\s+", "", neg, flags=re.I)
    neg = re.sub(r"\bdid not\s+", "did ", neg, flags=re.I)
    neg = re.sub(r"\bdoes not\s+", "does ", neg, flags=re.I)
    neg = re.sub(r"\bwas not\s+", "was ", neg, flags=re.I)
    neg = re.sub(r",?\s*nor did they\s+", ", and they did ", neg, flags=re.I)
    neg = re.sub(r",?\s*nor\s+", ", and ", neg, flags=re.I)
    if neg == constraint:
        neg = "It is not the case that: " + constraint
    probe_stmt = re.sub(
        r"\b([Tt]hey|[Hh]e|[Ss]he|[Tt]his (?:[a-z\-]+ ){0,2}"
        r"(?:person|scientist|scholar|individual|leader|researcher|academic|"
        r"figure|physicist|chemist|biologist|mathematician|engineer|"
        r"astrophysicist|economist|historian|philosopher|inventor|architect|"
        r"politician|statesperson))\b",
        "{NAME}", constraint, count=1)
    if "{NAME}" not in probe_stmt:
        probe_stmt = ("Consider {NAME}. The following statement is about "
                      "{NAME}: " + constraint)
    tail = " Who is this person?"
    return {
        "constraint": constraint,
        "ablated_question": (body + tail).strip(),
        "negated_question": (body + " " + neg + tail).strip(),
        "emphasized_question": ("Decisive fact: " + constraint + " " +
                                body + " " + constraint + tail).strip(),
        "paraphrased_question": ("Consider the following description. " +
                                 q + tail).strip(),
        "probe_question_template": ("Is the following statement true? \""
                                    + probe_stmt + "\" Answer Yes or No."),
        "violating_answer": "no",
        "_provenance": "heuristic",
    }

def validate_pert(kind: str, item: dict, p: dict) -> bool:
    if not isinstance(p, dict):
        return False
    if any(k not in p or p[k] in (None, "") for k in REQ_KEYS[kind]):
        return False
    if kind == "scientist":
        if p["violating_answer"].strip().lower() not in ("yes", "no"):
            return False
        if "{NAME}" not in p["probe_question_template"]:
            return False
        # the ablated question must actually drop the constraint
        c_norm = norm_text(p["constraint"])[:60]
        if c_norm and c_norm in norm_text(p["ablated_question"]):
            return False
    else:
        sat = p["option_satisfies"]
        if (not isinstance(sat, list) or len(sat) != 2
                or not all(isinstance(b, bool) for b in sat)):
            return False
        fq = p["feasibility_questions"]
        if (not isinstance(fq, list) or len(fq) != 2
                or not all(isinstance(q, str) and q.strip() for q in fq)):
            return False
        # standalone means no benchmark option labels may leak in
        if any(re.search(r"\boption\s*[12]\b", q, re.I) for q in fq):
            return False
    return True

def build_perturbations(item: dict, gen, cache) -> dict:
    kind = item["kind"]
    ck = sha1(f"pert|{getattr(gen, 'spec', 'gen')}|{item['key']}")
    hit = cache.get(ck)
    if hit is not None:
        return json.loads(hit)
    if kind == "scientist":
        user = GEN_USER_SCI.format(c1=item["cands"][0], c2=item["cands"][1],
                                   question=item["question_body"])
    else:
        user = GEN_USER_RL.format(scenario=item["scenario"],
                                  o1=item["options"][0], o2=item["options"][1])
    pert = None
    for _ in range(2):
        try:
            raw = gen.chat(GEN_SYS, user, temperature=0.0, max_tokens=2048,
                           tag="generate", meta={"item": item, "kind": kind})
            cand = extract_json_block(raw)
            if validate_pert(kind, item, cand):
                cand["_provenance"] = cand.get("_provenance", "llm")
                pert = cand
                break
        except ProviderError:
            time.sleep(1.0)
    if pert is None and kind == "scientist":
        pert = heuristic_scientist_perturbations(item["question_body"])
        if not validate_pert(kind, item, pert):
            pert = None
    if pert is None:
        pert = {"_provenance": "failed"}
    cache.put(ck, json.dumps(pert, ensure_ascii=False))
    return pert

# --------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------

def parse_scientist_answer(text: str, c1: str, c2: str):
    """Return 1, 2 or None (abstain/off-option)."""
    t = norm_text(text)
    n1, n2 = norm_text(c1), norm_text(c2)
    hit1, hit2 = n1 in t, n2 in t
    if hit1 and not hit2:
        return 1
    if hit2 and not hit1:
        return 2
    if hit1 and hit2:
        return 1 if t.find(n1) < t.find(n2) else 2
    s1, s2 = n1.split()[-1], n2.split()[-1]
    hit1, hit2 = re.search(rf"\b{re.escape(s1)}\b", t), \
                 re.search(rf"\b{re.escape(s2)}\b", t)
    if hit1 and not hit2:
        return 1
    if hit2 and not hit1:
        return 2
    m = re.search(r"\b([12])\b", text)
    return int(m.group(1)) if m else None

def parse_digit_answer(text: str):
    m = re.search(r"\b([12])\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"option\s*([12])", text, re.I)
    return int(m.group(1)) if m else None

def parse_yes_no(text: str):
    m = re.search(r"\b(yes|no)\b", text, re.I)
    return m.group(1).lower() if m else None

# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "probe_violation": 0.35,
    "probe_two_sided": 0.10,   # bonus: other option confirmed to satisfy
    "neg_invariant":   0.25,
    "abl_invariant":   0.15,
    "emph_rescue":     0.15,
    "para_flip":       0.05,
    "swap_flip":       0.05,
}
ALL_SIGNALS = ("probe", "negate", "ablate", "emphasize", "paraphrase", "swap")

@dataclass
class Config:
    signals: tuple = ALL_SIGNALS
    samples: int = 1
    threshold: float = 0.30
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    temperature_vote: float = 0.7   # used only when samples > 1

def _ask(subject, cache, system, user, tag, meta, cfg: Config):
    """Query subject (with cache + optional self-consistency vote).
    Returns list of raw responses (len == samples)."""
    outs = []
    n = cfg.samples if tag not in ("probe",) else 1
    for i in range(n):
        temp = 0.0 if n == 1 else cfg.temperature_vote
        ck = sha1(f"ans|{subject.spec}|{meta['key']}|{tag}|{i}|{temp}|{sha1(user)}")
        hit = cache.get(ck)
        if hit is None:
            hit = subject.chat(system, user, temperature=temp,
                               max_tokens=64, tag=tag, meta=meta)
            cache.put(ck, hit)
        outs.append(hit)
    return outs

def _vote(parsed):
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return None
    return max(set(parsed), key=parsed.count)

def detect_item(item: dict, subject, gen, cache, cfg: Config,
                provided_answer=None) -> dict:
    kind = item["kind"]
    pert = build_perturbations(item, gen, cache)
    rec = {"key": item["key"], "kind": kind,
           "pert_provenance": pert.get("_provenance", "?"),
           "signals": {}, "answers": {}, "evidence": {}}

    # ---- assemble prompts per variant ------------------------------------
    if kind == "scientist":
        c1, c2 = item["cands"]
        base_meta = {"key": item["key"], "kind": kind,
                     "gold_text": item["gold_text"],
                     "wrong_text": item["wrong_text"]}
        prompts = {"original": sci_prompt(c1, c2, item["question_body"])}
        if pert.get("_provenance") != "failed":
            prompts.update({
                "ablate": sci_prompt(c1, c2, pert["ablated_question"]),
                "negate": sci_prompt(c1, c2, pert["negated_question"]),
                "emphasize": sci_prompt(c1, c2, pert["emphasized_question"]),
                "paraphrase": sci_prompt(c1, c2, pert["paraphrased_question"]),
            })
        prompts["swap"] = sci_prompt(c2, c1, item["question_body"])
        parse = lambda txt: parse_scientist_answer(txt, c1, c2)
        subj_sys = SUBJ_SYS_SCI
    else:
        o1, o2 = item["options"]
        base_meta = {"key": item["key"], "kind": kind,
                     "gold_text": item["gold_text"],
                     "wrong_text": item["wrong_text"],
                     "shown_options": [o1, o2]}
        prompts = {"original": rl_prompt(item["scenario"], o1, o2)}
        if pert.get("_provenance") != "failed":
            prompts.update({
                "ablate": rl_prompt(pert["ablated_scenario"], o1, o2),
                "negate": rl_prompt(pert["negated_scenario"], o1, o2),
                "emphasize": rl_prompt(pert["emphasized_scenario"], o1, o2),
                "paraphrase": rl_prompt(pert["paraphrased_scenario"], o1, o2),
            })
        prompts["swap"] = rl_prompt(item["scenario"], o2, o1)
        parse = parse_digit_answer
        subj_sys = SUBJ_SYS_RL

    # ---- original answer --------------------------------------------------
    if provided_answer is not None:
        a0 = provided_answer
        rec["answers"]["original"] = f"[provided:{a0}]"
    else:
        outs = _ask(subject, cache, subj_sys, prompts["original"],
                    "original", base_meta, cfg)
        rec["answers"]["original"] = outs[0]
        a0 = _vote([parse(o) for o in outs])
    rec["a0"] = a0
    if a0 is None:                      # off-option -> hallucination per paper
        rec.update(score=1.0, flag=True, reason="off_option")
        return rec

    def answered(tag, meta_extra=None):
        if tag not in prompts:
            return None
        meta = dict(base_meta)
        if kind == "reallife" and tag == "swap":
            meta["shown_options"] = [item["options"][1], item["options"][0]]
        if meta_extra:
            meta.update(meta_extra)
        outs = _ask(subject, cache, subj_sys, prompts[tag], tag, meta, cfg)
        rec["answers"][tag] = outs[0]
        return _vote([parse(o) for o in outs])

    sig = rec["signals"]

    # ---- intervention signals ---------------------------------------------
    if "negate" in cfg.signals:
        a = answered("negate")
        sig["neg_invariant"] = None if a is None else (a == a0)
    if "ablate" in cfg.signals:
        a = answered("ablate")
        sig["abl_invariant"] = None if a is None else (a == a0)
    if "emphasize" in cfg.signals:
        a = answered("emphasize")
        sig["emph_rescue"] = None if a is None else (a != a0)
    if "paraphrase" in cfg.signals:
        a = answered("paraphrase")
        sig["para_flip"] = None if a is None else (a != a0)
    if "swap" in cfg.signals:
        # Parse the swapped-prompt response in the *shown* (swapped) order so
        # that both name replies and bare-digit replies resolve consistently,
        # then map back to original indexing (shown i -> original 3-i).
        meta = dict(base_meta)
        if kind == "reallife":
            meta["shown_options"] = [item["options"][1], item["options"][0]]
        outs = _ask(subject, cache, subj_sys, prompts["swap"], "swap",
                    meta, cfg)
        rec["answers"]["swap"] = outs[0]
        if kind == "scientist":
            swap_parse = lambda t: parse_scientist_answer(t, c2, c1)
        else:
            swap_parse = parse_digit_answer
        a_swap = _vote([swap_parse(o) for o in outs])
        sig["swap_flip"] = None if a_swap is None else ((3 - a_swap) != a0)

    # ---- closed-book probe signal ------------------------------------------
    if "probe" in cfg.signals and pert.get("_provenance") != "failed":
        if kind == "scientist":
            tmpl, viol = pert["probe_question_template"], \
                         pert["violating_answer"].strip().lower()
            chosen = item["cands"][a0 - 1]
            other = item["cands"][2 - a0]
            truths = {item["wrong_text"]: viol,
                      item["gold_text"]: ("no" if viol == "yes" else "yes")}
            pa = {}
            for name in (chosen, other):
                q = tmpl.replace("{NAME}", name)
                out = _ask(subject, cache, PROBE_SYS, q, "probe",
                           dict(base_meta, probe_name=name,
                                probe_truth=truths.get(name, "yes")), cfg)[0]
                rec["answers"][f"probe::{name}"] = out
                pa[name] = parse_yes_no(out)

            if pa[chosen] is None:
                sig["probe_violation"] = None
            else:
                sig["probe_violation"] = (pa[chosen] == viol)
                sig["probe_two_sided"] = (
                    sig["probe_violation"] and pa[other] is not None
                    and pa[other] != viol)
            rec["evidence"]["probe_template"] = tmpl
        else:
            # Real-Life: per-option closed-book feasibility probes.
            #
            # The earlier single probe ("Is condition C required for step
            # S?") fails here for two reasons. (1) Acquiescence bias: the
            # benchmark is built so C *is* required, and subjects agree-bias
            # toward "Yes" on any plausible requirement, so the answer
            # carries no information. (2) The decisive half of the old
            # signal, `option_satisfies[a0-1]`, is the *generator's* label,
            # not the subject's knowledge -- so the signal degenerated into a
            # re-grade by another model rather than a contradiction *within*
            # the subject.
            #
            # Instead we ask the subject, closed-book, one feasibility
            # question per option ("If someone <does action i>, can <the
            # constrained step> be completed?"). A violation requires the
            # subject's own answers to discriminate in the incriminating
            # direction: its CHOSEN action cannot complete the step, while
            # the alternative can. Requiring the two answers to *differ*
            # makes the signal immune to blanket yes-bias and blanket
            # no-bias by construction, exactly like the two-sided probe in
            # the scientist track.
            fq = pert["feasibility_questions"]
            sat = pert["option_satisfies"]
            pa = {}
            for idx in (1, 2):
                q = fq[idx - 1].strip()
                if not q.endswith("?"):
                    q += "?"
                q += " Answer Yes or No."
                out = _ask(subject, cache, PROBE_SYS, q, "probe",
                           dict(base_meta, probe_name=f"opt{idx}",
                                probe_truth="yes" if sat[idx - 1] else "no"),
                           cfg)[0]
                rec["answers"][f"probe::opt{idx}"] = out
                pa[idx] = parse_yes_no(out)

            chosen_pa, other_pa = pa[a0], pa[3 - a0]
            if chosen_pa is None:
                sig["probe_violation"] = None
            else:
                # subject's own physics: my chosen action cannot complete
                # the step ...
                sig["probe_violation"] = (chosen_pa == "no")
                # ... and the action I rejected can (discriminative).
                sig["probe_two_sided"] = (chosen_pa == "no"
                                          and other_pa == "yes")
            # auxiliary evidence: does the subject's feasibility judgment
            # agree with the generator's satisfies-labels? (not used in the
            # flag; useful for auditing generator quality)
            if chosen_pa is not None and other_pa is not None:
                rec["evidence"]["gen_agree"] = (
                    (chosen_pa == "yes") == bool(sat[a0 - 1])
                    and (other_pa == "yes") == bool(sat[2 - a0]))
            rec["evidence"]["feasibility_questions"] = fq

    rec["evidence"]["constraint"] = pert.get("constraint", "")

    # ---- aggregate ----------------------------------------------------------
    # Rationale for the rule below.
    #
    # An earlier rule -- `probe_violation AND neg_invariant` -- fired on a
    # probe measured *only on the chosen answer*. That probe is corrupted by a
    # blanket "yes-bias": asked "Was <famous scientist> awarded <prize X>?",
    # the subject tends to answer "Yes" for *any* famous name, so a "Yes" on
    # the chosen candidate is not evidence that the chosen candidate is the
    # constraint-violator. Empirically this produced the bulk of false
    # positives: items where the subject picked the CORRECT candidate but the
    # probe returned the violating answer for *both* candidates (a
    # non-discriminative probe).
    #
    # The current rule has two independent branches:
    #
    # Branch 1 (knowledge contradiction): the subject's closed-book knowledge
    # must DISCRIMINATE between the two candidates -- `probe_two_sided`
    # (chosen looks like a violation AND the alternative looks clean) -- while
    # the constraint is causally inert (`neg_invariant` or `abl_invariant`).
    # This is the genuine "known-fact hallucination" signature: the model's
    # own facts contradict its multiple-choice pick and confirm the
    # alternative.
    #
    # Branch 2 (salience rescue): `neg_invariant AND emph_rescue`. The answer
    # ignores a polarity flip of the constraint (causally inert), yet CHANGES
    # when the same constraint is restated with maximal salience. A faithful
    # answer can do neither-or-both, but not this combination: if the model
    # were actually reading the constraint, flipping its polarity would move
    # the answer at least as much as merely repeating it louder. This is the
    # salience-limited shortcut signature and needs no probe at all, so it is
    # immune to the probe yes-bias by construction.
    weighted = sum(cfg.weights.get(k, 0.0) for k, v in sig.items() if v)
    rec["weighted_score"] = round(min(weighted, 1.0), 4)

    constraint_inert = bool(sig.get("neg_invariant") or sig.get("abl_invariant"))
    knowledge_contradiction = bool(sig.get("probe_two_sided") and constraint_inert)
    salience_rescue = bool(sig.get("neg_invariant") and sig.get("emph_rescue"))
    rec["flag"] = knowledge_contradiction or salience_rescue
    rec["score"] = 1.0 if rec["flag"] else 0.0
    rec["detection_rule"] = ("(probe_two_sided AND (neg_invariant OR "
                             "abl_invariant)) OR (neg_invariant AND "
                             "emph_rescue)")
    return rec

# --------------------------------------------------------------------------
# Cache (thread-safe JSONL key-value store)
# --------------------------------------------------------------------------

class Cache:
    def __init__(self, path: str | None):
        self.path, self.mem, self.lock = path, {}, threading.Lock()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self.mem[d["k"]] = d["v"]
                    except Exception:
                        pass

    def get(self, k):
        return self.mem.get(k)

    def put(self, k, v):
        with self.lock:
            self.mem[k] = v
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": v},
                                       ensure_ascii=False) + "\n")

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def auroc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return float("nan")
    allv = sorted(pos + neg)
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2 + 1
        ranks[allv[i]] = r
        i = j + 1
    rsum = sum(ranks[s] for s in pos)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def prf(labels, flags):
    tp = sum(1 for l, f in zip(labels, flags) if l and f)
    fp = sum(1 for l, f in zip(labels, flags) if not l and f)
    fn = sum(1 for l, f in zip(labels, flags) if l and not f)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp, fp, fn

# --------------------------------------------------------------------------
# Runner / CLI
# --------------------------------------------------------------------------

def gold_index(item):
    if item["kind"] == "reallife":
        return item["gold_idx"]
    return 1 if item["cands"][0] == item["gold_text"] else 2

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SKID detector (discriminative two-sided probe rule)")
    ap.add_argument("--benchmark", choices=["scientist", "reallife"],
                    default="reallife")
    ap.add_argument("--data", required=True)
    ap.add_argument("--subject", required=True,
                    help="provider spec of the model under audit")
    ap.add_argument("--generator", default=None,
                    help="provider spec for perturbation generation "
                         "(default: same as subject)")
    ap.add_argument("--limit", type=int, default=500,
                    help="evaluate a random subsample of this size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1,
                    help="self-consistency votes per variant (>=1)")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="compatibility option; the explicit AND rule ignores it")
    ap.add_argument("--signals", default=",".join(ALL_SIGNALS),
                    help="comma list among: " + ",".join(ALL_SIGNALS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="claude_method/skid_cache.jsonl")
    ap.add_argument("--out", default="claude_method/skid_fixed_results_3.7_rl.jsonl")
    ap.add_argument("--answers-file", default=None,
                    help="optional JSON {item_key: option_index or text} of "
                         "pre-existing subject answers to audit")
    ap.add_argument("--dump-examples", type=int, default=3)
    args = ap.parse_args(argv)

    subject = make_provider(args.subject)
    gen = make_provider(args.generator) if args.generator else subject
    items = load_items(args.benchmark, args.data)
    if args.limit and args.limit < len(items):
        rng = random.Random(args.seed)
        items = rng.sample(items, args.limit)

    provided = {}
    if args.answers_file:
        with open(args.answers_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for it in items:
            if it["key"] in raw:
                v = raw[it["key"]]
                if isinstance(v, int):
                    provided[it["key"]] = v
                else:
                    tgt = norm_text(str(v))
                    opts = it["cands"] if it["kind"] == "scientist" \
                        else it["options"]
                    for idx, o in enumerate(opts, 1):
                        if norm_text(o) == tgt or norm_text(o) in tgt:
                            provided[it["key"]] = idx

    cfg = Config(signals=tuple(s.strip() for s in args.signals.split(",")),
                 samples=max(1, args.samples), threshold=args.threshold)
    cache = Cache(args.cache)

    t0, results, errors = time.time(), [], 0
    def work(it):
        return detect_item(it, subject, gen, cache, cfg,
                           provided_answer=provided.get(it["key"]))
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex, \
         open(args.out, "w", encoding="utf-8") as fout:
        futs = {ex.submit(work, it): it for it in items}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            it = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                errors += 1
                rec = {"key": it["key"], "error": repr(e), "score": None}
            rec["gold_idx"] = gold_index(it)
            if "mistake_models" in it:
                rec["mistake_models"] = it["mistake_models"]
            if rec.get("a0") is not None or "reason" in rec:
                rec["hallucinated"] = (rec.get("a0") != rec["gold_idx"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)
            if n % 25 == 0 or n == len(items):
                print(f"\r  processed {n}/{len(items)} "
                      f"({time.time()-t0:.0f}s, errors={errors})",
                      end="", flush=True)
    print()

    # ------------------------------ report ------------------------------
    ok = [r for r in results if r.get("score") is not None]
    labeled = [r for r in ok if "hallucinated" in r]
    labels = [1 if r["hallucinated"] else 0 for r in labeled]
    scores = [r["score"] for r in labeled]
    flags = [r["flag"] for r in labeled]
    n_h = sum(labels)
    print("\n=== SKID report ===")
    print(f"subject={subject.spec}  generator={getattr(gen,'spec','-')}")
    print(f"items scored: {len(ok)}   subject hallucination rate: "
          f"{n_h}/{len(labeled)} ({100*n_h/max(1,len(labeled)):.1f}%)")
    if n_h and n_h < len(labeled):
        p, r, f1, tp, fp, fn = prf(labels, flags)
        print(f"detector [two-sided probe rule]: precision={p:.3f} "
              f"recall={r:.3f} F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
        print(f"detector AUROC (continuous score): {auroc(labels, scores):.3f}")
        print("\nper-signal diagnostics "
              "(fire-rate on hallucinated vs on correct):")
        for s in DEFAULT_WEIGHTS:
            on_h, on_c = [], []
            for r_, l in zip(labeled, labels):
                v = r_.get("signals", {}).get(s)
                if v is None:
                    continue
                (on_h if l else on_c).append(1 if v else 0)
            if on_h or on_c:
                fh = sum(on_h) / len(on_h) if on_h else float("nan")
                fc = sum(on_c) / len(on_c) if on_c else float("nan")
                print(f"  {s:<16} halluc: {fh:5.1%}   correct: {fc:5.1%}   "
                      f"(n={len(on_h)}/{len(on_c)})")
    prov = {}
    for r_ in ok:
        prov[r_.get("pert_provenance", "?")] = \
            prov.get(r_.get("pert_provenance", "?"), 0) + 1
    print(f"perturbation provenance: {prov}")

    # optional: agreement with the error sets recorded in the benchmark file
    with_mm = [r_ for r_ in labeled if "mistake_models" in r_]
    if with_mm:
        fams = sorted({m for r_ in with_mm for m in r_["mistake_models"]})
        if fams:
            print("\noverlap of reproduced hallucinations with the "
                  "benchmark's recorded per-model errors:")
            for fam in fams:
                bench = [1 if fam in r_["mistake_models"] else 0
                         for r_ in with_mm]
                mine = [1 if r_["hallucinated"] else 0 for r_ in with_mm]
                both = sum(1 for b, m in zip(bench, mine) if b and m)
                print(f"  vs {fam:<9} benchmark errors={sum(bench):3d}  "
                      f"reproduced={sum(mine):3d}  overlap={both:3d}")

    shown = 0
    for r_ in labeled:
        if shown >= args.dump_examples:
            break
        if r_["hallucinated"] and r_["flag"]:
            shown += 1
            print(f"\n--- flagged hallucination example ({r_['key']}) ---")
            print(f"  constraint: {r_['evidence'].get('constraint','')[:140]}")
            print(f"  signals: { {k: v for k, v in r_['signals'].items()} }")
            print(f"  score={r_['score']}")
    print(f"\nfull evidence written to {args.out}")

if __name__ == "__main__":
    main()
