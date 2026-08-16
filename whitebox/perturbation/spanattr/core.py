# -*- coding: utf-8 -*-
"""
spanattr/core.py

Gradient / interaction based span attribution for teacher-forced hallucination margin.

------------------------------------------------------------------------------
CENTRAL OBJECT: the gate vector alpha in [0,1]^P over prompt token positions.

    E(alpha)_t = E_t + alpha_t * (Ebar_t - E_t)

where Ebar is the neutralization baseline (LENGTH PRESERVING -- no token is
deleted, so no position shift is introduced).

    alpha_t = 0  ->  original token
    alpha_t = 1  ->  fully neutralized token

Everything (first-order gradient, integrated gradients, finite-difference
interaction) is computed in this single space, so all quantities share units
and sign conventions by construction.

------------------------------------------------------------------------------
SCALAR OBJECTIVE (hallucination margin), teacher-forced:

    S(alpha) = logsumexp_v [ lp(pred_variant_v | E(alpha)) ]
             - logsumexp_v [ lp(gold_variant_v | E(alpha)) ]

GAIN FUNCTION (this is what we maximize when choosing spans):

    u(Set) = S(0) - S(1_Set)     >= 0 for evidence that supports the hallucination

SIGN CONVENTION -- stated once, used everywhere.  Under u:

    I_ij = u({i,j}) - u({i}) - u({j})
    I_ij <  0   ->  REDUNDANT   (either span alone already does the damage;
                                 set function is SUBMODULAR; greedy has 1-1/e)
    I_ij >  0   ->  SYNERGISTIC (neither alone matters, both together do;
                                 set function is SUPERMODULAR)

    objective(Set) = sum_i u_i + sum_{i<j} I_ij      -> MAXIMIZE

PRECONDITION.  The redundancy / synergy READING of I assumes u_i > 0, i.e. the
span supports the hallucinated answer.  For a span with u_i < 0 (context that
actually supports the GOLD answer, so that neutralizing it makes things worse)
the verbal interpretation inverts: "either alone suffices" no longer implies
I_ij < 0.  The arithmetic of I is unaffected, but any cluster-level narrative
must be restricted to the u_i > 0 subset.  61_ reports the sign distribution of
u so this can be checked before 62_ is interpreted.

Note the objective is expressed in u, never in S.  S appears only inside the
definition of u.  (An earlier draft of this framework stated the redundancy /
synergy signs against S rather than u and had them inverted; keeping S out of
all downstream expressions is the structural fix.)

------------------------------------------------------------------------------
Offline check of the torch-free portion:
    python -m spanattr.selftest
Full check (needs torch + a real or toy Llama):
    python -m spanattr.core --smoke
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import re
import string
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional for the pure-numpy selection helpers
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


# =============================================================================
# 1. torch-free utilities  (importable and testable without torch)
# =============================================================================

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks. No scipy dependency."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def bootstrap_ci(vals: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def norm_text(s: str) -> str:
    s = re.sub(r"<\|[^|>]+\|>", " ", s)
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stable_hash(t: str) -> int:
    """Deterministic across processes, unlike builtin hash()."""
    return zlib.crc32(t.encode("utf-8")) & 0xFFFFFFFF


# =============================================================================
# 2. data structures
# =============================================================================

DEFAULT_PREFIX = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                  "Context: ")
DEFAULT_MIDDLE = ("\n\nQuestion: {question}\n\nAnswer with a short phrase only."
                  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


@dataclass
class Item:
    item_id: str
    context: str                       # the ONLY perturbable region
    question: str
    gold: str
    pred: Optional[str] = None         # y-hat; greedily generated if None
    context_prefix: str = ""           # visible but excluded from perturbation
    gold_variants: List[str] = field(default_factory=list)
    pred_variants: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "Item":
        context = d.get("context", d.get("prompt"))
        if context is None:
            raise KeyError("item requires 'context' or dataset alias 'prompt'")
        gold = d.get("gold", d.get("rgt_ans"))
        if gold is None:
            raise KeyError("item requires 'gold' or dataset alias 'rgt_ans'")
        context_prefix = d.get("context_prefix", "")
        if "context" not in d and "prompt" in d:
            head, marker, body = context.partition("Question:\n")
            if marker:
                context_prefix = head + marker
                context = body
        return Item(
            item_id=str(d.get("item_id", d.get("id", d.get("key", "NA")))),
            context=context,
            question=d.get("question", "Which of the two options is correct?"),
            gold=gold,
            pred=d.get("pred", d.get("wrg_ans")),
            context_prefix=context_prefix,
            gold_variants=list(d.get("gold_variants", [])),
            pred_variants=list(d.get("pred_variants", [])),
        )


@dataclass
class Span:
    idx: int
    start: int          # absolute index into prompt_ids, inclusive
    end: int            # exclusive
    text: str = ""

    @property
    def width(self) -> int:
        return self.end - self.start

    def tokens(self) -> range:
        return range(self.start, self.end)

    def to_dict(self) -> dict:
        return {"idx": self.idx, "start": self.start, "end": self.end, "text": self.text}


@dataclass
class Prepared:
    item: Item
    prompt_ids: "torch.Tensor"      # [P]
    ctx_start: int
    ctx_end: int
    E: "torch.Tensor"               # [P, d] original prompt embeddings
    Ebar: "torch.Tensor"            # [P, d] neutralized inside ctx, identical outside
    pred_variant_ids: List["torch.Tensor"]
    gold_variant_ids: List["torch.Tensor"]
    spans: List[Span] = field(default_factory=list)


# =============================================================================
# 3. selection helpers  (pure numpy -- the part with the sign convention)
# =============================================================================

def nms_disjoint(u: np.ndarray, spans: List[Span], m: int) -> List[int]:
    """Greedy non-maximum suppression by |u|; selected spans share no token.

    Disjointness matters: the union semantics of alpha means overlapping spans
    would double-count the same neutralized token and make I_ij uninterpretable.
    """
    order = np.argsort(-np.abs(np.asarray(u, dtype=float)), kind="mergesort")
    taken: List[int] = []
    used: set = set()
    for i in order:
        toks = set(spans[int(i)].tokens())
        if toks & used:
            continue
        taken.append(int(i))
        used |= toks
        if len(taken) >= m:
            break
    return taken


def second_order_objective(S: Sequence[int], u: np.ndarray, I: np.ndarray) -> float:
    """obj(S) = sum_{i in S} u_i + sum_{i<j in S} I_ij.   MAXIMIZE."""
    S = list(S)
    if not S:
        return 0.0
    val = float(np.asarray(u, dtype=float)[S].sum())
    for a, b in itertools.combinations(S, 2):
        val += float(I[a, b])
    return val


def greedy_select(u: np.ndarray, I: np.ndarray, k: int) -> List[int]:
    m = len(u)
    S: List[int] = []
    for _ in range(min(k, m)):
        current = second_order_objective(S, u, I)
        best, best_v = None, current
        for c in range(m):
            if c in S:
                continue
            v = second_order_objective(S + [c], u, I)
            if v > best_v + 1e-12:
                best, best_v = c, v
        if best is None:
            break
        S.append(best)
    return S


def topk_first_order(u: np.ndarray, k: int) -> List[int]:
    """Best positive singleton gains, subject to the at-most-k budget."""
    a = np.asarray(u, dtype=float)
    return [int(i) for i in np.argsort(-a) if a[int(i)] > 0][:k]


def exhaustive_select(u: np.ndarray, I: np.ndarray, k: int,
                      cap: int = 50000) -> List[int]:
    """Exact argmax of the second-order objective; falls back to greedy if too big."""
    m = len(u)
    kk = min(k, m)
    n_sets = sum(math.comb(m, size) for size in range(kk + 1))
    if n_sets > cap:
        return greedy_select(u, I, k)
    best, best_v = [], 0.0
    for size in range(1, kk + 1):
        for S in itertools.combinations(range(m), size):
            v = second_order_objective(S, u, I)
            if v > best_v + 1e-12:
                best, best_v = list(S), v
    return best


def redundancy_clusters(I: np.ndarray, tau: float) -> List[List[int]]:
    """Connected components over REDUNDANT edges (I_ij < -tau).

    Under u, redundancy is negative interaction. Members of a cluster are
    substitutes for one another, so aggregation should happen at cluster level,
    not span level, to avoid double counting the same evidence.
    """
    m = I.shape[0]
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(m):
        for b in range(a + 1, m):
            if I[a, b] < -tau:
                union(a, b)
    groups: Dict[int, List[int]] = {}
    for x in range(m):
        groups.setdefault(find(x), []).append(x)
    return [sorted(g) for g in groups.values()]


def synergy_pairs(I: np.ndarray, tau: float) -> List[Tuple[int, int, float]]:
    """Pairs that only matter jointly (I_ij > tau). These are the multi-token
    'keywords' that a top-k saliency method structurally cannot find."""
    out = []
    for a in range(I.shape[0]):
        for b in range(a + 1, I.shape[0]):
            if I[a, b] > tau:
                out.append((a, b, float(I[a, b])))
    return sorted(out, key=lambda x: -x[2])


def leading_coalition(I: np.ndarray, thresh: float = 0.3) -> List[int]:
    """Heavy components of the dominant eigenvector of I.

    Eigen-decomposition is done in SPAN space and the EIGENVECTOR is kept, not a
    sorted eigenvalue spectrum -- span identity is preserved, which is exactly
    what sorting the spectrum would destroy.
    """
    if I.shape[0] == 0:
        return []
    Isym = (np.asarray(I, dtype=float) + np.asarray(I, dtype=float).T) / 2.0
    w, V = np.linalg.eigh(Isym)
    v = V[:, int(np.argmax(np.abs(w)))]
    a = np.abs(v)
    if a.max() <= 0:
        return []
    a = a / a.max()
    return [int(i) for i in np.where(a >= thresh)[0]]


def interaction_from_gains(u_single: np.ndarray, u_pair: Dict[Tuple[int, int], float]
                           ) -> np.ndarray:
    """I_ij = u({i,j}) - u({i}) - u({j}); zero diagonal, symmetric."""
    m = len(u_single)
    I = np.zeros((m, m), dtype=float)
    for (a, b), uab in u_pair.items():
        val = float(uab) - float(u_single[a]) - float(u_single[b])
        I[a, b] = I[b, a] = val
    return I


# =============================================================================
# 4. main engine  (requires torch)
# =============================================================================

class SpanAttributor:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        baseline: str = "mean",          # mean | unk | zero
        length_norm: bool = True,
        max_rows: int = 16,              # rows per forward pass
        prefix: str = DEFAULT_PREFIX,
        middle: str = DEFAULT_MIDDLE,
        answer_prefix: str = "",
    ):
        if not _HAS_TORCH:
            raise RuntimeError("SpanAttributor requires torch")
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.length_norm = length_norm
        self.max_rows = max_rows
        self.prefix = prefix
        self.middle = middle
        self.answer_prefix = answer_prefix
        self.emb_layer = model.get_input_embeddings()
        self.d = self.emb_layer.weight.shape[1]
        self.baseline_mode = baseline
        self._baseline_vec = self._make_baseline_vec(baseline)

    # ---------------- baseline ----------------
    def _make_baseline_vec(self, mode: str):
        W = self.emb_layer.weight.detach()
        if mode == "mean":
            v = W.mean(dim=0)
        elif mode == "zero":
            v = torch.zeros(self.d, device=W.device, dtype=W.dtype)
        elif mode == "unk":
            uid = getattr(self.tok, "unk_token_id", None)
            if uid is None:
                uid = getattr(self.tok, "pad_token_id", 0) or 0
            v = W[uid]
        else:
            raise ValueError(f"unknown baseline {mode}")
        return v.clone()

    # ---------------- tokenization / preparation ----------------
    def _enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def prepare(self, item: Item) -> Prepared:
        pre_ids = self._enc(self.prefix) + self._enc(item.context_prefix)
        ctx_ids = self._enc(item.context)
        post_ids = self._enc(self.middle.format(question=item.question))
        prompt_ids = torch.tensor(pre_ids + ctx_ids + post_ids, device=self.device)
        ctx_start, ctx_end = len(pre_ids), len(pre_ids) + len(ctx_ids)

        if item.pred is None:
            item.pred = self.greedy_answer(prompt_ids)

        def _variants(main: str, extra: List[str]):
            seen, out = set(), []
            for s in [main] + list(extra):
                ids = self._enc(self.answer_prefix + s)
                if not ids:
                    continue
                key = tuple(ids)
                if key in seen:
                    continue
                seen.add(key)
                out.append(torch.tensor(ids, device=self.device))
            if not out:  # degenerate guard
                out = [torch.tensor([getattr(self.tok, "eos_token_id", 1) or 1],
                                    device=self.device)]
            return out

        E = self.emb_layer(prompt_ids).detach()
        Ebar = E.clone()
        Ebar[ctx_start:ctx_end] = self._baseline_vec.to(E.dtype)

        return Prepared(
            item=item, prompt_ids=prompt_ids, ctx_start=ctx_start, ctx_end=ctx_end,
            E=E, Ebar=Ebar,
            pred_variant_ids=_variants(item.pred, item.pred_variants),
            gold_variant_ids=_variants(item.gold, item.gold_variants),
        )

    # ---------------- span proposal ----------------
    def build_spans(self, prep: Prepared, widths: Sequence[int] = (2, 3),
                    stride: int = 1) -> List[Span]:
        """Fixed 2/3-token sliding windows over the context region.

        Overlapping by design: overlap is resolved later by NMS, and genuine
        multi-token units are meant to reappear as SYNERGISTIC pairs in I.
        """
        spans, k = [], 0
        for w in widths:
            for s in range(prep.ctx_start, prep.ctx_end - w + 1, stride):
                txt = self.tok.decode(prep.prompt_ids[s:s + w].tolist())
                spans.append(Span(idx=k, start=s, end=s + w, text=txt))
                k += 1
        prep.spans = spans
        return spans

    def build_word_spans(self, prep: Prepared, widths: Sequence[int] = (2, 3),
                         stride: int = 1) -> List[Span]:
        """Word windows over raw context, mapped back to absolute token spans."""
        words = list(re.finditer(r"\b\w+(?:['’\-]\w+)*\b", prep.item.context,
                                 flags=re.UNICODE))
        if not words:
            prep.spans = []
            return []
        offsets = None
        try:
            enc = self.tok(prep.item.context, add_special_tokens=False,
                           return_offsets_mapping=True)
            ids, off = enc["input_ids"], enc["offset_mapping"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if off and isinstance(off[0], list) and off[0] and isinstance(off[0][0], list):
                off = off[0]
            if list(ids) == prep.prompt_ids[prep.ctx_start:prep.ctx_end].tolist():
                offsets = [(int(a), int(b)) for a, b in off]
        except (TypeError, KeyError, NotImplementedError):
            offsets = None
        if offsets is None:
            if len(words) != prep.ctx_end - prep.ctx_start:
                raise RuntimeError("word spans require a fast tokenizer with offset mappings")
            offsets = [(m.start(), m.end()) for m in words]
        spans, seen, idx = [], set(), 0
        for width in widths:
            for wi in range(0, len(words) - width + 1, stride):
                char_start, char_end = words[wi].start(), words[wi + width - 1].end()
                covered = [ti for ti, (a, b) in enumerate(offsets)
                           if b > char_start and a < char_end]
                if not covered:
                    continue
                start = prep.ctx_start + covered[0]
                end = prep.ctx_start + covered[-1] + 1
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                spans.append(Span(idx=idx, start=start, end=end,
                                  text=prep.item.context[char_start:char_end]))
                idx += 1
        prep.spans = spans
        return spans

    # ---------------- alpha construction ----------------
    def alpha_from_spans(self, prep: Prepared, span_ids: Sequence[int]):
        """UNION semantics: a token is neutralized if it lies in ANY selected span."""
        a = torch.zeros(prep.prompt_ids.shape[0], device=self.device)
        for i in span_ids:
            sp = prep.spans[int(i)]
            a[sp.start:sp.end] = 1.0
        return a

    def alpha_all(self, prep: Prepared):
        a = torch.zeros(prep.prompt_ids.shape[0], device=self.device)
        a[prep.ctx_start:prep.ctx_end] = 1.0
        return a

    def _embeds(self, prep: Prepared, alpha):
        """alpha: [B,P] -> [B,P,d]"""
        E = prep.E.unsqueeze(0)
        D = (prep.Ebar - prep.E).unsqueeze(0)
        return E + alpha.unsqueeze(-1).to(E.dtype) * D

    # ---------------- scoring ----------------
    def _variant_logprob(self, pe, ans_ids):
        """pe: [B,P,d] perturbed prompt embeds -> [B] logprob of ans_ids."""
        B, P, _ = pe.shape
        A = ans_ids.shape[0]
        ae = self.emb_layer(ans_ids).detach().unsqueeze(0).expand(B, A, self.d)
        seq = torch.cat([pe, ae.to(pe.dtype)], dim=1)
        mask = torch.ones(B, P + A, device=self.device, dtype=torch.long)
        logits = self.model(inputs_embeds=seq, attention_mask=mask).logits
        lg = logits[:, P - 1: P + A - 1, :].float()      # predicts answer tokens
        lp = torch.log_softmax(lg, dim=-1)
        tgt = ans_ids.unsqueeze(0).expand(B, A)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total = tok_lp.sum(dim=-1)
        return total / A if self.length_norm else total

    def _class_logprob(self, pe, variants: List["torch.Tensor"]):
        """Semantic-class score: aggregate over surface variants of the same answer.

        With length_norm=True this is a logsumexp over per-variant MEAN logprobs,
        i.e. a heuristic that keeps different-length variants comparable rather
        than a proper mixture. Set length_norm=False for the exact mixture form.
        """
        parts = [self._variant_logprob(pe, v) for v in variants]
        return torch.logsumexp(torch.stack(parts, dim=0), dim=0)

    def S(self, prep: Prepared, alpha, grad: bool = False):
        """alpha: [B,P] -> S: [B].  Tier-1 (teacher-forced) score."""
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            pe = self._embeds(prep, alpha)
            s = (self._class_logprob(pe, prep.pred_variant_ids)
                 - self._class_logprob(pe, prep.gold_variant_ids))
        return s

    def S_batched(self, prep: Prepared, alphas):
        out = []
        for i in range(0, alphas.shape[0], self.max_rows):
            out.append(self.S(prep, alphas[i:i + self.max_rows]).detach().float().cpu())
        return torch.cat(out) if out else torch.zeros(0)

    def S0(self, prep: Prepared) -> float:
        z = torch.zeros(1, prep.prompt_ids.shape[0], device=self.device)
        return float(self.S(prep, z)[0])

    def score_ids(self, prep: Prepared, prompt_ids) -> float:
        """Score an arbitrary (same-length) prompt id sequence. Used by 64 for
        discrete word substitution -- a HELD-OUT operator w.r.t. neutralization."""
        E = self.emb_layer(prompt_ids).detach().unsqueeze(0)
        with torch.no_grad():
            s = (self._class_logprob(E, prep.pred_variant_ids)
                 - self._class_logprob(E, prep.gold_variant_ids))
        return float(s[0])

    def score_ids_batched(self, prep: Prepared, prompt_ids_batch):
        """Score a batch of same-length discrete prompt substitutions."""
        out = []
        for i in range(0, prompt_ids_batch.shape[0], self.max_rows):
            ids = prompt_ids_batch[i:i + self.max_rows]
            E = self.emb_layer(ids).detach()
            with torch.no_grad():
                s = (self._class_logprob(E, prep.pred_variant_ids)
                     - self._class_logprob(E, prep.gold_variant_ids))
            out.append(s.detach().float().cpu())
        return torch.cat(out) if out else torch.zeros(0)

    # ---------------- gain function ----------------
    def u_of_sets(self, prep: Prepared, sets: Sequence[Sequence[int]],
                  S0: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """u(Set) = S(0) - S(1_Set) for a list of span-index sets."""
        if S0 is None:
            S0 = self.S0(prep)
        if len(sets) == 0:
            return np.zeros(0), S0
        A = torch.stack([self.alpha_from_spans(prep, s) for s in sets])
        return S0 - self.S_batched(prep, A).numpy(), S0

    # ---------------- first order ----------------
    def grad_alpha(self, prep: Prepared, alpha0=None) -> np.ndarray:
        """dS/dalpha_t at alpha0 (default 0). Returns [P]."""
        P = prep.prompt_ids.shape[0]
        a = torch.zeros(P, device=self.device) if alpha0 is None else alpha0.clone()
        a = a.unsqueeze(0).requires_grad_(True)
        s = self.S(prep, a, grad=True).sum()
        g, = torch.autograd.grad(s, a)
        return g.squeeze(0).detach().float().cpu().numpy()

    def grad_embed(self, prep: Prepared) -> np.ndarray:
        """dS/dE_t at alpha=0. Returns [P,d]. Used by 64 for vocabulary decoding."""
        E = prep.E.clone().unsqueeze(0).requires_grad_(True)
        with torch.enable_grad():
            s = (self._class_logprob(E, prep.pred_variant_ids)
                 - self._class_logprob(E, prep.gold_variant_ids)).sum()
        g, = torch.autograd.grad(s, E)
        return g.squeeze(0).detach().float().cpu().numpy()

    def u_hat_first_order(self, prep: Prepared, spans: List[Span],
                          g: Optional[np.ndarray] = None) -> np.ndarray:
        """First-order prediction of u_i.  u ~ -dS/dalpha, summed over the span."""
        if g is None:
            g = self.grad_alpha(prep)
        return np.array([-float(g[sp.start:sp.end].sum()) for sp in spans])

    def integrated_gradients(self, prep: Prepared, steps: int = 32) -> np.ndarray:
        """IG in gate space, with the neutralization baseline as the IG baseline.

        Completeness:  sum_t IG_t  ==  S(0) - S(1_all)  ==  u(all).
        This is checked at runtime and reported as `completeness_rel_err`.
        """
        P = prep.prompt_ids.shape[0]
        full = self.alpha_all(prep)
        acc = np.zeros(P, dtype=np.float64)
        for m in range(steps):
            s = (m + 0.5) / steps
            acc += -self.grad_alpha(prep, full * s)   # path alpha: 0 -> 1
        acc /= steps
        acc[:prep.ctx_start] = 0.0
        acc[prep.ctx_end:] = 0.0
        return acc

    # ---------------- attention baseline (the incumbent method) ----------------
    def attention_scores(self, prep: Prepared, layers: str = "all") -> np.ndarray:
        """Mean attention mass from answer positions onto each prompt position."""
        ans = prep.pred_variant_ids[0]
        ids = torch.cat([prep.prompt_ids, ans]).unsqueeze(0)
        with torch.no_grad():
            out = self.model(input_ids=ids, output_attentions=True)
        atts = out.attentions
        if atts is None:
            raise RuntimeError("model returned no attentions; load with "
                               "attn_implementation='eager'")
        if layers == "last":
            atts = atts[-1:]
        P = prep.prompt_ids.shape[0]
        acc = np.zeros(P, dtype=np.float64)
        for A in atts:                              # [1,H,T,T]
            a = A[0].float().mean(dim=0)            # [T,T]
            acc += a[P - 1:P + ans.shape[0] - 1, :P].mean(dim=0).cpu().numpy()
        acc /= max(len(atts), 1)
        acc[:prep.ctx_start] = 0.0
        acc[prep.ctx_end:] = 0.0
        return acc

    # ---------------- null distribution ----------------
    def null_sigma(self, prep: Prepared, widths: Sequence[int] = (2, 3),
                   n_draw: int = 24, S0: Optional[float] = None,
                   seed: int = 0) -> Tuple[float, np.ndarray]:
        """Position- and length-matched random spans -> noise floor for u.

        Required: with 2-3 token spans the single-span effect is small, and
        without sigma_null a large fraction of I would be noise that the
        clustering step would happily turn into structure.
        """
        rng = random.Random(seed)
        tmp: List[Span] = []
        for _ in range(n_draw):
            w = rng.choice(list(widths))
            if prep.ctx_end - prep.ctx_start <= w:
                continue
            s = rng.randrange(prep.ctx_start, prep.ctx_end - w + 1)
            tmp.append(Span(idx=-1, start=s, end=s + w))
        if not tmp:
            return 0.0, np.zeros(0)
        saved = prep.spans
        prep.spans = tmp
        u, _ = self.u_of_sets(prep, [[i] for i in range(len(tmp))], S0=S0)
        prep.spans = saved
        return (float(np.std(u, ddof=1)) if len(u) > 1 else 0.0), u

    def random_matched_set(self, prep: Prepared, ref: List[Span],
                           seed: int = 0) -> List[Span]:
        """Random spans matched to ref in width/context decile, disjoint."""
        rng = random.Random(seed)
        out: List[Span] = []
        used: set = set()
        for sp in ref:
            valid = list(range(prep.ctx_start, prep.ctx_end - sp.width + 1))
            ctx_len = max(prep.ctx_end - prep.ctx_start, 1)
            ref_bin = min(9, 10 * (sp.start - prep.ctx_start) // ctx_len)
            chosen = None
            for radius in range(10):
                pool = [s for s in valid
                        if abs(min(9, 10 * (s - prep.ctx_start) // ctx_len) - ref_bin)
                        <= radius and not (set(range(s, s + sp.width)) & used)]
                if pool:
                    chosen = rng.choice(pool)
                    break
            if chosen is not None:
                used.update(range(chosen, chosen + sp.width))
                out.append(Span(idx=len(out), start=chosen,
                                end=chosen + sp.width))
        return out

    # ---------------- tier 2: generation ----------------
    def greedy_answer(self, prompt_ids, max_new_tokens: int = 24) -> str:
        with torch.no_grad():
            out = self.model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=torch.ones_like(prompt_ids).unsqueeze(0),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=getattr(self.tok, "pad_token_id", 0) or 0)
        return self.tok.decode(out[0, prompt_ids.shape[0]:].tolist()).strip()

    def generate_under(self, prep: Prepared, span_ids: Sequence[int], n: int = 20,
                       temperature: float = 1.0, max_new_tokens: int = 24,
                       seed: int = 0) -> List[str]:
        """Tier-2 measurement. NOTE: with inputs_embeds, HF `generate` returns ONLY
        the newly generated tokens, so no prompt slicing is needed."""
        alpha = self.alpha_from_spans(prep, span_ids).unsqueeze(0)
        pe = self._embeds(prep, alpha)
        mask = torch.ones(1, pe.shape[1], device=self.device, dtype=torch.long)
        outs = []
        for k in range(n):
            torch.manual_seed(seed + k)
            with torch.no_grad():
                g = self.model.generate(
                    inputs_embeds=pe, attention_mask=mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0), temperature=max(temperature, 1e-5),
                    top_p=0.95, pad_token_id=getattr(self.tok, "pad_token_id", 0) or 0)
            outs.append(self.tok.decode(g[0].tolist()).strip())
        return outs

    @staticmethod
    def match_rate(gens: List[str], targets: List[str]) -> float:
        tg = [norm_text(t) for t in targets if norm_text(t)]
        if not tg or not gens:
            return float("nan")
        def matches(g: str, t: str) -> bool:
            ng = norm_text(g)
            if re.search(r"[a-z0-9]", t):
                pat = r"(?:^|\s)" + re.escape(t) + r"(?:$|\s)"
                return re.search(pat, ng) is not None
            return t in ng
        return sum(any(matches(g, t) for t in tg) for g in gens) / len(gens)


# =============================================================================
# 5. toy model / tokenizer for the torch smoke test
# =============================================================================

class ToyTokenizer:
    """Deterministic whitespace tokenizer, for the offline smoke test only."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = []
        for t in text.replace("\n", " \n ").split(" "):
            if t:
                ids.append(3 + stable_hash(t) % (self.vocab_size - 3))
        return ids

    def decode(self, ids) -> str:
        if _HAS_TORCH and isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(f"<{i}>" for i in ids)


def build_toy():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=512)
    model = LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ToyTokenizer(512)


def _smoke() -> None:
    set_seed(0)
    model, tok = build_toy()
    att = SpanAttributor(model, tok, device="cpu", max_rows=8,
                         prefix="ctx: ", middle=" q: {question} a: ")
    item = Item("t1", "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "which greek letter", "delta", "kappa")
    prep = att.prepare(item)
    spans = att.build_spans(prep, widths=(2, 3), stride=1)
    print(f"[smoke] P={prep.prompt_ids.shape[0]} "
          f"ctx=[{prep.ctx_start},{prep.ctx_end}) n_spans={len(spans)}")

    S0 = att.S0(prep)
    u_all = S0 - float(att.S(prep, att.alpha_all(prep).unsqueeze(0))[0])
    ig = att.integrated_gradients(prep, steps=16)
    rel = abs(ig.sum() - u_all) / (abs(u_all) + 1e-8)
    print(f"[smoke] S0={S0:.4f} u_all={u_all:.4f} sum(IG)={ig.sum():.4f} rel_err={rel:.3f}")
    assert rel < 0.30, "IG completeness violated"

    u_hat = att.u_hat_first_order(prep, spans)
    u_meas, _ = att.u_of_sets(prep, [[i] for i in range(len(spans))], S0=S0)
    print(f"[smoke] first-order rho={spearman(u_hat, u_meas):.3f} (random weights ~0 is fine)")

    cand = nms_disjoint(u_meas, spans, m=5)
    pairs = list(itertools.combinations(range(len(cand)), 2))
    sets = [[cand[i]] for i in range(len(cand))] + [[cand[a], cand[b]] for a, b in pairs]
    uv, _ = att.u_of_sets(prep, sets, S0=S0)
    u_c = uv[:len(cand)]
    I = interaction_from_gains(u_c, {p: uv[len(cand) + k] for k, p in enumerate(pairs)})
    print(f"[smoke] I in [{I.min():.4f},{I.max():.4f}] "
          f"exh={exhaustive_select(u_c, I, 3)} greedy={greedy_select(u_c, I, 3)}")

    print(f"[smoke] attn mass={att.attention_scores(prep)[prep.ctx_start:prep.ctx_end].sum():.4f}")
    print(f"[smoke] gens={att.generate_under(prep, cand[:2], n=2, max_new_tokens=4)}")
    print(f"[smoke] sigma_null={att.null_sigma(prep, n_draw=8, S0=S0)[0]:.4f}")
    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    if ap.parse_args().smoke:
        _smoke()
