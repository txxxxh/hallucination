# -*- coding: utf-8 -*-
"""
spanattr/selftest.py  --  torch-free validation of the selection layer.

Run:  python -m spanattr.selftest

This exercises the part of the framework where the sign convention lives.
It builds SYNTHETIC set functions with known ground-truth structure (a
redundant pair, a synergistic pair, an independent span) and asserts that:

  * redundancy really does produce I_ij < 0 under the gain function u
  * synergy really does produce I_ij > 0 under u
  * the second-order objective recovers the synergistic pair while
    first-order top-k provably cannot
  * greedy == exhaustive on a submodular (purely redundant) instance
  * clustering / coalition / NMS behave as documented

No model, no torch, no network required.
"""
from __future__ import annotations

import itertools
import numpy as np

from .core import (Span, SpanAttributor, spearman, pearson, nms_disjoint, second_order_objective,
                   greedy_select, exhaustive_select, topk_first_order,
                   redundancy_clusters, synergy_pairs, leading_coalition,
                   interaction_from_gains, norm_text, bootstrap_ci, stable_hash)

FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# -----------------------------------------------------------------------------
# ground-truth set function
# -----------------------------------------------------------------------------
# 5 spans:
#   0,1 -> REDUNDANT pair: each alone destroys 1.0; together still 1.0
#   2,3 -> SYNERGISTIC pair: each alone destroys 0.05; together 1.9
#   4   -> independent, destroys 0.6
def u_true(S) -> float:
    S = set(S)
    v = 0.0
    if S & {0, 1}:
        v += 1.0
    if 2 in S:
        v += 0.05
    if 3 in S:
        v += 0.05
    if {2, 3} <= S:
        v += 1.8
    if 4 in S:
        v += 0.6
    return v


def main() -> int:
    print("=" * 72)
    print("spanattr selftest (torch-free)")
    print("=" * 72)

    # ---- 1. sign convention -------------------------------------------------
    print("\n[1] interaction sign convention under the gain function u")
    m = 5
    u = np.array([u_true([i]) for i in range(m)])
    pairs = list(itertools.combinations(range(m), 2))
    I = interaction_from_gains(u, {p: u_true(p) for p in pairs})

    check("redundant pair (0,1) has I < 0", I[0, 1] < 0, f"I_01={I[0,1]:+.3f}")
    check("synergistic pair (2,3) has I > 0", I[2, 3] > 0, f"I_23={I[2,3]:+.3f}")
    check("independent span 4 has |I| small",
          all(abs(I[4, j]) < 1e-9 for j in range(m) if j != 4),
          f"max|I_4j|={max(abs(I[4, j]) for j in range(m) if j != 4):.3e}")
    check("I symmetric, zero diagonal",
          np.allclose(I, I.T) and np.allclose(np.diag(I), 0))

    # ---- 2. objective is consistent with the true set function --------------
    print("\n[2] second-order objective reproduces u_true on all subsets")
    errs = []
    for k in range(1, m + 1):
        for S in itertools.combinations(range(m), k):
            errs.append(abs(second_order_objective(S, u, I) - u_true(S)))
    check("exact for |S|<=2 (by construction)",
          max(abs(second_order_objective(S, u, I) - u_true(S))
              for k in (1, 2) for S in itertools.combinations(range(m), k)) < 1e-9)
    check("max abs error over ALL subsets is small (no 3rd-order terms here)",
          max(errs) < 1e-9, f"max_err={max(errs):.2e}")

    # ---- 3. first order cannot find synergy ---------------------------------
    print("\n[3] first-order top-k vs second-order selection, k=2")
    fo = sorted(topk_first_order(u, 2))
    so = sorted(exhaustive_select(u, I, 2))
    check("first-order top-2 MISSES the synergistic pair", so != fo,
          f"first_order={fo} (u={second_order_objective(fo,u,I):.2f})  "
          f"second_order={so} (u={second_order_objective(so,u,I):.2f})")
    check("second-order top-2 IS the synergistic pair {2,3}", so == [2, 3])
    check("second-order beats first-order in true gain",
          u_true(so) > u_true(fo), f"{u_true(so):.2f} > {u_true(fo):.2f}")
    uneg = np.array([1.0, -2.0, -3.0])
    Ineg = np.zeros((3, 3))
    check("at-most-k selection never adds negative marginal gain",
          topk_first_order(uneg, 3) == [0]
          and greedy_select(uneg, Ineg, 3) == [0]
          and exhaustive_select(uneg, Ineg, 3) == [0])

    # ---- 4. greedy vs exhaustive on a submodular instance --------------------
    print("\n[4] greedy vs exhaustive")
    # IMPORTANT documented limitation: greedy takes the highest single-gain span
    # first, so on a SUPERMODULAR (synergistic) instance it can never recover the
    # pair. Greedy is only safe when redundancy dominates. 63_ therefore uses
    # exhaustive search whenever C(m,k) is affordable and warns otherwise.
    gsel = sorted(greedy_select(u, I, 2))
    check("greedy PROVABLY fails on the synergistic instance (documented)",
          gsel != so, f"greedy={gsel} vs exhaustive={so}")
    sub = [0, 1, 4]                      # drop the synergy pair -> submodular
    us, Is = u[sub], I[np.ix_(sub, sub)]
    check("greedy == exhaustive on the submodular sub-instance {0,1,4}",
          sorted(greedy_select(us, Is, 2)) == sorted(exhaustive_select(us, Is, 2)),
          f"greedy={sorted(greedy_select(us,Is,2))}")
    rng = np.random.default_rng(0)
    ratios = []
    for t in range(200):                     # purely redundant => submodular
        uu = rng.uniform(0.2, 1.5, size=6)
        II = -np.abs(rng.uniform(0, 0.4, size=(6, 6)))
        II = (II + II.T) / 2
        np.fill_diagonal(II, 0.0)
        g = second_order_objective(greedy_select(uu, II, 3), uu, II)
        e = second_order_objective(exhaustive_select(uu, II, 3), uu, II)
        ratios.append(g / e if e > 0 else 1.0)
    check("greedy >= (1-1/e) * optimal on 200 submodular instances",
          min(ratios) >= 1 - 1 / np.e, f"min_ratio={min(ratios):.4f}")

    # ---- 5. clustering / coalition ------------------------------------------
    print("\n[5] redundancy clustering and leading coalition")
    cl = redundancy_clusters(I, tau=0.1)
    check("{0,1} land in one redundancy cluster",
          any(set(c) == {0, 1} for c in cl), f"clusters={cl}")
    check("2,3,4 are each their own cluster",
          all(any(c == [x] for c in cl) for x in (2, 3, 4)))
    sp = synergy_pairs(I, tau=0.1)
    check("synergy_pairs surfaces (2,3) first", sp and sp[0][:2] == (2, 3), f"{sp[:2]}")
    coal = leading_coalition(I, thresh=0.3)
    check("leading coalition is non-trivial", 1 <= len(coal) <= m, f"coalition={coal}")

    # ---- 6. NMS disjointness ------------------------------------------------
    print("\n[6] NMS over overlapping 2/3-token sliding windows")
    spans, k = [], 0
    for w in (2, 3):
        for s in range(10, 24 - w + 1):
            spans.append(Span(idx=k, start=s, end=s + w))
            k += 1
    uu = np.abs(np.random.default_rng(1).normal(size=len(spans)))
    keep = nms_disjoint(uu, spans, m=4)
    toks = [set(spans[i].tokens()) for i in keep]
    check("selected spans are pairwise token-disjoint",
          all(not (a & b) for a, b in itertools.combinations(toks, 2)),
          f"kept={[(spans[i].start, spans[i].end) for i in keep]}")
    check("NMS returns exactly m when feasible", len(keep) == 4)
    check("NMS keeps the global argmax",
          keep[0] == int(np.argmax(uu)))

    # ---- 7. stats utilities -------------------------------------------------
    print("\n[7] statistics utilities")
    x = np.arange(20, dtype=float)
    check("spearman monotone == 1", abs(spearman(x, x ** 3) - 1.0) < 1e-9)
    check("spearman anti-monotone == -1", abs(spearman(x, -x) + 1.0) < 1e-9)
    check("pearson linear == 1", abs(pearson(x, 2 * x + 5) - 1.0) < 1e-9)
    check("spearman on constant is nan", np.isnan(spearman(x, np.ones(20))))
    lo, hi = bootstrap_ci(np.random.default_rng(2).normal(0.5, 0.1, 200), n_boot=500)
    check("bootstrap CI brackets the mean", lo < 0.5 < hi, f"[{lo:.3f},{hi:.3f}]")
    check("norm_text strips articles/punctuation",
          norm_text("The  Eiffel Tower, Paris!") == "eiffel tower paris")
    check("norm_text removes chat special tokens with a boundary",
          norm_text("Emmanuelle Charpentier<|eot_id|>")
          == "emmanuelle charpentier")
    check("answer matching uses phrase boundaries",
          SpanAttributor.match_rate(["Russia", "US"], ["US"]) == 0.5)
    check("stable_hash is deterministic", stable_hash("abc") == stable_hash("abc"))

    print("\n" + "=" * 72)
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        return 1
    print("ALL SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
