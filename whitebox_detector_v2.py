"""
Token-indexed whitebox hallucination detector (v2).

Goal: detect hallucinations AND attribute them -- did the model draw on
the CONSTRAINT span or a SHORTCUT span when generating the answer?

Core idea, building on LapEigvals (Binkowski et al., EMNLP 2025,
arXiv:2502.17598): their Laplacian is lower-triangular, so its
eigenvalues lie on the diagonal and each eigenvalue is attached to a
specific token:

    lambda_i = d_ii - a_ii,   d_ii = (sum_{u>i} a_ui) / (T - i)

d_ii is the length-normalized attention token i RECEIVES from all
subsequent tokens -- i.e. how much later generation draws on token i.
The paper sorts eigenvalues (their Eq. 3), destroying token identity.
We instead POOL THE UNSORTED DIAGONAL BY TOKEN ROLE (constraint span
vs shortcut span), per layer and head. Same spectral quantity, but now
it answers "constraint or shortcut?" -- which sorted eigenvalues cannot.

Feature groups
  ROLE-LAP   role-pooled Laplacian diagonal per layer:
             lap_c[l] (constraint), lap_s[l] (shortcut), and their
             log-ratio. Also answer-row-restricted variant (attention
             received specifically FROM answer tokens).
  TOKENFLOW  per-answer-token constraint-vs-shortcut attention ratio
             rho_t = A(t->constraint) / (A(t->constraint)+A(t->shortcut));
             summarized (mean, min, frac shortcut-dominant, first-token,
             slope) + full trajectory returned for attribution/plots.
  LAPEIG     sorted top-k Laplacian eigenvalues per layer (head-avg),
             the paper's feature, kept as a strong baseline block.
  LOGIT      seq logprob stats, token entropy, top-2 margin stats.

Every feature is computed per layer, then summarized over early/mid/late
layer bands to keep dimensionality sane for small n.

Requires: transformers with attn_implementation="eager" (sdpa/flash do
not return attention weights; also noted by the paper -- materializing
attention disables FlashAttention).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------


def _band(x: np.ndarray, name: str) -> dict:
    """Summarize a per-layer vector into early/mid/late bands."""
    L = len(x)
    b = {
        f"{name}_early": float(x[: L // 3].mean()),
        f"{name}_mid": float(x[L // 3: 2 * L // 3].mean()),
        f"{name}_late": float(x[2 * L // 3:].mean()),
        f"{name}_max": float(x.max()),
    }
    return b


class TokenIndexedDetector:
    def __init__(self, model_name_or_path, device="cuda",
                 dtype=torch.bfloat16, lap_topk=10):
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device).eval()
        self.device = device
        self.lap_topk = lap_topk

    # ------------------------------------------------------------------ util
    def _char_mask(self, offsets, span, n_p):
        s, e = span
        m = torch.zeros(n_p, dtype=torch.bool)
        for i, (a, b) in enumerate(offsets[:n_p].tolist()):
            if a < e and b > s and b > a:
                m[i] = True
        return m

    # --------------------------------------------------------------- extract
    @torch.no_grad()
    def extract(self, prompt: str, answer: str,
                constraint_span: tuple[int, int],
                shortcut_span: tuple[int, int]) -> dict:
        """Returns (features: dict, attribution: dict).

        attribution contains per-answer-token arrays for explanation:
          rho[t]        constraint share among {constraint, shortcut} mass
          tokens[t]     answer token strings
          lap_c[l], lap_s[l]  role-pooled Laplacian diagonal per layer
        """
        enc_p = self.tok(prompt, return_tensors="pt",
                         return_offsets_mapping=True)
        enc_a = self.tok(answer, return_tensors="pt",
                         add_special_tokens=False)
        p_ids, a_ids = enc_p["input_ids"], enc_a["input_ids"]
        n_p, n_a = p_ids.shape[1], a_ids.shape[1]
        T = n_p + n_a
        ids = torch.cat([p_ids, a_ids], dim=1).to(self.device)

        out = self.model(input_ids=ids, output_attentions=True,
                         use_cache=False)
        logits = out.logits.float()
        attns = out.attentions                      # L x [1, H, T, T]
        L = len(attns)

        c_mask = self._char_mask(enc_p["offset_mapping"][0],
                                 constraint_span, n_p)
        s_mask = self._char_mask(enc_p["offset_mapping"][0],
                                 shortcut_span, n_p)
        if not c_mask.any() or not s_mask.any():
            raise ValueError("constraint/shortcut span matched no tokens")
        c_idx = c_mask.nonzero().squeeze(-1).to(self.device)
        s_idx = s_mask.nonzero().squeeze(-1).to(self.device)

        feats: dict = {}
        attrib: dict = {}

        # ---------------- ROLE-LAP: unsorted Laplacian diagonal by role ----
        # d_ii over ALL subsequent tokens (paper's Eq. 2), and an
        # answer-row-restricted variant d^ans_ii (attention from answer only)
        lap_c, lap_s = np.zeros(L), np.zeros(L)
        lap_c_ans, lap_s_ans = np.zeros(L), np.zeros(L)
        lap_sorted_topk = []
        for l, A in enumerate(attns):
            Ah = A[0].float().mean(0)               # head-avg [T, T]
            # receive-attention: column sums below the diagonal
            col = torch.arange(T, device=Ah.device)
            denom_all = (T - 1 - col).clamp(min=1).float()
            recv_all = (Ah.tril(-1).sum(0)) / denom_all       # [T]
            diag = torch.diagonal(Ah)
            lam = recv_all - diag                              # unsorted eigvals
            lap_c[l] = lam[c_idx].mean().item()
            lap_s[l] = lam[s_idx].mean().item()
            # answer-restricted receive: rows n_p..T-1 only
            recv_ans = Ah[n_p:, :].sum(0) / max(n_a, 1)        # [T]
            lap_c_ans[l] = recv_ans[c_idx].mean().item()
            lap_s_ans[l] = recv_ans[s_idx].mean().item()
            # paper-style sorted top-k (baseline block)
            k = min(self.lap_topk, T)
            lap_sorted_topk.append(
                torch.sort(lam, descending=True).values[:k].cpu().numpy())

        eps = 1e-9
        feats.update(_band(lap_c, "lapc"))
        feats.update(_band(lap_s, "laps"))
        feats.update(_band(np.log((lap_c + eps) / (lap_s + eps)), "lap_logratio"))
        feats.update(_band(lap_c_ans, "lapc_ans"))
        feats.update(_band(lap_s_ans, "laps_ans"))
        feats.update(_band(np.log((lap_c_ans + eps) / (lap_s_ans + eps)),
                           "lap_ans_logratio"))
        lap_sorted = np.stack(lap_sorted_topk)      # [L, k]
        for j in range(lap_sorted.shape[1]):
            feats.update(_band(lap_sorted[:, j], f"lapeig{j}"))
        attrib["lap_c"], attrib["lap_s"] = lap_c, lap_s

        # ---------------- TOKENFLOW: per-answer-token role attribution -----
        # use late-layer band (grounding lives late); rho_t in [0,1],
        # >0.5 means the answer token drew more on constraint than shortcut
        late = attns[2 * L // 3:]
        Aa = torch.stack([A[0].float().mean(0)[n_p:, :n_p] for A in late]
                         ).mean(0)                  # [n_a, n_p]
        mass_c = Aa[:, c_idx].sum(-1)
        mass_s = Aa[:, s_idx].sum(-1)
        rho = (mass_c / (mass_c + mass_s + eps)).cpu().numpy()   # [n_a]
        feats["rho_mean"] = float(rho.mean())
        feats["rho_min"] = float(rho.min())
        feats["rho_first"] = float(rho[0])
        feats["rho_frac_shortcut"] = float((rho < 0.5).mean())
        if n_a > 2:
            feats["rho_slope"] = float(np.polyfit(np.arange(n_a), rho, 1)[0])
        attrib["rho"] = rho
        attrib["tokens"] = self.tok.convert_ids_to_tokens(a_ids[0])

        # ---------------- LOGIT ------------------------------------------
        sl = logits[0, n_p - 1: T - 1]
        logp = F.log_softmax(sl, dim=-1)
        tgt = ids[0, n_p:T]
        tok_lp = logp.gather(-1, tgt[:, None]).squeeze(-1)
        probs = logp.exp()
        ent = -(probs * logp).sum(-1)
        top2 = sl.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        feats.update({
            "lp_mean": tok_lp.mean().item(), "lp_min": tok_lp.min().item(),
            "ent_mean": ent.mean().item(), "ent_max": ent.max().item(),
            "margin_first": margin[0].item(),
            "margin_mean": margin.mean().item(),
            "margin_min": margin.min().item(),
        })
        return feats, attrib

    # ---------------------------------------------------------- explanation
    @staticmethod
    def explain(attrib, threshold=0.5) -> str:
        """Human-readable attribution: which answer tokens leaned on the
        shortcut instead of the constraint."""
        rho, toks = attrib["rho"], attrib["tokens"]
        bad = [(t, r) for t, r in zip(toks, rho) if r < threshold]
        if not bad:
            return "All answer tokens drew primarily on the constraint span."
        lines = [f"  {t!r}: constraint share {r:.2f}" for t, r in bad]
        return ("Answer tokens drawing more on the SHORTCUT span:\n"
                + "\n".join(lines))
