"""
Whitebox hallucination detector: logit + attention + gradient features.

Drop-in replacement for the single-layer constraint_share / logit_margin
pipeline. Extracts a feature vector per (prompt, answer) pair from a
HF causal LM, then classifies with a small logistic-regression head.

Feature groups (each grounded in recent literature):

  LOGIT  - seq mean logprob, min token logprob, perplexity,
           mean/max token entropy, first-answer-token margin
           (standard UE baselines; Guerreiro'23, Vashurin'25)

  ATTN   - lookback ratio: attention mass on the prompt/context vs the
           generated prefix, averaged per layer (Lookback-Lens,
           Chuang et al. 2024)
         - constraint-sentence attention share, but taken per-layer and
           summarized (mean over MIDDLE layers, not one layer -- single
           late-layer shares are what gave you AUROC 0.55)
         - attention entropy over prompt sentences, per layer
         - top-k Laplacian eigenvalues of the last-layer attention map
           (LapEigvals, Binkowski et al., EMNLP 2025)

  GRAD   - norm of gradients of answer NLL w.r.t. the LAST k transformer
           layers (Grad Detect, ICML'26 ws: final ~5 layers hold >97% of
           the discriminative signal). Cheap: one backward pass, only
           last-k layer params require grad.

Usage:
    det = WhiteboxDetector("Qwen/Qwen2.5-7B-Instruct", grad_last_k=5)
    feats = det.extract(prompt, answer, constraint_span=(s, e))  # dict
    # collect feats for a labeled dev set, then:
    head = DetectorHead().fit(list_of_feat_dicts, labels)
    score = head.score(feats)   # P(hallucination)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------


class WhiteboxDetector:
    def __init__(self, model_name_or_path, device="cuda", dtype=torch.bfloat16,
                 grad_last_k=5, lap_topk=10):
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype,
            attn_implementation="eager",  # required: sdpa/flash return no attn weights
        ).to(device).eval()
        self.device = device
        self.grad_last_k = grad_last_k
        self.lap_topk = lap_topk

        # freeze everything except the last k layers (for cheap grad features)
        layers = self._layers()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for layer in layers[-grad_last_k:]:
            for p in layer.parameters():
                p.requires_grad_(True)

    def _layers(self):
        m = self.model
        for attr in ("model", "transformer"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                for lattr in ("layers", "h"):
                    if hasattr(inner, lattr):
                        return getattr(inner, lattr)
        raise ValueError("could not locate transformer layers")

    # ---------------------------------------------------------------- extract
    @torch.no_grad()
    def _forward(self, input_ids, attn=True):
        return self.model(input_ids=input_ids,
                          output_attentions=attn,
                          use_cache=False)

    def extract(self, prompt: str, answer: str,
                constraint_span: tuple[int, int] | None = None,
                sentence_spans: list[tuple[int, int]] | None = None) -> dict:
        """
        constraint_span / sentence_spans: (start_char, end_char) offsets
        into `prompt` for the constraint sentence / all prompt sentences.
        """
        enc_p = self.tok(prompt, return_tensors="pt", return_offsets_mapping=True)
        enc_a = self.tok(answer, return_tensors="pt", add_special_tokens=False)
        p_ids, a_ids = enc_p["input_ids"], enc_a["input_ids"]
        n_p, n_a = p_ids.shape[1], a_ids.shape[1]
        ids = torch.cat([p_ids, a_ids], dim=1).to(self.device)

        out = self._forward(ids)
        logits = out.logits.float()          # [1, T, V]
        attns = out.attentions              # tuple(L) of [1, H, T, T]

        feats = {}
        feats.update(self._logit_feats(logits, n_p, n_a, ids))
        feats.update(self._attn_feats(attns, n_p, n_a,
                                      enc_p["offset_mapping"][0],
                                      constraint_span, sentence_spans))
        feats.update(self._grad_feats(ids, n_p))
        return feats

    # ---------------------------------------------------------------- logits
    def _logit_feats(self, logits, n_p, n_a, ids):
        # predictions for answer tokens: positions n_p-1 .. n_p+n_a-2
        sl = logits[0, n_p - 1: n_p + n_a - 1]              # [n_a, V]
        logp = F.log_softmax(sl, dim=-1)
        tgt = ids[0, n_p: n_p + n_a]
        tok_lp = logp.gather(-1, tgt[:, None]).squeeze(-1)   # [n_a]
        probs = logp.exp()
        ent = -(probs * logp).sum(-1)                        # [n_a]
        top2 = sl.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1])                   # [n_a]
        return {
            "lp_mean": tok_lp.mean().item(),
            "lp_min": tok_lp.min().item(),
            "ppl": (-tok_lp.mean()).exp().item(),
            "ent_mean": ent.mean().item(),
            "ent_max": ent.max().item(),
            "margin_first": margin[0].item(),
            "margin_mean": margin.mean().item(),
            "margin_min": margin.min().item(),
        }

    # ------------------------------------------------------------- attention
    def _attn_feats(self, attns, n_p, n_a, offsets,
                    constraint_span, sentence_spans):
        L = len(attns)
        feats = {}

        # (1) lookback ratio per layer (Lookback-Lens): for answer tokens,
        # attention mass on prompt vs on generated prefix, averaged over heads
        lookback = []
        for A in attns:                       # [1, H, T, T]
            Aa = A[0, :, n_p:, :]             # answer rows [H, n_a, T]
            to_prompt = Aa[..., :n_p].sum(-1)             # [H, n_a]
            to_prefix = Aa[..., n_p:].sum(-1) + 1e-9
            lookback.append((to_prompt / (to_prompt + to_prefix)).mean().item())
        lookback = np.array(lookback)
        feats["lookback_mean"] = lookback.mean()
        feats["lookback_late"] = lookback[int(L * 2 / 3):].mean()
        feats["lookback_slope"] = np.polyfit(np.arange(L), lookback, 1)[0]

        # (2) constraint-share per layer, summarized over middle layers
        if constraint_span is not None:
            mask = self._char_mask(offsets, constraint_span, n_p)
            if mask.any():
                shares = []
                for A in attns:
                    Aa = A[0, :, n_p:, :n_p].mean(0)      # [n_a, n_p] head-avg
                    total = Aa.sum(-1) + 1e-9
                    shares.append((Aa[:, mask].sum(-1) / total).mean().item())
                shares = np.array(shares)
                mid = shares[L // 4: 3 * L // 4]
                feats["cshare_mid"] = mid.mean()
                feats["cshare_max"] = shares.max()
                feats["cshare_std"] = shares.std()

        # (3) attention entropy over prompt positions (answer rows), late layers
        ents = []
        for A in attns[int(L * 2 / 3):]:
            Aa = A[0, :, n_p:, :n_p].mean(0)
            p = Aa / (Aa.sum(-1, keepdim=True) + 1e-9)
            ents.append((-(p * (p + 1e-12).log()).sum(-1)).mean().item())
        feats["attn_ent_late"] = float(np.mean(ents))

        # (4) LapEigvals on last-layer head-averaged attention (full seq)
        A_last = attns[-1][0].mean(0).float()             # [T, T]
        A_sym = 0.5 * (A_last + A_last.T)
        deg = A_sym.sum(-1)
        Lap = torch.diag(deg) - A_sym
        try:
            ev = torch.linalg.eigvalsh(Lap.cpu()).numpy()
            top = np.sort(ev)[::-1][: self.lap_topk]
            for i, v in enumerate(top):
                feats[f"lap_ev{i}"] = float(v)
        except Exception:
            pass
        return feats

    @staticmethod
    def _char_mask(offsets, span, n_p):
        s, e = span
        m = torch.zeros(n_p, dtype=torch.bool)
        for i, (a, b) in enumerate(offsets[:n_p].tolist()):
            if a < e and b > s and b > a:
                m[i] = True
        return m

    # ------------------------------------------------------------- gradients
    def _grad_feats(self, ids, n_p):
        """Grad Detect-style: per-layer grad norms of answer NLL over the
        last k layers. One backward pass."""
        self.model.zero_grad(set_to_none=True)
        labels = ids.clone()
        labels[:, :n_p] = -100                    # loss on answer tokens only
        out = self.model(input_ids=ids, labels=labels, use_cache=False)
        out.loss.backward()

        feats = {"grad_loss": out.loss.item()}
        norms = []
        for li, layer in enumerate(self._layers()[-self.grad_last_k:]):
            sq = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    sq += p.grad.float().pow(2).sum().item()
            n = sq ** 0.5
            norms.append(n)
            feats[f"gnorm_l{li}"] = n
        norms = np.array(norms)
        feats["gnorm_mean"] = norms.mean()
        feats["gnorm_ratio"] = norms[-1] / (norms[0] + 1e-9)
        self.model.zero_grad(set_to_none=True)
        return feats


# --------------------------------------------------------------------------
# classifier head
# --------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DetectorHead:
    """Logistic regression over the extracted features with internal CV
    for regularization strength. L1 penalty prunes dead features (like a
    flat constraint_share) automatically."""

    def __init__(self):
        self.keys = None
        self.pipe = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=10, cv=5, penalty="l1",
                                 solver="liblinear", max_iter=5000,
                                 scoring="roc_auc"),
        )

    def _mat(self, dicts):
        if self.keys is None:
            self.keys = sorted(set().union(*[d.keys() for d in dicts]))
        return np.array([[d.get(k, 0.0) for k in self.keys] for d in dicts])

    def fit(self, feat_dicts, labels):
        self.pipe.fit(self._mat(feat_dicts), np.asarray(labels, dtype=int))
        return self

    def score(self, feat_dict_or_list):
        single = isinstance(feat_dict_or_list, dict)
        X = self._mat([feat_dict_or_list] if single else feat_dict_or_list)
        p = self.pipe.predict_proba(X)[:, 1]
        return float(p[0]) if single else p

    def report(self):
        lr = self.pipe.named_steps["logisticregressioncv"]
        w = lr.coef_[0]
        kept = [(k, c) for k, c in zip(self.keys, w) if abs(c) > 1e-8]
        return sorted(kept, key=lambda z: -abs(z[1]))
