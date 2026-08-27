#!/usr/bin/env python3
"""Frozen SAPLMA/ICR transfer from Scientist known/full to multidomain v6."""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "315_saplma_icr_scientist_to_multidomain"
TARGET_CACHE = OUT / "target_features"
MODEL = "/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
SEEDS = (42, 43, 44)

sys.path.insert(0, str(HERE / "third_party" / "ICR_Probe_official"))
from src.utils import ICRProbe  # noqa: E402


def target_rows():
    return importlib.import_module("150_multidomain_v6_scientist_frozen_transfer").rows()


def scientist_rows(source):
    mod = importlib.import_module("100_collect_multilayer_trajectory")
    if source == "scientist_known":
        return mod._scientist_rows("known")
    valid = {json.loads(s)["key"] for s in
             (RUNS / "273_full_scientist_saplma_paper" / "predictions.jsonl").open()}
    return [r for r in mod._scientist_rows("all") if r["key"] in valid]


def metric(y, score):
    pred = score >= .5
    return {
        "n": int(len(y)), "correct": int(y.sum()),
        "incorrect": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "accuracy_at_0.5": float(accuracy_score(y, pred)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)),
        "confusion_tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }


def collect_saplma():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = TARGET_CACHE / "saplma"
    cache.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True).eval()
    rows = target_rows()
    for i, row in enumerate(rows, 1):
        path = cache / f"{row['key']}.npz"
        if path.exists():
            continue
        text = row["prompt"] + " " + str(row["pred"])
        enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            h = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[28][0, -1]
        np.savez_compressed(path, key=row["key"], x=h.float().cpu().numpy().astype(np.float16))
        if i % 25 == 0 or i == len(rows):
            print(f"SAPLMA target {i}/{len(rows)}", flush=True)


def collect_icr():
    # Reuse the audited official adapter so prompt boundaries and ICR computation
    # remain identical to the Scientist feature collection.
    adapter = importlib.import_module("293_icr_probe_official")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = TARGET_CACHE / "icr"
    cache.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager", local_files_only=True).eval()
    rows = target_rows()
    for i, row in enumerate(rows, 1):
        path = cache / f"{row['key']}.npz"
        if path.exists():
            continue
        # Scientist-style prompts: the stored prompt is already the exact user text.
        wrapped = {"raw": {"prompt": row["prompt"]}, "pred": row["pred"]}
        x, heads, top_p, ni, no, boundaries = adapter.official_feature(
            model, tok, "scientist_known", wrapped)
        np.savez_compressed(path, key=row["key"], x=x, induction_heads=heads,
                            top_p_mean=top_p, input_tokens=ni, answer_tokens=no,
                            boundaries=json.dumps(boundaries))
        print(f"ICR target {i}/{len(rows)}", flush=True)


class SAPLMA(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_saplma(x, y, tx, seed):
    seed_all(seed)
    model = SAPLMA(x.shape[1]).cuda()
    opt = torch.optim.Adam(model.parameters())
    lossfn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(5):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda(); opt.zero_grad()
            lossfn(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(tx).cuda())).cpu().numpy()


def fit_icr(x, y, tx, seed):
    seed_all(seed)
    tr, va = train_test_split(np.arange(len(y)), test_size=.1, stratify=y,
                              random_state=seed)
    model = ICRProbe(x.shape[1]).cuda()
    lossfn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=.5, patience=5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr].astype(np.float32))),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    xv = torch.from_numpy(x[va]).cuda(); yv = torch.from_numpy(y[va].astype(np.float32)).cuda()
    best, best_loss = None, float("inf")
    for _ in range(50):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda(); opt.zero_grad()
            lossfn(model(xb).squeeze(1), yb).backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            value = lossfn(model(xv).squeeze(1), yv).item()
        sched.step(value)
        if value < best_loss:
            best_loss = value
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(tx).cuda()).squeeze(1).cpu().numpy()


def load_source(method, source):
    rows = scientist_rows(source)
    if method == "saplma":
        hidden = torch.load(RUNS / "286_aiersilan_full_scientist" / "hidden_states.pt",
                            map_location="cpu")
        lookup = {k: hidden["hidden_states"][i, 28].float().numpy()
                  for i, k in enumerate(hidden["keys"])}
        for row in rows:
            if row["key"] in lookup:
                continue
            path = RUNS / "100_scientist_trajectory_l8" / f"{row['key']}.npz"
            with np.load(path, allow_pickle=True) as z:
                layer = z["layers"].astype(int)
                lookup[row["key"]] = z["last"][np.flatnonzero(layer == 28)[0]].astype(np.float32)
        x = np.stack([lookup[r["key"]] for r in rows]).astype(np.float32)
    else:
        root = RUNS / "298_icr_probe_paper_strict"
        if not root.exists():
            root = RUNS / "293_icr_probe_official"
        values = {}
        for sub in ("scientist_full", "scientist_known"):
            for path in (root / sub / "features").glob("*.npz"):
                with np.load(path) as z:
                    values[str(z["key"])] = z["icr"].astype(np.float32)
        x = np.stack([values[r["key"]] for r in rows])
    # Train scores predict correctness, matching the transfer table convention.
    y = np.asarray([r["correct"] for r in rows], dtype=int)
    return x, y


def load_target(method):
    rows = target_rows(); cache = TARGET_CACHE / method
    x = []
    for row in rows:
        path = cache / f"{row['key']}.npz"
        if not path.exists():
            raise RuntimeError(f"missing target feature {path}")
        with np.load(path) as z:
            x.append(z["x"].astype(np.float32))
    return np.stack(x), np.asarray([r["correct"] for r in rows]), rows


def evaluate(method):
    tx, ty, rows = load_target(method)
    results, predictions = {}, []
    for source in ("scientist_known", "scientist_full"):
        x, y = load_source(method, source)
        scores = [fit_saplma(x, y, tx, seed) if method == "saplma"
                  else fit_icr(x, y, tx, seed) for seed in SEEDS]
        score = np.mean(scores, axis=0)
        subsets = {}
        masks = {"all": np.ones(len(rows), bool)}
        masks.update({d: np.asarray([r["domain"] == d for r in rows])
                      for d in ("athlete", "musician", "building")})
        for name, mask in masks.items():
            subsets[name] = metric(ty[mask], score[mask])
            subsets[name]["per_seed_auroc"] = [float(roc_auc_score(ty[mask], p[mask]))
                                                 for p in scores]
        results[source] = {"source_n": len(y), "source_correct": int(y.sum()),
                           "target_n": len(ty), "subsets": subsets}
        predictions.extend({"method": method, "source": source, "key": r["key"],
                            "domain": r["domain"], "correct": int(label),
                            "prob_correct": float(prob)}
                           for r, label, prob in zip(rows, ty, score))
    report = {
        "protocol": "strict task split; frozen Scientist-only training and multidomain-only testing",
        "method": method, "seeds": list(SEEDS),
        "target": "multidomain_v6_fixed500_musician_opt; probe_state=knows_both; unmatched excluded",
        "target_labels_used_for_fitting_or_tuning": False,
        "training": ("published SAPLMA MLP; Adam; BCE; five epochs; batch 32" if method == "saplma"
                     else "published ICR MLP; source-only stratified 90/10 checkpoint validation; Adam lr=5e-4; up to 50 epochs; batch 32"),
        "results": results,
    }
    out = OUT / method; out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("collect-saplma", "collect-icr", "eval-saplma", "eval-icr", "all"))
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("collect-saplma", "all"): collect_saplma()
    if a.stage in ("collect-icr", "all"): collect_icr()
    if a.stage in ("eval-saplma", "all"): evaluate("saplma")
    if a.stage in ("eval-icr", "all"): evaluate("icr")


if __name__ == "__main__":
    main()
