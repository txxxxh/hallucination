#!/usr/bin/env python3
"""Paper-faithful SAPLMA training/evaluation on the four frozen benchmarks.

The four benchmarks act as the paper's topics: train on three and test on the
held-out fourth. Activations are the candidate's last-token states collected by
282 with the released direct-concatenation/tokenization protocol.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
FEATURES = Path("/tmp/hpd_original_benchmarks/hidden_states")
OUT = HERE / "runs" / "283_saplma_exact_original_benchmarks"
DATASETS = ("scientist", "trivia", "gsm8k", "drop")
LAYERS = (32, 28, 24, 20, 16)  # last, -4, -8, -12, middle for a 32-block model
SEEDS = (42, 43, 44)


class SAPLMA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_all():
    result = {}
    for dataset in DATASETS:
        obj = torch.load(FEATURES / f"llama3.1-8b__{dataset}.pt", map_location="cpu")
        result[dataset] = (
            obj["hidden_states"].float().numpy(),
            obj["labels"].numpy().astype(np.int64),
        )
    return result


def train_once(x_train, y_train, x_test, y_test, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SAPLMA(x_train.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    for _ in range(5):
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        prob = model(torch.from_numpy(x_test).cuda()).cpu().numpy()
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auroc": float(roc_auc_score(y_test, prob)),
        "auprc": float(average_precision_score(y_test, prob)),
    }


def main():
    data = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for held_out in DATASETS:
        train_names = [name for name in DATASETS if name != held_out]
        y_train = np.concatenate([data[name][1] for name in train_names])
        y_test = data[held_out][1]
        per_layer = {}
        for layer in LAYERS:
            x_train = np.concatenate([data[name][0][:, layer] for name in train_names]).astype(np.float32)
            x_test = data[held_out][0][:, layer].astype(np.float32)
            runs = [train_once(x_train, y_train, x_test, y_test, seed) for seed in SEEDS]
            per_layer[str(layer)] = {
                "per_seed": runs,
                "mean": {key: float(np.mean([run[key] for run in runs])) for key in runs[0]},
                "std": {key: float(np.std([run[key] for run in runs])) for key in runs[0]},
            }
            print(held_out, layer, per_layer[str(layer)]["mean"], flush=True)
        best_layer = max(LAYERS, key=lambda layer: per_layer[str(layer)]["mean"]["auroc"])
        report = {
            "dataset": held_out,
            "n_test": int(len(y_test)),
            "train_benchmarks": train_names,
            "n_train": int(len(y_train)),
            "protocol": "SAPLMA leave-one-topic-out mapped to leave-one-benchmark-out",
            "target_model": "Llama-3.1-8B-Instruct (same target as original matrix)",
            "features": "candidate last-token hidden state; direct prompt-answer concatenation",
            "layers": list(LAYERS),
            "architecture": "D-256-128-64-1; ReLU; sigmoid",
            "training": "Adam defaults; BCE; 5 epochs; batch 32; seeds 42,43,44",
            "per_layer": per_layer,
            "best_layer_by_auroc": int(best_layer),
            "best": per_layer[str(best_layer)],
        }
        (OUT / f"{held_out}.json").write_text(json.dumps(report, indent=2) + "\n")
        summary[held_out] = {
            "best_layer": int(best_layer),
            **per_layer[str(best_layer)]["mean"],
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
