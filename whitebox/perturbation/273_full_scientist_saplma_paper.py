#!/usr/bin/env python3
"""Paper-faithful SAPLMA on all 2,894 parse-valid Scientist rows."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
OUT = RUNS / "273_full_scientist_saplma_paper"
SEEDS = (42, 43, 44)
LAYER = 28


class SAPLMA(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def read_jsonl(path):
    return [json.loads(x) for x in path.open() if x.strip()]


def metrics(y, score):
    pred = score >= .5
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, np.clip(score, 1e-9, 1-1e-9))),
        "accuracy_at_0.5": float(accuracy_score(y, pred)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)),
        "tn": int(np.sum((~pred) & (y == 0))),
        "fp": int(np.sum(pred & (y == 0))),
        "fn": int(np.sum((~pred) & (y == 1))),
        "tp": int(np.sum(pred & (y == 1))),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {x["key"]: x for x in read_jsonl(
        ROOT / "tool_gate_correctness_names_llama31_8b" / "records.jsonl")}
    manifest = {x["key"]: x for x in read_jsonl(
        RUNS / "76_closedbook_fact_probe_manifest.jsonl")}
    rows = []
    for path in sorted((RUNS/"141_scientist_all_trajectory_l8").glob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            key = str(z["key"].item())
            if (key not in records or key not in manifest or
                    not records[key].get("parse_valid", True)):
                continue
            layers = z["layers"].astype(int)
            index = int(np.flatnonzero(layers == LAYER)[0])
            hidden = z["last"].astype(np.float32)[index]
        rows.append({"key": key, "group": manifest[key]["right_qid"],
                     "error": int(not records[key]["correct"]), "x": hidden})
    if len(rows) != 2894:
        raise RuntimeError(f"aligned rows {len(rows)}/2894")
    x = np.stack([r["x"] for r in rows])
    y = np.asarray([r["error"] for r in rows])
    groups = np.asarray([r["group"] for r in rows])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_predictions = []
    per_seed = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prediction = np.zeros(len(y))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(cv.split(x, y, groups)):
            torch.manual_seed(seed*10+fold)
            model = SAPLMA(x.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            loss_function = nn.BCEWithLogitsLoss()
            dataset = TensorDataset(torch.from_numpy(x[train]),
                                    torch.from_numpy(y[train].astype(np.float32)))
            loader = DataLoader(dataset, batch_size=32, shuffle=True,
                                generator=torch.Generator().manual_seed(seed*10+fold))
            model.train()
            for _ in range(5):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = loss_function(model(xb), yb)
                    loss.backward(); optimizer.step()
            model.eval()
            with torch.inference_mode():
                value = torch.sigmoid(model(torch.from_numpy(x[test]).to(device)))
                prediction[test] = value.cpu().numpy()
            print(f"seed={seed} fold={fold+1}/5", flush=True)
        all_predictions.append(prediction)
        per_seed.append(metrics(y, prediction))
    mean_prediction = np.mean(all_predictions, axis=0)
    report = {
        "protocol": ("SAPLMA paper architecture; full 2894 parse-valid Scientist; "
                     "generated-answer final-token layer28; Adam; 5 epochs; "
                     "right-person grouped 3x5 OOF"),
        "n": len(y), "errors": int(y.sum()), "correct": int((1-y).sum()),
        "groups": len(set(groups)), "device": device,
        "per_seed": per_seed, "mean_probability": metrics(y, mean_prediction),
    }
    (OUT/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT/"predictions.jsonl").open("w") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({"key": row["key"], "error": int(y[i]),
                                     "saplma_error_probability":
                                     float(mean_prediction[i])}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
