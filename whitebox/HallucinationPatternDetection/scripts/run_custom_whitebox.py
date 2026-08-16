"""Reproduce the paper's hidden-state probe on the two local white-box datasets.

The script deliberately reports three protocols:

1. ``item_random``: the paper/repository's stratified item-level 70/10/20 split.
2. ``question_grouped``: keeps the truthful and false answer for a question in
   the same split, preventing paired-question leakage.
3. ``cross_dataset``: trains on one dataset and evaluates on the other.

Only the linear probe is used: this is the paper's headline method and avoids
adding a more expressive detector than the one under test.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_loader import PromptItem
from src.detection.probes import LinearProbe
from src.models.hidden_state_extractor import ExtractionConfig, HiddenStateExtractor
from src.models.model_loader import load_quantized_model
from src.utils.helpers import set_seed


LOCAL_DATASETS = {
    "shuffled_prepend_names": ROOT.parent / "shuffled_prepend_names_question.json",
    "question_and_result": ROOT.parent / "question_and_result.json",
}

MODEL_CFG = {
    "short_name": "qwen2.5-7b",
    "hf_id": "Qwen/Qwen2.5-7B-Instruct",
    "family": "qwen",
    "chat_template": "qwen",
}

QUANT_CFG = {
    "bits": 4,
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "bfloat16",
}


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def build_pairs(
    dataset: str, limit_pairs: int | None, seed: int
) -> Tuple[List[PromptItem], List[str]]:
    """Build truthful/false candidate-answer pairs without splitting a pair."""
    rows = _read_json(LOCAL_DATASETS[dataset])
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    if limit_pairs is not None:
        order = order[:limit_pairs]

    items: List[PromptItem] = []
    group_ids: List[str] = []
    for row_idx in order:
        row = rows[row_idx]
        if dataset == "shuffled_prepend_names":
            prompt = str(row["prompt"])
            group = str(row.get("key", f"row_{row_idx:04d}"))
            candidates = [
                (str(row["rgt_ans"]), 1, "truthful"),
                (str(row["wrg_ans"]), 0, "hallucinated"),
            ]
        else:
            prompt = str(row.get("benchmark_prompt") or row["question"])
            group = f"row_{row_idx:04d}"
            answer_idx = int(row["answer"]) - 1
            options = [str(x) for x in row["options"]]
            if len(options) != 2 or answer_idx not in (0, 1):
                raise ValueError(f"Expected a binary question at row {row_idx}")
            candidates = [
                (options[answer_idx], 1, "truthful"),
                (options[1 - answer_idx], 0, "hallucinated"),
            ]

        for answer, label, kind in candidates:
            items.append(
                PromptItem(
                    prompt=prompt,
                    answer=answer,
                    label=label,
                    dataset=dataset,
                    meta={"group_id": group, "kind": kind, "source_row": row_idx},
                )
            )
            group_ids.append(group)

    # The official loaders shuffle paired items before the item-level split.
    perm = list(range(len(items)))
    rng.shuffle(perm)
    return [items[i] for i in perm], [group_ids[i] for i in perm]


def save_manifest(
    dataset: str, items: Sequence[PromptItem], group_ids: Sequence[str], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item, group in zip(items, group_ids):
            row = asdict(item)
            row["group_id"] = group
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_dataset(
    bundle: Any,
    dataset: str,
    items: Sequence[PromptItem],
    group_ids: Sequence[str],
    out_path: Path,
    batch_size: int,
    max_input_length: int,
) -> None:
    cfg = ExtractionConfig(
        pool="last_token",
        capture_attention=False,
        batch_size=batch_size,
        max_input_length=max_input_length,
    )
    result = HiddenStateExtractor(bundle, cfg).extract(items)
    result["group_ids"] = list(group_ids)
    result["paper_protocol"] = {
        "candidate_conditioned": True,
        "pooling": "last_answer_token",
        "quantization": "4-bit NF4",
        "compute_dtype": "bfloat16",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)


def _metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
    }


def _group_split(
    group_ids: Sequence[str], seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.array(sorted(set(group_ids)), dtype=object)
    train_val, test = train_test_split(groups, test_size=0.2, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.1 / 0.8, random_state=seed
    )
    gid = np.asarray(group_ids, dtype=object)
    return (
        np.flatnonzero(np.isin(gid, train)),
        np.flatnonzero(np.isin(gid, val)),
        np.flatnonzero(np.isin(gid, test)),
    )


def _item_split(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_val, test = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1 / 0.8,
        stratify=y[train_val],
        random_state=seed,
    )
    return np.asarray(train), np.asarray(val), np.asarray(test)


def _fit_linear(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: str,
    seed: int,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> LinearProbe:
    """Match the official PyTorch affine probe and val-AUROC checkpointing."""
    torch.manual_seed(seed)
    model = LinearProbe(X.shape[1], n_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X[train_idx].astype(np.float32)),
            torch.from_numpy(y[train_idx].astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_auc = -np.inf
    best_state = None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        prob = _predict(model, X[val_idx], device)
        auc = roc_auc_score(y[val_idx], prob)
        if auc > best_auc:
            best_auc = float(auc)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.inference_mode()
def _predict(model: LinearProbe, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
    return F.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def evaluate_within(
    data: Dict[str, Any], protocol: str, seeds: Sequence[int], device: str
) -> Dict[str, Any]:
    X_all = data["hidden_states"].numpy()
    y = data["labels"].numpy()
    groups = data["group_ids"]
    n_layers = X_all.shape[1]
    per_layer: Dict[str, Any] = {}

    for layer in range(n_layers):
        X = X_all[:, layer, :]
        seed_rows = []
        for seed in seeds:
            if protocol == "item_random":
                tr, va, te = _item_split(y, seed)
            elif protocol == "question_grouped":
                tr, va, te = _group_split(groups, seed)
            else:
                raise ValueError(protocol)
            model = _fit_linear(X, y, tr, va, device, seed)
            val_metrics = _metrics(y[va], _predict(model, X[va], device))
            test_metrics = _metrics(y[te], _predict(model, X[te], device))
            seed_rows.append(
                {
                    "seed": seed,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "n_test": int(len(te)),
                    "val": val_metrics,
                    "test": test_metrics,
                }
            )
        per_layer[str(layer)] = {
            "seed_results": seed_rows,
            "val_auroc_mean": float(np.mean([r["val"]["auroc"] for r in seed_rows])),
            "val_auroc_std": float(np.std([r["val"]["auroc"] for r in seed_rows])),
            "test_auroc_mean": float(np.mean([r["test"]["auroc"] for r in seed_rows])),
            "test_auroc_std": float(np.std([r["test"]["auroc"] for r in seed_rows])),
            "test_auprc_mean": float(np.mean([r["test"]["auprc"] for r in seed_rows])),
            "test_accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in seed_rows])),
            "test_f1_mean": float(np.mean([r["test"]["f1"] for r in seed_rows])),
        }

    oracle_layer = max(per_layer, key=lambda k: per_layer[k]["test_auroc_mean"])
    val_layer = max(per_layer, key=lambda k: per_layer[k]["val_auroc_mean"])
    return {
        "protocol": protocol,
        "n_items": int(len(y)),
        "n_groups": int(len(set(groups))),
        "seeds": list(seeds),
        "per_layer": per_layer,
        "paper_style_test_oracle": {
            "best_layer": int(oracle_layer),
            **per_layer[oracle_layer],
        },
        "validation_selected": {
            "best_layer": int(val_layer),
            **per_layer[val_layer],
        },
    }


def evaluate_cross(
    source: Dict[str, Any],
    target: Dict[str, Any],
    source_name: str,
    target_name: str,
    seeds: Sequence[int],
    device: str,
) -> Dict[str, Any]:
    Xs = source["hidden_states"].numpy()
    ys = source["labels"].numpy()
    gs = source["group_ids"]
    Xt = target["hidden_states"].numpy()
    yt = target["labels"].numpy()
    if Xs.shape[1:] != Xt.shape[1:]:
        raise ValueError("Cross-dataset representations have incompatible shapes")

    per_layer: Dict[str, Any] = {}
    for layer in range(Xs.shape[1]):
        rows = []
        for seed in seeds:
            tr, va, _ = _group_split(gs, seed)
            model = _fit_linear(Xs[:, layer], ys, tr, va, device, seed)
            rows.append(
                {
                    "seed": seed,
                    "source_val": _metrics(
                        ys[va], _predict(model, Xs[va, layer], device)
                    ),
                    "target_test": _metrics(
                        yt, _predict(model, Xt[:, layer], device)
                    ),
                }
            )
        per_layer[str(layer)] = {
            "seed_results": rows,
            "source_val_auroc_mean": float(
                np.mean([r["source_val"]["auroc"] for r in rows])
            ),
            "target_auroc_mean": float(
                np.mean([r["target_test"]["auroc"] for r in rows])
            ),
            "target_auroc_std": float(
                np.std([r["target_test"]["auroc"] for r in rows])
            ),
            "target_auprc_mean": float(
                np.mean([r["target_test"]["auprc"] for r in rows])
            ),
            "target_accuracy_mean": float(
                np.mean([r["target_test"]["accuracy"] for r in rows])
            ),
            "target_f1_mean": float(
                np.mean([r["target_test"]["f1"] for r in rows])
            ),
        }
    selected = max(per_layer, key=lambda k: per_layer[k]["source_val_auroc_mean"])
    oracle = max(per_layer, key=lambda k: per_layer[k]["target_auroc_mean"])
    return {
        "protocol": "cross_dataset",
        "source": source_name,
        "target": target_name,
        "source_n": int(len(ys)),
        "target_n": int(len(yt)),
        "per_layer": per_layer,
        "source_validation_selected": {"best_layer": int(selected), **per_layer[selected]},
        "target_test_oracle_diagnostic": {"best_layer": int(oracle), **per_layer[oracle]},
    }


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["prepare", "extract", "evaluate", "all"], default="all"
    )
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(42, deterministic=True)
    manifest_dir = ROOT / "data" / "custom"
    hs_dir = ROOT / "results" / "custom_hidden_states"
    metrics_dir = ROOT / "results" / "custom_metrics"

    prepared: Dict[str, Tuple[List[PromptItem], List[str]]] = {}
    for name in LOCAL_DATASETS:
        items, groups = build_pairs(name, args.limit_pairs, seed=42)
        prepared[name] = (items, groups)
        save_manifest(name, items, groups, manifest_dir)
        print(f"[prepare] {name}: {len(items)} items / {len(set(groups))} questions")

    if args.stage == "prepare":
        return

    if args.stage in ("extract", "all"):
        bundle = load_quantized_model(
            MODEL_CFG, QUANT_CFG, device="cuda", cache_dir=None
        )
        print(
            f"[model] {bundle.hf_id}: {bundle.num_hidden_layers} blocks, "
            f"hidden={bundle.hidden_size}, 4-bit NF4"
        )
        for name, (items, groups) in prepared.items():
            out = hs_dir / f"{MODEL_CFG['short_name']}__{name}.pt"
            if out.exists() and not args.force:
                print(f"[skip] {out} exists")
                continue
            print(f"[extract] {name} -> {out}")
            extract_dataset(
                bundle,
                name,
                items,
                groups,
                out,
                args.batch_size,
                args.max_input_length,
            )
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

    if args.stage == "extract":
        return

    loaded: Dict[str, Dict[str, Any]] = {}
    for name in LOCAL_DATASETS:
        path = hs_dir / f"{MODEL_CFG['short_name']}__{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Run extraction first: {path}")
        loaded[name] = torch.load(path, map_location="cpu", weights_only=False)

    seeds = [42, 43, 44]
    for name, data in loaded.items():
        for protocol in ("item_random", "question_grouped"):
            print(f"[evaluate] {name} / {protocol}")
            result = evaluate_within(data, protocol, seeds, device="cuda")
            result.update(
                {
                    "dataset": name,
                    "model": MODEL_CFG["hf_id"],
                    "quantization": "4-bit NF4",
                    "pooling": "last answer token",
                }
            )
            _save_json(result, metrics_dir / f"{name}__{protocol}.json")

    names = list(LOCAL_DATASETS)
    for source_name, target_name in ((names[0], names[1]), (names[1], names[0])):
        print(f"[evaluate] {source_name} -> {target_name}")
        result = evaluate_cross(
            loaded[source_name],
            loaded[target_name],
            source_name,
            target_name,
            seeds,
            device="cuda",
        )
        result.update(
            {
                "model": MODEL_CFG["hf_id"],
                "quantization": "4-bit NF4",
                "pooling": "last answer token",
            }
        )
        _save_json(
            result, metrics_dir / f"{source_name}__to__{target_name}.json"
        )


if __name__ == "__main__":
    main()
