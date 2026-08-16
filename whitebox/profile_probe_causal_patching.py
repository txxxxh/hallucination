#!/usr/bin/env python3
"""Causal patching along cross-fitted gold-probe directions.

Targets items where the out-of-fold L16 probe is gold-correct while the saved
generation modal is wrong. Patches the last prompt-token residual stream at
L16/L24. Directions for every item come from a fold that excluded that item.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import profile_perturbation_unsupervised as pp
from profile_order_generation_check import mode, parse_choice


HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_crossfit_directions(X, y, layers, folds, pca_components, seed):
    """Return raw-space unit directions and train projection SD per item/layer."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    directions = np.zeros_like(X, dtype=np.float32)
    scales = np.zeros((len(y), len(layers)), dtype=np.float32)
    random_directions = np.zeros_like(X, dtype=np.float32)
    rng = np.random.default_rng(seed + 991)
    for lp, _ in enumerate(layers):
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            scaler = StandardScaler().fit(X[tr, lp])
            Ztr0 = scaler.transform(X[tr, lp])
            nc = min(pca_components, len(tr) - 2, X.shape[2])
            pca = PCA(n_components=nc, svd_solver="randomized", random_state=seed).fit(Ztr0)
            Ztr = pca.transform(Ztr0)
            clf = LogisticRegression(C=.1, class_weight="balanced", max_iter=2000,
                                     random_state=seed).fit(Ztr, y[tr])
            w = (clf.coef_[0] @ pca.components_) / scaler.scale_
            w = w / max(np.linalg.norm(w), 1e-12)
            r = rng.normal(size=len(w))
            r -= np.dot(r, w) * w
            r /= max(np.linalg.norm(r), 1e-12)
            directions[te, lp] = w
            random_directions[te, lp] = r
            scales[te, lp] = max(float(np.std(X[tr, lp] @ w)), 1e-6)
    return directions, random_directions, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=HERE / "profile_perturbation_forward_output" / "items")
    ap.add_argument("--probe-results", type=Path,
                    default=HERE / "profile_layerwise_threeway_probe_output" /
                            "cross_fitted_predictions.npz")
    ap.add_argument("--generation-labels", type=Path,
                    default=HERE / "profile_likelihood_generation_m3_output" / "items.jsonl")
    ap.add_argument("--data", type=Path, default=pp.DEFAULT_DATA)
    ap.add_argument("--output", type=Path,
                    default=HERE / "profile_probe_causal_patching_output")
    ap.add_argument("--model", default=pp.DEFAULT_MODEL)
    ap.add_argument("--cache-dir", default=pp.DEFAULT_CACHE)
    ap.add_argument("--layers", default="16,24")
    ap.add_argument("--doses", default="0.5,1,2,4")
    ap.add_argument("--control-dose", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pca-components", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=.7)
    ap.add_argument("--condition-filter", default=None)
    ap.add_argument("--results-name", default="greedy_results.jsonl")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    saved = np.load(args.probe_results)
    keys = [str(x) for x in saved["keys"]]
    all_layers = [int(x) for x in saved["layers"]]
    y = saved["gold"].astype(int)
    generation = saved["generation"].astype(int)
    probe_pred = saved["probe_predictions"].astype(int)
    wanted_layers = [int(x) for x in args.layers.split(",")]
    layer_pos = [all_layers.index(x) for x in wanted_layers]
    target = (generation != y) & (probe_pred[all_layers.index(16)] == y)
    target_indices = np.flatnonzero(target)

    X = np.empty((len(keys), len(all_layers), 4096), np.float32)
    metadata = {}
    for i, key in enumerate(keys):
        record = pp.load_item_npz(args.features / f"{key}.npz")
        md = record["metadata"]
        full = md["condition_names"].index("full_context")
        X[i] = record["hidden"][full]
        metadata[key] = md
    directions, random_dirs, scales = build_crossfit_directions(
        X, y, all_layers, args.folds, args.pca_components, args.seed)

    raw = json.loads(args.data.read_text(encoding="utf-8"))
    raw_by_key = {str(r["key"]): r for r in raw}
    prompts, names = {}, {}
    for i in target_indices:
        item = pp.parse_item(raw_by_key[keys[i]])
        prompts[keys[i]] = {c.name: c.prompt for c in pp.build_conditions(item)}["full_context"]
        names[keys[i]] = [p.name for p in item.profiles]

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(args.cache_dir) / "hub")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True).eval()

    output_path = args.output / args.results_name
    done = {}
    if output_path.exists():
        for row in read_jsonl(output_path):
            done[(row["key"], row["condition"])] = row

    conditions = [("baseline", None, None, 0.0)]
    for layer in wanted_layers:
        for dose in [float(x) for x in args.doses.split(",")]:
            conditions.append((f"L{layer}_probe_d{dose:g}", layer, "probe", dose))
        conditions.append((f"L{layer}_reverse_d{args.control_dose:g}",
                           layer, "reverse", args.control_dose))
        conditions.append((f"L{layer}_random_d{args.control_dose:g}",
                           layer, "random", args.control_dose))
    if args.condition_filter:
        selected = set(args.condition_filter.split(","))
        conditions = [c for c in conditions if c[0] in selected]

    def run_batch(batch_indices, layer, kind, dose):
        batch_keys = [keys[i] for i in batch_indices]
        chats = [tok.apply_chat_template([{"role": "user", "content": prompts[k]}],
                                         tokenize=False, add_generation_prompt=True)
                 for k in batch_keys]
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
        handle = None
        if layer is not None:
            lp = all_layers.index(layer)
            delta = []
            for i in batch_indices:
                sign = 1.0 if y[i] == 1 else -1.0
                vec = directions[i, lp]
                if kind == "reverse":
                    sign *= -1
                elif kind == "random":
                    vec = random_dirs[i, lp]
                delta.append(sign * dose * scales[i, lp] * vec)
            delta = torch.as_tensor(np.stack(delta), device=model.device, dtype=torch.bfloat16)
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.shape[1] <= 1:
                    return output
                patched = tensor.clone()
                active_delta = delta
                if tensor.shape[0] != delta.shape[0]:
                    if tensor.shape[0] % delta.shape[0] != 0:
                        raise RuntimeError("Generation batch expansion is not an integer multiple")
                    active_delta = delta.repeat_interleave(
                        tensor.shape[0] // delta.shape[0], dim=0)
                patched[:, -1, :] += active_delta
                return (patched,) + output[1:] if isinstance(output, tuple) else patched
            handle = model.model.layers[layer - 1].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=args.samples > 1,
                                     temperature=args.temperature if args.samples > 1 else None,
                                     top_p=.95 if args.samples > 1 else None,
                                     num_return_sequences=args.samples, max_new_tokens=32,
                                     pad_token_id=tok.pad_token_id)
        finally:
            if handle is not None:
                handle.remove()
        width = enc["input_ids"].shape[1]
        decoded = [tok.decode(seq[width:], skip_special_tokens=True).strip() for seq in out]
        return [decoded[i:i + args.samples] for i in range(0, len(decoded), args.samples)]

    for ci, (condition, layer, kind, dose) in enumerate(conditions, 1):
        pending = [i for i in target_indices if (keys[i], condition) not in done]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            output_groups = run_batch(batch, layer, kind, dose)
            with output_path.open("a", encoding="utf-8") as f:
                for i, outputs in zip(batch, output_groups):
                    choices = [parse_choice(text, names[keys[i]]) for text in outputs]
                    choice = mode(choices)
                    row = {"key": keys[i], "condition": condition, "layer": layer,
                           "kind": kind, "dose": dose, "gold": int(y[i]),
                           "old_generation_modal": int(generation[i]),
                           "outputs": outputs, "choices": choices, "choice": choice,
                           "correct": choice == int(y[i])}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[(keys[i], condition)] = row
        print(f"[{ci}/{len(conditions)}] {condition}: {len(pending)} new", flush=True)

    rows = list(done.values())
    summary = {"target_definition": "old generation modal wrong and cross-fitted L16 probe correct",
               "n_targets": int(len(target_indices)), "greedy": {}}
    by_condition = {}
    for condition, _, _, _ in conditions:
        rr = [r for r in rows if r["condition"] == condition]
        by_condition[condition] = rr
        summary["greedy"][condition] = {
            "n": len(rr), "parsed": sum(r["choice"] is not None for r in rr),
            "correct": sum(r["correct"] for r in rr),
            "accuracy": sum(r["correct"] for r in rr) / len(rr),
        }
    baseline_rows = [r for r in rows if r["condition"] == "baseline"]
    baseline_wrong = {r["key"] for r in baseline_rows if not r["correct"]}
    for condition in summary["greedy"]:
        rr = [r for r in by_condition[condition] if r["key"] in baseline_wrong]
        summary["greedy"][condition]["baseline_wrong_n"] = len(rr)
        summary["greedy"][condition]["wrong_to_correct"] = sum(r["correct"] for r in rr)
        summary["greedy"][condition]["wrong_to_correct_rate"] = (
            sum(r["correct"] for r in rr) / len(rr) if rr else None)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
