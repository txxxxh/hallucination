#!/usr/bin/env python3
"""Causal steering test for the knowledge -> search gate.

This script consumes the output of ``tool_gate_calibration.py`` and the
``known_subset_search_probe.py`` analysis.  It fits the same standardized
linear probes as the offline analysis, then intervenes on the residual stream
at the selected transformer block while the model is generating the tool
decision.

Primary test
------------
* necessity: baseline-SEARCH/unknown items, ``h <- h - alpha * d_unknown``;
  success means SEARCH changes to a direct answer.
* sufficiency: baseline-answer/known items, ``h <- h + alpha * d_unknown``;
  success means a SEARCH decision is induced.
* random control: the same interventions with a norm-matched random direction.

``hidden_states[l]`` in the saved files is the output after transformer block
``l - 1`` (layer 0 is the embedding output).  Therefore a probe layer l is
implemented by hooking ``model.model.layers[l - 1]``.  This convention is
printed and saved in the result file to make off-by-one errors explicit.

The strength alpha is measured in units of the per-example RMS of the hooked
residual stream, so results are comparable across examples and directions.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("causal_search_steering")


def load_data(out: Path):
    import torch

    records = [json.loads(x) for x in (out / "records.jsonl").open()
               if x.strip()]
    kept, hidden = [], []
    for r in records:
        p = out / "hidden" / f"{r['qid']}.pt"
        if not p.exists():
            continue
        item = torch.load(p, map_location="cpu", weights_only=False)
        kept.append(r)
        hidden.append(item["hidden"].float().numpy())
    if not hidden:
        raise ValueError(f"no hidden states under {out / 'hidden'}")
    return kept, np.stack(hidden)


def raw_space_direction(x: np.ndarray, y: np.ndarray, c: float, seed: int):
    """Return a unit vector in the original hidden-state coordinates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        max_iter=2000, C=c, class_weight="balanced", solver="liblinear",
        random_state=seed,
    ).fit(scaler.transform(x), y)
    # coef @ ((x - mean) / scale) => raw coefficient is coef / scale.
    d = model.coef_[0] / scaler.scale_
    norm = np.linalg.norm(d)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("probe produced a zero or non-finite direction")
    return d / norm


def selected_layers(out: Path, args):
    probe = json.loads((out / "known_subset_search_probe.json").read_text())
    analysis = json.loads((out / "analysis.json").read_text())
    know_layer = args.know_layer
    search_layer = args.search_layer
    if know_layer is None:
        know_layer = analysis["probe_knows_dontknow"]["layer"]
    if search_layer is None:
        search_layer = probe["best_auroc_layer"]["layer"]
    if know_layer is None or search_layer is None:
        raise ValueError("could not infer layers; pass --know-layer and --search-layer")
    return int(know_layer), int(search_layer)


def make_directions(records, hidden, args, know_layer, search_layer):
    known = np.array([r["know_prior"] == "known" for r in records])
    search = np.array([r["action"] == "search" for r in records])
    d_unknown = -raw_space_direction(
        hidden[:, know_layer], known.astype(int), args.c, args.seed
    )
    known_search = known
    d_search = raw_space_direction(
        hidden[known_search, search_layer], search[known_search].astype(int),
        args.c, args.seed,
    )
    rng = np.random.default_rng(args.seed + 991)
    random_dirs = {
        "knowledge_layer": rng.normal(size=hidden.shape[-1]),
        "search_layer": rng.normal(size=hidden.shape[-1]),
    }
    random_dirs = {k: v / np.linalg.norm(v) for k, v in random_dirs.items()}
    return d_unknown, d_search, random_dirs


def model_layers(model):
    """Find the decoder blocks for common HF causal-LM layouts."""
    for obj in (getattr(model, "model", None), model):
        if obj is not None and hasattr(obj, "layers"):
            return obj.layers
    if hasattr(getattr(model, "transformer", None), "h"):
        return model.transformer.h
    raise AttributeError("cannot find transformer decoder layers on this model")


def steer_generate(engine, prompt, layer, direction, alpha, seed, max_new):
    """Generate one tool decision with a last-token residual intervention."""
    import torch

    torch.manual_seed(seed)
    text = engine._fmt(prompt)
    enc = engine.tok(
        text, return_tensors="pt", truncation=True,
        max_length=engine.max_input, add_special_tokens=False,
    ).to(engine.device)
    blocks = model_layers(engine.model)
    # Saved hidden index l is block l-1; layer 0 is embeddings.
    block_index = layer - 1
    if block_index < 0 or block_index >= len(blocks):
        raise ValueError(f"hidden layer {layer} maps to invalid block {block_index}")
    vec = torch.as_tensor(direction, device=engine.device, dtype=next(
        engine.model.parameters()).dtype)

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # During cached decoding seq_len is one; -1 is the decision position.
        rms = hs[:, -1, :].float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        delta = (float(alpha) * rms).to(hs.dtype) * vec
        hs = hs.clone()
        hs[:, -1, :] = hs[:, -1, :] + delta
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = blocks[block_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            generated = engine.model.generate(
                **enc, max_new_tokens=max_new, do_sample=False,
                pad_token_id=engine.tok.pad_token_id,
            )
    finally:
        handle.remove()
    gen_ids = generated[0, enc.input_ids.shape[1]:]
    return engine.tok.decode(gen_ids, skip_special_tokens=True).strip()


def summarize(rows):
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == condition]
        n = len(sub)
        base_search = sum(r["baseline_action"] == "search" for r in sub)
        steered_search = sum(r["steered_action"] == "search" for r in sub)
        flips = sum(r["flip_success"] for r in sub)
        out[condition] = {
            "n": n,
            "baseline_search_rate": round(base_search / n, 4) if n else None,
            "steered_search_rate": round(steered_search / n, 4) if n else None,
            "search_rate_delta": round((steered_search - base_search) / n, 4) if n else None,
            "successful_flips": flips,
            "flip_rate": round(flips / n, 4) if n else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--know-layer", type=int)
    ap.add_argument("--search-layer", type=int)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-group", type=int, default=100,
                     help="paired samples per baseline group; 0 means all")
    ap.add_argument("--strengths", type=float, nargs="+",
                     default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--result-file", default="causal_search_steering.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    out = Path(args.output_dir)
    records, hidden = load_data(out)
    know_layer, search_layer = selected_layers(out, args)
    d_unknown, d_search, random_dirs = make_directions(
        records, hidden, args, know_layer, search_layer
    )
    LOG.info("knowledge probe: hidden layer %d -> block %d; search probe: hidden layer %d -> block %d",
             know_layer, know_layer - 1, search_layer, search_layer - 1)

    # Use only unambiguous baseline actions, as required by the paired design.
    unknown = [r for r in records if r["know_prior"] == "unknown" and r["action"] == "search"]
    known = [r for r in records if r["know_prior"] == "known" and r["action"] == "answer"]
    rng = np.random.default_rng(args.seed + 123)
    rng.shuffle(unknown); rng.shuffle(known)
    if args.max_per_group:
        unknown = unknown[:args.max_per_group]
        known = known[:args.max_per_group]
    if not unknown or not known:
        raise ValueError(f"need baseline unknown/SEARCH and known/answer groups: {len(unknown)}, {len(known)}")

    # Importing this module reuses exactly the benchmark's prompt and action parser.
    from tool_gate_calibration import Engine, TOOL_INSTR, classify_action
    engine = Engine(args.model, args.device, args.dtype, args.max_input_tokens,
                    args.quantize_4bit, args.trust_remote_code)
    rows = []
    for strength in args.strengths:
        conditions = [
            ("necessity_unknown_direction", unknown, know_layer, d_unknown, -1),
            ("sufficiency_unknown_direction", known, know_layer, d_unknown, +1),
            ("necessity_random_control", unknown, know_layer, random_dirs["knowledge_layer"], -1),
            ("sufficiency_random_control", known, know_layer, random_dirs["knowledge_layer"], +1),
            # This is an additional diagnostic: if the SEARCH probe direction is
            # causal, it should behave like an action-specific direction.
            ("necessity_search_direction", unknown, search_layer, d_search, -1),
            ("sufficiency_search_direction", known, search_layer, d_search, +1),
            ("necessity_search_random_control", unknown, search_layer, random_dirs["search_layer"], -1),
            ("sufficiency_search_random_control", known, search_layer, random_dirs["search_layer"], +1),
        ]
        for condition, group, layer, direction, sign in conditions:
            for i, r in enumerate(group):
                generated = steer_generate(
                    engine, TOOL_INSTR.format(q=r["question"]), layer,
                    direction, sign * strength, args.seed + i * 1009,
                    args.max_new_tokens,
                )
                action = classify_action(generated)
                target = "answer" if r["action"] == "search" else "search"
                rows.append({
                    "qid": r["qid"], "strength": strength, "condition": condition,
                    "layer": layer, "block_index": layer - 1,
                    "baseline_action": r["action"], "steered_action": action,
                    "flip_success": action == target,
                    "generation": generated[:300],
                })
        LOG.info("completed strength %.3g", strength)

    result = {
        "design": {
            "necessity": "unknown baseline SEARCH; subtract direction; success=SEARCH->answer",
            "sufficiency": "known baseline answer; add direction; success=answer->SEARCH",
            "random_control": "norm-matched Gaussian direction at the same layer",
            "strength_units": "alpha times per-example RMS of hooked residual stream",
            "hidden_layer_to_block": "hidden_states[l] is hooked at decoder block l-1",
        },
        "layers": {"knowledge_hidden_layer": know_layer, "search_hidden_layer": search_layer},
        "n_unknown_search": len(unknown), "n_known_answer": len(known),
        "strengths": args.strengths,
        "summary_by_strength": {
            str(s): summarize([r for r in rows if r["strength"] == s])
            for s in args.strengths
        },
        "rows": rows,
    }
    path = out / args.result_file
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"layers": result["layers"], "summary_by_strength": result["summary_by_strength"]},
                     indent=2, ensure_ascii=False))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
