#!/usr/bin/env python3
"""Gold-trained, cross-fitted layer probes compared with generation and likelihood.

No model inference is performed. The script consumes saved full-context hidden
states plus actual generation-modal labels. The diagnostic subset is where
generation and teacher-forced likelihood disagree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import profile_perturbation_unsupervised as pp


HERE = Path(__file__).resolve().parent


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_modal(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("generation_mode") is not None:
                out[str(row["key"])] = row
    return out


def bootstrap_delta(probe, generation, likelihood, mask, seed, n_boot=5000):
    """Paired bootstrap CI for agreement(probe,generation)-agreement(probe,LL)."""
    idx = np.flatnonzero(mask)
    contributions = ((probe[idx] == generation[idx]).astype(float)
                     - (probe[idx] == likelihood[idx]).astype(float))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = np.mean(rng.choice(contributions, len(contributions), replace=True))
    return [float(x) for x in np.quantile(means, [.025, .975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=HERE / "profile_perturbation_forward_output" / "items")
    ap.add_argument("--generation-labels", type=Path,
                    default=HERE / "profile_likelihood_generation_m3_output" / "items.jsonl")
    ap.add_argument("--output", type=Path,
                    default=HERE / "profile_layerwise_threeway_probe_output")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca-components", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permutations", type=int, default=100)
    args = ap.parse_args()

    modal = load_modal(args.generation_labels)
    keys, ys, generations, likelihoods, tensors = [], [], [], [], []
    layer_ids = None
    for key in sorted(modal):
        path = args.features / f"{key}.npz"
        if not path.exists():
            continue
        record = pp.load_item_npz(path)
        md = record["metadata"]
        full = md["condition_names"].index("full_context")
        current_layers = [int(x) for x in md["layers"]]
        if layer_ids is None:
            layer_ids = current_layers
        if current_layers != layer_ids:
            raise ValueError(f"Inconsistent layers for {key}: {current_layers} != {layer_ids}")
        row = modal[key]
        keys.append(key)
        ys.append(int(md["right_index"]))
        generations.append(int(row["generation_mode"]))
        likelihoods.append(int(np.argmax(record["candidate_scores"][full])))
        tensors.append(record["hidden"][full].astype(np.float32))

    X = np.stack(tensors)                 # item, layer, hidden
    y = np.asarray(ys, int)
    generation = np.asarray(generations, int)
    likelihood = np.asarray(likelihoods, int)
    disagree = generation != likelihood
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    splits = list(cv.split(X, y))
    rng = np.random.default_rng(args.seed)
    permuted_y = [rng.permutation(y) for _ in range(args.permutations)]
    results = []
    predictions = np.empty((len(layer_ids), len(y)), np.int8)
    probabilities = np.empty((len(layer_ids), len(y)), np.float32)

    for lp, layer in enumerate(layer_ids):
        pred = np.empty(len(y), int)
        prob = np.empty(len(y), float)
        null_correct = np.zeros(args.permutations, float)
        for tr, te in splits:
            nc = min(args.pca_components, len(tr) - 2, X.shape[2])
            transform = make_pipeline(
                StandardScaler(),
                PCA(n_components=nc, svd_solver="randomized", random_state=args.seed),
            )
            Ztr = transform.fit_transform(X[tr, lp])
            Zte = transform.transform(X[te, lp])
            clf = LogisticRegression(C=.1, class_weight="balanced",
                                     max_iter=2000, random_state=args.seed)
            clf.fit(Ztr, y[tr])
            pred[te] = clf.predict(Zte)
            prob[te] = clf.predict_proba(Zte)[:, 1]
            for b, yp in enumerate(permuted_y):
                null_clf = LogisticRegression(C=.1, class_weight="balanced",
                                              max_iter=1000, random_state=args.seed)
                null_clf.fit(Ztr, yp[tr])
                null_correct[b] += np.sum(null_clf.predict(Zte) == yp[te])
        null_acc = null_correct / len(y)
        predictions[lp] = pred
        probabilities[lp] = prob
        n_gen = int(np.sum((pred == generation) & disagree))
        n_ll = int(np.sum((pred == likelihood) & disagree))
        diagnostic_n = int(disagree.sum())
        pvalue = float(binomtest(n_gen, diagnostic_n, .5).pvalue)
        result = {
            "layer": layer,
            "probe_gold_accuracy": float(np.mean(pred == y)),
            "permuted_label_accuracy_mean": float(np.mean(null_acc)),
            "permuted_label_accuracy_q95": float(np.quantile(null_acc, .95)),
            "agreement_generation_all": float(np.mean(pred == generation)),
            "agreement_likelihood_all": float(np.mean(pred == likelihood)),
            "disagreement_subset_n": diagnostic_n,
            "agreement_generation_on_disagreement": n_gen / diagnostic_n,
            "agreement_likelihood_on_disagreement": n_ll / diagnostic_n,
            "delta_generation_minus_likelihood_on_disagreement": (n_gen - n_ll) / diagnostic_n,
            "delta_bootstrap_95ci": bootstrap_delta(
                pred, generation, likelihood, disagree, args.seed + layer),
            "two_sided_binomial_p": pvalue,
        }
        results.append(result)

    # Item-level alignment trajectories on the diagnostic subset.
    align = np.where(predictions[:, disagree] == generation[disagree], "G", "L")
    trajectories = {}
    for col in range(align.shape[1]):
        name = "->".join(align[:, col])
        trajectories[name] = trajectories.get(name, 0) + 1
    trajectory_rows = [
        {"trajectory": k, "count": v, "fraction": v / int(disagree.sum())}
        for k, v in sorted(trajectories.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    out = {
        "n_items": len(y),
        "layers": layer_ids,
        "hidden_token": "last_prompt_token",
        "probe_target": "gold_profile_index",
        "probe_protocol": f"{args.folds}-fold cross-fitted StandardScaler+PCA({args.pca_components})+L2 logistic",
        "generation_likelihood_agree": int((~disagree).sum()),
        "generation_likelihood_disagree": int(disagree.sum()),
        "generation_accuracy": float(np.mean(generation == y)),
        "likelihood_accuracy": float(np.mean(likelihood == y)),
        "gold_index_balance": {"index_0": int(np.sum(y == 0)), "index_1": int(np.sum(y == 1))},
        "layer_results": results,
        "alignment_trajectories_on_disagreement": trajectory_rows,
        "interpretation_guardrail": (
            "A gold-trained linear probe establishes decodable answer information, not that the "
            "model causally uses it. Generation-vs-likelihood comparisons are diagnostic only on "
            "their disagreement subset."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "summary.json", out)
    np.savez_compressed(args.output / "cross_fitted_predictions.npz",
                        keys=np.asarray(keys), layers=np.asarray(layer_ids), gold=y,
                        generation=generation, likelihood=likelihood,
                        probe_predictions=predictions, probe_probabilities=probabilities)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
