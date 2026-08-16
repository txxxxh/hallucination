#!/usr/bin/env python3
"""Fit current127 on all Scientist rows and evaluate frozen on TennisQA."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def ch(scores):
    u = scores[0] - scores[1:]
    scale = abs(float(scores[0])) + 1e-6
    return np.r_[scores[0], u, u / scale, u.max(initial=0), u.min(initial=0),
                 np.abs(u).mean(), u.std(), np.mean(u > 0)]


def ch2(scores):
    return np.r_[scores[0], scores[0] - scores[1:]]


def wd(hidden, u):
    delta = hidden[1:].astype(np.float32) - hidden[0].astype(np.float32)
    return (delta * u[:, None]).sum(0) / (np.abs(u).sum() + 1e-9)


def source_rows():
    import importlib
    mod = importlib.import_module("101_fuse_sota_trajectory")
    keys, _, y, _, _, _, _ = mod.load_response("scientist")
    _, _, last, _ = mod.trajectory("scientist", keys)
    dual, physical = {}, {}
    for fp in (RUNS / "116_dual_candidate_hidden_top5").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
            dual[str(z["key"].item())] = (ph[0], wd(ph, z["pred_u"]), oh[0], wd(oh, z["other_u"]))
    for fp in (RUNS / "120_physical_delete_rerank").glob("*.npz"):
        with np.load(fp, allow_pickle=True) as z:
            p, o = z["stage1_pred_scores"], z["stage1_other_scores"]
            q, r = z["stage2_pred_scores"], z["stage2_other_scores"]
            scalar = np.r_[ch(p), ch(o), ch2(q), ch2(r), p[0]-q[0], o[0]-r[0],
                           (p[0]-o[0])-(q[0]-r[0])]
            physical[str(z["key"].item())] = scalar
    missing = [k for k in keys if k not in dual or k not in physical]
    if missing:
        raise RuntimeError(f"missing {len(missing)} Scientist rows")
    return (np.asarray(y), np.stack([physical[k] for k in keys]),
            [np.stack([dual[k][j] for k in keys]) for j in range(4)], last[:, 3].astype(np.float32))


def target_rows():
    rows = []
    for fp in sorted((RUNS / "134_tennis_current127").glob("*.npz")):
        with np.load(fp, allow_pickle=True) as z:
            p, o, q, r = z["stage1_pred"], z["stage1_other"], z["stage2_pred"], z["stage2_other"]
            scalar = np.r_[ch(p), ch(o), ch2(q), ch2(r), p[0]-q[0], o[0]-r[0],
                           (p[0]-o[0])-(q[0]-r[0])]
            ph, oh = z["pred_hidden"].astype(np.float32), z["other_hidden"].astype(np.float32)
            rows.append((str(z["key"].item()), int(z["correct"]), str(z["probe_state"].item()),
                         scalar, (ph[0], wd(ph, z["pred_u"]), oh[0], wd(oh, z["other_u"])),
                         z["layer14"].astype(np.float32)))
    if not rows:
        raise RuntimeError("no TennisQA feature rows found")
    return rows


def metrics(y, p):
    pred = p >= .5
    return {"n": int(len(y)), "correct": int(y.sum()), "incorrect": int(len(y)-y.sum()),
            "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, pred)),
            "mean_score_correct": float(p[y == 1].mean()), "mean_score_incorrect": float(p[y == 0].mean())}


def main():
    sy, ss, sh, sl = source_rows()
    rows = target_rows()
    ty = np.asarray([x[1] for x in rows])
    ts = np.stack([x[3] for x in rows])
    th = [np.stack([x[4][j] for x in rows]) for j in range(4)]
    tl = np.stack([x[5] for x in rows])
    seed_probs = []
    for seed in (42, 43, 44):
        source_parts, target_parts = [], []
        for src, tgt, dim in [(ss, ts, None), *[(sh[j], th[j], 8) for j in range(4)], (sl, tl, 48)]:
            scaler = StandardScaler().fit(src)
            a, b = scaler.transform(src), scaler.transform(tgt)
            if dim is not None:
                pca = PCA(dim, whiten=True, svd_solver="randomized", random_state=seed).fit(a)
                a, b = pca.transform(a), pca.transform(b)
            source_parts.append(a); target_parts.append(b)
        clf = LogisticRegression(C=.03, max_iter=5000, class_weight="balanced",
                                 solver="liblinear", random_state=seed).fit(np.concatenate(source_parts, 1), sy)
        seed_probs.append(clf.predict_proba(np.concatenate(target_parts, 1))[:, 1])
    probs = np.mean(seed_probs, axis=0)
    ids = np.asarray([int(x[0].rsplit("_", 1)[1]) for x in rows])
    known = np.asarray([x[2] == "knows_both" for x in rows])
    masks = {"all": np.ones(len(rows), dtype=bool),
             "probe_known_both": known,
             "original_100": ids < 100,
             "original_100_probe_known_both": (ids < 100) & known,
             "new_100": ids >= 100,
             "new_100_probe_known_both": (ids >= 100) & known}
    report = {"protocol": "zero-shot frozen transfer: fit preprocessing/PCA/LR on all 1084 Scientist rows; no TennisQA labels used",
              "source_detector": "current127: scalar47 + four candidate-hidden PCA8 + layer14 PCA48; LR C=.03",
              "ensemble_seeds": [42, 43, 44], "subsets": {}, "per_item": []}
    for name, mask in masks.items():
        report["subsets"][name] = metrics(ty[mask], probs[mask])
        report["subsets"][name]["per_seed_auroc"] = [float(roc_auc_score(ty[mask], p[mask])) for p in seed_probs]
    for row, score in zip(rows, probs):
        report["per_item"].append({"id": row[0], "correct": bool(row[1]), "probe_state": row[2],
                                   "scientist_detector_score": float(score)})
    out = RUNS / "135_scientist_frozen_on_tennis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "per_item"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
