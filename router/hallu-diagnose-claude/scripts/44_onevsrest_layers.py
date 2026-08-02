#!/usr/bin/env python3
"""
44_onevsrest_layers.py — 各 stressor 的 one-vs-rest 逐层扫描 (形态学证据)

目的: router 是单层/多族拼接的分类器, 其"读到什么"不透明。
      对每一类做 one-vs-rest 的逐层扫描, 看各类信号的**峰值层**落在哪里。
      若知识类(Z1)峰值靠前、决策/校准类(Z6)峰值靠后, 则与 tool-gate 在独立数据上
      观察到的 "知道@早层 / 行动@深层" 结构**形态一致** —— 这是形态学证据,
      不是同一性证据, 但比无连接强得多。

必做的控制:
  * 域内重复: 同一扫描在单一 domain 内重跑, 排除"峰值差异其实来自领域差异"。
  * 峰值层的 bootstrap CI: 逐层曲线常有平台, 单点 argmax 不稳。
  * 曲线形状指标: 除峰值外报告"达到 95% 峰值性能的最早层" (更稳健的 onset 指标)。
  * 随机标签对照: 打乱标签后的峰值分布, 确认观察到的结构非偶然。

用法:
  python 44_onevsrest_layers.py --features data/features/<model> --out onevsrest.json
"""
from __future__ import annotations
import argparse, json, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def load_features(feat_dir: Path, drop_clean=True):
    idx = read_jsonl(feat_dir / "index.jsonl")
    rows, F1 = [], []
    for r in idx:
        if drop_clean and r["label"] == "CLEAN":
            continue
        p = feat_dir / f"{r['sid']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        if "f1" not in d:
            raise SystemExit(f"{p} 缺少 f1 字段; 请确认特征由 40_extract_features.py 生成")
        F1.append(np.nan_to_num(d["f1"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        rows.append(r)
    if not rows:
        raise SystemExit("未载入任何样本")
    L = F1[0].shape[0]
    if any(f.shape[0] != L for f in F1):
        raise SystemExit("各样本层数不一致")
    return rows, np.stack(F1), L


def layer_auroc(X_layer, y, groups, seed=0, n_splits=4):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    ng = len(set(groups))
    gkf = GroupKFold(n_splits=min(n_splits, ng))
    prob = np.zeros(len(y))
    for tr, te in gkf.split(X_layer, y, groups):
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X_layer[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(X_layer[tr]), y[tr])
        prob[te] = clf.predict_proba(sc.transform(X_layer[te]))[:, 1]
    try:
        return float(roc_auc_score(y, prob)), prob
    except ValueError:
        return float("nan"), prob


def scan(F1, y, groups, layers, seed=0):
    curve = []
    for l in layers:
        a, _ = layer_auroc(F1[:, l], y, groups, seed)
        curve.append((int(l), round(a, 4) if np.isfinite(a) else None))
    return curve


def summarize_curve(curve, onset_frac=0.95):
    vals = [(l, a) for l, a in curve if a is not None]
    if not vals:
        return {}
    peak_l, peak_a = max(vals, key=lambda t: t[1])
    base = 0.5
    thr = base + onset_frac * (peak_a - base)
    onset = next((l for l, a in vals if a >= thr), peak_l)
    # 曲线重心 (以 AUROC-0.5 为权重), 比 argmax 稳
    w = np.array([max(a - base, 0.0) for _, a in vals])
    ls = np.array([l for l, _ in vals], float)
    centroid = float((w * ls).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return {"peak_layer": int(peak_l), "peak_auroc": round(float(peak_a), 4),
            "onset_layer_95pct": int(onset), "centroid_layer": round(centroid, 2)}


def bootstrap_peak(F1, y, groups, layers, n_boot, seed):
    """按 item 重采样, 估计峰值层与 onset 层的不确定性。"""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups)))
    peaks, onsets = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        m = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if len(set(y[m])) < 2:
            continue
        c = scan(F1[m], y[m], groups[m], layers, seed)
        s = summarize_curve(c)
        if s:
            peaks.append(s["peak_layer"]); onsets.append(s["onset_layer_95pct"])
    q = lambda v: [int(np.quantile(v, .025)), int(np.quantile(v, .975))] if v else None
    return {"peak_layer_ci95": q(peaks), "onset_layer_ci95": q(onsets),
            "n_boot_effective": len(peaks)}


def main(args):
    feat_dir = Path(args.features)
    rows, F1, L = load_features(feat_dir)
    labels = np.array([r["label"] for r in rows])
    domains = np.array([r.get("domain", "") for r in rows])
    groups = np.array([r["sid"].replace("__clean", "") for r in rows])
    classes = sorted(set(labels))
    step = max(1, L // args.max_layers_scanned)
    layers = list(range(1, L, step))          # 排除 embedding 层
    print(f"n={len(rows)}  层数={L}  扫描层={layers}")
    print(f"类别分布: {dict(Counter(labels))}")

    res = {"n": len(rows), "n_layers": L, "layers_scanned": layers,
           "label_counts": dict(Counter(labels)),
           "domain_counts": dict(Counter(domains)), "one_vs_rest": {}}

    # ---------- 主扫描: 全数据 one-vs-rest ----------
    for c in classes:
        y = (labels == c).astype(int)
        if y.sum() < 15 or (y == 0).sum() < 15:
            res["one_vs_rest"][c] = {"skipped": f"n_pos={int(y.sum())}"}
            continue
        curve = scan(F1, y, groups, layers, args.seed)
        s = summarize_curve(curve)
        s["curve"] = curve
        s["n_pos"] = int(y.sum())
        if args.n_boot > 0:
            s.update(bootstrap_peak(F1, y, groups, layers, args.n_boot, args.seed))
        res["one_vs_rest"][c] = s
        print(f"  {c}: peak L{s['peak_layer']} (AUROC {s['peak_auroc']}) "
              f"onset L{s['onset_layer_95pct']} centroid {s['centroid_layer']}")

    # ---------- 域内重复 (排除领域驱动的峰值差异) ----------
    within = {}
    for dom in sorted(set(domains)):
        m = domains == dom
        if m.sum() < 60:
            continue
        sub_lab = labels[m]
        blk = {}
        for c in sorted(set(sub_lab)):
            y = (sub_lab == c).astype(int)
            if y.sum() < 15 or (y == 0).sum() < 15:
                continue
            curve = scan(F1[m], y, groups[m], layers, args.seed)
            s = summarize_curve(curve); s["n_pos"] = int(y.sum()); s["curve"] = curve
            blk[c] = s
        if blk:
            within[dom] = blk
    res["within_domain"] = within
    res["within_domain_note"] = ("域内重复: 若峰值层的相对顺序在域内保持, 说明形态差异"
                                 "不是由 stressor-domain 混淆造成的。")

    # ---------- 随机标签对照 ----------
    rng = np.random.RandomState(args.seed)
    null_peaks = []
    for _ in range(args.n_null):
        perm = rng.permutation(len(labels))
        y = (labels[perm] == classes[0]).astype(int)
        c = scan(F1, y, groups, layers, args.seed)
        s = summarize_curve(c)
        if s:
            null_peaks.append((s["peak_layer"], s["peak_auroc"]))
    if null_peaks:
        res["null_control"] = {
            "n": len(null_peaks),
            "mean_peak_auroc": round(float(np.mean([a for _, a in null_peaks])), 4),
            "peak_layer_spread": [int(np.min([l for l, _ in null_peaks])),
                                  int(np.max([l for l, _ in null_peaks]))],
            "note": "标签打乱后的峰值 AUROC 应接近 0.5; 峰值层应随机散布。"}

    # ---------- 形态判读 ----------
    ok = {c: v for c, v in res["one_vs_rest"].items() if "peak_layer" in v}
    if len(ok) >= 2:
        order_peak = sorted(ok, key=lambda c: ok[c]["peak_layer"])
        order_onset = sorted(ok, key=lambda c: ok[c]["onset_layer_95pct"])
        order_cent = sorted(ok, key=lambda c: ok[c]["centroid_layer"])
        res["morphology"] = {
            "by_peak_layer": [(c, ok[c]["peak_layer"]) for c in order_peak],
            "by_onset_layer": [(c, ok[c]["onset_layer_95pct"]) for c in order_onset],
            "by_centroid": [(c, ok[c]["centroid_layer"]) for c in order_cent],
            "layer_span_peak": int(ok[order_peak[-1]]["peak_layer"]
                                   - ok[order_peak[0]]["peak_layer"]),
            "note": ("与 tool-gate 独立观察到的 '知道@早层 / 行动@深层' 比较: "
                     "若知识类(Z1)靠前、校准/决策类(Z6)靠后, 即形态一致。"
                     "峰值常有平台, 请以 onset 与 centroid 为主, peak 为辅; "
                     "并核对 bootstrap CI 是否重叠 —— 重叠则不能声称顺序差异。"),
        }
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n形态:", json.dumps(res.get("morphology", {}), ensure_ascii=False, indent=1)[:900])
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="onevsrest_layers.json")
    ap.add_argument("--max-layers-scanned", type=int, default=33)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
