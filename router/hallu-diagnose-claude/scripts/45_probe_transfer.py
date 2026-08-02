#!/usr/bin/env python3
"""
45_probe_transfer.py — 跨 probe 迁移: tool-gate 的"知识边界方向"能否点亮 router 的 Z1?

问题: router 与 tool-gate 是**两套独立测量**(不同数据、不同标签构造、不同任务)。
      L12/L30 的层位分离不能自动说明 router 读到了同样的东西, 而 router 甚至可能在读 domain。

检验: 在 tool-gate 数据(PopQA/synthetic, know vs dontknow)上求出方向 w,
      **不做任何再训练**, 直接投影到 router 特征上, 看:
        (a) Z1(知识缺失)样本的投影是否显著高于 Z2/Z4/Z6;
        (b) 仅用这一维投影做 Z1-vs-rest 的 AUROC 是多少。
      迁移成功 => router 的 Z1 信号有跨数据集的内部实在性, 不是 domain 捷径,
                 且与 tool-gate 的知识边界表征是同一个东西。
      迁移失败 => 两者读的是不同东西; §6 须按"独立测量"改写(这也是诚实的发现)。

必备对照:
  * 随机方向 (n 次): 同维度随机单位向量的投影 AUROC 分布 -> 迁移是否超出偶然。
  * 标签打乱方向: 在 tool-gate 上打乱 know/dontknow 后求方向 -> 排除"任意方向都行"。
  * 反向迁移: router 的 Z1-vs-rest 方向 -> tool-gate 的 know/dontknow, 双向一致更强。
  * 域内检验: 仅在 Z1 所在 domain 内比较, 排除领域差异伪装成知识信号。

前置条件: 两边必须是**同一模型**(层数与隐藏维一致)。脚本会自检, 不一致时明确报错。

用法:
  python 45_probe_transfer.py \\
      --gate-dir out_tool_gate/ --features data/features/<model> \\
      --gate-layer 12 --out probe_transfer.json
"""
from __future__ import annotations
import argparse, json, warnings
from collections import Counter
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


# ---------------- 载入 tool-gate ----------------
def load_gate(gate_dir: Path):
    import torch
    recs = read_jsonl(gate_dir / "records.jsonl")
    H, y, qids = [], [], []
    for r in recs:
        prior = r.get("know_prior")
        if prior not in ("known", "unknown"):
            continue
        p = gate_dir / "hidden" / f"{r['qid']}.pt"
        if not p.exists():
            continue
        h = torch.load(p, map_location="cpu", weights_only=False)["hidden"]
        H.append(np.nan_to_num(h.float().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
        y.append(1 if prior == "unknown" else 0)      # 1 = dontknow
        qids.append(r["qid"])
    if not H:
        raise SystemExit(f"{gate_dir} 未载入任何 tool-gate 隐藏状态")
    return np.stack(H), np.array(y), np.array(qids), recs


# ---------------- 载入 router ----------------
def load_router(feat_dir: Path):
    idx = read_jsonl(feat_dir / "index.jsonl")
    H, lab, dom, grp = [], [], [], []
    for r in idx:
        if r["label"] == "CLEAN":
            continue
        p = feat_dir / f"{r['sid']}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        H.append(np.nan_to_num(d["f1"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        lab.append(r["label"]); dom.append(r.get("domain", ""))
        grp.append(r["sid"].replace("__clean", ""))
    if not H:
        raise SystemExit(f"{feat_dir} 未载入任何 router 特征")
    return np.stack(H), np.array(lab), np.array(dom), np.array(grp)


# ---------------- 方向 ----------------
def diff_of_means(X, y):
    """标准化后的类均值差方向 (对迁移比 logistic 更鲁棒: 不依赖尺度与正则)。"""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    n = np.linalg.norm(w)
    return (w / (n + 1e-12)), mu, sd


def project(X, w, mu, sd):
    return ((X - mu) / sd) @ w


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def main(args):
    from scipy.stats import mannwhitneyu
    gate_dir, feat_dir = Path(args.gate_dir), Path(args.features)
    Hg, yg, qg, _ = load_gate(gate_dir)
    Hr, lab, dom, grp = load_router(feat_dir)
    Lg, Dg = Hg.shape[1], Hg.shape[2]
    Lr, Dr = Hr.shape[1], Hr.shape[2]
    print(f"tool-gate: n={len(yg)} 层={Lg} 维={Dg} | dontknow={int(yg.sum())}")
    print(f"router   : n={len(lab)} 层={Lr} 维={Dr} | {dict(Counter(lab))}")

    res = {"gate": {"n": int(len(yg)), "layers": Lg, "dim": Dg,
                    "n_dontknow": int(yg.sum())},
           "router": {"n": int(len(lab)), "layers": Lr, "dim": Dr,
                      "label_counts": dict(Counter(lab)),
                      "domain_counts": dict(Counter(dom))}}

    if Dg != Dr:
        res["error"] = (f"隐藏维不一致 ({Dg} vs {Dr}) —— 两边不是同一模型, 方向无法迁移。"
                        f"请把 tool-gate 与 router 特征提取统一到同一模型后重跑。")
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("\n" + res["error"]); return
    if Lg != Lr:
        res["warning"] = (f"层数不一致 ({Lg} vs {Lr}); 将按**相对深度**对齐层索引, "
                          f"结论需谨慎。")
        print("\n[warn] " + res["warning"])

    gl = args.gate_layer
    rl = gl if Lg == Lr else int(round(gl / (Lg - 1) * (Lr - 1)))
    res["layer_mapping"] = {"gate_layer": gl, "router_layer": rl}

    # ---- 在 tool-gate 上求方向 ----
    w, mu, sd = diff_of_means(Hg[:, gl], yg)
    res["direction_quality_on_source"] = {
        "auroc_on_gate_data": round(auroc(yg, project(Hg[:, gl], w, mu, sd)), 4),
        "note": "源域自评; 高不代表能迁移。"}

    # ---- 迁移到 router ----
    s = project(Hr[:, rl], w, mu, sd)
    is_z1 = (lab == "Z1").astype(int)
    a_z1 = auroc(is_z1, s)
    per_class = {}
    for c in sorted(set(lab)):
        v = s[lab == c]
        per_class[c] = {"n": int(len(v)), "mean_projection": round(float(v.mean()), 4),
                        "std": round(float(v.std()), 4)}
    others = s[lab != "Z1"]
    z1 = s[lab == "Z1"]
    u, pu = mannwhitneyu(z1, others, alternative="greater") if len(z1) and len(others) else (np.nan, np.nan)
    res["transfer_gate_to_router"] = {
        "auroc_Z1_vs_rest_using_transferred_direction_only": round(a_z1, 4),
        "per_class_projection": per_class,
        "cohens_d_Z1_vs_rest": round(cohens_d(z1, others), 4),
        "mannwhitney_p_one_sided_greater": float(pu),
        "note": ("仅用一维投影(无再训练)。显著高于随机方向基线 => tool-gate 的知识边界方向"
                 "在 router 数据上依然点亮 Z1 => 跨数据集的内部实在性。"),
    }

    # ---- 对照 1: 随机方向 ----
    rng = np.random.RandomState(args.seed)
    rnd = []
    for _ in range(args.n_random):
        v = rng.randn(Dr); v /= np.linalg.norm(v)
        rnd.append(auroc(is_z1, ((Hr[:, rl] - Hr[:, rl].mean(0)) /
                                 (Hr[:, rl].std(0) + 1e-8)) @ v))
    rnd = np.array([x for x in rnd if np.isfinite(x)])
    res["control_random_directions"] = {
        "n": len(rnd), "mean_auroc": round(float(rnd.mean()), 4),
        "p95_auroc": round(float(np.quantile(rnd, .95)), 4),
        "max_auroc": round(float(rnd.max()), 4),
        "empirical_p": round(float((np.abs(rnd - 0.5) >= abs(a_z1 - 0.5)).mean()), 4),
        "note": "随机方向的 |AUROC-0.5| 分布; empirical_p 小 => 迁移非偶然。"}

    # ---- 对照 2: 源域标签打乱 ----
    sh = []
    for _ in range(args.n_shuffle):
        yp = rng.permutation(yg)
        w2, mu2, sd2 = diff_of_means(Hg[:, gl], yp)
        sh.append(auroc(is_z1, project(Hr[:, rl], w2, mu2, sd2)))
    sh = np.array([x for x in sh if np.isfinite(x)])
    res["control_shuffled_source_labels"] = {
        "n": len(sh), "mean_auroc": round(float(sh.mean()), 4),
        "p95_auroc": round(float(np.quantile(sh, .95)), 4) if len(sh) else None,
        "note": "打乱 know/dontknow 后求方向再迁移; 应退化到 0.5 附近。"}

    # ---- 对照 3: 域内检验 ----
    wd = {}
    for d in sorted(set(dom)):
        m = dom == d
        if m.sum() < 40 or len(set(lab[m])) < 2 or (lab[m] == "Z1").sum() < 10:
            continue
        wd[d] = {"n": int(m.sum()), "n_z1": int((lab[m] == "Z1").sum()),
                 "auroc": round(auroc((lab[m] == "Z1").astype(int), s[m]), 4)}
    res["within_domain_transfer"] = {
        "by_domain": wd,
        "note": "领域在此恒定; 仍显著 => 迁移信号不是领域差异伪装的。"}

    # ---- 逐层迁移曲线 (源层固定, 扫描目标层) ----
    curve = []
    for l in range(1, Lr, max(1, Lr // 20)):
        curve.append((int(l), round(auroc(is_z1, project(Hr[:, l], w, mu, sd)), 4)))
    res["transfer_by_router_layer"] = {
        "curve": curve,
        "best": max(curve, key=lambda t: t[1]) if curve else None,
        "note": "源方向固定于 gate_layer; 扫描它在 router 各层的迁移效果。"}

    # ---- 反向迁移: router Z1 方向 -> tool-gate ----
    if args.reverse:
        w_r, mu_r, sd_r = diff_of_means(Hr[:, rl], is_z1)
        a_rev = auroc(yg, project(Hg[:, gl], w_r, mu_r, sd_r))
        res["reverse_transfer_router_to_gate"] = {
            "auroc_know_vs_dontknow": round(a_rev, 4),
            "note": "双向迁移都成立 => 两个 probe 捕捉的是同一表征方向。"}

    # ---- 判读 ----
    thr = res["control_random_directions"]["p95_auroc"]
    if a_z1 > max(thr, 0.65) and res["control_shuffled_source_labels"]["mean_auroc"] < 0.6:
        v = ("迁移成立: tool-gate 的知识边界方向在 router 数据上显著点亮 Z1, "
             "且超过随机方向与打乱标签对照 => router 的 Z1 信号具跨数据集内部实在性")
    elif a_z1 > thr:
        v = "迁移边缘: 超过随机方向但幅度有限, 需扩大样本或谨慎表述"
    else:
        v = ("迁移不成立: 两个 probe 读的可能是不同表征。§6 应按'独立测量'改写, "
             "不得声称 L12/L30 结构解释 router")
    res["verdict"] = v
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("transfer_by_router_layer",)}, indent=2,
                     ensure_ascii=False)[:3500])
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", required=True, help="tool_gate 输出目录 (含 records.jsonl 与 hidden/)")
    ap.add_argument("--features", required=True, help="router 特征目录 (含 index.jsonl 与 *.npz)")
    ap.add_argument("--gate-layer", type=int, default=12, help="源方向所在层 (tool-gate 的最优层)")
    ap.add_argument("--out", default="probe_transfer.json")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-shuffle", type=int, default=50)
    ap.add_argument("--reverse", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
