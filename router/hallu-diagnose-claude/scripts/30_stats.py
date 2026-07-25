"""统计分析: 治愈率矩阵 + McNemar(matched vs mismatched) + 交互 logistic 回归 + 热图。
用法: python scripts/30_stats.py --result data/results/matrix_DeepSeek-R1-Distill-Llama-8B.jsonl
"""
import argparse, itertools
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from common import DATA, read_jsonl

MATCHED = {"Z1": "T-RAG", "Z2": "T-Clean", "Z3": "T-CF", "Z4": "T-Budget", "Z6": "T-Abstain"}

def main(result_path, metric="strict"):
    df = pd.DataFrame(read_jsonl(result_path))
    df[metric] = df[metric].astype(int)

    # ---- 1. 治愈率矩阵(strict 与 honest 各一张)
    for m in ("strict", "honest"):
        mat = df.pivot_table(index="stressor", columns="treatment", values=m, aggfunc="mean")
        mat = mat.reindex(index=sorted(df.stressor.unique()))
        print(f"\n===== 治愈率矩阵 ({m}) =====\n", (mat * 100).round(1))
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(mat.index)), mat.index)
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Cure rate ({m})")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(DATA / f"results/matrix_{m}.png", dpi=200)

    # ---- 2. 对角优势 Delta (go/no-go 指标)
    # Z6 用 honest, 其余用 strict (README 双结局设计)
    diag, off = [], []
    for z, t_match in MATCHED.items():
        m = "honest" if z == "Z6" else "strict"
        sub = df[df.stressor == z]
        if sub.empty:
            continue
        d = sub[sub.treatment == t_match][m].mean()
        o = sub[~sub.treatment.isin([t_match, "none", "T-CleanOracle"])].groupby("treatment")[m].mean().mean()
        diag.append(d); off.append(o)
        print(f"[Delta] {z}: matched({t_match})={d:.1%}  mismatched均值={o:.1%}")
    delta = np.mean(diag) - np.mean(off)
    print(f"\n>>> 对角优势 Delta = {delta:.3f}  (GO 标准 >= 0.25; No-Go < 0.15) <<<")

    # ---- 3. McNemar: 每个 stressor 上 matched vs 每个 mismatched (配对样本)
    print("\n===== McNemar (matched vs mismatched, 同一样本配对) =====")
    n_tests = sum(1 for z in MATCHED for t in df.treatment.unique()
                  if t not in (MATCHED[z], "none", "T-CleanOracle"))
    alpha = 0.05 / max(n_tests, 1)  # Bonferroni
    for z, tm in MATCHED.items():
        m = "honest" if z == "Z6" else "strict"
        sub = df[df.stressor == z]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="sid", columns="treatment", values=m)
        for t in piv.columns:
            if t in (tm, "none", "T-CleanOracle") or tm not in piv.columns:
                continue
            pair = piv[[tm, t]].dropna()
            b = ((pair[tm] == 1) & (pair[t] == 0)).sum()  # matched治愈而mismatched没治愈
            c = ((pair[tm] == 0) & (pair[t] == 1)).sum()
            if b + c == 0:
                continue
            res = mcnemar([[0, b], [c, 0]], exact=(b + c < 25))
            sig = "*" if res.pvalue < alpha else " "
            print(f"  {z}: {tm} vs {t}: b={b} c={c} p={res.pvalue:.2e} {sig}")

    # ---- 4. stressor x treatment 交互 logistic 回归
    sub = df[~df.treatment.isin(["none", "T-CleanOracle"])].copy()
    sub["y"] = np.where(sub.stressor == "Z6", sub.honest, sub.strict)
    try:
        model = smf.logit("y ~ C(stressor) * C(treatment)", data=sub).fit(disp=0, maxiter=200)
        lr_null = smf.logit("y ~ C(stressor) + C(treatment)", data=sub).fit(disp=0, maxiter=200)
        from scipy.stats import chi2
        lr_stat = 2 * (model.llf - lr_null.llf)
        dof = model.df_model - lr_null.df_model
        print(f"\n===== 交互项 LR 检验: chi2={lr_stat:.1f}, dof={dof}, "
              f"p={chi2.sf(lr_stat, dof):.2e} =====")
    except Exception as exc:
        print(f"\n[warning] 交互 logistic 回归不可估计: {exc}")

    # ---- 5. 剂量-反应 (Z2 干扰句数)
    z2 = df[(df.stressor == "Z2") & (df.treatment == "none")]
    if not z2.empty and z2.intensity.nunique() > 1:
        print("\nZ2 剂量-反应 (none 治疗下的 strict 正确率):")
        print(z2.groupby("intensity")["strict"].mean().round(3))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    a = ap.parse_args()
    main(a.result)
