#!/usr/bin/env python3
"""Plot the phase-1 and confirmation results for multi-cue interactions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ATLAS = RUNS / "230_scientist_factorial_interaction_atlas/cluster_report.json"
ROBUST = RUNS / "233_confirm_multicue_repairs/report.json"
GEN = RUNS / "234_paired_generation_multicue_controls/report.json"
OUT = RUNS / "235_multicue_results_overview.png"


def annotate(ax, bars, fmt=lambda x: f"{x:.0f}"):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                fmt(value), ha="center", va="bottom", fontsize=9,
                fontweight="bold")


def main():
    atlas = json.loads(ATLAS.read_text())
    robust = json.loads(ROBUST.read_text())
    gen = json.loads(GEN.read_text())
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    blue, orange, green, red, gray = "#3274A1", "#E1812C", "#3A923A", "#C03D3E", "#8A8A8A"

    # A. Dataset and repair funnel.
    ax = axes[0, 0]
    cov = atlas["coverage"]
    rep = atlas["systematic_error_repair"]
    labels = ["Parse-valid\ngeneration errors", "Analyzed\n(>=2 cues)",
              "Systematic\nerrors", "Any cue-set\nrepair", "Multi-cue\nonly repair"]
    values = [cov["generation_errors_parse_valid"], cov["analysed_m_ge_2"],
              cov["likelihood_systematic"],
              rep["single_cue_repair_available"] + rep["multi_cue_only_repair"],
              rep["multi_cue_only_repair"]]
    bars = ax.bar(labels, values, color=[blue, blue, orange, green, red])
    annotate(ax, bars)
    ax.set_ylabel("Questions")
    ax.set_title("A. Coverage and repair funnel", loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

    # B. Robust interaction taxonomy; 'other' is shown separately in gray.
    ax = axes[0, 1]
    counts = atlas["robust_pair_taxonomy"]["pair_counts"]
    order = ["competition", "robust_redundancy", "robust_synergy",
             "pure_combination", "other"]
    names = ["Competition", "Redundancy", "Synergy", "Pure\ncombination", "Other"]
    vals = [counts[x] for x in order]
    bars = ax.bar(names, vals, color=[orange, blue, green, red, gray])
    annotate(ax, bars)
    ax.set_ylabel("Keyword pairs")
    ax.set_title("B. Pair relationships (robust rule)", loc="left", fontweight="bold")
    ax.text(.99, .96, "Robust = local FD and Banzhaf\nagree, |effect| > 0.1",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#444444")

    # C. Cross-operator flip rates.
    ax = axes[1, 0]
    methods = ["Length-preserving\nneutralization", "Physical\ndeletion"]
    target = np.array([robust["neutralisation"]["target_flip_rate"],
                       robust["physical_deletion"]["target_flip_rate"]]) * 100
    random = np.array([robust["neutralisation"]["matched_random_flip_rate_mean"],
                       robust["physical_deletion"]["matched_random_flip_rate_mean"]]) * 100
    x = np.arange(2); width = .34
    b1 = ax.bar(x-width/2, target, width, label="Target cue set", color=green)
    b2 = ax.bar(x+width/2, random, width, label="Matched random", color=gray)
    annotate(ax, b1, lambda z: f"{z:.1f}%"); annotate(ax, b2, lambda z: f"{z:.1f}%")
    ax.set_xticks(x, methods); ax.set_ylim(0, 108); ax.set_ylabel("Wrong-to-right margin flips")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("C. Repair does not transfer equally across operators",
                 loc="left", fontweight="bold")

    # D. Paired-seed free generation.
    ax = axes[1, 1]
    pg = gen["mean_p_right"]
    labels = ["Original", "Best single\ncue", "Preselected\nrandom set", "Target\nmulti-cue set"]
    vals = np.array([pg["base"], pg["best_single"],
                     pg["preselected_random"], pg["joint"]]) * 100
    bars = ax.bar(labels, vals, color=[gray, blue, orange, green])
    annotate(ax, bars, lambda z: f"{z:.1f}%")
    ax.set_ylim(0, 65); ax.set_ylabel("Correct-person generation rate")
    ax.set_title("D. Multi-cue intervention improves free generation",
                 loc="left", fontweight="bold")
    ci = gen["joint_minus_preselected_random"]["ci95"]
    ax.text(.02, .96,
            f"Target - random: +18.9 pp\n95% CI [{100*ci[0]:.1f}, {100*ci[1]:.1f}] pp",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#bbbbbb"})

    fig.suptitle("Scientist Multi-Cue Interaction Pilot and Confirmation",
                 fontsize=17, fontweight="bold")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
