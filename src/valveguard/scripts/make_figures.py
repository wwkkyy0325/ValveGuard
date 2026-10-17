"""Figure set for the detector report.

Charts are deliberately simple: one axis per panel, direct labels where
possible, a validated categorical palette, and explicit n/caveats in titles.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from valveguard.data.tar_index import load_cache
from valveguard.features.geometry import KILL_ROW, TICK_RATE, recover

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d3"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 8,
})

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

WEAPON_NAMES = {
    0: "knife", 1: "auto rifle", 2: "semi rifle",
    3: "pistol", 4: "grenade", 5: "smg", 6: "shotgun",
}


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_roc() -> None:
    d = pd.read_parquet("reports/cascade/kill_table.parquet")
    series = [("GBDT", "p_g", BLUE), ("Transformer", "p_t", ORANGE),
              ("Ensemble", "p_ens", AQUA)]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    for name, col, color in series:
        p = d.groupby(["match_key", "player_name"], as_index=False).agg(
            label=("label", "max"), score=(col, "mean"))
        fpr, tpr, _ = roc_curve(p["label"], p["score"])
        auc = roc_auc_score(p["label"], p["score"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Player-level ROC — 5-fold OOF, calibrated, match-disjoint")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="lower right")
    _save(fig, "fig01_player_roc.png")


def fig_risk_coverage() -> None:
    p = pd.read_parquet("reports/cascade/player_table.parquet")
    series = [("mean", "mean", BLUE), ("HMM", "hmm", ORANGE),
              ("noisy-OR", "noisy_or", AQUA)]
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for name, col, color in series:
        fpr, tpr, _ = roc_curve(p["label"], p[col])
        keep = fpr <= 0.2
        ax.plot(fpr[keep], tpr[keep], color=color, lw=2, label=name)
    ax.axvline(0.01, color=RED, lw=1, ls=":")
    ax.text(0.011, 0.05, "1% FPR", color=RED, fontsize=8, rotation=90)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 0.2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate (log scale)")
    ax.set_ylabel("Player recall")
    ax.set_title("Player risk–coverage — low-FPR operating region")
    ax.grid(True, alpha=0.5, which="both")
    ax.legend(loc="upper left")
    _save(fig, "fig02_risk_coverage.png")


def fig_calibration() -> None:
    cal = pd.read_csv("reports/cascade/calibration_by_decile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--", label="perfect")
    ax.plot(cal["pred"], cal["obs"], color=BLUE, lw=2, marker="o", ms=4,
            label="observed")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed cheat rate")
    ax.set_title("Kill-level calibration by decile")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper left")

    ax = axes[1]
    d = pd.read_parquet("reports/cascade/kill_table.parquet")
    for label, color, name in [(0, BLUE, "human"), (1, ORANGE, "cheater")]:
        ax.hist(d.loc[d.label == label, "p_ens"], bins=40, color=color,
                alpha=0.55, label=name, density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by label")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center")
    _save(fig, "fig03_calibration.png")


def fig_subgroup_fpr() -> None:
    d = json.load(open("reports/subgroup_cal/subgroup_calibration.json"))
    base = {g["weapon"]: g for g in d["weapon_fpr"]["p_ens"]["groups"]}
    sub = {g["weapon"]: g for g in d["weapon_fpr"]["p_sub"]["groups"]}
    weapons = sorted(set(base) & set(sub))
    x = np.arange(len(weapons))
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    w = 0.36
    ax.bar(x - w / 2, [base[k]["fpr"] * 100 for k in weapons], w,
           color=BLUE, label="global model")
    ax.bar(x + w / 2, [sub[k]["fpr"] * 100 for k in weapons], w,
           color=ORANGE, label="per-weapon isotonic")
    ax.axhline(1.0, color=RED, lw=1, ls=":")
    ax.text(len(weapons) - 0.4, 1.1, "1% target", color=RED, fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{WEAPON_NAMES.get(k, str(k))}\n(n={base[k]['n']})" for k in weapons],
        rotation=0, fontsize=7,
    )
    ax.set_ylabel("FPR at global cutoff (%)")
    ax.set_title("Residual subgroup bias — per-weapon FPR")
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend(loc="upper right")
    for i, k in enumerate(weapons):
        ax.text(i - w / 2, base[k]["fpr"] * 100 + 0.1,
                f"{base[k]['fpr']*100:.1f}", ha="center", fontsize=7, color=MUTED)
        ax.text(i + w / 2, sub[k]["fpr"] * 100 + 0.1,
                f"{sub[k]['fpr']*100:.1f}", ha="center", fontsize=7, color=MUTED)
    _save(fig, "fig04_subgroup_fpr.png")


def fig_fold_variability() -> None:
    d = pd.read_csv("reports/cascade/player_fold_metrics.csv")
    aggs = ["mean", "top3", "noisy_or", "hmm"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "auc", "Player AUC by fold"),
        (axes[1], "recall_fpr1", "Player recall@1% FPR by fold"),
    ]:
        for i, (agg, color) in enumerate(zip(aggs, colors)):
            g = d[d["agg"] == agg]
            ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)),
                       g[metric], color=color, s=28, alpha=0.85, label=agg)
            ax.hlines(g[metric].mean(), i - 0.22, i + 0.22, color=color, lw=2)
        ax.set_xticks(range(len(aggs)))
        ax.set_xticklabels(aggs, rotation=15)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.5)
    axes[0].set_ylabel("AUC")
    axes[1].set_ylabel("Recall")
    _save(fig, "fig05_fold_variability.png")


def fig_residual_compare() -> None:
    d = json.load(open("reports/mouse_residual_eval2/residual_eval.json"))
    sets = ["base", "base_tail", "base_mr", "mr_only"]
    labels = ["base", "+tail residual", "+all residual", "residual only"]
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "auc", "Player AUC — 160-match subset"),
        (axes[1], "recall_at_fpr1", "Player recall@1% FPR — 160-match subset"),
    ]:
        vals = [d[s]["player"][key] for s in sets]
        bars = ax.bar(range(len(sets)), vals, color=colors, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=MUTED)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.5)
        ax.set_ylim(0, 1.0 if key == "auc" else max(vals) * 1.35)
    _save(fig, "fig06_residual_compare.png")


def fig_map_holdout() -> None:
    d = pd.read_csv("reports/analysis_relative/map_holdout_gbdt.csv")
    d = d.sort_values("player_auc")
    fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
    y = np.arange(len(d))
    ax.hlines(y, 0.8, d["player_auc"], color=GRID, lw=1)
    ax.scatter(d["player_auc"], y, color=BLUE, s=42, zorder=3)
    ax.axvline(0.9, color=RED, lw=1, ls=":")
    ax.text(0.902, -0.6, "0.90", color=RED, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{m}  (n={int(n)})" for m, n in zip(d["map"], d["n_test"])]
    )
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("Held-out-map player AUC")
    ax.set_title("Map holdout — aim-only GBDT")
    ax.grid(True, axis="x", alpha=0.5)
    _save(fig, "fig07_map_holdout.png")


def _human_band(cache: str, weapon_idx: int, dist_bin: tuple[float, float],
                n: int = 1200):
    mm, meta, names = load_cache(cache)
    feats = pd.read_parquet(
        "reports/baseline_clean/summary_features.parquet",
        columns=["row", "weapon_idx"],
    )
    meta = meta.merge(feats, on="row", how="left")
    clean = meta[(meta.label == 0) & (meta.weapon_idx == weapon_idx)]
    if len(clean) < 200:
        clean = meta[meta.label == 0]
    rng = np.random.default_rng(0)
    take = clean.sample(n=min(n, len(clean)), random_state=0)
    errs, speeds = [], []
    for row in take["row"]:
        w = pd.DataFrame(mm[row], columns=names)
        g = recover(w)
        errs.append(np.asarray(g["err_origin"])[KILL_ROW - 63:KILL_ROW + 1])
        d = np.asarray(g["dyaw"])[KILL_ROW - 63:KILL_ROW + 1] * TICK_RATE
        p = np.asarray(g["dpitch"])[KILL_ROW - 63:KILL_ROW + 1] * TICK_RATE
        speeds.append(np.hypot(d, p))
    return np.asarray(errs), np.asarray(speeds)


def fig_trajectory() -> None:
    cache = "data/processed/cw256"
    d = pd.read_parquet("reports/cascade/kill_table.parquet")
    mm, meta, names = load_cache(cache)
    feats = pd.read_parquet(
        "reports/baseline_clean/summary_features.parquet",
        columns=["row", "weapon_idx"],
    )
    meta = meta.merge(feats, on="row", how="left")
    idx = meta.set_index("row")
    # A high-scoring cheater kill and a low-scoring human kill in the same
    # weapon group.
    top = d[d.label == 1].sort_values("p_ens", ascending=False).iloc[0]
    weapon = int(idx.loc[top.row, "weapon_idx"])
    human_pool = d[(d.label == 0) & (d.weapon_idx == weapon)]
    human = human_pool.sort_values("p_ens").iloc[len(human_pool) // 2]
    errs, speeds = _human_band(cache, weapon, (0, 4000))
    t = (np.arange(64) - 63) / TICK_RATE

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for ax, arr, sample_col, ylabel, title in [
        (axes[0], errs, "err_origin", "Crosshair-to-target error (deg)",
         "Angular error — last 64 ticks"),
        (axes[1], speeds, None, "Angular speed (deg/s)",
         "Angular speed — last 64 ticks"),
    ]:
        lo = np.percentile(arr, 10, axis=0)
        hi = np.percentile(arr, 90, axis=0)
        med = np.percentile(arr, 50, axis=0)
        ax.fill_between(t, lo, hi, color=BLUE, alpha=0.16, label="human 10–90%")
        ax.plot(t, med, color=BLUE, lw=1.6, label="human median")
        for row, color, name in [(top.row, ORANGE, "cheater kill"),
                                 (human.row, AQUA, "human kill")]:
            g = recover(pd.DataFrame(mm[row], columns=names))
            if sample_col:
                y = np.asarray(g[sample_col])[KILL_ROW - 63:KILL_ROW + 1]
            else:
                y = np.hypot(np.asarray(g["dyaw"])[KILL_ROW - 63:KILL_ROW + 1],
                             np.asarray(g["dpitch"])[KILL_ROW - 63:KILL_ROW + 1]) * TICK_RATE
            ax.plot(t, y, color=color, lw=2, label=name)
        ax.axvline(0, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("Time relative to kill (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper left")
    _save(fig, "fig08_trajectory_deviation.png")


def fig_feature_percentiles() -> None:
    feats = pd.read_parquet("reports/baseline_clean/summary_features.parquet")
    labels = pd.read_parquet(
        "data/processed/cw256/index.parquet", columns=["row", "label"]
    )
    feats = feats.merge(labels, on="row", how="left")
    d = pd.read_parquet("reports/cascade/kill_table.parquet")
    top = d[d.label == 1].sort_values("p_ens", ascending=False).iloc[0]
    row = int(top.row)
    human = feats[feats.label == 0]
    sample = feats[feats.row == row].iloc[0]
    specs = [
        ("time_to_target", "time to target", True),
        ("final_err", "final error", False),
        ("sparc", "smoothness (SPARC)", True),
        ("ldlj", "smoothness (LDLJ)", False),
        ("submovements", "submovements", False),
        ("psd_slope", "PSD slope", False),
        ("lock_ratio", "lock ratio", False),
        ("lead_lag_corr", "lead–lag corr", False),
    ]
    names, pct = [], []
    for col, label, higher_is_better in specs:
        if col not in human.columns or not np.isfinite(sample[col]):
            continue
        vals = human[col].dropna().to_numpy()
        if len(vals) < 100:
            continue
        sample_val = sample[col]
        # LDLJ is negative; negate it so a higher percentile always means
        # "smoother / stronger" in the plotted direction.
        if col == "ldlj":
            vals = -vals
            sample_val = -sample_val
        p = float(np.mean(vals <= sample_val) * 100)
        names.append(label)
        pct.append(p)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    y = np.arange(len(names))
    colors = [ORANGE if p > 90 or p < 10 else BLUE for p in pct]
    ax.barh(y, pct, color=colors, height=0.55)
    ax.axvline(50, color=GRID, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile within human kill windows")
    ax.set_title("Sample cheater kill — feature deviation\n(orange = outside human 10–90%)")
    ax.grid(True, axis="x", alpha=0.4)
    for i, p in enumerate(pct):
        ax.text(p + 1.5, i, f"{p:.0f}", va="center", fontsize=8, color=MUTED)
    _save(fig, "fig09_feature_percentiles.png")


def fig_rank_strata() -> None:
    d = json.load(open("reports/skill/skill_metrics.json"))
    byq = pd.DataFrame(d["by_rank_quartile"])
    fpr = pd.DataFrame(d["fpr_by_rank_quartile_at_global_1pct"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    x = np.arange(len(byq))
    colors = [BLUE, BLUE, BLUE, RED]
    axes[0].bar(x, byq["auc"], color=colors, width=0.6)
    for i, v in enumerate(byq["auc"]):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=MUTED)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(byq["quartile"], rotation=15)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_ylabel("Player AUC")
    axes[0].set_title("AUC by rank quartile")
    axes[0].grid(True, axis="y", alpha=0.5)

    axes[1].bar(x, fpr["fpr"] * 100, color=colors, width=0.6)
    axes[1].axhline(1.0, color=RED, lw=1, ls=":")
    for i, v in enumerate(fpr["fpr"] * 100):
        axes[1].text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=8, color=MUTED)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fpr["quartile"], rotation=15)
    axes[1].set_ylabel("FPR at global 1% cutoff (%)")
    axes[1].set_title("False-positive rate by rank quartile")
    axes[1].grid(True, axis="y", alpha=0.5)
    _save(fig, "fig10_rank_strata.png")


if __name__ == "__main__":
    fig_roc()
    fig_risk_coverage()
    fig_calibration()
    fig_subgroup_fpr()
    fig_fold_variability()
    fig_residual_compare()
    fig_map_holdout()
    fig_trajectory()
    fig_feature_percentiles()
    fig_rank_strata()
