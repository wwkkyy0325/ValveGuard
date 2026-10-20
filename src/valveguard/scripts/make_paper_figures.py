"""Build the figures used by the final paper draft."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.scripts.compare_cnn_gbdt import _players

COLORS = {
    "champion": "#2f6fed",
    "mean": "#f9ab00",
    "gbdt": "#9aa0a6",
    "flag": "#d93025",
    "review": "#f9ab00",
    "below": "#5f6368",
    "human": "#9aa0a6",
    "cheater": "#d93025",
}
INK = "#202124"
MUTED = "#5f6368"


def _use_font(zh: bool) -> None:
    if zh:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "DejaVu Sans"
        ]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _t(zh: bool, en: str, cn: str) -> str:
    return cn if zh else en


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _score(models: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: _players(frame, "score") for name, frame in models.items()}


def _box(ax, x, y, width, height, text, face, edge, fontsize=10.5,
         dashed=False) -> None:
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4 if not dashed else 1.1,
        linestyle="--" if dashed else "-",
        facecolor=face, edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=INK, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color="#5f6368") -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.6, color=color, shrinkA=0, shrinkB=0,
    ))


def figure_architecture(out: Path, zh: bool = False) -> None:
    _use_font(zh)
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")

    blue_face, blue_edge = "#e8f0fe", "#2f6fed"
    gold_face, gold_edge = "#fff4e5", "#f9ab00"
    gray_face, gray_edge = "#f1f3f4", "#5f6368"

    rows = [
        (10.3, _t(zh,
                  "CS2CD kill window\n256 ticks × 29 aim-only channels\n(kill at tick 223, 64 Hz)",
                  "CS2CD 击杀窗口\n256 tick × 29 个纯瞄准通道\n（击杀在第 223 tick，64 Hz）"),
         blue_face, blue_edge),
        (9.0, _t(zh,
                 "Robust normalization\nper-channel median / IQR, clip ±10",
                 "鲁棒归一化\n逐通道 median / IQR，裁剪 ±10"),
         blue_face, blue_edge),
        (7.7, _t(zh,
                 "Convolutional encoder\n4 blocks: 96 → 192 → 256 → 256\nBatchNorm · ReLU · max-pool",
                 "卷积编码器\n4 个卷积块：96 → 192 → 256 → 256\nBatchNorm · ReLU · 最大池化"),
         blue_face, blue_edge),
        (6.4, _t(zh,
                 "Temporal attention\nsoftmax over 256 ticks → weighted summary z",
                 "时间注意力\n对 256 tick 做 softmax → 加权摘要 z"),
         blue_face, blue_edge),
        (5.1, _t(zh,
                 "MLP head\nkill-window score  p = σ(ℓ)",
                 "MLP 头\n击杀窗口分数  p = σ(ℓ)"),
         blue_face, blue_edge),
        (3.8, _t(zh,
                 "Player aggregation\nshrunk mean: (Σ p + k₀π) / (n + k₀)\nk₀ = 5, π = 0.302",
                 "玩家聚合\n收缩均值：(Σ p + k0π) / (n + k0)\nk0 = 5，π = 0.302"),
         gold_face, gold_edge),
        (2.5, _t(zh,
                 "Player threshold\nτ = human p99 = 0.608",
                 "玩家阈值\nτ = 人类 p99 = 0.608"),
         gray_face, gray_edge),
    ]
    width, height = 7.4, 0.92
    for index, (y, text, face, edge) in enumerate(rows):
        _box(ax, 6.0, y, width, height, text, face, edge)
        if index:
            _arrow(ax, 6.0, rows[index - 1][0] - height / 2,
                   6.0, y + height / 2)

    decisions = [
        (2.9, _t(zh, "FLAG\np ≥ τ", "FLAG\np ≥ τ"), "#fce8e6", COLORS["flag"]),
        (6.0, _t(zh, "REVIEW\np95 ≤ p < τ", "REVIEW\np95 ≤ p < τ"),
         "#fef7e0", COLORS["review"]),
        (9.1, _t(zh, "below\np < p95", "低于阈值\np < p95"),
         "#f1f3f4", COLORS["below"]),
    ]
    for x, text, face, edge in decisions:
        _box(ax, x, 1.12, 2.55, 0.82, text, face, edge, fontsize=11)
        _arrow(ax, 6.0, 2.5 - height / 2, x, 1.12 + 0.41)

    ax.text(6.0, 0.34,
            _t(zh,
               "REVIEW reasons (HS, shots/kill, 4s multikill, burst, wallbang, smoke) "
               "only order the queue.\nThey never change the score or the flag.",
               "REVIEW 理由（HS、shots/kill、4 秒连杀、burst、穿墙、烟雾）只用于排序复核队列。\n"
               "它们不会改变分数或 FLAG。"),
            ha="center", va="center", fontsize=10, color=MUTED,
            linespacing=1.3)

    _box(ax, 11.75, 8.05,
         3.5, 3.4,
         _t(zh,
            "Aim-only by design\n\nExcluded from the main score:\n"
            "• weapon and kill outcome\n"
            "• headshot / smoke / wallbang\n"
            "• raw mouse and recoil\n"
            "• visibility and reaction\n\n"
            "These are review context or\nnegative ablations, not features.",
            "设计上只用瞄准信息\n\n主分数排除：\n"
            "• 武器与击杀结果\n"
            "• 爆头 / 烟雾 / 穿墙\n"
            "• 原始鼠标与后坐力\n"
            "• 可见性与反应\n\n"
            "这些只作为复核上下文或\n负结果消融，不是特征。"),
         "none", "#9aa0a6", fontsize=9.2, dashed=True)
    _arrow(ax, 10.0, 8.05, 9.7, 8.05, color="#9aa0a6")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_roc(players: dict[str, pd.DataFrame], out: Path,
               zh: bool = False) -> None:
    _use_font(zh)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    style = {
        "ValveGuard CNN": ("champion", "-"),
        "CNN mean ensemble": ("mean", "--"),
        "GBDT baseline": ("gbdt", ":"),
    }
    labels_zh = {
        "ValveGuard CNN": "ValveGuard CNN",
        "CNN mean ensemble": "CNN 均值集成",
        "GBDT baseline": "GBDT 基线",
    }
    for name, frame in players.items():
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["score"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, scores)
        color = COLORS[style[name][0]]
        ax.plot(fpr, tpr, style[name][1], color=color, linewidth=2,
                label=f"{_t(zh, name, labels_zh[name])}  "
                      f"AUC {roc_auc_score(labels, scores):.4f}")
    ax.plot([0, 1], [0, 1], color="#dadce0", linewidth=1, linestyle="-")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(_t(zh, "Player false-positive rate", "玩家假阳性率"), fontsize=9)
    ax.set_ylabel(_t(zh, "Player true-positive rate", "玩家真阳性率"), fontsize=9)
    ax.set_title(_t(zh, "Player-level ROC on the fixed match folds",
                    "固定比赛折上的玩家级 ROC"), loc="left",
                 fontsize=11, color=INK)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(color="#f1f3f4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_low_fpr(players: dict[str, pd.DataFrame], out: Path,
                   zh: bool = False) -> None:
    _use_font(zh)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    style = {
        "ValveGuard CNN": ("champion", "-"),
        "CNN mean ensemble": ("mean", "--"),
        "GBDT baseline": ("gbdt", ":"),
    }
    labels_zh = {
        "ValveGuard CNN": "ValveGuard CNN",
        "CNN mean ensemble": "CNN 均值集成",
        "GBDT baseline": "GBDT 基线",
    }
    for name, frame in players.items():
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["score"].to_numpy(dtype=float)
        human = np.sort(scores[labels == 0])
        points = []
        for quantile in np.linspace(0.90, 0.9995, 80):
            threshold = np.quantile(human, quantile)
            fpr = float((scores[labels == 0] >= threshold).mean())
            recall = float((scores[labels == 1] >= threshold).mean())
            points.append((fpr, recall))
        points = np.array(points)
        points = points[points[:, 0] > 0]
        color = COLORS[style[name][0]]
        ax.plot(points[:, 0] * 100, points[:, 1] * 100, style[name][1],
                color=color, linewidth=2, label=_t(zh, name, labels_zh[name]))
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 12)
    ax.set_ylim(0, 100)
    ax.set_xlabel(_t(zh, "Player false-positive rate (%)",
                     "玩家假阳性率（%）"), fontsize=9)
    ax.set_ylabel(_t(zh, "Player recall (%)", "玩家召回率（%）"), fontsize=9)
    ax.set_title(_t(zh, "Low-FPR operating region", "低 FPR 工作区间"),
                 loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(color="#f1f3f4", linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_scores(players: dict[str, pd.DataFrame], out: Path,
                  zh: bool = False) -> None:
    _use_font(zh)
    frame = players["ValveGuard CNN"]
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    threshold = float(np.quantile(scores[labels == 0], 0.99))
    p95 = float(np.quantile(scores[labels == 0], 0.95))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bins = np.linspace(0, 1, 50)
    ax.hist(scores[labels == 0], bins=bins, density=True, color=COLORS["human"],
            alpha=0.45, edgecolor="white", linewidth=0.4,
            label=_t(zh, "human players", "干净玩家"))
    ax.hist(scores[labels == 1], bins=bins, density=True,
            color=COLORS["cheater"], alpha=0.45, edgecolor="white",
            linewidth=0.4, label=_t(zh, "cheater players", "作弊玩家"))
    ax.axvline(p95, color=MUTED, linestyle=":", linewidth=1)
    ax.axvline(threshold, color=COLORS["flag"], linestyle="--", linewidth=1.2)
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.86,
            _t(zh, f"p99 = {threshold:.3f}", f"p99 = {threshold:.3f}"),
            fontsize=7.5, color=COLORS["flag"])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(_t(zh, "Shrunk player score", "收缩后的玩家分数"), fontsize=9)
    ax.set_ylabel(_t(zh, "density", "密度"), fontsize=8, color=MUTED)
    ax.set_title(_t(zh, "Player score distribution (OOF)",
                    "玩家分数分布（OOF）"), loc="left", fontsize=11,
                 color=INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_rank(players: pd.DataFrame, ranks: pd.DataFrame, out: Path,
                zh: bool = False) -> None:
    _use_font(zh)
    frame = players.merge(
        ranks[["match_key", "player_name", "rank_q"]],
        on=["match_key", "player_name"], how="inner",
    )
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    threshold = float(np.quantile(scores[labels == 0], 0.99))
    order = ["Q1 low", "Q2", "Q3", "Q4 high"]
    fpr, recall, counts = [], [], []
    for name in order:
        subset = frame[frame["rank_q"] == name]
        human = subset[subset["label"] == 0]
        positive = subset[subset["label"] == 1]
        fpr.append(float((human["score"] >= threshold).mean()) * 100)
        recall.append(float((positive["score"] >= threshold).mean()) * 100)
        counts.append(len(subset))
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    width = 0.36
    ax.bar(x - width / 2, fpr, width, color=COLORS["gbdt"],
           label=_t(zh, "human FPR", "干净玩家 FPR"))
    ax.bar(x + width / 2, recall, width, color=COLORS["champion"],
           label=_t(zh, "cheater recall", "作弊玩家召回"))
    for index, value in enumerate(fpr):
        ax.text(index - width / 2, value + 0.6, f"{value:.1f}", ha="center",
                fontsize=7.5, color=INK)
    for index, value in enumerate(recall):
        ax.text(index + width / 2, value + 0.6, f"{value:.1f}", ha="center",
                fontsize=7.5, color=INK)
    order_zh = {
        "Q1 low": "Q1 低分", "Q2": "Q2", "Q3": "Q3", "Q4 high": "Q4 高分",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{_t(zh, name, order_zh[name])}\n(n={n})"
        for name, n in zip(order, counts)
    ], fontsize=8)
    ax.set_ylim(0, 46)
    ax.set_ylabel(_t(zh, "percent at the 1% threshold", "超过 1% 阈值的百分比"),
                  fontsize=9)
    ax.set_title(_t(zh, "Rank-stratified error at the champion threshold",
                    "冠军阈值下的排名分层错误率"), loc="left",
                 fontsize=11, color=INK)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", color="#f1f3f4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_worked_kills(demo_root: Path, out: Path,
                        zh: bool = False) -> None:
    _use_font(zh)
    result = json.loads((demo_root / "cdata_0" / "result.json").read_text())
    p95 = result["human_reference"]["p95"]
    threshold = result["threshold"]
    selected = [
        ("Player_3", "#2f6fed", "-", "cheater, 29 kills", "作弊者，29 杀"),
        ("Player_4", "#f9ab00", "--", "cheater, 19 kills", "作弊者，19 杀"),
        ("Player_2", "#5f6368", ":", "clean, 4 kills", "干净玩家，4 杀"),
        ("Player_1", "#9aa0a6", "-.", "clean, 6 kills", "干净玩家，6 杀"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for player, color, style, label, label_zh in selected:
        kills = sorted(
            [row for row in result["kills"] if row["player"] == player],
            key=lambda row: row["kill_tick"],
        )
        x = np.arange(1, len(kills) + 1)
        y = [row["score"] for row in kills]
        ax.plot(x, y, style, color=color, linewidth=1.6, marker="o",
                markersize=3.5,
                label=_t(zh, f"{player} ({label})", f"{player}（{label_zh}）"))
    ax.axhline(p95, color=MUTED, linestyle=":", linewidth=1)
    ax.axhline(threshold, color=COLORS["flag"], linestyle="--", linewidth=1.2)
    ax.text(1, p95 + 0.015,
            _t(zh, f"human p95 = {p95:.3f}", f"人类 p95 = {p95:.3f}"),
            fontsize=7.5, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.text(1, threshold + 0.015,
            _t(zh, f"p99 threshold = {threshold:.3f}",
               f"p99 阈值 = {threshold:.3f}"),
            fontsize=7.5, color=COLORS["flag"],
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xlabel(_t(zh, "the player's kill index (chronological)",
                     "玩家内击杀序号（按时间）"), fontsize=9)
    ax.set_ylabel(_t(zh, "kill-window score", "击杀窗口分数"), fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title(_t(zh, "Per-kill scores in cdata:0",
                    "cdata:0 的逐击杀分数"), loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, ncol=2)
    ax.grid(color="#f1f3f4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_example_windows(out: Path, zh: bool = False) -> None:
    _use_font(zh)
    index = pd.read_parquet("data/processed/cw256/index.parquet")
    ticks = pd.read_parquet("data/processed/kill_ticks.parquet")[["row", "kill_tick"]]
    frame = index[index["match_key"] == "cdata:0"].merge(ticks, on="row", how="inner")
    result = json.loads(
        (Path("reports") / "demo_dataset" / "cdata_0" / "result.json").read_text()
    )
    scores = {
        (row["player"], int(row["kill_tick"])): row["score"]
        for row in result["kills"]
    }
    frame["score"] = [
        scores.get((row.player_name, int(row.kill_tick)), np.nan)
        for row in frame.itertuples()
    ]
    high = frame[frame["player_name"] == "Player_3"].sort_values("score").iloc[-1]
    low = frame[frame["player_name"] == "Player_1"].sort_values("score").iloc[0]
    names = json.loads(
        (Path("data/processed/cw256_channels/channel_names.json")).read_text()
    )
    position = {name: i for i, name in enumerate(names)}
    source = np.load("data/processed/cw256_channels/channels.npy", mmap_mode="r")

    time = (np.arange(256) - 223) / 64.0
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for row, label, color in (
        (high, _t(zh,
                  f"high-score kill: {high.player_name}, score {high.score:.3f}",
                  f"高分击杀：{high.player_name}，分数 {high.score:.3f}"),
         "#2f6fed"),
        (low, _t(zh,
                 f"low-score kill: {low.player_name}, score {low.score:.3f}",
                 f"低分击杀：{low.player_name}，分数 {low.score:.3f}"),
         "#9aa0a6"),
    ):
        window = np.asarray(source[int(row.row)], dtype=np.float32)
        error = np.hypot(
            window[:, position["err_yaw"]], window[:, position["err_pitch"]]
        )
        speed = np.hypot(
            window[:, position["dyaw_dps"]], window[:, position["dpitch_dps"]]
        )
        axes[0].plot(time, error, color=color, linewidth=1.6, label=label)
        axes[1].plot(time, speed, color=color, linewidth=1.6)
    for ax in axes:
        ax.axvline(0, color=COLORS["flag"], linestyle="--", linewidth=1)
        ax.grid(color="#f1f3f4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(_t(zh, "angular error to target (deg)",
                          "到目标的角误差（度）"), fontsize=9)
    axes[1].set_ylabel(_t(zh, "angular speed (deg/s)", "角速度（度/秒）"),
                       fontsize=9)
    axes[1].set_xlabel(_t(zh, "time relative to the kill (s)",
                          "相对击杀的时间（秒）"), fontsize=9)
    axes[0].set_title(_t(zh, "Two kill windows as the model sees them",
                         "模型看到的两个击杀窗口"), loc="left",
                      fontsize=11, color=INK)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_demo(root: Path, out: Path, zh: bool = False) -> None:
    _use_font(zh)
    matches = [("cdata:0", "cdata_0"), ("ncdata:36", "ncdata_36")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, (title, folder) in zip(axes, matches):
        result = json.loads((root / folder / "result.json").read_text())
        players = sorted(result["players"],
                         key=lambda row: row["shrunk_score"], reverse=True)
        y = np.arange(len(players))[::-1]
        colors = [COLORS[row["tier"]] for row in players]
        ax.barh(y, [row["shrunk_score"] for row in players], color=colors,
                height=0.62)
        for index, row in zip(y, players):
            ax.text(row["shrunk_score"] + 0.008, index,
                    f"{row['shrunk_score']:.3f}", va="center", fontsize=7,
                    color=INK)
        ax.axvline(result["threshold"], color=COLORS["flag"], linestyle="--",
                   linewidth=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels([row["player"] for row in players], fontsize=7.5)
        ax.set_xlim(0, 0.85)
        ax.set_xlabel(_t(zh, "shrunk score", "收缩分数"), fontsize=8)
        ax.set_title(f"{title} ({result['map']})", loc="left", fontsize=10,
                     color=INK)
        ax.grid(axis="x", color="#f1f3f4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle(_t(zh, "CS2CD demo report: flag threshold and review queue",
                    "CS2CD 演示报告：FLAG 阈值与复核队列"),
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path("reports")
    out = root / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    raw = {
        "ValveGuard CNN": _load(str(root / "cnn29_hn5_ensemble" / "oof.parquet")),
        "CNN mean ensemble": _load(str(root / "cnn29_fixed_deploy" / "oof_ensemble.parquet")),
        "GBDT baseline": _load(str(root / "cv_gbdt" / "cv_predictions.parquet")),
    }
    players = _score(raw)
    ranks = pd.read_parquet(root / "rank_replication" / "rank_players.parquet")
    for zh, suffix in ((False, ""), (True, "_zh")):
        figure_architecture(out / f"fig0_architecture{suffix}.png", zh=zh)
        figure_roc(players, out / f"fig1_player_roc{suffix}.png", zh=zh)
        figure_low_fpr(players, out / f"fig2_low_fpr{suffix}.png", zh=zh)
        figure_scores(players, out / f"fig3_player_scores{suffix}.png", zh=zh)
        figure_rank(players["ValveGuard CNN"], ranks,
                    out / f"fig4_rank_strata{suffix}.png", zh=zh)
        figure_demo(root / "demo_dataset",
                    out / f"fig5_demo_queue{suffix}.png", zh=zh)
        figure_worked_kills(root / "demo_dataset",
                            out / f"fig6_worked_kills{suffix}.png", zh=zh)
        figure_example_windows(out / f"fig7_example_windows{suffix}.png", zh=zh)
    print("wrote", out)


if __name__ == "__main__":
    main()
