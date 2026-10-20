"""Score one demo with a packaged artifact and draw per-player reports."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

from valveguard.models.artifact import load_detector
from valveguard.scripts.score_demo_cnn import demo_channels

HUMAN = "#9aa0a6"
PLAYER = "#2f6fed"
FLAG = "#d93025"
WARN = "#f9ab00"
INK = "#202124"
MUTED = "#5f6368"


def _player_names(demo: Path) -> dict[str, str]:
    try:
        from demoparser2 import DemoParser
        info = DemoParser(str(demo)).parse_player_info()
        return {
            str(int(s)): str(n)
            for s, n in zip(info["steamid"], info["name"])
            if pd.notna(s)
        }
    except Exception:
        return {}


def _display_name(steamid, names: dict[str, str]) -> str:
    name = names.get(str(steamid))
    return name[:12] if name and name.strip() else str(steamid)[-6:]


def _tier(score: float, threshold: float, p95: float) -> str:
    if score >= threshold:
        return "flag"
    if score >= p95:
        return "review"
    return "below"


def _review_reasons(row) -> list[str]:
    reasons = []
    kills = int(row["kills"])
    if kills >= 3:
        if row["headshot_rate"] >= 0.80:
            reasons.append("HS>=80%")
        if row["shots_per_kill"] <= 1.5:
            reasons.append("<=1.5 shots/kill")
        if row["sniper_share"] >= 0.50 and row["headshot_rate"] >= 0.80:
            reasons.append("sniper HS")
        if row["wallbang_rate"] >= 0.20:
            reasons.append("wallbang>=20%")
        if row["smoke_rate"] >= 0.20:
            reasons.append("smoke>=20%")
    if row["max_multikill_4s"] >= 3:
        reasons.append("4s multikill>=3")
    if kills >= 4 and row["burst_fraction"] >= 0.30:
        reasons.append("burst>=30%")
    return reasons


def _behavior_context(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for player, group in frame.groupby("player_name", sort=False):
        group = group.sort_values("kill_tick")
        ticks = group["kill_tick"].to_numpy(dtype=float)
        gaps = np.diff(ticks)
        max_multi = 1
        for tick in ticks:
            max_multi = max(max_multi, int(((ticks >= tick) & (ticks <= tick + 256)).sum()))
        burst = gaps <= 256 if len(gaps) else np.array([], dtype=bool)
        kills = len(group)
        shots = float(group["shots"].sum())
        hs_rate = float(group["headshot"].mean()) if kills else 0.0
        burst_frac = float(burst.mean()) if len(burst) else 0.0
        gap = float(np.median(gaps) / 64.0) if len(gaps) else float("nan")
        context = (f"HS {hs_rate:.0%} · {((shots + 1) / max(1, kills)):.1f} shots/kill"
                   f" · multi {max_multi} · burst {burst_frac:.0%}")
        rows.append({
            "player_name": player,
            "headshot_rate": hs_rate,
            "headshot_kills": int(group["headshot"].sum()),
            "shots_per_kill": (shots + 1.0) / max(1, kills),
            "kill_per_shot": kills / (shots + kills),
            "max_multikill_4s": max_multi,
            "burst_fraction": burst_frac,
            "median_kill_gap_s": gap,
            "sniper_share": float((group["weapon_class"] == 2).mean()),
            "wallbang_rate": float(group["wallbang"].mean()),
            "smoke_rate": float(group["through_smoke"].mean()),
            "behavior_context": context,
        })
    return pd.DataFrame(rows)


def _summary_plot(players: pd.DataFrame, reference: dict, threshold: float,
                  p95: float, names: dict[str, str], out: Path) -> None:
    n = len(players)
    fig = plt.figure(figsize=(12, 2.2 + 0.42 * n))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, max(2.0, 0.42 * n)], hspace=0.08)

    top = fig.add_subplot(grid[0])
    human = reference["human_player_scores"]
    top.hist(human, bins=60, density=True, color=HUMAN, alpha=0.35,
             edgecolor="white", linewidth=0.4)
    for value, label in ((p95, "human p95"), (threshold, "human p99")):
        top.axvline(value, color=MUTED, linestyle=":", linewidth=1)
        top.text(value, top.get_ylim()[1] * 0.92, label, rotation=90,
                 va="top", ha="right", fontsize=7, color=MUTED)
    top.axvspan(threshold, max(1.0, float(np.max(human)) + 0.05),
                color=FLAG, alpha=0.06)
    top.set_xlim(0, 0.85)
    top.set_yticks([])
    top.set_ylabel("human density", fontsize=8, color=MUTED)
    top.set_title("Where this demo's players sit in the CS2CD human distribution",
                  loc="left", fontsize=11, color=INK, pad=8)
    top.spines[["top", "right", "left"]].set_visible(False)
    top.tick_params(axis="x", labelbottom=False)

    rows = fig.add_subplot(grid[1], sharex=top)
    rows.axvspan(threshold, 0.85, color=FLAG, alpha=0.06)
    rows.axvline(threshold, color=FLAG, linestyle="--", linewidth=1.2)
    rows.text(threshold + 0.005, -0.95, "flag threshold = human p99",
              fontsize=7.5, color=FLAG, va="bottom")
    order = players.sort_values("review_rank").reset_index(drop=True)
    for i, row in order.iterrows():
        if row["flagged"]:
            color = FLAG
        elif row["shrunk_score"] >= p95:
            color = WARN
        else:
            color = PLAYER
        rows.plot([row["shrunk_score"]], [i], "o", color=color, markersize=7,
                  markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        tier = "FLAG" if row["flagged"] else ("REVIEW" if row["shrunk_score"] >= p95 else "")
        if tier and int(row.get("review_reason_count", 0)):
            tier = f"{tier} · {int(row['review_reason_count'])} reasons"
        label = (f"{int(row['kills'])} kills  |  {row['shrunk_score']:.3f}"
                 f"  |  p{row['percentile']:.0f}" + (f"  |  {tier}" if tier else ""))
        if "behavior_context" in row and isinstance(row["behavior_context"], str):
            label += "\n" + str(row["behavior_context"])
        rows.text(0.86, i, label, va="center", fontsize=6.8, linespacing=1.2,
                  clip_on=False, color=color if tier else MUTED)
    rows.set_yticks(range(n))
    rows.set_yticklabels([_display_name(s, names) for s in order["steamid"]], fontsize=7.5)
    rows.set_ylim(-1.0, n - 0.3)
    rows.invert_yaxis()
    rows.set_xlim(0, 0.85)
    rows.text(0.86, -0.95, "kills | score | human percentile", fontsize=7,
              color=MUTED, clip_on=False, va="bottom")
    by_score = players.sort_values("shrunk_score", ascending=False).reset_index(drop=True)
    leader = by_score.iloc[0]
    second = float(by_score.iloc[1]["shrunk_score"]) if len(by_score) > 1 else 0.0
    gap = float(leader["shrunk_score"]) - second
    rows.text(0.0, -0.95,
              f"top: {_display_name(leader['steamid'], names)}  "
              f"+{gap:.3f} over second  ({leader['shrunk_score']:.3f} vs {second:.3f})",
              fontsize=7.5, color=WARN if gap > 0.05 else MUTED, va="bottom")
    rows.set_xlabel("player risk score (shrunk mean of kill-window scores)",
                    fontsize=8, labelpad=12)
    rows.spines[["top", "right", "left"]].set_visible(False)
    rows.grid(axis="x", color="#e8eaed", linewidth=0.8)
    rows.set_axisbelow(True)

    note = (
        "Reading: gray is the distribution of CS2CD human players; the dashed line "
        "is the 99th percentile of that distribution. Each row is one player in this "
        "demo. The score shrinks the mean kill-window risk toward the human prior "
        "(k0=5), so a few kills cannot dominate. Blue = below the human p95, "
        "amber = p95-p99 review band, red = at or above the p99 flag line. This is "
        "a risk score, not a probability; a 10-player demo is expected to show 0-1 flags."
    )
    fig.text(0.01, 0.008, textwrap.fill(note, 120), fontsize=7, color=MUTED,
             ha="left", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.62, top=0.92, bottom=0.20)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _player_panels(players: pd.DataFrame, frame: pd.DataFrame,
                   reference: dict, names: dict[str, str], threshold: float,
                   p95: float, out: Path) -> None:
    n = len(players)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), squeeze=False)
    human = reference["human_window_scores"]
    for ax in axes.flat:
        ax.set_visible(False)
    for i, player in enumerate(players.itertuples()):
        ax = axes[i // cols][i % cols]
        ax.set_visible(True)
        ax.hist(human, bins=50, density=True, color=HUMAN, alpha=0.3,
                edgecolor="white", linewidth=0.3)
        scores = frame.loc[frame["player_name"] == player.steamid, "score"].to_numpy()
        jitter = np.zeros(len(scores)) if len(scores) == 1 else np.random.default_rng(i).normal(0, 0.02, len(scores))
        ax.plot(scores, jitter, "o", color=PLAYER, markersize=4.5, alpha=0.8)
        ax.axvline(player.shrunk_score, color=PLAYER, linewidth=1.4)
        ax.axvline(player.max_kill_score, color=WARN, linewidth=1, linestyle=":")
        ax.set_xlim(0, 1.0)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
        tier = _tier(float(player.shrunk_score), threshold, p95)
        label = {"flag": "FLAG", "review": "REVIEW", "below": "below threshold"}[tier]
        color = FLAG if tier == "flag" else (WARN if tier == "review" else INK)
        ax.set_title(
            f"{_display_name(player.steamid, names)}  |  {int(player.kills)} kills  |  "
            f"shrunk {player.shrunk_score:.3f} (p{player.percentile:.0f})  |  {label}",
            loc="left", fontsize=8, color=color,
        )
    note = (
        "Each panel: gray is the CS2CD human kill-window score distribution; blue dots "
        "are this player's kill windows; the blue line is the shrunk player score; the "
        "dotted gold line is the highest single window. One high window can be noise; "
        "a cluster of high windows is more concerning."
    )
    fig.text(0.01, 0.01, textwrap.fill(note, 150), fontsize=7, color=MUTED)
    fig.suptitle("Per-player kill-window scores", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _behavior_table(players: pd.DataFrame, names: dict[str, str],
                    p95: float, out: Path) -> None:
    order = players.sort_values("review_rank").reset_index(drop=True)
    headers = ["#", "Player", "Score", "Tier", "Kills", "HS%", "Shots/K",
               "4s max", "Burst%", "Gap s", "Sniper%", "Reasons"]
    reason_codes = {
        "HS>=80%": "HS",
        "<=1.5 shots/kill": "eff",
        "sniper HS": "sniper",
        "wallbang>=20%": "wall",
        "smoke>=20%": "smoke",
        "4s multikill>=3": "multi4",
        "burst>=30%": "burst",
    }
    table_rows = []
    for _, row in order.iterrows():
        tier = "FLAG" if row["flagged"] else (
            "REVIEW" if row["shrunk_score"] >= p95 else ""
        )
        gap = row["median_kill_gap_s"]
        table_rows.append([
            str(int(row["review_rank"])),
            _display_name(row["steamid"], names),
            f"{row['shrunk_score']:.3f}",
            tier,
            str(int(row["kills"])),
            f"{row['headshot_rate']:.0%}",
            f"{row['shots_per_kill']:.1f}",
            str(int(row["max_multikill_4s"])),
            f"{row['burst_fraction']:.0%}",
            "—" if not np.isfinite(gap) else f"{gap:.1f}",
            f"{row['sniper_share']:.0%}",
            ", ".join(reason_codes.get(value, value)
                      for value in row["review_reasons"]) if row["review_reasons"] else "—",
        ])
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(order) + 1.6))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    widths = [0.03, 0.14, 0.06, 0.08, 0.05, 0.05, 0.06, 0.06,
              0.06, 0.05, 0.07, 0.23]
    for (row, col), cell in table.get_celld().items():
        cell.set_width(widths[col])
        cell.set_edgecolor("#e8eaed")
        if row == 0:
            cell.set_facecolor("#f1f3f4")
            cell.set_text_props(weight="bold", color=INK)
        elif col == 3:
            label = table_rows[row - 1][3]
            cell.set_text_props(color=FLAG if label.startswith("FLAG")
                                else (WARN if label.startswith("REVIEW") else MUTED))
    ax.set_title("Behavior context (review only; not part of the score)",
                 loc="left", fontsize=11, color=INK, pad=12)
    note = (
        "HS% = headshot share of this player's kill windows. Shots/K = shots "
        "fired per kill (lower is more efficient). 4s max = maximum kills "
        "inside any four-second window. Burst% = share of kills within four "
        "seconds of the previous kill. Gap s = median seconds between kills. "
        "Sniper% = share of kills with a sniper rifle. Reason codes: HS=headshot "
        "rate, eff=shots per kill, multi4=4-second multikill, burst=burst share, "
        "wall=wallbang, smoke=smoke. They only order the review queue and never "
        "change the risk score or the flag; full text is in review_queue.csv."
    )
    fig.text(0.01, 0.01, textwrap.fill(note, 150), fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _add_review_order(players: pd.DataFrame, threshold: float,
                      p95: float) -> pd.DataFrame:
    players = players.copy()
    players["tier"] = [
        _tier(float(value), threshold, p95) for value in players["shrunk_score"]
    ]
    players["review_reasons"] = players.apply(_review_reasons, axis=1)
    players["review_reason_count"] = players["review_reasons"].apply(len)
    tier_rank = {"flag": 0, "review": 1, "below": 2}
    players["_tier_rank"] = players["tier"].map(tier_rank)
    players["_reason_sort"] = players["review_reason_count"].where(
        players["tier"] != "below", 0
    )
    players = players.sort_values(
        ["_tier_rank", "_reason_sort", "shrunk_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    players["review_rank"] = np.arange(1, len(players) + 1)
    return players.drop(columns=["_tier_rank", "_reason_sort"])


def _combine_review_queues(queues: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(queues, ignore_index=True)
    tier_rank = {"flag": 0, "review": 1, "below": 2}
    combined["_tier_rank"] = combined["tier"].map(tier_rank)
    combined["_reason_sort"] = combined["review_reason_count"].where(
        combined["tier"] != "below", 0
    )
    combined = combined.sort_values(
        ["_tier_rank", "_reason_sort", "shrunk_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    combined.insert(0, "queue_rank", np.arange(1, len(combined) + 1))
    return combined.drop(columns=["_tier_rank", "_reason_sort"])


def run_one(demo: Path, detector, out: Path, threshold: float | None) -> dict:
    frame, channels, names = demo_channels(demo)
    if len(frame) == 0:
        raise SystemExit(f"no valid kill windows in {demo}")
    scores = detector.score_channels(channels, names)
    frame["score"] = scores
    players = detector.aggregate(frame, scores)
    players = players.merge(_behavior_context(frame), on="player_name", how="left")
    players = players.rename(columns={"player_name": "steamid"})
    names = _player_names(demo)
    players["name"] = players["steamid"].map(
        lambda value: _display_name(value, names)
    )
    threshold = threshold if threshold is not None else detector.threshold
    p95 = float(np.quantile(detector.human_player_scores, 0.95))
    players["flagged"] = players["shrunk_score"] >= threshold
    players["percentile"] = [
        float((detector.human_player_scores < value).mean() * 100.0)
        for value in players["shrunk_score"]
    ]
    players = _add_review_order(players, threshold, p95)

    out.mkdir(parents=True, exist_ok=True)
    reference = {"human_player_scores": detector.human_player_scores,
                 "human_window_scores": detector.human_window_scores}
    result = {
        "demo": str(demo),
        "map": str(frame["map"].iloc[0]),
        "artifact": detector.manifest["version"],
        "threshold": threshold,
        "human_reference": {
            "players": int(len(detector.human_player_scores)),
            "p95": float(np.quantile(detector.human_player_scores, 0.95)),
            "p99": float(np.quantile(detector.human_player_scores, 0.99)),
        },
        "players": [
            {
                "steamid": row.steamid,
                "name": row.name,
                "tier": row.tier,
                "review_rank": int(row.review_rank),
                "review_reasons": list(row.review_reasons),
                "review_reason_count": int(row.review_reason_count),
                "kills": int(row.kills),
                "headshot_rate": round(float(row.headshot_rate), 4),
                "headshot_kills": int(row.headshot_kills),
                "shots_per_kill": round(float(row.shots_per_kill), 3),
                "kill_per_shot": round(float(row.kill_per_shot), 3),
                "max_multikill_4s": int(row.max_multikill_4s),
                "burst_fraction": round(float(row.burst_fraction), 4),
                "median_kill_gap_s": None if not np.isfinite(row.median_kill_gap_s)
                else round(float(row.median_kill_gap_s), 3),
                "sniper_share": round(float(row.sniper_share), 4),
                "wallbang_rate": round(float(row.wallbang_rate), 4),
                "behavior_context": row.behavior_context,
                "mean_score": round(float(row.mean), 6),
                "max_kill_score": round(float(row.max_kill_score), 6),
                "shrunk_score": round(float(row.shrunk_score), 6),
                "percentile": round(float(row.percentile), 1),
                "flagged": bool(row.flagged),
                "reliable": bool(row.reliable),
            }
            for row in players.itertuples()
        ],
        "kills": [
            {"steamid": str(row.player_name), "kill_tick": int(row.kill_tick),
             "score": round(float(row.score), 6)}
            for row in frame.sort_values(["player_name", "kill_tick"]).itertuples()
        ],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    queue = players[[
        "review_rank", "steamid", "name", "tier", "shrunk_score",
        "percentile", "kills", "review_reason_count", "review_reasons",
    ]].copy()
    queue["review_reasons"] = queue["review_reasons"].apply(
        lambda values: "; ".join(values)
    )
    queue.to_csv(out / "review_queue.csv", index=False)
    _summary_plot(players, reference, threshold, p95, names, out / "players.png")
    _player_panels(players, frame, reference, names, threshold, p95,
                   out / "kill_windows.png")
    _behavior_table(players, names, p95, out / "behavior.png")
    return {
        "demo": result["demo"], "map": result["map"],
        "players": len(result["players"]),
        "flagged": sum(p["flagged"] for p in result["players"]),
        "out": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True,
                    help="a .dem file or a directory scanned recursively")
    ap.add_argument("--artifact", default="artifacts/valveguard_cnn_v1")
    ap.add_argument("--out", default="reports/demo_reports")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    detector = load_detector(args.artifact)
    demo = Path(args.demo)
    if demo.is_dir():
        demos = sorted(demo.rglob("*.dem"))
        if not demos:
            raise SystemExit(f"no .dem files under {demo}")
        root = Path(args.out)
        summaries = []
        queues = []
        for path in demos:
            summary = run_one(path, detector, root / path.stem, args.threshold)
            summaries.append(summary)
            queue_path = root / path.stem / "review_queue.csv"
            if queue_path.exists():
                queue = pd.read_csv(queue_path)
                queue.insert(0, "demo", path.stem)
                queues.append(queue)
        if queues:
            _combine_review_queues(queues).to_csv(
                root / "review_queue.csv", index=False
            )
        (root / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summaries, indent=2))
    else:
        print(json.dumps(
            run_one(demo, detector, Path(args.out), args.threshold), indent=2
        ))


if __name__ == "__main__":
    main()
