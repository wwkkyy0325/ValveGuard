"""Review-only behavior context, reason rules, and queue ordering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tier(score: float, threshold: float, p95: float) -> str:
    if score >= threshold:
        return "flag"
    if score >= p95:
        return "review"
    return "below"


def review_reasons(row) -> list[str]:
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


def behavior_context(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for player, group in frame.groupby("player_name", sort=False):
        group = group.sort_values("kill_tick")
        ticks = group["kill_tick"].to_numpy(dtype=float)
        gaps = np.diff(ticks)
        max_multi = 1
        for tick in ticks:
            max_multi = max(
                max_multi, int(((ticks >= tick) & (ticks <= tick + 256)).sum())
            )
        burst = gaps <= 256 if len(gaps) else np.array([], dtype=bool)
        kills = len(group)
        shots = float(group["shots"].sum())
        hs_rate = float(group["headshot"].mean()) if kills else 0.0
        burst_frac = float(burst.mean()) if len(burst) else 0.0
        gap = float(np.median(gaps) / 64.0) if len(gaps) else float("nan")
        context = (
            f"HS {hs_rate:.0%} · {((shots + 1) / max(1, kills)):.1f} shots/kill"
            f" · multi {max_multi} · burst {burst_frac:.0%}"
        )
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


def add_review_order(players: pd.DataFrame, threshold: float,
                     p95: float) -> pd.DataFrame:
    players = players.copy()
    players["tier"] = [
        tier(float(value), threshold, p95) for value in players["shrunk_score"]
    ]
    players["review_reasons"] = players.apply(review_reasons, axis=1)
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


def combine_review_queues(queues: list[pd.DataFrame]) -> pd.DataFrame:
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
