"""Player-level score aggregation shared by the PyTorch and ONNX loaders."""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_players(frame: pd.DataFrame, scores: np.ndarray,
                      aggregation: dict, human_player_scores: np.ndarray,
                      threshold: float) -> pd.DataFrame:
    """Aggregate kill scores into shrunk player scores and flags."""
    k0 = float(aggregation["k0"])
    prior = float(aggregation["prior"])
    out = frame.assign(score=scores)
    grouped = out.groupby("player_name", sort=False).agg(
        kills=("score", "size"),
        total=("score", "sum"),
        mean=("score", "mean"),
        max_kill_score=("score", "max"),
    ).reset_index()
    grouped["shrunk_score"] = (
        grouped["total"] + k0 * prior
    ) / (grouped["kills"] + k0)
    grouped["percentile"] = [
        float((human_player_scores < value).mean() * 100.0)
        for value in grouped["shrunk_score"]
    ]
    grouped["flagged"] = grouped["shrunk_score"] >= threshold
    grouped["reliable"] = grouped["kills"] >= 3
    return grouped.sort_values("shrunk_score", ascending=False).reset_index(drop=True)
