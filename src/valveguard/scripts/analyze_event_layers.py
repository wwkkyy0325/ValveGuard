"""Player-rank and weapon layers for the full event-feature pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

K0 = 5.0
PRIOR = 0.3018746014732038
WEAPONS = ["knife", "auto_rifle", "sniper", "pistol", "grenade", "smg", "shotgun"]


def _players(oof: pd.DataFrame, score_col: str) -> pd.DataFrame:
    grouped = oof.groupby(["match_key", "player_name"], sort=False).agg(
        label=("label", "max"),
        n=(score_col, "size"),
        total=(score_col, "sum"),
    )
    grouped["score"] = (grouped["total"] + K0 * PRIOR) / (grouped["n"] + K0)
    return grouped.reset_index()


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "recall_at_fpr1": float((positive >= threshold).mean()),
        "fpr": float((human >= threshold).mean()),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-column", default="base_plus_events")
    args = parser.parse_args()

    oof = pd.read_parquet(args.oof)
    ranks = pd.read_parquet(args.ranks)[["match_key", "player_name", "rank_q"]]
    features = pd.read_parquet(args.features)[["row", "weapon_idx"]]
    frame = oof.merge(features, on="row", how="left")

    players = _players(frame, args.score_column)
    overall = _metrics(
        players["label"].to_numpy(dtype=int),
        players["score"].to_numpy(dtype=float),
    )
    ranked = players.merge(ranks, on=["match_key", "player_name"], how="inner")
    human = ranked[ranked["label"] == 0]
    positive = ranked[ranked["label"] == 1]
    thresholds = {}
    rank_report = {}
    threshold = float(np.quantile(human["score"], 0.99))
    for label in ["Q1 low", "Q2", "Q3", "Q4 high"]:
        h = human[human["rank_q"] == label]
        p = positive[positive["rank_q"] == label]
        rank_report[label] = {
            "human_players": int(len(h)),
            "positive_players": int(len(p)),
            "fpr": float((h["score"] >= threshold).mean()) if len(h) else None,
            "recall": float((p["score"] >= threshold).mean()) if len(p) else None,
        }
        thresholds[label] = threshold

    kill_labels = frame["label"].to_numpy(dtype=int)
    kill_scores = frame[args.score_column].to_numpy(dtype=float)
    kill_threshold = float(np.quantile(kill_scores[kill_labels == 0], 0.99))
    weapon_report = {}
    for idx, name in enumerate(WEAPONS):
        mask = frame["weapon_idx"].to_numpy(dtype=float) == idx
        if mask.sum() == 0:
            continue
        labels = kill_labels[mask]
        scores = kill_scores[mask]
        human = scores[labels == 0]
        positive = scores[labels == 1]
        weapon_report[name] = {
            "kills": int(mask.sum()),
            "human_kills": int(len(human)),
            "positive_kills": int(len(positive)),
            "recall_at_global_fpr1": float((positive >= kill_threshold).mean()) if len(positive) else None,
            "fpr_at_global_fpr1": float((human >= kill_threshold).mean()) if len(human) else None,
        }

    result = {
        "score_column": args.score_column,
        "overall_player": overall,
        "rank_strata": rank_report,
        "rank_threshold": threshold,
        "kill_threshold": kill_threshold,
        "weapon": weapon_report,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "layers.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
