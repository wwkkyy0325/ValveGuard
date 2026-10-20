"""Evaluate kill-information weighting for player-level aggregation.

The champion aggregates kill scores with a plain mean and empirical-Bayes
shrinkage. This script asks whether down-weighting low-information kills
(close range, stationary or already-damaged victim, smoke/wallbang/flash)
improves player-level low-FPR recall or the high-rank false-positive rate.

Inputs are out-of-fold kill scores and the summary-feature table, so no model
is retrained; the experiment isolates the aggregation rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

K0 = 5.0

FEATURE_COLUMNS = [
    "row", "dist_final", "victim_speed_p95", "victim_health", "headshot",
    "through_smoke", "wallbang", "midair", "flashed_at_kill", "shot_count",
    "weapon_idx",
]


def _z(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    median = values.median()
    iqr = values.quantile(0.75) - values.quantile(0.25)
    return (values - median) / (iqr if iqr > 0 else 1.0)


def load_inputs(oof_path: Path, features_path: Path, ranks_path: Path | None):
    oof = pd.read_parquet(oof_path)
    features = pd.read_parquet(features_path)
    columns = [c for c in FEATURE_COLUMNS if c in features.columns]
    features = features[columns]
    df = oof.merge(features, on="row", how="left", validate="one_to_one")
    ranks = pd.read_parquet(ranks_path) if ranks_path is not None else None
    return df, ranks


def context_weights(df: pd.DataFrame, variant: str) -> np.ndarray:
    if variant == "uniform":
        return np.ones(len(df), dtype=float)

    distance = df["dist_final"].astype(float)
    distance = distance.fillna(distance.median())
    speed = df["victim_speed_p95"].astype(float).fillna(0.0)
    health = df["victim_health"].astype(float).fillna(100.0)
    smoke = df.get("through_smoke", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
    wall = df.get("wallbang", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
    flash = df.get("flashed_at_kill", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)

    if variant == "distance":
        weight = (distance / distance.median()).clip(0.5, 2.0)
    elif variant == "difficulty":
        difficulty = (
            0.5 * _z(distance)
            + 0.5 * _z(speed)
            + 0.5 * (health / 100.0)
            - 0.3 * smoke
            - 0.3 * wall
            - 0.3 * flash
        )
        weight = (1.0 + 0.5 * difficulty).clip(0.25, 3.0)
    elif variant == "difficulty_soft":
        difficulty = (
            0.5 * _z(distance)
            + 0.5 * _z(speed)
            + 0.5 * (health / 100.0)
            - 0.3 * smoke
            - 0.3 * wall
            - 0.3 * flash
        )
        weight = (1.0 + 0.25 * difficulty).clip(0.5, 2.0)
    elif variant == "info":
        weight = (
            (distance / distance.median()).clip(0.5, 2.0)
            * (1.0 + speed / 300.0).clip(0.5, 2.0)
            * (0.5 + health / 100.0)
            * (1.0 - 0.3 * smoke)
            * (1.0 - 0.3 * wall)
        ).clip(0.25, 3.0)
    elif variant == "no_smoke_wall":
        weight = ((1.0 - 0.5 * smoke) * (1.0 - 0.5 * wall)).clip(0.25, 1.0)
    else:
        raise ValueError(f"unknown variant: {variant}")
    return weight.to_numpy(dtype=float)


def aggregate(df: pd.DataFrame, weights: np.ndarray) -> tuple[pd.DataFrame, float]:
    frame = df.assign(_w=weights, _ws=df["score"].to_numpy() * weights)
    players = frame.groupby(["match_key", "player_name"], sort=False).agg(
        label=("label", "max"),
        n=("score", "size"),
        weight_sum=("_w", "sum"),
        weighted_sum=("_ws", "sum"),
        uniform=("score", "mean"),
    )
    prior = float(df.loc[df["label"] == 0, "score"].mean())
    players["weighted"] = (players["weighted_sum"] + K0 * prior) / (players["weight_sum"] + K0)
    return players.reset_index(), prior


def evaluate(players: pd.DataFrame, ranks: pd.DataFrame | None, score_col: str) -> dict:
    y = players["label"].to_numpy()
    score = players[score_col].to_numpy()
    auc = float(roc_auc_score(y, score))
    human = score[y == 0]
    positive = score[y == 1]
    threshold = float(np.quantile(human, 0.99))
    result = {
        "auc": auc,
        "recall_at_fpr1": float((positive >= threshold).mean()),
        "fpr": float((human >= threshold).mean()),
        "threshold": threshold,
        "players": int(len(players)),
        "human_players": int(len(human)),
        "positive_players": int(len(positive)),
        "q4_fpr": None,
        "q4_recall": None,
        "q4_human_players": None,
        "q4_positive_players": None,
    }
    if ranks is not None:
        merged = players.merge(
            ranks[["match_key", "player_name", "rank_q"]],
            on=["match_key", "player_name"],
            how="inner",
        )
        ranked_human = merged[merged["label"] == 0]
        ranked_positive = merged[merged["label"] == 1]
        ranked_threshold = float(np.quantile(ranked_human[score_col], 0.99))
        q4_human = ranked_human[ranked_human["rank_q"].astype(str).str.startswith("Q4")]
        q4_positive = ranked_positive[
            ranked_positive["rank_q"].astype(str).str.startswith("Q4")
        ]
        result["q4_fpr"] = float((q4_human[score_col] >= ranked_threshold).mean())
        result["q4_recall"] = (
            float((q4_positive[score_col] >= ranked_threshold).mean())
            if len(q4_positive) else None
        )
        result["q4_human_players"] = int(len(q4_human))
        result["q4_positive_players"] = int(len(q4_positive))
    return result


def _score_metrics(
    y: np.ndarray,
    score: np.ndarray,
    ranked_human: np.ndarray,
    q4_human: np.ndarray,
) -> tuple[float, float, float]:
    human = y == 0
    positive = y == 1
    threshold = float(np.quantile(score[human], 0.99))
    recall = float((score[positive] >= threshold).mean())
    q4_threshold = float(np.quantile(score[ranked_human], 0.99))
    q4_fpr = float((score[q4_human] >= q4_threshold).mean())
    return float(roc_auc_score(y, score)), recall, q4_fpr


def bootstrap_deltas(
    player_table: pd.DataFrame,
    variants: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    matches = player_table["match_key"].unique()
    groups = {
        match: player_table.index[player_table["match_key"] == match].to_numpy()
        for match in matches
    }
    rng = np.random.default_rng(seed)
    samples = {variant: [] for variant in variants}
    for _ in range(n_boot):
        chosen = rng.choice(matches, size=len(matches), replace=True)
        index = np.concatenate([groups[match] for match in chosen])
        frame = player_table.loc[index]
        y = frame["label"].to_numpy()
        rank_q = frame["rank_q"].fillna("").astype(str)
        ranked_human = (y == 0) & (rank_q != "").to_numpy()
        q4_human = (y == 0) & rank_q.str.startswith("Q4").to_numpy()
        base = None
        for variant in variants:
            score = frame[variant].to_numpy(dtype=float)
            metrics = _score_metrics(y, score, ranked_human, q4_human)
            if base is None:
                base = metrics
            samples[variant].append(tuple(m - b for m, b in zip(metrics, base)))
    report = {}
    for variant, values in samples.items():
        if variant == "uniform":
            continue
        array = np.asarray(values, dtype=float)
        report[variant] = {
            "delta_auc": [float(np.quantile(array[:, 0], 0.025)), float(np.quantile(array[:, 0], 0.975))],
            "delta_recall_at_fpr1": [float(np.quantile(array[:, 1], 0.025)), float(np.quantile(array[:, 1], 0.975))],
            "delta_q4_fpr": [float(np.quantile(array[:, 2], 0.025)), float(np.quantile(array[:, 2], 0.975))],
            "replicates": n_boot,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--ranks", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        default="uniform,distance,difficulty,difficulty_soft,info,no_smoke_wall",
    )
    args = parser.parse_args()

    df, ranks = load_inputs(
        Path(args.oof), Path(args.features),
        Path(args.ranks) if args.ranks else None,
    )
    if df["row"].duplicated().any():
        raise SystemExit("duplicate rows after merging OOF and features")
    mixed = df.groupby(["match_key", "player_name"])["label"].nunique()
    if int(mixed.max()) > 1:
        raise SystemExit("a player has mixed labels; aggregation assumption broken")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    player_table = None
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.bootstrap and "uniform" not in variant_names:
        variant_names.insert(0, "uniform")
    for variant in variant_names:
        weights = context_weights(df, variant)
        players, prior = aggregate(df, weights)
        metrics = evaluate(players, ranks, "weighted")
        metrics["mean_weight"] = float(weights.mean())
        metrics["median_weight"] = float(np.median(weights))
        metrics["prior"] = prior
        results[variant] = metrics
        column = players[["match_key", "player_name", "label", "n"]].copy()
        column[variant] = players["weighted"].to_numpy()
        if player_table is None:
            player_table = column
        else:
            player_table = player_table.merge(
                column[["match_key", "player_name", variant]],
                on=["match_key", "player_name"],
                how="outer",
            )

    if player_table is not None and ranks is not None:
        player_table = player_table.merge(
            ranks[["match_key", "player_name", "rank_q"]],
            on=["match_key", "player_name"],
            how="left",
        )
    if player_table is not None:
        player_table.to_csv(out / "player_scores.csv", index=False)
    if args.bootstrap and player_table is not None and "rank_q" in player_table.columns:
        results["_bootstrap"] = bootstrap_deltas(
            player_table, variant_names, args.bootstrap, args.seed
        )
    (out / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
