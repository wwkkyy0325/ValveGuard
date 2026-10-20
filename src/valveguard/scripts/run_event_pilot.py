"""Pilot: do raw recoil/fire/mouse features add signal over the 79 summaries?

Runs match-disjoint 5-fold CV on the raw subset only, with the same GBDT for
the base feature set and base + event features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from valveguard.models.baseline import feature_columns

K0 = 5.0
PRIOR = 0.3018746014732038


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int):
    X_train = train[columns].replace([np.inf, -np.inf], np.nan)
    X_test = test[columns].replace([np.inf, -np.inf], np.nan)
    y_train = train["label"].to_numpy(dtype=int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, int(len(y_train) - y_train.sum()))
    weights = np.where(
        y_train == 1,
        len(y_train) / (2.0 * positives),
        len(y_train) / (2.0 * negatives),
    )
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=0,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    return model.predict_proba(X_test)[:, 1]


def _kill_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return {
        "kill_auc": float(roc_auc_score(labels, scores)),
        "kill_ap": float(average_precision_score(labels, scores)),
        "kill_recall_at_fpr1": float((positive >= threshold).mean()),
        "kill_fpr": float((human >= threshold).mean()),
        "kills": int(len(labels)),
        "positive_kills": int(len(positive)),
    }


def _players(frame: pd.DataFrame, scores: np.ndarray, prior: float) -> pd.DataFrame:
    tmp = frame.assign(score=scores)
    grouped = tmp.groupby(["match_key", "player_name"], sort=False).agg(
        label=("label", "max"),
        n=("score", "size"),
        total=("score", "sum"),
    )
    grouped["shrunk"] = (grouped["total"] + K0 * prior) / (grouped["n"] + K0)
    return grouped.reset_index()


def _player_metrics(players: pd.DataFrame) -> dict:
    labels = players["label"].to_numpy(dtype=int)
    scores = players["shrunk"].to_numpy(dtype=float)
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return {
        "player_auc": float(roc_auc_score(labels, scores)),
        "player_recall_at_fpr1": float((positive >= threshold).mean()),
        "player_fpr": float((human >= threshold).mean()),
        "players": int(len(players)),
        "positive_players": int(len(positive)),
    }


def run(data: pd.DataFrame, columns: list[str], seed: int) -> dict:
    folds = GroupKFold(n_splits=5)
    oof = np.full(len(data), np.nan, dtype=float)
    player_parts = []
    for fold, (train_idx, test_idx) in enumerate(
        folds.split(data, groups=data["match_key"])
    ):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        scores = _fit_predict(train, test, columns, seed + fold)
        oof[test_idx] = scores
        player_parts.append(_players(test, scores, PRIOR))
    valid = np.isfinite(oof)
    result = _kill_metrics(data.loc[valid, "label"].to_numpy(dtype=int), oof[valid])
    pooled = pd.concat(player_parts, ignore_index=True)
    result.update(_player_metrics(pooled))
    return result, oof, pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = pd.read_parquet(args.index)[
        ["row", "label", "match_key", "player_name", "source"]
    ]
    base = pd.read_parquet(args.features)
    events = pd.read_parquet(args.events)
    events = events[["row"] + [c for c in events.columns if c.startswith("ev_")]]
    if base["row"].duplicated().any() or events["row"].duplicated().any():
        raise SystemExit("duplicate row keys")
    data = index.merge(base, on="row", how="inner").merge(events, on="row", how="inner")
    # Match the champion: drop label-0 windows from cheater-present matches.
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    if data.empty:
        raise SystemExit("no rows after joining")
    base_columns = [c for c in feature_columns(data) if not c.startswith("ev_")]
    event_columns = [c for c in data.columns if c.startswith("ev_")]
    base_result, base_oof, base_players = run(data, base_columns, args.seed)
    event_result, event_oof, event_players = run(
        data, base_columns + event_columns, args.seed
    )
    result = {
        "base": base_result,
        "base_plus_events": event_result,
        "rows": int(len(data)),
        "matches": int(data["match_key"].nunique()),
        "base_features": len(base_columns),
        "event_features": len(event_columns),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    oof = data[["row", "label", "match_key", "player_name"]].copy()
    oof["base"] = base_oof
    oof["base_plus_events"] = event_oof
    oof.to_parquet(out / "oof.parquet", index=False)
    base_players.to_csv(out / "players_base.csv", index=False)
    event_players.to_csv(out / "players_events.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
