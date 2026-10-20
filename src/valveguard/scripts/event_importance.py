"""Permutation importance for the raw event-feature pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from valveguard.models.baseline import feature_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=3000)
    args = parser.parse_args()

    index = pd.read_parquet(args.index)[
        ["row", "label", "match_key", "player_name", "source"]
    ]
    base = pd.read_parquet(args.features)
    events = pd.read_parquet(args.events)
    events = events[["row"] + [c for c in events.columns if c.startswith("ev_")]]
    data = index.merge(base, on="row", how="inner").merge(events, on="row", how="inner")
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()

    columns = feature_columns(data)
    split = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(split.split(data, groups=data["match_key"]))
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

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
        random_state=args.seed,
        verbose=0,
    )
    model.fit(
        train[columns].replace([np.inf, -np.inf], np.nan),
        y_train,
        sample_weight=weights,
    )
    test_scores = model.predict_proba(
        test[columns].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    test_auc = float(roc_auc_score(test["label"].to_numpy(dtype=int), test_scores))

    rng = np.random.default_rng(args.seed)
    sample_idx = (
        rng.choice(len(test), size=args.max_rows, replace=False)
        if len(test) > args.max_rows else np.arange(len(test))
    )
    sample = test.iloc[sample_idx]
    importance = permutation_importance(
        model,
        sample[columns].replace([np.inf, -np.inf], np.nan),
        sample["label"].to_numpy(dtype=int),
        scoring="roc_auc",
        n_repeats=args.repeats,
        random_state=args.seed,
        n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "importance.csv", index=False)
    result = {
        "test_kill_auc": test_auc,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": len(columns),
        "top": table.head(30).to_dict(orient="records"),
    }
    (out / "importance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
