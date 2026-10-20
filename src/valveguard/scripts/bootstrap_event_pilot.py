"""Match-grouped bootstrap for the raw event-feature pilot."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

K0 = 5.0
PRIOR = 0.3018746014732038


def _player_table(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    grouped = frame.groupby(["match_key", "player_name"], sort=False).agg(
        label=("label", "max"),
        n=(score_col, "size"),
        total=(score_col, "sum"),
    )
    grouped["score"] = (grouped["total"] + K0 * PRIOR) / (grouped["n"] + K0)
    return grouped.reset_index()


def _metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return (
        float(roc_auc_score(labels, scores)),
        float((positive >= threshold).mean()),
        float((human >= threshold).mean()),
    )


def _evaluate(frame: pd.DataFrame) -> dict:
    result = {}
    for variant in ("base", "base_plus_events"):
        kill_auc, _, _ = _metrics(
            frame["label"].to_numpy(dtype=int), frame[variant].to_numpy(dtype=float)
        )
        players = _player_table(frame, variant)
        player_auc, recall, fpr = _metrics(
            players["label"].to_numpy(dtype=int),
            players["score"].to_numpy(dtype=float),
        )
        result[variant] = {
            "kill_auc": kill_auc,
            "player_auc": player_auc,
            "player_recall_at_fpr1": recall,
            "player_fpr": fpr,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet(args.oof)
    point = _evaluate(frame)
    matches = frame["match_key"].unique()
    groups = {
        match: frame.index[frame["match_key"] == match].to_numpy()
        for match in matches
    }
    rng = np.random.default_rng(args.seed)
    deltas = {
        key: []
        for key in ("kill_auc", "player_auc", "player_recall_at_fpr1", "player_fpr")
    }
    for _ in range(args.bootstrap):
        chosen = rng.choice(matches, size=len(matches), replace=True)
        index = np.concatenate([groups[match] for match in chosen])
        sample = _evaluate(frame.loc[index])
        for key in deltas:
            deltas[key].append(
                sample["base_plus_events"][key] - sample["base"][key]
            )

    report = {"point": point, "bootstrap": {}, "replicates": args.bootstrap}
    for key, values in deltas.items():
        array = np.asarray(values, dtype=float)
        report["bootstrap"][key] = {
            "mean": float(np.mean(array)),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }
    print(json.dumps(report, indent=2))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
