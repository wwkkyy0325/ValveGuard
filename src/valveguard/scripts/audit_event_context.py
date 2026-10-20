"""Audit raw event features for label and map shortcuts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 10 or len(np.unique(labels[finite])) < 2:
        return np.nan
    return float(roc_auc_score(labels[finite], values[finite]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index = pd.read_parquet(args.index)[
        ["row", "label", "match_key", "player_name", "source"]
    ]
    events = pd.read_parquet(args.events)
    events = events[["row"] + [c for c in events.columns if c.startswith("ev_")]]
    frame = index.merge(events, on="row", how="inner")
    if args.features:
        base = pd.read_parquet(args.features)[["row", "map"]]
        frame = frame.merge(base, on="row", how="left")
    frame = frame[~((frame["source"] == "cdata") & (frame["label"] == 0))].copy()

    rows = []
    for column in [c for c in frame.columns if c.startswith("ev_")]:
        values = frame[column].to_numpy(dtype=float)
        labels = frame["label"].to_numpy(dtype=int)
        human = values[labels == 0]
        cheater = values[labels == 1]
        human = human[np.isfinite(human)]
        cheater = cheater[np.isfinite(cheater)]
        if len(human) < 10 or len(cheater) < 10:
            continue
        pooled = np.sqrt((np.var(human) + np.var(cheater)) / 2.0)
        rows.append({
            "feature": column,
            "human_mean": float(np.mean(human)),
            "cheater_mean": float(np.mean(cheater)),
            "smd": float((np.mean(cheater) - np.mean(human)) / (pooled + 1e-12)),
            "auc": _auc(labels, values),
            "human_std": float(np.std(human)),
            "cheater_std": float(np.std(cheater)),
        })
    table = pd.DataFrame(rows)
    table["abs_smd"] = table["smd"].abs()
    table = table.sort_values("abs_smd", ascending=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "event_shortcuts.csv", index=False)
    top = table.head(15)
    by_map = {}
    if "map" in frame.columns:
        for feature in top["feature"].head(5):
            by_map[feature] = (
                frame.groupby("map")[feature].mean()
                .sort_values(ascending=False).head(10).to_dict()
            )
    report = {
        "rows": int(len(frame)),
        "top": top.to_dict(orient="records"),
        "by_map": by_map,
    }
    (out / "event_shortcuts.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
