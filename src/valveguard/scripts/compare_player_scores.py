"""Paired match-grouped comparison of two player-level score files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

K0 = 5.0
PRIOR = 0.3018746014732038
KEYS = ["match_key", "player_name"]


def load(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "row" in frame.columns:
        grouped = frame.groupby(KEYS, sort=False).agg(
            label=("label", "max"), n=("score", "size"), total=("score", "sum")
        )
        grouped["score"] = (grouped["total"] + K0 * PRIOR) / (grouped["n"] + K0)
        return grouped.reset_index()[KEYS + ["label", "score"]]
    return frame[KEYS + ["label", "score"]]


def metrics(frame: pd.DataFrame, q4: pd.DataFrame) -> dict:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    merged = frame.merge(q4, on=KEYS, how="inner")
    q4h = merged[(merged["label"] == 0)
                 & (merged["rank_q"].astype(str).str.startswith("Q4"))]
    q4p = merged[(merged["label"] == 1)
                 & (merged["rank_q"].astype(str).str.startswith("Q4"))]
    return {
        "player_auc": float(roc_auc_score(labels, scores)),
        "recall_fpr1": float((positive >= threshold).mean()),
        "fpr": float((human >= threshold).mean()),
        "q4_fpr": float((q4h["score"] >= threshold).mean()),
        "q4_recall": float((q4p["score"] >= threshold).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = load(args.a).rename(columns={"score": "a"})
    b = load(args.b).rename(columns={"score": "b"})
    frame = a.merge(b, on=KEYS, suffixes=("", "_b"))
    if not (frame["label"] == frame["label_b"]).all():
        raise SystemExit("label mismatch")
    frame = frame[KEYS + ["label", "a", "b"]]
    q4 = pd.read_parquet(args.ranks)[KEYS + ["rank_q"]]

    def point(column: str) -> dict:
        tmp = frame.rename(columns={column: "score"})
        return metrics(tmp, q4)

    matches = frame["match_key"].unique()
    groups = {m: frame[frame["match_key"] == m] for m in matches}
    rng = np.random.default_rng(args.seed)
    deltas = {key: [] for key in ("player_auc", "recall_fpr1", "q4_fpr")}
    for _ in range(args.bootstrap):
        chosen = rng.choice(matches, size=len(matches), replace=True)
        sample = pd.concat([groups[m] for m in chosen], ignore_index=True)
        ma = metrics(sample.rename(columns={"a": "score"}), q4)
        mb = metrics(sample.rename(columns={"b": "score"}), q4)
        for key in deltas:
            deltas[key].append(mb[key] - ma[key])

    report = {
        "point": {"a": point("a"), "b": point("b")},
        "bootstrap": {
            key: {"mean": float(np.mean(v)),
                  "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]}
            for key, v in deltas.items()
        },
        "players": int(len(frame)),
        "matches": int(len(matches)),
        "replicates": args.bootstrap,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
