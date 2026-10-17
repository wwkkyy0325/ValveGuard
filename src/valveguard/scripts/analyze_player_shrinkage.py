"""Empirical-Bayes shrinkage of player scores by kill count.

The player score is the mean of kill scores, which is unstable for players
with one or two kills.  The Q4 false positives are dominated by exactly those
players.  Shrink each player's mean toward the training human mean with a
pseudo-count k0:

    score = (sum(kill scores) + k0 * prior) / (n + k0)

The prior is the mean score of human kills in the training folds, so no test
label is used.  A minimum-kill filter is reported as the simpler alternative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.metrics import binary_metrics, recall_at_fpr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_parquet(args.pred)
    skill = pd.read_parquet(args.skill)
    keys = ["match_key", "player_name"]
    agg = pred.groupby(keys + ["fold"]).agg(
        n=("score", "size"), total=("score", "sum"), label=("label", "max")
    ).reset_index()

    # Human prior from the other folds only (per-fold training data).
    priors = {
        fold: float(
            pred[(pred["label"] == 0) & (pred["fold"] != fold)]["score"].mean()
        )
        for fold in pred["fold"].unique()
    }

    q4 = skill[["match_key", "player_name", "rank_q"]]
    rows = []
    for k0 in (0, 1, 2, 3, 5, 8, 12, 20):
        s = agg.copy()
        s["score"] = (s["total"] + k0 * s["fold"].map(priors)) / (s["n"] + k0)
        s = s.merge(q4, on=keys, how="left")
        m = binary_metrics(s["label"], s["score"])

        human = s[s["label"] == 0]["score"]
        thr = float(human.quantile(0.99))
        res = {
            "k0": k0,
            "player_auc": m["auc"],
            "player_ap": m["ap"],
            "player_recall_fpr1": float(
                recall_at_fpr(s["label"].to_numpy(), s["score"].to_numpy())
            ),
            "fpr_at_tpr95": m["fpr_at_tpr95"],
        }
        for q in ("Q1 low", "Q2", "Q3", "Q4 high"):
            d = s[(s["rank_q"] == q) & (s["label"] == 0)]
            res[f"{q}_fpr"] = float((d["score"] >= thr).mean()) if len(d) else np.nan
            d2 = s[s["rank_q"] == q]
            if len(d2) and d2["label"].nunique() > 1:
                res[f"{q}_auc"] = float(
                    binary_metrics(d2["label"], d2["score"])["auc"]
                )
        rows.append(res)

    table = pd.DataFrame(rows)
    table.to_csv(out / "shrinkage.csv", index=False)
    print(table.to_string(index=False))

    # Minimum-kill filter reference, using the unshrunk mean.
    filt = []
    for min_k in (1, 3, 5, 8):
        s = agg[agg["n"] >= min_k].copy()
        s["score"] = s["total"] / s["n"]
        s = s.merge(q4, on=keys, how="left")
        m = binary_metrics(s["label"], s["score"])
        human = s[s["label"] == 0]["score"]
        thr = float(human.quantile(0.99))
        row = {
            "min_kills": min_k,
            "players": int(len(s)),
            "q4_clean": int(((s["rank_q"] == "Q4 high") & (s["label"] == 0)).sum()),
            "player_auc": m["auc"],
            "player_recall_fpr1": float(
                recall_at_fpr(s["label"].to_numpy(), s["score"].to_numpy())
            ),
        }
        d = s[(s["rank_q"] == "Q4 high") & (s["label"] == 0)]
        row["q4_fpr"] = float((d["score"] >= thr).mean()) if len(d) else np.nan
        filt.append(row)
    ftable = pd.DataFrame(filt)
    ftable.to_csv(out / "min_kills.csv", index=False)
    print()
    print(ftable.to_string(index=False))

    with open(out / "shrinkage.json", "w", encoding="utf-8") as f:
        json.dump({"shrinkage": rows, "min_kills": filt}, f, indent=2)


if __name__ == "__main__":
    main()
