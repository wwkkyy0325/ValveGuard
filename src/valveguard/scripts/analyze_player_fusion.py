"""Post-hoc fusion of player-level views over the same window scores."""
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


def player_table(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame[KEYS + ["label", "score"]].copy()
    ranked["rank"] = ranked.groupby(KEYS)["score"].rank(ascending=False, method="first")
    top5 = ranked[ranked["rank"] <= 5].groupby(KEYS, sort=False).agg(
        top5_sum=("score", "sum"), top5_n=("rank", "size")
    )
    grouped = frame.groupby(KEYS, sort=False).agg(
        label=("label", "max"),
        n=("score", "size"),
        total=("score", "sum"),
        top1=("score", "max"),
    )
    grouped = grouped.join(top5)
    grouped["mean"] = (grouped["total"] + K0 * PRIOR) / (grouped["n"] + K0)
    grouped["top5"] = (grouped["top5_sum"] + K0 * PRIOR) / (grouped["top5_n"] + K0)
    grouped["gap"] = grouped["top1"] - grouped["mean"]
    grouped["match_pct"] = grouped.groupby(level="match_key")["mean"].rank(pct=True)
    return grouped.reset_index()


def add_views(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    for col in ("mean", "top5", "match_pct", "gap"):
        p[f"r_{col}"] = p[col].rank(pct=True)
    p["fusion_raw"] = 0.5 * p["mean"] + 0.5 * p["top5"]
    p["fusion_rank"] = 0.5 * p["r_mean"] + 0.5 * p["r_top5"]
    p["mean_match"] = 0.5 * p["r_mean"] + 0.5 * p["r_match_pct"]
    p["fusion3"] = (p["r_mean"] + p["r_top5"] + p["r_match_pct"]) / 3.0
    p["disp"] = 0.5 * p["r_mean"] + 0.5 * p["r_gap"]
    return p


def metrics(p: pd.DataFrame, column: str, q4: pd.DataFrame) -> dict:
    labels = p["label"].to_numpy(dtype=int)
    scores = p[column].to_numpy(dtype=float)
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    merged = p.merge(q4, on=KEYS, how="inner")
    mh = merged[merged["label"] == 0]
    mp = merged[merged["label"] == 1]
    q4h = mh[mh["rank_q"].astype(str).str.startswith("Q4")]
    q4p = mp[mp["rank_q"].astype(str).str.startswith("Q4")]
    return {
        "player_auc": float(roc_auc_score(labels, scores)),
        "recall_fpr1": float((positive >= threshold).mean()),
        "fpr": float((human >= threshold).mean()),
        "q4_fpr": float((q4h[column] >= threshold).mean()),
        "q4_recall": float((q4p[column] >= threshold).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", required=True)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frame = pd.read_parquet(args.cnn)[["row", "match_key", "player_name", "label", "score"]]
    q4 = pd.read_parquet(args.ranks)[KEYS + ["rank_q"]]
    players = add_views(player_table(frame))
    variants = ["mean", "top5", "fusion_raw", "fusion_rank", "mean_match", "fusion3", "disp"]
    point = {name: metrics(players, name, q4) for name in variants}

    matches = players["match_key"].unique()
    groups = {m: players[players["match_key"] == m] for m in matches}
    rng = np.random.default_rng(args.seed)
    deltas = {name: {key: [] for key in ("player_auc", "recall_fpr1", "q4_fpr")}
              for name in variants if name != "mean"}
    for _ in range(args.bootstrap):
        chosen = rng.choice(matches, size=len(matches), replace=True)
        sample = pd.concat([groups[m] for m in chosen], ignore_index=True)
        base = metrics(sample, "mean", q4)
        for name in deltas:
            value = metrics(sample, name, q4)
            for key in deltas[name]:
                deltas[name][key].append(value[key] - base[key])

    report = {"point": point, "bootstrap": {}, "rows": int(len(frame)),
              "players": int(len(players)), "replicates": args.bootstrap}
    for name, values in deltas.items():
        report["bootstrap"][name] = {
            key: {"mean": float(np.mean(v)),
                  "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]}
            for key, v in values.items()
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
