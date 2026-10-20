"""Paired comparison of the GPU CNN and the GBDT champion on the same rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

K0 = 5.0
PRIOR = 0.3018746014732038
WEAPONS = ["knife", "auto_rifle", "sniper", "pistol", "grenade", "smg", "shotgun"]


def _players(frame: pd.DataFrame, column: str, topk: int = 0) -> pd.DataFrame:
    keys = ["match_key", "player_name"]
    if topk > 0:
        ranked = frame[keys + ["label", column]].copy()
        ranked["rank"] = ranked.groupby(keys)[column].rank(
            ascending=False, method="first"
        )
        top = ranked[ranked["rank"] <= topk]
        grouped = top.groupby(keys, sort=False).agg(
            label=("label", "max"),
            n=("rank", "size"),
            total=(column, "sum"),
        )
    else:
        grouped = frame.groupby(keys, sort=False).agg(
            label=("label", "max"),
            n=(column, "size"),
            total=(column, "sum"),
        )
    grouped["score"] = (grouped["total"] + K0 * PRIOR) / (grouped["n"] + K0)
    return grouped.reset_index()


def _kill_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return {
        "kill_auc": float(roc_auc_score(labels, scores)),
        "kill_ap": float(average_precision_score(labels, scores)),
        "kill_recall_at_fpr1": float((positive >= threshold).mean()),
        "kill_fpr": float((human >= threshold).mean()),
    }


def _player_metrics(players: pd.DataFrame) -> dict:
    labels = players["label"].to_numpy(dtype=int)
    scores = players["score"].to_numpy(dtype=float)
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return {
        "player_auc": float(roc_auc_score(labels, scores)),
        "player_recall_at_fpr1": float((positive >= threshold).mean()),
        "player_fpr": float((human >= threshold).mean()),
    }


def _q4(players: pd.DataFrame, ranks: pd.DataFrame) -> dict:
    merged = players.merge(
        ranks[["match_key", "player_name", "rank_q"]],
        on=["match_key", "player_name"], how="inner",
    )
    human = merged[merged["label"] == 0]
    positive = merged[merged["label"] == 1]
    threshold = float(np.quantile(human["score"], 0.99))
    q4_human = human[human["rank_q"].astype(str).str.startswith("Q4")]
    q4_positive = positive[positive["rank_q"].astype(str).str.startswith("Q4")]
    return {
        "q4_fpr": float((q4_human["score"] >= threshold).mean()),
        "q4_recall": float((q4_positive["score"] >= threshold).mean()),
        "q4_human_players": int(len(q4_human)),
        "q4_positive_players": int(len(q4_positive)),
    }


def _weapons(frame: pd.DataFrame, column: str) -> dict:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame[column].to_numpy(dtype=float)
    threshold = float(np.quantile(scores[labels == 0], 0.99))
    result = {}
    for index, name in enumerate(WEAPONS):
        mask = frame["weapon_idx"].to_numpy(dtype=float) == index
        if not mask.any():
            continue
        h = scores[mask & (labels == 0)]
        p = scores[mask & (labels == 1)]
        result[name] = {
            "kills": int(mask.sum()),
            "fpr": float((h >= threshold).mean()) if len(h) else None,
            "recall": float((p >= threshold).mean()) if len(p) else None,
        }
    return result


def _evaluate(frame: pd.DataFrame, column: str, ranks: pd.DataFrame, topk: int) -> dict:
    players = _players(frame, column, topk)
    result = _kill_metrics(frame["label"].to_numpy(dtype=int), frame[column].to_numpy(dtype=float))
    result.update(_player_metrics(players))
    result.update(_q4(players, ranks))
    result["weapon"] = _weapons(frame, column)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", required=True)
    parser.add_argument("--gbdt", required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=0,
                        help="player aggregation over the k highest window scores; 0 = mean")
    args = parser.parse_args()

    cnn = pd.read_parquet(args.cnn)
    gbdt = pd.read_parquet(args.gbdt)
    cnn = cnn.rename(columns={"score": "cnn"})
    gbdt = gbdt.rename(columns={"score": "gbdt"})
    frame = cnn.merge(gbdt[["row", "gbdt", "label"]], on="row", suffixes=("", "_gbdt"))
    if "label_gbdt" in frame.columns:
        if int((frame["label"] != frame["label_gbdt"]).sum()):
            raise SystemExit("label mismatch between CNN and GBDT OOF")
        frame = frame.drop(columns=["label_gbdt"])
    features = pd.read_parquet(args.features)[["row", "weapon_idx"]]
    frame = frame.merge(features, on="row", how="left")
    ranks = pd.read_parquet(args.ranks)[["match_key", "player_name", "rank_q"]]

    point = {
        "cnn": _evaluate(frame, "cnn", ranks, args.topk),
        "gbdt": _evaluate(frame, "gbdt", ranks, args.topk),
    }
    matches = frame["match_key"].unique()
    groups = {
        match: frame.index[frame["match_key"] == match].to_numpy() for match in matches
    }
    rng = np.random.default_rng(args.seed)
    deltas = {key: [] for key in ("kill_auc", "player_auc", "player_recall_at_fpr1", "q4_fpr")}
    for _ in range(args.bootstrap):
        chosen = rng.choice(matches, size=len(matches), replace=True)
        index = np.concatenate([groups[match] for match in chosen])
        sample = frame.loc[index]
        for column in ("cnn", "gbdt"):
            players = _players(sample, column, args.topk)
            values = {
                "kill_auc": float(roc_auc_score(
                    sample["label"].to_numpy(dtype=int), sample[column].to_numpy(dtype=float)
                )),
                "player_auc": float(roc_auc_score(
                    players["label"].to_numpy(dtype=int), players["score"].to_numpy(dtype=float)
                )),
            }
            labels = players["label"].to_numpy(dtype=int)
            scores = players["score"].to_numpy(dtype=float)
            threshold = float(np.quantile(scores[labels == 0], 0.99))
            values["player_recall_at_fpr1"] = float((scores[labels == 1] >= threshold).mean())
            q4 = _q4(players, ranks)
            values["q4_fpr"] = q4["q4_fpr"]
            if column == "cnn":
                cnn_values = values
            else:
                gbdt_values = values
        for key in deltas:
            deltas[key].append(cnn_values[key] - gbdt_values[key])

    report = {
        "point": point,
        "bootstrap": {},
        "replicates": args.bootstrap,
        "topk": args.topk,
        "rows": int(len(frame)),
        "matches": int(len(matches)),
    }
    for key, values in deltas.items():
        array = np.asarray(values, dtype=float)
        report["bootstrap"][key] = {
            "mean": float(np.mean(array)),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
