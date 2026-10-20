"""Player-level outcome/behavior fusion on top of the aim champion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

K0 = 5.0
PRIOR = 0.3018746014732038
KEYS = ["match_key", "player_name"]


def player_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(KEYS, sort=False):
        group = group.sort_values("kill_tick")
        ticks = group["kill_tick"].to_numpy(dtype=float)
        gaps = np.diff(ticks)
        max_multi = 1
        for tick in ticks:
            max_multi = max(max_multi, int(((ticks >= tick) & (ticks <= tick + 256)).sum()))
        burst = np.zeros(len(ticks), dtype=bool)
        if len(gaps):
            burst[1:] = gaps <= 256
        cluster = np.ones(len(ticks), dtype=int)
        if len(gaps):
            current = 1
            for i, adjacent in enumerate(burst[1:], start=1):
                current = current + 1 if adjacent else 1
                cluster[i] = current
        shots = group["shot_count"].sum()
        kills = len(group)
        rows.append({
            "match_key": key[0],
            "player_name": key[1],
            "label": int(group["label"].max()),
            "fold": int(group["fold"].iloc[0]),
            "kills": kills,
            "hs_rate": float(group["headshot"].mean()),
            "hs_count": float(group["headshot"].sum()),
            "shots_per_kill": float((shots + 1.0) / max(1, kills)),
            "kill_per_shot": float(kills / (shots + kills)),
            "max_multikill_4s": int(max_multi),
            "burst_frac": float(burst.mean()) if kills > 1 else 0.0,
            "burst3_frac": float((cluster >= 3).mean()) if kills > 1 else 0.0,
            "min_gap": float(gaps.min()) if len(gaps) else 4096.0,
            "median_gap": float(np.median(gaps)) if len(gaps) else 4096.0,
            "sniper_frac": float((group["weapon_idx"] == 2).mean()),
            "pistol_frac": float((group["weapon_idx"] == 3).mean()),
            "rifle_frac": float((group["weapon_idx"] == 1).mean()),
            "sniper_hs": float(group.loc[group["weapon_idx"] == 2, "headshot"].mean())
            if (group["weapon_idx"] == 2).any() else np.nan,
            "smoke_rate": float(group["through_smoke"].mean()),
            "wallbang_rate": float(group["wallbang"].mean()),
            "midair_rate": float(group["midair"].mean()),
            "full_hp_rate": float((group["victim_health"] >= 99).mean()),
            "victim_health": float(group["victim_health"].mean()),
            "mean_window_score": float(group["score"].mean()),
            "sum_window_score": float(group["score"].sum()),
        })
    players = pd.DataFrame(rows)
    players["shrunk_score"] = (
        players["sum_window_score"] + K0 * PRIOR
    ) / (players["kills"] + K0)
    players["log_median_gap"] = np.log1p(players["median_gap"])
    players["log_min_gap"] = np.log1p(players["min_gap"])
    return players


def _threshold(values: np.ndarray, labels: np.ndarray) -> float:
    return float(np.quantile(values[labels == 0], 0.99))


def metrics(frame: pd.DataFrame, column: str, q4: pd.DataFrame) -> dict:
    labels = frame["label"].to_numpy(dtype=int)
    values = frame[column].to_numpy(dtype=float)
    threshold = _threshold(values, labels)
    merged = frame.merge(q4, on=KEYS, how="inner")
    q4h = merged[(merged["label"] == 0)
                 & (merged["rank_q"].astype(str).str.startswith("Q4"))]
    q4p = merged[(merged["label"] == 1)
                 & (merged["rank_q"].astype(str).str.startswith("Q4"))]
    return {
        "auc": float(roc_auc_score(labels, values)),
        "ap": float(average_precision_score(labels, values)),
        "recall_at_fpr1": float((values[labels == 1] >= threshold).mean()),
        "fpr": float((values[labels == 0] >= threshold).mean()),
        "q4_fpr": float((q4h[column] >= threshold).mean()),
        "q4_recall": float((q4p[column] >= threshold).mean()),
        "threshold": threshold,
        "players": int(len(frame)),
        "positive_players": int(labels.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/cw256/index.parquet")
    ap.add_argument("--features", default="reports/baseline_clean/summary_features.parquet")
    ap.add_argument("--kills", default="data/processed/kill_ticks.parquet")
    ap.add_argument("--champion", default="reports/cnn29_fixed_deploy/oof_ensemble.parquet")
    ap.add_argument("--folds", default="reports/cv_gbdt/cv_predictions.parquet")
    ap.add_argument("--ranks", default="reports/rank_replication/rank_players.parquet")
    ap.add_argument("--out", default="reports/behavior_fusion")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    index = pd.read_parquet(args.index)
    index = index[~((index["source"] == "cdata") & (index["label"] == 0))].copy()
    features = pd.read_parquet(args.features)
    kills = pd.read_parquet(args.kills)[["row", "kill_tick"]]
    champion = pd.read_parquet(args.champion)[["row", "score"]]
    folds = pd.read_parquet(args.folds)[["match_key", "fold"]].drop_duplicates()
    q4 = pd.read_parquet(args.ranks)[KEYS + ["rank_q"]]

    frame = (index.merge(features, on="row", how="left")
             .merge(kills, on="row", how="left")
             .merge(champion, on="row", how="left")
             .merge(folds, on="match_key", how="left"))
    players = player_features(frame)
    print("players", len(players), "positive", int(players.label.sum()), flush=True)

    behavior_columns = [
        "hs_rate", "hs_count", "shots_per_kill", "kill_per_shot",
        "max_multikill_4s", "burst_frac", "burst3_frac",
        "log_min_gap", "log_median_gap",
        "sniper_frac", "pistol_frac", "rifle_frac", "sniper_hs",
        "smoke_rate", "wallbang_rate", "midair_rate",
        "full_hp_rate", "victim_health",
    ]
    oof = np.full(len(players), np.nan, dtype=float)
    for fold in sorted(players["fold"].unique()):
        train = players["fold"] != fold
        test = players["fold"] == fold
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=args.seed,
            class_weight="balanced",
        )
        model.fit(players.loc[train, behavior_columns],
                  players.loc[train, "label"])
        oof[test] = model.predict_proba(players.loc[test, behavior_columns])[:, 1]
    players["behavior_score"] = oof

    univariate = {}
    for column in behavior_columns:
        values = players[column].replace([np.inf, -np.inf], np.nan)
        valid = values.notna()
        if players.loc[valid, "label"].nunique() > 1:
            univariate[column] = float(roc_auc_score(
                players.loc[valid, "label"], values[valid]
            ))

    metrics_out = {
        "champion": metrics(players, "shrunk_score", q4),
        "behavior": metrics(players, "behavior_score", q4),
    }
    for column in ("shrunk_score", "behavior_score"):
        players[f"rank_{column}"] = players[column].rank(pct=True)
    for weight in (0.25, 0.5, 0.75):
        name = f"fused_{weight:.2f}"
        players[name] = (weight * players["rank_shrunk_score"]
                         + (1.0 - weight) * players["rank_behavior_score"])
        metrics_out[name] = metrics(players, name, q4)

    matches = players["match_key"].unique()
    groups = {m: players[players["match_key"] == m] for m in matches}
    rng = np.random.default_rng(args.seed)
    deltas = {key: [] for key in ("auc", "recall_at_fpr1", "q4_fpr")}
    for _ in range(args.bootstrap):
        chosen = rng.choice(matches, size=len(matches), replace=True)
        sample = pd.concat([groups[m] for m in chosen], ignore_index=True)
        base = metrics(sample, "shrunk_score", q4)
        for weight in (0.25, 0.5, 0.75):
            name = f"fused_{weight:.2f}"
            value = metrics(sample, name, q4)
            for key in deltas:
                deltas[key].append(value[key] - base[key])
    bootstrap = {
        key: {"mean": float(np.mean(v)),
              "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]}
        for key, v in deltas.items()
    }

    champion_threshold = metrics_out["champion"]["threshold"]
    behavior_threshold = metrics_out["behavior"]["threshold"]
    missed = players[(players["label"] == 1)
                     & (players["shrunk_score"] < champion_threshold)]
    rescued = missed[missed["behavior_score"] >= behavior_threshold]
    high_activity = players[(players["kills"] >= 15) & (players["hs_rate"] >= 0.6)]
    subgroup = {
        "players": int(len(high_activity)),
        "positive": int(high_activity["label"].sum()),
        "champion_recall": float(
            (high_activity.loc[high_activity.label == 1, "shrunk_score"]
             >= champion_threshold).mean()
        ) if (high_activity.label == 1).any() else None,
        "behavior_recall": float(
            (high_activity.loc[high_activity.label == 1, "behavior_score"]
             >= behavior_threshold).mean()
        ) if (high_activity.label == 1).any() else None,
    }
    result = {
        "metrics": metrics_out,
        "bootstrap_fused_vs_champion": bootstrap,
        "univariate_auc": univariate,
        "missed_by_champion": int(len(missed)),
        "rescued_by_behavior": int(len(rescued)),
        "high_activity_subgroup": subgroup,
        "features": behavior_columns,
        "players": int(len(players)),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    players.to_parquet(out / "players.parquet", index=False)
    pd.DataFrame({
        "feature": list(univariate),
        "univariate_auc": list(univariate.values()),
    }).sort_values("univariate_auc", ascending=False).to_csv(
        out / "univariate.csv", index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
