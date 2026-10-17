"""Add professional human windows as extra negatives during training.

Five-fold match-disjoint CV on the CS2CD development 80%.  Pro windows are
added to every training fold and are never part of a test fold.  The question
is whether high-skill human negatives change the CS2CD ranking, especially the
top-rank quartile false positives.

Evaluation is dev-only; the locked holdout is not touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from valveguard.metrics import binary_metrics, player_scores
from valveguard.scripts.ablation_features import META_DROP
from valveguard.scripts.run_self_baseline import q4_table

OUTCOME = {
    "headshot", "through_smoke", "wallbang", "midair",
    "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
}


def feature_cols(df: pd.DataFrame) -> list[str]:
    weapon = [c for c in df.columns if c.startswith("weapon_")]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    return [
        c for c in df.columns
        if c not in META_DROP and c not in OUTCOME and c != "fold"
        and c not in set(weapon + motion) and df[c].notna().any()
    ]


def balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--pro", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    idx = pd.read_parquet(Path(args.cache) / "index.parquet")
    meta = idx[["row", "match_key", "player_name", "label", "source", "kill_idx"]]
    data = feats.merge(meta, on="row", how="inner")
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    data = data.merge(
        pd.read_parquet(args.folds)[["match_key", "fold"]].drop_duplicates(),
        on="match_key", how="left",
    )
    hm = pd.read_parquet(args.holdout)
    dev_keys = set(hm.loc[hm["split"] == "dev", "match_key"])
    data = data[data["match_key"].isin(dev_keys)].dropna(subset=["fold"]).copy()
    data["fold"] = data["fold"].astype(int)
    data = data.reset_index(drop=True)

    pro = pd.read_parquet(args.pro)
    cols = [c for c in feature_cols(data) if c in pro.columns]
    print(f"cs2cd {len(data)} pro {len(pro)} features {len(cols)}", flush=True)

    oof = np.full(len(data), np.nan)
    for fold in sorted(data["fold"].unique()):
        tr = data[data["fold"] != fold]
        te = data[data["fold"] == fold]
        Xtr = pd.concat([tr[cols], pro[cols]], ignore_index=True).replace(
            [np.inf, -np.inf], np.nan
        )
        ytr = np.concatenate([tr["label"].to_numpy(int), np.zeros(len(pro), dtype=int)])
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=args.seed + int(fold),
        )
        clf.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
        oof[te.index.to_numpy()] = clf.predict_proba(
            te[cols].replace([np.inf, -np.inf], np.nan)
        )[:, 1]
        print(f"fold {fold} done", flush=True)

    data["score"] = oof
    kill = binary_metrics(data["label"].to_numpy(), oof)
    players = player_scores(data, oof, method="mean")
    player = binary_metrics(players["label"].to_numpy(), players["score"].to_numpy())
    skill = pd.read_parquet(args.skill)
    q4 = q4_table(data, oof, skill)

    res = {"kill": kill, "player": player, "q4": q4, "n_features": len(cols)}
    with open(out / "pro_augment.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    data[["row", "match_key", "player_name", "label", "fold", "score"]].to_parquet(
        out / "cv_predictions.parquet", index=False
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
