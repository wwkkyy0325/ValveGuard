"""Fit and save the deployment model artifact.

The artifact contains the GBDT, the exact feature order, the player-level 1%
threshold, the human score prior, and the shrinkage pseudo-count.  Training
uses all clean windows (ncdata label 0 plus cheater windows); the reported
metrics still come from the frozen dev/holdout evaluation.

Usage:
    python -m valveguard.scripts.train_final \
        --features reports/baseline_clean/summary_features.parquet \
        --cache data/processed/cw256 \
        --out artifacts/valveguard_gbdt_v1.joblib
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from valveguard.metrics import player_scores
from valveguard.scripts.ablation_features import META_DROP

OUTCOME = {
    "headshot", "through_smoke", "wallbang", "midair",
    "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
}
K0 = 5


def feature_cols(df: pd.DataFrame) -> list[str]:
    weapon = [c for c in df.columns if c.startswith("weapon_")]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    return [
        c for c in df.columns
        if c not in META_DROP and c not in OUTCOME and c != "fold"
        and c not in set(weapon + motion) and df[c].notna().any()
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    idx = pd.read_parquet(Path(args.cache) / "index.parquet")
    meta = idx[["row", "match_key", "player_name", "label", "source"]]
    data = feats.merge(meta, on="row", how="inner")
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    cols = feature_cols(data)
    y = data["label"].to_numpy(int)

    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=args.seed,
    )
    clf.fit(data[cols].replace([np.inf, -np.inf], np.nan), y, sample_weight=sw)

    scores = clf.predict_proba(
        data[cols].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    data = data.assign(score=scores)
    players = player_scores(data, scores, method="mean")
    human = players.loc[players["label"] == 0, "score"]
    threshold = float(human.quantile(0.99))
    prior = float(data.loc[data["label"] == 0, "score"].mean())

    artifact = {
        "version": f"valveguard-gbdt-{date.today().isoformat()}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": clf,
        "features": cols,
        "threshold": threshold,
        "prior": prior,
        "k0": K0,
        "n_windows": int(len(data)),
        "n_matches": int(data["match_key"].nunique()),
        "train_fpr_target": 0.01,
        "train_human_fpr": float((human >= threshold).mean()),
        "notes": "GBDT on aim-only window features; player score is shrunk by n",
    }
    joblib.dump(artifact, out)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in artifact.items() if k != "model"}, f, indent=2)
    print(json.dumps({
        "artifact": str(out),
        "version": artifact["version"],
        "windows": artifact["n_windows"],
        "matches": artifact["n_matches"],
        "features": len(cols),
        "threshold": threshold,
        "prior": prior,
    }, indent=2))


if __name__ == "__main__":
    main()
