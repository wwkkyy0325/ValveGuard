"""Score professional CS2 kills with the champion detector.

The champion is fitted on the CS2CD development 80%.  Thresholds are set at a
1% false-positive rate on CS2CD human kills and human players, then applied to
pro windows, which are treated as clean human negatives.  This is a diagnostic:
it measures how often top-level human aim looks like aim assist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from valveguard.metrics import player_scores
from valveguard.scripts.ablation_features import META_DROP

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    idx = pd.read_parquet(Path(args.cache) / "index.parquet")
    meta = idx[["row", "match_key", "player_name", "label", "source"]]
    data = feats.merge(meta, on="row", how="inner")
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    hm = pd.read_parquet(args.holdout)
    dev_keys = set(hm.loc[hm["split"] == "dev", "match_key"])
    dev = data[data["match_key"].isin(dev_keys)].copy()
    pro = pd.read_parquet(args.pro)

    cols = [c for c in feature_cols(dev) if c in pro.columns]
    y = dev["label"].to_numpy(int)
    pos, neg = max(1, int(y.sum())), max(1, len(y) - int(y.sum()))
    sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=args.seed,
    )
    clf.fit(dev[cols].replace([np.inf, -np.inf], np.nan), y, sample_weight=sw)

    dev = dev.copy()
    dev["score"] = clf.predict_proba(dev[cols].replace([np.inf, -np.inf], np.nan))[:, 1]
    pro = pro.copy()
    pro["score"] = clf.predict_proba(pro[cols].replace([np.inf, -np.inf], np.nan))[:, 1]

    neg_kill = dev.loc[dev["label"] == 0, "score"].to_numpy()
    dev_players = player_scores(dev, dev["score"].to_numpy(), method="mean")
    human_players = dev_players[dev_players["label"] == 0]["score"].to_numpy()
    pro_players = player_scores(pro, pro["score"].to_numpy(), method="mean")

    operating = {}
    for target in (0.001, 0.01, 0.05):
        kill_thr = float(np.quantile(neg_kill, 1.0 - target))
        player_thr = float(np.quantile(human_players, 1.0 - target))
        operating[f"fpr_{target}"] = {
            "kill_threshold": kill_thr,
            "dev_kill_fpr": float(np.mean(neg_kill >= kill_thr)),
            "pro_kill_fpr": float(np.mean(pro["score"].to_numpy() >= kill_thr)),
            "player_threshold": player_thr,
            "dev_player_fpr": float(np.mean(human_players >= player_thr)),
            "pro_player_fpr": float(
                np.mean(pro_players["score"].to_numpy() >= player_thr)
            ),
        }
    kill_thr = operating["fpr_0.01"]["kill_threshold"]

    by_demo = (
        pro.groupby("match_key")
        .agg(n=("score", "size"), mean=("score", "mean"),
             frac_above=("score", lambda s: float(np.mean(s >= kill_thr))))
        .reset_index()
    )
    res = {
        "n_features": len(cols),
        "dev_kills": int(len(dev)),
        "pro_kills": int(len(pro)),
        "pro_demos": int(pro["match_key"].nunique()),
        "pro_players": int(len(pro_players)),
        "operating_points": operating,
        "score_means": {
            "dev_human_kills": float(neg_kill.mean()),
            "dev_cheater_kills": float(dev.loc[dev["label"] == 1, "score"].mean()),
            "pro_kills": float(pro["score"].mean()),
        },
        "by_demo": by_demo.to_dict(orient="records"),
    }
    with open(out / "pro_fpr.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "by_demo"}, indent=2))
    print(by_demo.to_string(index=False))


if __name__ == "__main__":
    main()
