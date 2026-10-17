"""One-shot evaluation of the frozen champion on the locked holdout.

The champion pipeline is fitted on the development 80% only and scored once
on the held-out 20% of matches.  The holdout was locked after v1 selection, so
this is a clean out-of-sample estimate for a fixed pipeline, but it cannot
undo the earlier selection that produced that pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import binary_metrics, player_scores
from valveguard.scripts.ablation_features import META_DROP


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
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
    meta = idx[["row", "match_key", "player_name", "label", "source",
                "match_idx", "kill_idx"]]
    data = feats.merge(meta, on="row", how="inner")
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()

    hm = pd.read_parquet(args.holdout)
    split = hm.set_index("match_key")["split"]
    data["split"] = data["match_key"].map(split)
    if data["split"].isna().any():
        raise RuntimeError("holdout map does not cover every match")

    weapon = [c for c in data.columns if c.startswith("weapon_")]
    outcome = ["headshot", "through_smoke", "wallbang", "midair",
               "shot_count", "victim_noise", "flashed_at_kill", "victim_health"]
    motion = [c for c in data.columns if c.startswith("attacker_vel")]
    cols = [c for c in data.columns if c not in META_DROP
            and c not in {"split"} and data[c].notna().any()
            and c not in set(weapon + outcome + motion)]

    tr = data[data.split == "dev"]
    te = data[data.split == "holdout"]
    ytr = tr["label"].to_numpy(int)
    pos = max(1, int(ytr.sum()))
    neg = max(1, len(ytr) - pos)
    sw = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg))
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=args.seed,
    )
    clf.fit(tr[cols].replace([np.inf, -np.inf], np.nan), ytr, sample_weight=sw)

    te = te.copy()
    te["score"] = clf.predict_proba(
        te[cols].replace([np.inf, -np.inf], np.nan)
    )[:, 1]
    kill = binary_metrics(te["label"].to_numpy(), te["score"].to_numpy())
    players = player_scores(te, te["score"].to_numpy(), method="mean")
    player = binary_metrics(players["label"].to_numpy(), players["score"].to_numpy())

    rng = np.random.default_rng(args.seed)
    matches = players["match_key"].unique()
    aucs, recs = [], []
    for _ in range(1000):
        pick = rng.choice(matches, size=len(matches), replace=True)
        b = pd.concat([players[players.match_key == m] for m in pick], ignore_index=True)
        yy = b["label"].to_numpy()
        ss = b["score"].to_numpy()
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, ss))
        recs.append(recall_at_fpr(yy, ss))

    def ci(x):
        x = np.asarray(x)
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    res = {
        "train_matches": int(tr.match_key.nunique()),
        "holdout_matches": int(te.match_key.nunique()),
        "n_features": len(cols),
        "kill": kill,
        "player": player,
        "player_auc_ci95": ci(aucs),
        "player_recall_fpr1_ci95": ci(recs),
    }
    with open(out / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
