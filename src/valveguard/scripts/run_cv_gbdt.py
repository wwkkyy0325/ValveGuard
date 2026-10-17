"""5-fold match-disjoint CV for the aim-only GBDT.

Match-level folds avoid the optimistic single-split estimate and give a
more stable player-level AUC / low-FPR recall.  Folds are stratified by
whether the match contains a cheater.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.data.tar_index import load_cache
from valveguard.metrics import binary_metrics, player_scores
from valveguard.scripts.ablation_features import META_DROP


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    _, meta, _ = load_cache(args.cache)
    data = feats.merge(
        meta[["row", "match_key", "player_name", "label", "source",
              "match_idx", "player", "kill_idx"]],
        on="row", how="inner",
    )
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    data = data.reset_index(drop=True)

    weapon = [c for c in data.columns if c.startswith("weapon_")]
    outcome = [
        "headshot", "through_smoke", "wallbang", "midair",
        "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
    ]
    motion = [c for c in data.columns if c.startswith("attacker_vel")]
    cols = [c for c in data.columns if c not in META_DROP
            and data[c].notna().any()
            and c not in set(weapon + outcome + motion)]

    # Stratified match folds.
    match_label = data.groupby("match_key")["label"].max()
    rng = np.random.default_rng(args.seed)
    fold_of: dict[str, int] = {}
    for label, keys in match_label.groupby(match_label).groups.items():
        keys = np.array(sorted(keys))
        rng.shuffle(keys)
        for i, k in enumerate(keys):
            fold_of[k] = i % args.folds
    data["fold"] = data["match_key"].map(fold_of)

    oof = np.full(len(data), np.nan)
    fold_metrics = []
    for fold in range(args.folds):
        tr = data[data.fold != fold]
        te = data[data.fold == fold]
        Xtr = tr[cols].replace([np.inf, -np.inf], np.nan)
        Xte = te[cols].replace([np.inf, -np.inf], np.nan)
        ytr = tr["label"].to_numpy(int)
        yte = te["label"].to_numpy(int)
        pos = max(1, int(ytr.sum()))
        neg = max(1, len(ytr) - pos)
        sw = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg))
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=args.seed + fold,
        )
        clf.fit(Xtr, ytr, sample_weight=sw)
        p = clf.predict_proba(Xte)[:, 1]
        oof[te.index.to_numpy()] = p
        m = binary_metrics(yte, p)
        fold_metrics.append({"fold": fold, **m})

    data["score"] = oof
    y = data["label"].to_numpy()
    s = data["score"].to_numpy()
    kill = binary_metrics(y, s)
    players = player_scores(data, s, method="mean")
    player = binary_metrics(players["label"].to_numpy(), players["score"].to_numpy())

    # Match-grouped bootstrap for the player-level AUC and recall@1% FPR.
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

    def ci(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    results = {
        "folds": fold_metrics,
        "kill_oof": kill,
        "player_oof": player,
        "player_auc_ci95": ci(aucs),
        "player_recall_fpr1_ci95": ci(recs),
        "n_matches": int(data.match_key.nunique()),
        "n_players": int(len(players)),
        "n_features": len(cols),
    }
    with open(out / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    data[["row", "match_key", "player_name", "label", "fold", "score"]].to_parquet(
        out / "cv_predictions.parquet", index=False
    )
    players.to_parquet(out / "cv_player_scores.parquet", index=False)
    print(json.dumps({
        "kill": kill, "player": player,
        "player_auc_ci95": results["player_auc_ci95"],
        "player_recall_fpr1_ci95": results["player_recall_fpr1_ci95"],
        "n_matches": results["n_matches"], "n_features": len(cols),
    }, indent=2))


if __name__ == "__main__":
    main()
