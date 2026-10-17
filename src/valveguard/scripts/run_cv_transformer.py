"""5-fold match-disjoint CV for the relative-only Transformer.

This is the honest primary estimate: every match is scored once by a model
that never saw it.  Folds are stratified by whether the match has a cheater.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.data.tar_index import load_cache
from valveguard.metrics import binary_metrics, player_scores
from valveguard.models.transformer import train_transformer


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--channels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-groups", default="weapon,outcome,motion,position")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(Path(args.cache) / "index.parquet")
    index = index[~((index["source"] == "cdata") & (index["label"] == 0))].copy()
    index = index.reset_index(drop=True)

    match_label = index.groupby("match_key")["label"].max()
    rng = np.random.default_rng(args.seed)
    fold_of: dict[str, int] = {}
    for _, keys in match_label.groupby(match_label).groups.items():
        keys = np.array(sorted(keys))
        rng.shuffle(keys)
        for i, k in enumerate(keys):
            fold_of[k] = i % args.folds
    index["fold"] = index["match_key"].map(fold_of)

    parts = []
    for fold in range(args.folds):
        fd = out / f"fold{fold}"
        fd.mkdir(parents=True, exist_ok=True)
        idx = index.copy()
        idx["split"] = "train"
        idx.loc[idx.fold == (fold + 1) % args.folds, "split"] = "val"
        idx.loc[idx.fold == fold, "split"] = "test"
        print(f"fold {fold}: train={sum(idx.split=='train')} val={sum(idx.split=='val')} test={sum(idx.split=='test')}", flush=True)
        train_transformer(
            args.cache, args.channels, fd,
            epochs=args.epochs, batch_size=args.batch_size,
            workers=args.workers, seed=args.seed,
            drop_groups=[g.strip() for g in args.drop_groups.split(",") if g.strip()],
            index_df=idx,
        )
        p = pd.read_parquet(fd / "predictions.parquet")
        p["fold"] = fold
        parts.append(p)

    pred = pd.concat(parts, ignore_index=True)
    pred.to_parquet(out / "cv_predictions.parquet", index=False)
    kill = binary_metrics(pred["label"].to_numpy(), pred["score"].to_numpy())
    players = player_scores(pred, pred["score"].to_numpy(), method="mean")
    player = binary_metrics(players["label"].to_numpy(), players["score"].to_numpy())

    rng = np.random.default_rng(args.seed)
    matches = players["match_key"].unique()
    aucs, recs = [], []
    for _ in range(1000):
        pick = rng.choice(matches, size=len(matches), replace=True)
        b = pd.concat([players[players.match_key == m] for m in pick], ignore_index=True)
        yy = b["label"].to_numpy(); ss = b["score"].to_numpy()
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, ss))
        recs.append(recall_at_fpr(yy, ss))

    def ci(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    results = {
        "kill_oof": kill,
        "player_oof": player,
        "player_auc_ci95": ci(aucs),
        "player_recall_fpr1_ci95": ci(recs),
        "n_matches": int(pred.match_key.nunique()),
        "n_players": int(len(players)),
    }
    with open(out / "cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
