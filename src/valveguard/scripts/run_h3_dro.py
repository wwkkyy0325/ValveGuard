"""H3 pilot: DRO-style reweighting across rank groups.

High-rank humans are the main source of false positives (Q4 FPR 4.76% at the
global 1% cut).  Earlier experiments upweighted Q4 clean windows by a fixed
factor and did not help.  This pilot replaces the fixed factor with adaptive
group reweighting: groups are rank quartiles (plus an unknown group), and the
weight of a group grows when its held-out loss is high.

The weights are estimated from inner 2-fold predictions inside each outer
training set, so no outer-test information is used to set them.  The model is
still a GBDT; this is a proxy for group DRO, not the exact minimax objective.

Pilot subset: the 160 matches with raw rank data.  Not a final result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from valveguard.metrics import player_scores
from valveguard.scripts.ablation_features import META_DROP

GROUPS = ["Q1 low", "Q2", "Q3", "Q4 high", "unknown"]


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def make_clf(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=seed,
    )


def balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


def group_losses(X, y, gidx, wg, seed) -> np.ndarray:
    """Held-out log-loss per group from a 2-fold inner split."""
    pred = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = make_clf(seed)
        clf.fit(
            X.iloc[tr], y[tr],
            sample_weight=balanced_weights(y[tr]) * wg[gidx[tr]],
        )
        pred[te] = clf.predict_proba(X.iloc[te])[:, 1]
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    out = np.array([
        loss[gidx == g].mean() if np.any(gidx == g) else 0.0
        for g in range(len(wg))
    ])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--index", default="data/processed/cw256/index.parquet")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--max-matches", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    index = pd.read_parquet(args.index)
    index = index[
        index["source"].isin(["cdata", "ncdata"])
        & (index["match_idx"] < args.max_matches)
    ]
    meta = index[["row", "match_key", "player_name", "label", "source"]]
    meta = meta[~((meta["source"] == "cdata") & (meta["label"] == 0))]
    df = feats.merge(meta, on="row", how="inner")
    df = df.merge(
        pd.read_parquet(args.folds)[["match_key", "fold"]].drop_duplicates(),
        on="match_key", how="left",
    )
    df = df.merge(
        pd.read_parquet(args.skill)[["match_key", "player_name", "rank_q"]],
        on=["match_key", "player_name"], how="left",
    )
    df = df[df["fold"].notna()].reset_index(drop=True)
    df["fold"] = df["fold"].astype(int)
    df["group"] = df["rank_q"].fillna("unknown").astype(str)
    df.loc[~df["group"].isin(GROUPS), "group"] = "unknown"
    gmap = {g: i for i, g in enumerate(GROUPS)}
    gidx_all = df["group"].map(gmap).to_numpy()

    weapon = [c for c in df.columns if c.startswith("weapon_")]
    outcome = ["headshot", "through_smoke", "wallbang", "midair",
               "shot_count", "victim_noise", "flashed_at_kill", "victim_health"]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    cols = [c for c in df.columns if c not in META_DROP
            and c not in {"rank_q", "group", "fold"}
            and df[c].notna().any()
            and c not in set(weapon + outcome + motion)]

    print(f"rows {len(df)} players {df.groupby(['match_key','player_name']).ngroups} "
          f"features {len(cols)} groups {df['group'].value_counts().to_dict()}",
          flush=True)

    def run(name: str, dro: bool) -> dict:
        oof = np.full(len(df), np.nan)
        for fold in sorted(df["fold"].unique()):
            tr = (df["fold"] != fold).to_numpy()
            te = ~tr
            Xtr = df.loc[tr, cols].replace([np.inf, -np.inf], np.nan)
            Xte = df.loc[te, cols].replace([np.inf, -np.inf], np.nan)
            ytr = df.loc[tr, "label"].to_numpy(int)
            gidx = gidx_all[tr]
            wg = np.ones(len(GROUPS))
            if dro:
                for it in range(args.iterations):
                    losses = group_losses(Xtr, ytr, gidx, wg, seed=42 + fold * 10 + it)
                    wg = wg * np.exp(args.eta * (losses - losses.mean()))
                    wg = wg / wg.mean()
            clf = make_clf(42 + fold)
            clf.fit(Xtr, ytr,
                    sample_weight=balanced_weights(ytr) * wg[gidx])
            oof[te] = clf.predict_proba(Xte)[:, 1]
            print(f"{name} fold {fold} weights "
                  f"{dict(zip(GROUPS, np.round(wg, 3)))}", flush=True)

        d = df.copy()
        d["score"] = oof
        players = player_scores(d, oof, method="mean")
        players = players.merge(
            d.groupby(["match_key", "player_name"])["group"].first().reset_index(),
            on=["match_key", "player_name"], how="left",
        )
        res = {
            "player_auc": float(roc_auc_score(players["label"], players["score"])),
            "player_recall_fpr1": recall_at_fpr(
                players["label"].to_numpy(), players["score"].to_numpy()
            ),
            "kill_auc": float(roc_auc_score(d["label"], oof)),
        }
        fpr, _, thr = roc_curve(d["label"], oof)
        ok = np.where(fpr <= 0.01)[0]
        cut = float(thr[ok[-1]]) if len(ok) else 1.0
        for g in GROUPS:
            p = players[players["group"] == g]
            if len(p) and p["label"].nunique() > 1:
                pred = (p["score"] >= cut).to_numpy()
                y = p["label"].to_numpy()
                fp = int(((pred == 1) & (y == 0)).sum())
                tn = int(((pred == 0) & (y == 0)).sum())
                res[f"{g}_auc"] = float(roc_auc_score(y, p["score"]))
                res[f"{g}_fpr"] = fp / max(fp + tn, 1)
                res[f"{g}_n"] = int(len(p))
        return res

    results = {"baseline": run("base", False), "dro": run("dro", True)}
    with open(out / "h3_dro.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
