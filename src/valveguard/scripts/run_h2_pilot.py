"""H2 pilot: does 1024-tick context add signal over the near-kill window?

The published cache only has 256-tick windows.  This pilot rebuilds 1024-tick
windows from the raw 160-match subset and compares three feature sets on the
same match-disjoint folds:

  near     -- standard summary over the last 128 ticks before the kill
  near+macro -- near plus whole-window (1024-tick) macro features
  macro    -- macro features alone

A fourth arm, ``cached256``, uses the published cache features on the same
matches as a reconstruction sanity check.

This is a diagnostic on a subset, not a final candidate: the main dataset has
no 1024-tick windows.
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

OUTCOME = {
    "headshot", "through_smoke", "wallbang", "midair",
    "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
}


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def aim_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    drop = set(META_DROP) | OUTCOME
    return [
        c for c in cols
        if c not in drop and not c.startswith("weapon_")
        and not c.startswith("attacker_vel") and df[c].notna().any()
    ]


def run_cv(data: pd.DataFrame, cols: list[str], seed: int = 42) -> np.ndarray:
    oof = np.full(len(data), np.nan)
    folds = sorted(data["fold"].unique())
    for fold in folds:
        tr = data[data.fold != fold]
        te = data[data.fold == fold]
        Xtr = tr[cols].replace([np.inf, -np.inf], np.nan)
        Xte = te[cols].replace([np.inf, -np.inf], np.nan)
        ytr = tr["label"].to_numpy(int)
        pos = max(1, int(ytr.sum()))
        neg = max(1, len(ytr) - pos)
        sw = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg))
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed + int(fold),
        )
        clf.fit(Xtr, ytr, sample_weight=sw)
        oof[te.index.to_numpy()] = clf.predict_proba(Xte)[:, 1]
    return oof


def score(data: pd.DataFrame, cols: list[str]) -> tuple[dict, pd.DataFrame]:
    s = run_cv(data, cols)
    d = data.copy()
    d["score"] = s
    kill = binary_metrics(d["label"].to_numpy(), s)
    players = player_scores(d, s, method="mean")
    player = binary_metrics(players["label"].to_numpy(), players["score"].to_numpy())
    return {"kill": kill, "player": player, "n_features": len(cols)}, players


def with_key(players: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    out["player_key"] = out["match_key"] + ":" + out["player_name"]
    return out


def paired(players_a: pd.DataFrame, players_b: pd.DataFrame, seed: int = 42):
    a = with_key(players_a).set_index("player_key")["score"]
    b = with_key(players_b).set_index("player_key")["score"]
    common = a.index.intersection(b.index)
    p = with_key(players_a).set_index("player_key").loc[common]
    y = p["label"].to_numpy()
    matches = p["match_key"].to_numpy()
    sa = a.loc[common].to_numpy()
    sb = b.loc[common].to_numpy()

    rng = np.random.default_rng(seed)
    uniq = np.unique(matches)
    aucs, recs = [], []
    for _ in range(1000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(matches == m)[0] for m in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, sb[idx]) - roc_auc_score(yy, sa[idx]))
        recs.append(recall_at_fpr(yy, sb[idx]) - recall_at_fpr(yy, sa[idx]))

    def ci(x):
        x = np.asarray(x)
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {"delta_auc": ci(aucs), "delta_recall_fpr1": ci(recs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features", required=True,
                    help="cached 256 summary_features.parquet for the sanity arm")
    ap.add_argument("--folds", required=True,
                    help="cv_predictions.parquet with match_key/fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    idx = pd.read_parquet(Path(args.cache) / "index.parquet")
    meta = idx[["row", "match_key", "player_name", "label", "source",
                "match_idx", "kill_idx"]]
    meta = meta[~((meta["source"] == "cdata") & (meta["label"] == 0))]
    folds = pd.read_parquet(args.folds)[["match_key", "fold"]].drop_duplicates()

    raw = pd.read_parquet(args.windows)
    data = raw.merge(meta.drop(columns=["match_key", "player_name", "kill_idx"]),
                     on="row", how="inner")
    data = data.merge(folds, on="match_key", how="left")
    data["player_key"] = data["match_key"] + ":" + data["player_name"]
    if data["fold"].isna().any():
        raise RuntimeError("fold map does not cover every rebuilt window")
    data["fold"] = data["fold"].astype(int)

    cached = pd.read_parquet(args.features)
    cached = cached.merge(
        meta[["row", "match_key", "player_name", "label", "source", "match_idx", "kill_idx"]],
        on="row", how="inner", suffixes=("", "_m"),
    )
    cached = cached[~((cached["source"] == "cdata") & (cached["label"] == 0))]
    # Keep the sanity arm on exactly the rebuilt subset so the comparison is
    # like-for-like (same matches, same folds, same number of training rows).
    cached = cached[cached["row"].isin(set(data["row"]))].copy()
    cached = cached.merge(folds, on="match_key", how="left")
    cached["player_key"] = cached["match_key"] + ":" + cached["player_name"]

    near = aim_cols(data, [c for c in raw.columns if not c.startswith("macro_")])
    macro = [c for c in data.columns if c.startswith("macro_")]
    sets = {
        "near": near,
        "near+macro": near + macro,
        "macro": macro,
    }

    rows, players = [], {}
    for name, cols in sets.items():
        print(f"training {name}: {len(cols)} features", flush=True)
        res, pl = score(data, cols)
        players[name] = pl
        rows.append({"set": name, "n_features": len(cols),
                     "player_auc": res["player"]["auc"],
                     "player_ap": res["player"]["ap"],
                     "player_recall_fpr1": res["player"]["recall_at_fpr1"],
                     "player_fpr_tpr95": res["player"]["fpr_at_tpr95"],
                     "kill_auc": res["kill"]["auc"],
                     "kill_recall_fpr1": res["kill"]["recall_at_fpr1"]})

    cached_cols = aim_cols(cached, near)
    print(f"training cached256: {len(cached_cols)} features", flush=True)
    res, pl = score(cached, cached_cols)
    players["cached256"] = pl
    rows.append({"set": "cached256", "n_features": len(cached_cols),
                 "player_auc": res["player"]["auc"],
                 "player_ap": res["player"]["ap"],
                 "player_recall_fpr1": res["player"]["recall_at_fpr1"],
                 "player_fpr_tpr95": res["player"]["fpr_at_tpr95"],
                 "kill_auc": res["kill"]["auc"],
                 "kill_recall_fpr1": res["kill"]["recall_at_fpr1"]})

    table = pd.DataFrame(rows)
    table.to_csv(out / "h2_pilot.csv", index=False)
    print(table.to_string(index=False))

    comparisons = {
        "near+macro vs near": paired(players["near"], players["near+macro"]),
        "macro vs near": paired(players["near"], players["macro"]),
        "cached256 vs near": paired(players["near"], players["cached256"]),
    }
    with open(out / "h2_paired.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)
    print(json.dumps(comparisons, indent=2))

    data[["row", "match_key", "player_name", "fold", "label"]].to_parquet(
        out / "h2_windows.parquet", index=False
    )


if __name__ == "__main__":
    main()
