"""Does the mouse-view residual improve low-FPR detection?

5-fold match-disjoint CV on the 160-match raw subset.  Compares:
  - base: aim-only summary features (no weapon/outcome/motion);
  - base + mouse residual;
  - residual only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import binary_metrics, player_scores
from valveguard.scripts.ablation_features import META_DROP


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def make_folds(df: pd.DataFrame, k: int, seed: int) -> pd.Series:
    match_label = df.groupby("match_key")["label"].max()
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, keys in match_label.groupby(match_label).groups.items():
        keys = np.array(sorted(keys))
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            fold_of[key] = i % k
    return df["match_key"].map(fold_of)


def cv_eval(df: pd.DataFrame, cols: list[str], k: int, seed: int) -> tuple[dict, pd.DataFrame]:
    df = df.copy()
    df["fold"] = make_folds(df, k, seed)
    oof = np.full(len(df), np.nan)
    for fold in range(k):
        tr = df.fold != fold
        te = df.fold == fold
        Xtr = df.loc[tr, cols].replace([np.inf, -np.inf], np.nan)
        Xte = df.loc[te, cols].replace([np.inf, -np.inf], np.nan)
        ytr = df.loc[tr, "label"].to_numpy(int)
        pos = max(1, int(ytr.sum()))
        neg = max(1, len(ytr) - pos)
        sw = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg))
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed + fold,
        )
        clf.fit(Xtr, ytr, sample_weight=sw)
        oof[te.to_numpy()] = clf.predict_proba(Xte)[:, 1]
    df["score"] = oof
    kill = binary_metrics(df["label"], df["score"])
    players = player_scores(df, df["score"], method="mean")
    player = binary_metrics(players["label"], players["score"])

    rng = np.random.default_rng(seed)
    matches = players["match_key"].unique()
    aucs, recs = [], []
    for _ in range(1000):
        pick = rng.choice(matches, size=len(matches), replace=True)
        b = pd.concat([players[players.match_key == m] for m in pick], ignore_index=True)
        yy = b["label"].to_numpy(); ss = b["score"].to_numpy()
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, ss)); recs.append(recall_at_fpr(yy, ss))

    def ci(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "kill": kill, "player": player,
        "player_auc_ci95": ci(aucs),
        "player_recall_fpr1_ci95": ci(recs),
        "n_matches": int(df.match_key.nunique()),
        "n_features": len(cols),
    }, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--residual", required=True)
    ap.add_argument("--index", default="data/processed/cw256/index.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    resid = pd.read_parquet(args.residual)
    mr_cols = [c for c in resid.columns if c.startswith("mr_")]
    index = pd.read_parquet(args.index)
    # The summary feature file carries only `row`; bring the metadata from
    # the cache index before joining the residual features.
    meta = index[["row", "match_key", "player_name", "kill_idx", "source", "label"]]
    df = feats.merge(meta, on="row", how="inner")
    df = df.merge(resid[["row"] + mr_cols], on="row", how="inner")
    df = df[~((df["source"] == "cdata") & (df["label"] == 0))].copy()
    df = df.reset_index(drop=True)

    weapon = [c for c in df.columns if c.startswith("weapon_")]
    outcome = [
        "headshot", "through_smoke", "wallbang", "midair",
        "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
    ]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    base_cols = [
        c for c in df.columns
        if c not in META_DROP and c not in mr_cols
        and df[c].notna().any()
        and c not in set(weapon + outcome + motion)
    ]
    core_mr = [
        "mr_valid", "mr_shot", "mr_last8_max",
        "mr_last64_mean", "mr_last64_p95", "mr_last64_frac3",
    ]
    tail_mr = ["mr_valid", "mr_shot", "mr_last8_max", "mr_last64_frac3"]
    fit_mr = ["mr_valid", "mr_r2_yaw", "mr_r2_pitch"]
    sets = {
        "base": base_cols,
        "base_mr": base_cols + mr_cols,
        "base_core": base_cols + [c for c in core_mr if c in mr_cols],
        "base_tail": base_cols + [c for c in tail_mr if c in mr_cols],
        "base_fit": base_cols + [c for c in fit_mr if c in mr_cols],
        "mr_only": mr_cols,
    }

    results = {}
    for name, cols in sets.items():
        print(f"{name}: {len(cols)} features", flush=True)
        res, scored = cv_eval(df, cols, args.folds, args.seed)
        results[name] = res
        if name == "base":
            scored[["row", "match_key", "player_name", "label", "fold",
                    "score"]].to_parquet(out / "base_predictions.parquet",
                                         index=False)
            player_scores(scored, scored["score"], method="mean").to_parquet(
                out / "base_player_scores.parquet", index=False
            )
        if name in ("base", "base_mr"):
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(scored), size=min(5000, len(scored)), replace=False)
            # Fit one diagnostic model on 80% and measure permutation
            # importance on the held-out 20%.
            tr = scored.index.isin(idx[: int(0.8 * len(idx))])
            te = scored.index.isin(idx[int(0.8 * len(idx)):])
            Xtr = scored.loc[tr, cols].replace([np.inf, -np.inf], np.nan)
            Xte = scored.loc[te, cols].replace([np.inf, -np.inf], np.nan)
            clf = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                random_state=args.seed,
            )
            clf.fit(Xtr, scored.loc[tr, "label"])
            imp = permutation_importance(
                clf, Xte, scored.loc[te, "label"], scoring="roc_auc",
                n_repeats=3, random_state=args.seed, n_jobs=1,
            )
            imp_df = pd.DataFrame({
                "feature": cols, "importance": imp.importances_mean,
            }).sort_values("importance", ascending=False)
            imp_df.to_csv(out / f"importance_{name}.csv", index=False)
            results[name]["top_features"] = imp_df.head(20).to_dict(orient="records")

    with open(out / "residual_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({
        k: {
            "kill": {kk: v["kill"][kk] for kk in
                     ["auc", "ap", "recall_at_fpr1", "fpr_at_tpr95"]},
            "player": {kk: v["player"][kk] for kk in
                       ["auc", "ap", "recall_at_fpr1", "fpr_at_tpr95"]},
            "player_auc_ci95": v["player_auc_ci95"],
            "player_recall_fpr1_ci95": v["player_recall_fpr1_ci95"],
            "n_matches": v["n_matches"],
        }
        for k, v in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
