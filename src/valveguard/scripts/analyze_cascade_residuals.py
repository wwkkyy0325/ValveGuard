"""Cascade and residual analysis on the 5-fold OOF predictions.

Cascade concerns:
  - player aggregation must not be trained on the same fold used to score it;
  - calibration is cross-fitted per fold;
  - compare label-free aggregators (mean/top-k/max/noisy-OR/HMM) instead of
    a learned meta-model that could leak through base-model folds.

Residual concerns:
  - calibration residuals by score decile;
  - error rates by map, weapon group, distance, and kills-per-player;
  - player-level false positives/negatives at a validation-selected FPR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import binary_metrics


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def crossfit_isotonic(df: pd.DataFrame, score_col: str) -> np.ndarray:
    """Calibrate each fold with isotonic fitted on the other folds."""
    out = np.full(len(df), np.nan)
    for fold in sorted(df["fold"].unique()):
        tr = df.fold != fold
        te = ~tr
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(df.loc[tr, score_col], df.loc[tr, "label"])
        out[te.to_numpy()] = iso.predict(df.loc[te, score_col])
    return np.clip(out, 1e-4, 1 - 1e-4)


def agg_table(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
    keys = ["match_key", "player_name"]
    base = df.groupby(keys, as_index=False).agg(
        label=("label", "max"), n_kills=("label", "size"),
        fold=("fold", "max"),
    )
    scores = df.groupby(keys, as_index=False)[p_col].agg(
        mean="mean",
        max="max",
        top3=lambda x: float(np.mean(np.sort(x)[-3:])),
        # Noisy-OR: 1 - prod(1-p).
        noisy_or=lambda x: float(1.0 - np.prod(1.0 - np.clip(x, 0, 1))),
    )
    return base.merge(scores, on=keys, how="left")


def hmm_player_score(scores: np.ndarray, stay: float = 0.9) -> float:
    """Two-state HMM with Bernoulli emissions from calibrated kill scores."""
    a = np.array([[stay, 1 - stay], [1 - stay, stay]])
    prior = np.array([0.9, 0.1])
    alpha = prior * np.array([1 - scores[0], scores[0]])
    alpha = alpha / max(alpha.sum(), 1e-12)
    for p in scores[1:]:
        emit = np.array([1 - p, p])
        alpha = (alpha @ a) * emit
        alpha = alpha / max(alpha.sum(), 1e-12)
    return float(1.0 - alpha[0])


def player_metrics(pt: pd.DataFrame, col: str) -> dict:
    m = binary_metrics(pt["label"].to_numpy(), pt[col].to_numpy())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt", required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    g = pd.read_parquet(args.gbdt)
    t = pd.read_parquet(args.transformer)
    df = g.merge(t[["row", "score"]].rename(columns={"score": "score_t"}),
                 on="row", how="inner", validate="one_to_one")
    df = df.rename(columns={"score": "score_g"})
    df["fold"] = df["fold_x"] if "fold_x" in df else df["fold"]
    df["p_g"] = crossfit_isotonic(df, "score_g")
    df["p_t"] = crossfit_isotonic(df, "score_t")
    df["p_ens"] = 0.5 * (df["p_g"] + df["p_t"])

    feats = pd.read_parquet(args.features, columns=["row", "map", "weapon_idx", "dist_final"])
    df = df.merge(feats, on="row", how="left")

    player = agg_table(df, "p_ens")
    player["hmm"] = [
        hmm_player_score(group.sort_index()["p_ens"].to_numpy())
        for _, group in df.groupby(["match_key", "player_name"])
    ]

    # Player-level metrics for several aggregators.
    results = {"player": {}}
    for col in ["mean", "max", "top3", "noisy_or", "hmm"]:
        m = player_metrics(player, col)
        results["player"][col] = m

    # Per-fold variability for the two primary aggregators.
    fold_rows = []
    for fold, fg in df.groupby("fold"):
        p = agg_table(fg, "p_ens")
        p["hmm"] = [
            hmm_player_score(x.sort_index()["p_ens"].to_numpy())
            for _, x in fg.groupby(["match_key", "player_name"])
        ]
        for col in ["mean", "top3", "noisy_or", "hmm"]:
            m = player_metrics(p, col)
            fold_rows.append({"fold": int(fold), "agg": col, "auc": m["auc"],
                              "recall_fpr1": m["recall_at_fpr1"]})
    pd.DataFrame(fold_rows).to_csv(out / "player_fold_metrics.csv", index=False)

    df[["row", "match_key", "player_name", "label", "fold", "p_g", "p_t",
        "p_ens", "map", "weapon_idx", "dist_final"]].to_parquet(
        out / "kill_table.parquet", index=False
    )

    # Kill-level calibration residuals by decile.
    df["bin"] = pd.qcut(df["p_ens"], 10, labels=False, duplicates="drop")
    cal = df.groupby("bin").agg(n=("label", "size"), pred=("p_ens", "mean"),
                                obs=("label", "mean")).reset_index()
    cal["residual"] = cal["obs"] - cal["pred"]
    cal.to_csv(out / "calibration_by_decile.csv", index=False)

    # Error rates by map / weapon / distance / kills-per-player at a
    # threshold selected for ~1% FPR on the OOF kill scores.
    fpr, tpr, thr = roc_curve(df["label"], df["p_ens"])
    idx = np.where(fpr <= 0.01)[0][-1]
    cutoff = float(thr[idx])
    df["pred"] = (df["p_ens"] >= cutoff).astype(int)
    subgroups = {
        "map": df["map"],
        "weapon": df["weapon_idx"],
        "distance": pd.qcut(df["dist_final"], 4, labels=False, duplicates="drop"),
    }
    sub_rows = []
    for name, keys in subgroups.items():
        tmp = df.assign(group=keys)
        for gkey, gdf in tmp.groupby("group"):
            y = gdf["label"].to_numpy()
            pred = gdf["pred"].to_numpy()
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            sub_rows.append({
                "group_type": name, "group": str(gkey), "n": len(gdf),
                "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            })
    pd.DataFrame(sub_rows).to_csv(out / "subgroup_errors.csv", index=False)

    # Cross-fitted player threshold: choose the cutoff on the other folds,
    # apply it to the held-out fold.  This avoids the optimistic threshold
    # that comes from picking it on the same player table.
    crossfit = {}
    for col in ["mean", "hmm"]:
        rows = []
        for fold in sorted(player["fold"].unique()):
            tr = player[player.fold != fold]
            te = player[player.fold == fold]
            fpr, tpr, thr = roc_curve(tr["label"], tr[col])
            ok = np.where(fpr <= 0.01)[0]
            cutoff = float(thr[ok[-1]]) if len(ok) else 1.0
            pred = (te[col] >= cutoff).astype(int)
            y = te["label"].to_numpy()
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            rows.append({
                "fold": int(fold), "cutoff": cutoff,
                "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
                "n_pos": tp + fn, "n_neg": fp + tn,
            })
        crossfit[col] = rows
    results["player_crossfit_threshold"] = crossfit

    # Player-level false positives/negatives at the global mean cutoff
    # (diagnostic only; the cross-fitted table above is the honest one).
    fpr, tpr, thr = roc_curve(player["label"], player["mean"])
    idx = np.where(fpr <= 0.01)[0][-1]
    pcut = float(thr[idx])
    player["pred"] = (player["mean"] >= pcut).astype(int)
    player["error"] = np.select(
        [(player.pred == 1) & (player.label == 0),
         (player.pred == 0) & (player.label == 1)],
        ["FP", "FN"], default="correct",
    )
    player.to_parquet(out / "player_table.parquet", index=False)

    results["kill_cutoff_at_1pct_fpr"] = cutoff
    results["player_cutoff_at_1pct_fpr"] = pcut
    results["player_error_counts"] = player["error"].value_counts().to_dict()
    results["calibration"] = cal.to_dict(orient="records")
    results["subgroups"] = sub_rows
    with open(out / "cascade_residuals.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "player": {k: {kk: round(vv, 4) for kk, vv in v.items()
                       if isinstance(vv, (int, float))}
                   for k, v in results["player"].items()},
        "player_cutoff": pcut,
        "player_errors": results["player_error_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
