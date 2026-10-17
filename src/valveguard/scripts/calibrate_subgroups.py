"""Cross-fitted per-weapon calibration and residual-FPR check.

The global 1% FPR threshold produced ~4.8% FPR on the semi-auto rifle
group.  This script calibrates each weapon group on the other folds and
measures whether the residual bias shrinks without destroying ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve

from valveguard.metrics import binary_metrics
from valveguard.scripts.analyze_cascade_residuals import (
    agg_table,
    hmm_player_score,
)


def crossfit_global(df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for fold in sorted(df.fold.unique()):
        tr = df.fold != fold
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(df.loc[tr, "p_ens"], df.loc[tr, "label"])
        out[(~tr).to_numpy()] = iso.predict(df.loc[~tr, "p_ens"])
    return np.clip(out, 1e-4, 1 - 1e-4)


def crossfit_weapon(df: pd.DataFrame, min_n: int = 200) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for fold in sorted(df.fold.unique()):
        tr_all = df.fold != fold
        te_all = ~tr_all
        iso_g = IsotonicRegression(out_of_bounds="clip")
        iso_g.fit(df.loc[tr_all, "p_ens"], df.loc[tr_all, "label"])
        global_pred = iso_g.predict(df.loc[te_all, "p_ens"])
        out[te_all.to_numpy()] = global_pred
        for w in df.weapon_idx.dropna().unique():
            tr = tr_all & (df.weapon_idx == w)
            te = te_all & (df.weapon_idx == w)
            if tr.sum() < min_n or te.sum() == 0:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(df.loc[tr, "p_ens"], df.loc[tr, "label"])
            out[te.to_numpy()] = iso.predict(df.loc[te, "p_ens"])
    return np.clip(out, 1e-4, 1 - 1e-4)


def cutoff_at_fpr(y, s, fpr_target=0.01):
    fpr, _, thr = roc_curve(y, s)
    ok = np.where(fpr <= fpr_target)[0]
    return float(thr[ok[-1]]) if len(ok) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill-table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.kill_table).reset_index(drop=True)
    df["p_global"] = crossfit_global(df)
    df["p_sub"] = crossfit_weapon(df)

    results = {"kill": {}, "player": {}, "weapon_fpr": {}}
    for col in ["p_ens", "p_global", "p_sub"]:
        results["kill"][col] = binary_metrics(df["label"], df[col])

    for col in ["p_ens", "p_sub"]:
        player = agg_table(df, col)
        player["hmm"] = [
            hmm_player_score(g.sort_index()[col].to_numpy())
            for _, g in df.groupby(["match_key", "player_name"])
        ]
        results["player"][col] = {
            "mean": binary_metrics(player["label"], player["mean"]),
            "hmm": binary_metrics(player["label"], player["hmm"]),
        }

        # Global cutoff at ~1% FPR, then residual FPR by weapon group.
        cut = cutoff_at_fpr(df["label"], df[col])
        per_w = []
        for w, g in df.groupby("weapon_idx", dropna=True):
            y = g["label"].to_numpy()
            pred = (g[col] >= cut).to_numpy()
            fp = int(((pred == 1) & (y == 0)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            tp = int(((pred == 1) & (y == 1)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            per_w.append({
                "weapon": int(w), "n": int(len(g)),
                "recall": tp / max(tp + fn, 1),
                "fpr": fp / max(fp + tn, 1),
            })
        results["weapon_fpr"][col] = {"cutoff": cut, "groups": per_w}

    # Cross-fitted player threshold for p_sub mean/HMM.
    results["player_crossfit_threshold"] = {}
    for agg in ["mean", "hmm"]:
        player = agg_table(df, "p_sub")
        player["hmm"] = [
            hmm_player_score(g.sort_index()["p_sub"].to_numpy())
            for _, g in df.groupby(["match_key", "player_name"])
        ]
        rows = []
        for fold in sorted(player.fold.unique()):
            tr = player[player.fold != fold]
            te = player[player.fold == fold]
            cut = cutoff_at_fpr(tr["label"], tr[agg])
            pred = (te[agg] >= cut).astype(int)
            y = te["label"].to_numpy()
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            rows.append({
                "fold": int(fold), "cutoff": cut,
                "recall": tp / max(tp + fn, 1),
                "fpr": fp / max(fp + tn, 1),
                "n_pos": tp + fn, "n_neg": fp + tn,
            })
        results["player_crossfit_threshold"][agg] = rows

    with open(out / "subgroup_calibration.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "kill": {k: {"auc": v["auc"], "ap": v["ap"],
                     "recall_at_fpr1": v["recall_at_fpr1"],
                     "ece": v["ece"]}
                 for k, v in results["kill"].items()},
        "player": {k: {a: {"auc": v[a]["auc"],
                           "recall_at_fpr1": v[a]["recall_at_fpr1"],
                           "ece": v[a]["ece"]}
                       for a in v}
                   for k, v in results["player"].items()},
        "weapon_fpr": results["weapon_fpr"],
        "player_crossfit_threshold": results["player_crossfit_threshold"],
    }, indent=2))


if __name__ == "__main__":
    main()
