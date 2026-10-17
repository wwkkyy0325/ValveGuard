"""Geometric and tail analyses for the detector.

- PCA scree / participation ratio (intrinsic dimension)
- principal angles between human and cheater subspaces
- PCA-whitened Mahalanobis distance as a one-class score
- EVT (generalized Pareto) tail p-values on player scores
- cross-fitted conformal p-values at the player level
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from valveguard.metrics import binary_metrics, player_scores
from sklearn.metrics import average_precision_score


def anomaly_metrics(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= 0.01
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "recall_at_fpr1": float(tpr[ok].max()) if ok.any() else 0.0,
        "fpr_at_tpr95": float(fpr[np.searchsorted(tpr, 0.95)]) if np.searchsorted(tpr, 0.95) < len(fpr) else 1.0,
    }
from valveguard.scripts.ablation_features import META_DROP


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    index = pd.read_parquet("data/processed/cw256/index.parquet")
    meta = index[["row", "match_key", "player_name", "kill_idx", "source", "label"]]
    df = feats.merge(meta, on="row", how="inner")
    df = df[~((df.source == "cdata") & (df.label == 0))].reset_index(drop=True)
    weapon = [c for c in df.columns if c.startswith("weapon_")]
    outcome = ["headshot","through_smoke","wallbang","midair","shot_count","victim_noise","flashed_at_kill","victim_health"]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    cols = [c for c in df.columns if c not in META_DROP and c not in {"fold"} and df[c].notna().any()
            and c not in set(weapon+outcome+motion)]
    X = df[cols].replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median()).to_numpy(float)
    y = df.label.to_numpy(int)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    # PCA scree / participation ratio.
    pca = PCA(n_components=min(30, Z.shape[1]), random_state=0).fit(Z)
    ev = pca.explained_variance_ratio_
    pr = float((ev.sum() ** 2) / np.sum(ev ** 2))
    # Principal angles between human/cheater top subspaces.
    ph = PCA(n_components=10, random_state=0).fit(Z[y == 0])
    pc = PCA(n_components=10, random_state=0).fit(Z[y == 1])
    s = np.linalg.svd(ph.components_ @ pc.components_.T, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(s, -1, 1)))
    # Mahalanobis to human mean/cov.
    mu = Z[y == 0].mean(axis=0)
    cov = np.cov(Z[y == 0], rowvar=False) + np.eye(Z.shape[1]) * 1e-3
    inv = np.linalg.pinv(cov)
    d2 = np.einsum("ij,jk,ik->i", Z - mu, inv, Z - mu)
    mah = np.sqrt(np.maximum(d2, 0))
    kill_mah = anomaly_metrics(y, mah)
    players_mah = player_scores(df, mah, method="mean")
    player_mah = anomaly_metrics(players_mah.label, players_mah.score)

    # Player scores + folds for EVT/conformal.
    sc = pd.read_parquet(args.scores)
    if "match_key" not in sc.columns:
        sc = sc.merge(meta[["row","match_key","player_name"]], on="row", how="inner")
    pl = player_scores(sc, sc.score.to_numpy(), method="mean")
    pl = pl.merge(sc.groupby(["match_key","player_name"])["fold"].first().reset_index(),
                  on=["match_key","player_name"], how="left")

    # EVT tail on human player scores.
    human = pl[pl.label == 0].score.to_numpy()
    u = float(np.quantile(human, 0.90))
    excess = human[human > u] - u
    c, loc, scale = genpareto.fit(excess, floc=0)
    def evt_score(x):
        x = np.asarray(x, float)
        p = np.where(x > u, genpareto.sf(x - u, c, loc=0, scale=scale), 1.0)
        return -np.log(np.clip(p, 1e-12, 1.0))
    pl["evt"] = evt_score(pl.score.to_numpy())
    evt_metrics = anomaly_metrics(pl.label, pl.evt)

    # Cross-fitted conformal p-values on player scores.
    pl["conformal"] = np.nan
    for fold in sorted(pl.fold.dropna().unique()):
        calib = pl[(pl.fold != fold) & (pl.label == 0)].score.to_numpy()
        te = pl.fold == fold
        if len(calib) < 50:
            continue
        x = pl.loc[te, "score"].to_numpy()
        # p = (1 + #{calib >= x}) / (n + 1); anomaly = -log p
        p = (1 + (calib[None, :] >= x[:, None]).sum(axis=1)) / (len(calib) + 1)
        pl.loc[te, "conformal"] = -np.log(np.clip(p, 1e-12, 1.0))
    conf_metrics = anomaly_metrics(pl.loc[pl.conformal.notna(), "label"],
                                  pl.loc[pl.conformal.notna(), "conformal"])

    results = {
        "pca": {"explained_variance_ratio": ev.tolist(), "participation_ratio": pr},
        "principal_angles_deg": angles.tolist(),
        "mahalanobis": {"kill": kill_mah, "player": player_mah,
                        "player_auc": player_mah["auc"],
                        "player_recall_fpr1": player_mah["recall_at_fpr1"]},
        "evt": {"threshold": u, "gpd_c": float(c), "gpd_scale": float(scale),
                "metrics": evt_metrics},
        "conformal": {"metrics": conf_metrics},
    }
    with open(out/"math_analysis.json","w",encoding="utf-8") as f: json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
