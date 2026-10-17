"""Systematic CPU experiment runner.

Every variant uses the same 5 match-disjoint folds and the same player-level
metrics.  A variant replaces the champion only if it improves player AUC
and/or recall@FPR=1% with non-overlapping bootstrap intervals; otherwise the
current baseline stays.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.kernel_approximation import Nystroem

from valveguard.metrics import binary_metrics, player_scores
from valveguard.scripts.ablation_features import META_DROP


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def bootstrap_player(y, s, groups, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    idx = np.searchsorted(keys, groups)
    order = np.argsort(idx)
    idx_s = idx[order]
    y_s = np.asarray(y)[order]
    s_s = np.asarray(s)[order]
    starts = np.searchsorted(idx_s, np.arange(len(keys) + 1))
    aucs, recs = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([np.arange(starts[j], starts[j + 1]) for j in pick])
        yy, ss = y_s[rows], s_s[rows]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, ss))
        recs.append(recall_at_fpr(yy, ss))
    def ci(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return ci(aucs), ci(recs)


def build_df(features_path: str) -> pd.DataFrame:
    feats = pd.read_parquet(features_path)
    index = pd.read_parquet("data/processed/cw256/index.parquet")
    meta = index[["row", "match_key", "player_name", "kill_idx", "source", "label"]]
    df = feats.merge(meta, on="row", how="inner")
    df = df[~((df["source"] == "cdata") & (df["label"] == 0))].copy()
    df = df.reset_index(drop=True)
    folds = pd.read_parquet("reports/cv_gbdt/cv_predictions.parquet",
                            columns=["row", "fold"])
    df = df.merge(folds, on="row", how="inner")
    return df


def feature_groups(df: pd.DataFrame):
    weapon = [c for c in df.columns if c.startswith("weapon_")]
    outcome = ["headshot", "through_smoke", "wallbang", "midair",
               "shot_count", "victim_noise", "flashed_at_kill", "victim_health"]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    base = [c for c in df.columns if c not in META_DROP
            and c not in {"fold"}
            and df[c].notna().any()
            and c not in set(weapon + outcome + motion)]
    return base, weapon, outcome, motion


def model_pipeline(kind: str, n_features: int, seed: int):
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    if kind == "logreg":
        return Pipeline([("imp", imputer), ("sc", scaler), ("clf", logistic)])
    if kind == "logreg_pca32":
        return Pipeline([
            ("imp", imputer), ("sc", scaler),
            ("pca", PCA(n_components=32, random_state=seed)),
            ("clf", logistic),
        ])
    if kind == "logreg_rff":
        return Pipeline([
            ("imp", imputer), ("sc", scaler),
            ("rff", Nystroem(gamma=1.0 / max(n_features, 1), n_components=300,
                             random_state=seed)),
            ("clf", logistic),
        ])
    if kind == "logreg_poly2":
        return Pipeline([
            ("imp", imputer), ("sc", scaler),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True,
                                        include_bias=False)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
        ])
    if kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
    if kind == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed,
        )
    if kind == "gbdt15":
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
            l2_regularization=2.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed,
        )
    if kind == "gbdt63":
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=63,
            l2_regularization=5.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed,
        )
    raise ValueError(kind)


def run_variant(df: pd.DataFrame, cols: list[str], kind: str, name: str,
                seed: int = 42, n_boot: int = 300) -> dict:
    oof = np.full(len(df), np.nan)
    t0 = time.time()
    for fold in sorted(df.fold.unique()):
        tr = df.fold != fold
        te = df.fold == fold
        Xtr = df.loc[tr, cols]
        Xte = df.loc[te, cols]
        ytr = df.loc[tr, "label"].to_numpy(int)
        if kind.startswith("gbdt"):
            clf = model_pipeline(kind, len(cols), seed + fold)
            pos = max(1, int(ytr.sum())); neg = max(1, len(ytr) - pos)
            sw = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg))
            clf.fit(Xtr, ytr, sample_weight=sw)
        else:
            clf = model_pipeline(kind, len(cols), seed + fold)
            clf.fit(Xtr, ytr)
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(Xte)[:, 1]
        else:
            p = clf.decision_function(Xte)
        oof[te.to_numpy()] = p

    kill = binary_metrics(df["label"], oof)
    players = player_scores(df, oof, method="mean")
    player = binary_metrics(players["label"], players["score"])
    # group bootstrap by match
    pm = players.merge(
        df.groupby(["match_key", "player_name"])["fold"].first().reset_index(),
        on=["match_key", "player_name"], how="left",
    )
    auc_ci, rec_ci = bootstrap_player(
        pm["label"].to_numpy(), pm["score"].to_numpy(),
        pm["match_key"].to_numpy(), n_boot=n_boot, seed=seed,
    )
    out = {
        "name": name, "model": kind, "n_features": len(cols),
        "kill_auc": kill["auc"], "kill_ap": kill["ap"],
        "kill_recall_fpr1": kill["recall_at_fpr1"],
        "player_auc": player["auc"], "player_ap": player["ap"],
        "player_recall_fpr1": player["recall_at_fpr1"],
        "player_fpr_tpr95": player["fpr_at_tpr95"],
        "player_ece": player["ece"],
        "player_auc_ci": auc_ci, "player_recall_ci": rec_ci,
        "seconds": time.time() - t0,
    }
    pred = df[["row", "match_key", "player_name", "label", "fold"]].copy()
    pred["score"] = oof
    pred["variant"] = name
    return out, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oof").mkdir(exist_ok=True)
    df = build_df(args.features)
    base, weapon, outcome, motion = feature_groups(df)

    # A compact, interpretable grid: model family x feature group, plus a
    # few targeted hyperparameter/kernel variants.
    feature_sets = {
        "base": base,
        "base_weapon": base + weapon,
        "base_outcome": base + outcome,
        "base_context": base + weapon + outcome,
        "base_motion": base + motion,
    }
    variants = []
    for fs, cols in feature_sets.items():
        for kind in ["gbdt", "logreg", "logreg_pca32", "logreg_rff", "extratrees"]:
            variants.append((f"{fs}__{kind}", fs, cols, kind))
    for kind in ["gbdt15", "gbdt63"]:
        variants.append((f"base__{kind}", "base", base, kind))
    # Polynomial interactions on the top-20 GBDT features only; full 59
    # features would create 1.7k columns and overfit.
    imp_path = Path("reports/baseline_clean/feature_importance.csv")
    if imp_path.exists():
        top = pd.read_csv(imp_path)["feature"].head(20).tolist()
        top = [c for c in top if c in base]
        if len(top) >= 10:
            variants.append(("base__logreg_poly2_top20", "base", top, "logreg_poly2"))

    if args.variants != "all":
        keep = set(args.variants.split(","))
        variants = [v for v in variants if v[0] in keep]

    rows = []
    for name, fs, cols, kind in variants:
        print(f"running {name} ({kind}, {len(cols)} features)", flush=True)
        try:
            res, pred = run_variant(df, cols, kind, name, seed=args.seed)
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
            continue
        rows.append(res)
        pred.to_parquet(out / "oof" / f"{name}.parquet", index=False)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in res.items()
                          if k not in {"seconds", "player_auc_ci", "player_recall_ci"}}),
              flush=True)

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "results.csv", index=False)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(res_df.sort_values("player_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
