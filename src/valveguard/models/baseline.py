"""Cheap handcrafted-feature baseline.

Purpose: validate the labels, the split, and the feature extraction before
training a sequence model.  If this baseline is near chance, no transformer
will fix the data problem.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ..data.split import assign_match_splits, player_key
from ..data.tar_index import load_cache
from ..features.summarize import summarize
from ..metrics import binary_metrics, player_scores

_WORKER_MM = None
_WORKER_NAMES = None


def _init_worker(cache_dir: str, names: list[str]) -> None:
    global _WORKER_MM, _WORKER_NAMES
    _WORKER_MM = np.load(Path(cache_dir) / "windows.npy", mmap_mode="r")
    _WORKER_NAMES = names


def _summarize_row(i: int) -> dict:
    window = pd.DataFrame(_WORKER_MM[i], columns=_WORKER_NAMES)
    feat = summarize(window)
    feat["row"] = int(i)
    return feat


def build_summary(cache_dir: str | Path, out_path: str | Path | None = None,
                  rows: np.ndarray | None = None,
                  n_jobs: int | None = None) -> pd.DataFrame:
    mm, meta, names = load_cache(cache_dir)
    if rows is None:
        rows = np.arange(len(meta))
    rows = np.asarray(rows)

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)
    if n_jobs <= 1 or len(rows) < 200:
        records = []
        for i in rows:
            window = pd.DataFrame(mm[i], columns=names)
            feat = summarize(window)
            feat["row"] = int(i)
            records.append(feat)
            if (len(records) % 5000) == 0:
                print(f"summarized {len(records)}/{len(rows)}", flush=True)
    else:
        print(f"summarizing {len(rows)} windows with {n_jobs} workers", flush=True)
        records = []
        # Spawn is the only start method available on Windows; use the
        # platform default context explicitly.
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(str(cache_dir), names),
        ) as pool:
            for feat in pool.imap_unordered(_summarize_row, rows, chunksize=128):
                records.append(feat)
                if (len(records) % 5000) == 0:
                    print(f"summarized {len(records)}/{len(rows)}", flush=True)
        records.sort(key=lambda r: r["row"])

    df = pd.DataFrame(records)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"row", "map", "weapon_idx", "label", "split", "match_key",
            "player_name", "player_key", "source", "match_idx", "player",
            "kill_idx", "augmented"}
    cols = [c for c in df.columns if c not in drop]
    # Drop columns that are entirely missing.
    cols = [c for c in cols if df[c].notna().any()]
    return cols


def run_baseline(cache_dir: str | Path, out_dir: str | Path,
                 seed: int = 42, limit: int | None = None,
                 clean_labels: bool = True,
                 n_jobs: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mm, meta, names = load_cache(cache_dir)
    if limit is not None and limit < len(meta):
        # Stratified pilot sample so both classes are represented.
        rng = np.random.default_rng(seed)
        idx = []
        for label, g in meta.groupby("label"):
            n = max(1, int(round(limit * len(g) / len(meta))))
            idx.extend(rng.choice(g.index.to_numpy(), size=min(n, len(g)), replace=False))
        rows = np.array(sorted(idx))
    else:
        rows = np.arange(len(meta))

    feat_name = "summary_features.parquet" if limit is None else f"summary_features_limit{limit}.parquet"
    feat_path = out_dir / feat_name
    if feat_path.exists():
        feats = pd.read_parquet(feat_path)
        if "row" not in feats.columns or len(feats) != len(rows):
            print("cached summary does not match requested rows; rebuilding", flush=True)
            feats = build_summary(cache_dir, feat_path, rows=rows, n_jobs=n_jobs)
    else:
        feats = build_summary(cache_dir, feat_path, rows=rows, n_jobs=n_jobs)

    data = feats.merge(
        meta[["row", "match_key", "player_name", "label", "source",
              "match_idx", "player", "kill_idx"]],
        on="row", how="inner",
    )
    # The dataset card reports only ~55.6% precision for "not cheater"
    # labels inside matches that contain a cheater.  Those windows carry
    # the cdata prefix; exclude them from the negative class by default.
    if clean_labels:
        before = len(data)
        data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
        print(f"clean labels: kept {len(data)}/{before} windows", flush=True)

    data = assign_match_splits(data, seed=seed)
    data["player_key"] = player_key(data)

    cols = feature_columns(data)
    train = data[data.split == "train"]
    val = data[data.split == "val"]
    test = data[data.split == "test"]

    def fit_xy(d):
        X = d[cols].replace([np.inf, -np.inf], np.nan)
        return X, d["label"].to_numpy().astype(int)

    Xtr, ytr = fit_xy(train)
    Xval, yval = fit_xy(val)
    Xte, yte = fit_xy(test)

    # Balanced sample weights.
    pos = max(1, int(ytr.sum()))
    neg = max(1, int(len(ytr) - ytr.sum()))
    w_pos = len(ytr) / (2.0 * pos)
    w_neg = len(ytr) / (2.0 * neg)
    sw = np.where(ytr == 1, w_pos, w_neg)

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=0,
    )
    clf.fit(Xtr, ytr, sample_weight=sw)

    results = {}
    preds = {}
    for name, X, y, d in [("val", Xval, yval, val), ("test", Xte, yte, test)]:
        p = clf.predict_proba(X)[:, 1]
        preds[name] = p
        m = binary_metrics(y, p, threshold=0.5)
        ps = player_scores(d, p, method="mean")
        m["player_auc_mean"] = binary_metrics(
            ps["label"].to_numpy(), ps["score"].to_numpy()
        )["auc"]
        ps_max = player_scores(d, p, method="max")
        m["player_auc_max"] = binary_metrics(
            ps_max["label"].to_numpy(), ps_max["score"].to_numpy()
        )["auc"]
        results[name] = m

    # Permutation importance on a manageable validation subset.
    rng = np.random.default_rng(seed)
    if len(val) > 5000:
        idx = rng.choice(len(val), size=5000, replace=False)
        Xi = Xval.iloc[idx]
        yi = yval[idx]
    else:
        Xi, yi = Xval, yval
    try:
        imp = permutation_importance(
            clf, Xi, yi, scoring="roc_auc", n_repeats=3,
            random_state=seed, n_jobs=1,
        )
        imp_df = pd.DataFrame({
            "feature": cols,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
        }).sort_values("importance_mean", ascending=False)
        imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
        results["top_features"] = imp_df.head(30).to_dict(orient="records")
    except Exception as exc:  # pragma: no cover - diagnostic only
        results["importance_error"] = repr(exc)

    results["n_features"] = len(cols)
    results["n_rows"] = int(len(data))
    results["clean_labels"] = bool(clean_labels)
    results["split_counts"] = data.groupby("split").size().to_dict()
    results["label_counts"] = data.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save predictions for error analysis.
    pred_df = data[["row", "match_key", "player_name", "kill_idx", "label",
                    "split", "source", "match_idx", "player"]].copy()
    pred_df["score"] = np.nan
    for name, p in preds.items():
        pred_df.loc[pred_df.split == name, "score"] = p
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

    return results
