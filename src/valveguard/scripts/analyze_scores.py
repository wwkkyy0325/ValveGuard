"""Compare GBDT and Transformer, calibrate, ensemble, and aggregate to players.

This is deliberately cheap: it reuses saved predictions and does not train
anything.  It answers the deployment question (player-level, low FPR) rather
than only kill-level AUC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, roc_curve
from sklearn.preprocessing import label_binarize

from valveguard.metrics import binary_metrics, player_scores


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    z = _logit(p)

    def loss(log_t):
        t = np.exp(log_t)
        return log_loss(y, _sigmoid(z / t))

    res = minimize_scalar(loss, bounds=(-2.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target: float = 0.01):
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    if len(ok) == 0:
        return 1.0, 0.0, 0.0
    i = ok[-1]
    return float(thr[i]), float(tpr[i]), float(fpr[i])


def player_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.groupby(["match_key", "player_name"], as_index=False).agg(
        label=("label", "max"),
        n_kills=("label", "size"),
        mean=(score_col, "mean"),
        max=(score_col, "max"),
    )
    top3 = (
        df.sort_values(score_col, ascending=False)
          .groupby(["match_key", "player_name"])[score_col]
          .apply(lambda x: float(np.mean(x.head(3))))
          .rename("top3")
          .reset_index()
    )
    return out.merge(top3, on=["match_key", "player_name"], how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt", required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    g = pd.read_parquet(args.gbdt)
    t = pd.read_parquet(args.transformer)
    df = g.merge(
        t[["row", "score"]].rename(columns={"score": "score_t"}),
        on="row", how="inner", validate="one_to_one",
    )
    df = df.rename(columns={"score": "score_g"})
    if len(df) != len(g) or len(df) != len(t):
        print(f"warning: merged {len(df)} of gbdt={len(g)} transformer={len(t)}")
    assert np.allclose(df["label"], df["label"])

    val = df[df.split == "val"].copy()
    test = df[df.split == "test"].copy()

    # Temperature scaling on validation, applied to test.
    t_g = fit_temperature(val["label"].to_numpy(), val["score_g"].to_numpy())
    t_t = fit_temperature(val["label"].to_numpy(), val["score_t"].to_numpy())
    df["cal_g"] = _sigmoid(_logit(df["score_g"]) / t_g)
    df["cal_t"] = _sigmoid(_logit(df["score_t"]) / t_t)
    # Rank average is robust to different calibration scales.
    df["rank_g"] = df.groupby("split")["score_g"].rank(pct=True)
    df["rank_t"] = df.groupby("split")["score_t"].rank(pct=True)
    df["ens_prob"] = 0.5 * (df["cal_g"] + df["cal_t"])
    df["ens_rank"] = 0.5 * (df["rank_g"] + df["rank_t"])
    val = df[df.split == "val"].copy()
    test = df[df.split == "test"].copy()

    results = {"temperatures": {"gbdt": t_g, "transformer": t_t}, "kill": {}, "player": {}}
    for col in ["score_g", "score_t", "cal_g", "cal_t", "ens_prob", "ens_rank"]:
        y = test["label"].to_numpy()
        s = test[col].to_numpy()
        results["kill"][col] = binary_metrics(y, s)
        pt = player_table(test, col)
        results["player"][col] = {
            "mean": binary_metrics(pt["label"].to_numpy(), pt["mean"].to_numpy()),
            "max": binary_metrics(pt["label"].to_numpy(), pt["max"].to_numpy()),
            "top3": binary_metrics(pt["label"].to_numpy(), pt["top3"].to_numpy()),
        }
        # Threshold chosen on validation players at FPR=1%.
        pv = player_table(val, col)
        thr, _, _ = threshold_at_fpr(pv["label"].to_numpy(), pv["mean"].to_numpy(), 0.01)
        pred = pt["mean"].to_numpy() >= thr
        tp = int((pred & (pt["label"].to_numpy() == 1)).sum())
        fp = int((pred & (pt["label"].to_numpy() == 0)).sum())
        pos = int((pt["label"].to_numpy() == 1).sum())
        neg = int((pt["label"].to_numpy() == 0).sum())
        results["player"][col]["mean_at_val_fpr1"] = {
            "threshold": thr,
            "test_recall": tp / max(pos, 1),
            "test_fpr": fp / max(neg, 1),
            "n_players": int(len(pt)),
        }

    pt = player_table(test, "ens_prob")
    pt.to_parquet(out / "player_scores.parquet", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
