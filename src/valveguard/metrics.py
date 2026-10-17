"""Evaluation metrics with an anti-cheat operating point in mind."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def fpr_at_tpr(y_true, y_score, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def recall_at_fpr(y_true, y_score, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= target_fpr
    if not np.any(ok):
        return 0.0
    return float(np.max(tpr[ok]))


def ece(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    total = len(y_true)
    out = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        out += (mask.sum() / total) * abs(acc - conf)
    return float(out)


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr_at_tpr95": fpr_at_tpr(y_true, y_score, 0.95),
        "recall_at_fpr1": recall_at_fpr(y_true, y_score, 0.01),
        "recall_at_fpr01": recall_at_fpr(y_true, y_score, 0.001),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": ece(y_true, y_score),
    }


def player_scores(meta: pd.DataFrame, scores: np.ndarray,
                  method: str = "mean") -> pd.DataFrame:
    df = meta[["match_key", "player_name", "label"]].copy()
    df["score"] = np.asarray(scores, dtype=float)
    agg = {"score": method, "label": "max"}
    out = df.groupby(["match_key", "player_name"], as_index=False).agg(agg)
    return out
