"""Paired match-grouped bootstrap between two OOF prediction files.

Both files must cover the same windows and folds.  Reports the delta in
player-level AUC and recall@FPR=1% for B minus A, with 95% CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import player_scores


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = pd.read_parquet(args.a)[["row", "match_key", "player_name", "label", "score"]]
    b = pd.read_parquet(args.b)[["row", "match_key", "player_name", "label", "score"]]
    d = a.merge(b, on="row", suffixes=("_a", "_b"))
    if not (d["match_key_a"] == d["match_key_b"]).all():
        raise AssertionError("files do not cover the same windows")
    if not (d["label_a"] == d["label_b"]).all():
        raise AssertionError("label mismatch")

    pa = player_scores(d.rename(columns={"match_key_a": "match_key",
                                         "player_name_a": "player_name",
                                         "label_a": "label"}), d["score_a"].to_numpy())
    pb = player_scores(d.rename(columns={"match_key_b": "match_key",
                                         "player_name_b": "player_name",
                                         "label_b": "label"}), d["score_b"].to_numpy())
    key = ["match_key", "player_name"]
    p = pa.merge(pb, on=key, suffixes=("_a", "_b"))
    y = p["label_a"].to_numpy()
    matches = p["match_key"].to_numpy()
    sa = p["score_a"].to_numpy()
    sb = p["score_b"].to_numpy()

    rng = np.random.default_rng(args.seed)
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

    res = {
        "n_players": int(len(p)),
        "a_auc": float(roc_auc_score(y, sa)),
        "b_auc": float(roc_auc_score(y, sb)),
        "a_recall_fpr1": recall_at_fpr(y, sa),
        "b_recall_fpr1": recall_at_fpr(y, sb),
        "delta_auc": ci(aucs),
        "delta_recall_fpr1": ci(recs),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
