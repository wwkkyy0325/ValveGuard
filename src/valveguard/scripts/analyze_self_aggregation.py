"""Compare player aggregation rules on self-baseline kill scores.

The self-baseline model ranks individual kills much better than the champion
(kill AUC ~0.92 vs 0.85) but the player mean washes the gain out.  This checks
whether max / top-k aggregation recovers it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import binary_metrics


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(args.oof)

    methods = ["mean", "max", "top2", "top3", "top5"]
    rows, player_frames = [], {}
    for how in methods:
        if how == "mean":
            score = d.groupby(["match_key", "player_name"])["score"].transform("mean")
        elif how == "max":
            score = d.groupby(["match_key", "player_name"])["score"].transform("max")
        else:
            k = int(how[3:])
            score = d.groupby(["match_key", "player_name"])["score"].transform(
                lambda x: np.mean(np.sort(x.to_numpy())[-k:])
            )
        p = pd.DataFrame({
            "match_key": d["match_key"],
            "player_name": d["player_name"],
            "label": d["label"],
            "score": score,
        }).drop_duplicates(["match_key", "player_name"])
        player_frames[how] = p.set_index(["match_key", "player_name"])["score"]
        m = binary_metrics(p["label"].to_numpy(), p["score"].to_numpy())
        rows.append({"agg": how, "player_auc": m["auc"], "player_ap": m["ap"],
                     "recall_fpr1": m["recall_at_fpr1"],
                     "fpr_tpr95": m["fpr_at_tpr95"]})
    table = pd.DataFrame(rows)
    table.to_csv(out / "aggregation.csv", index=False)
    print(table.to_string(index=False))

    base = player_frames["mean"]
    labels = d.groupby(["match_key", "player_name"])["label"].max()
    matches = labels.index.get_level_values(0).to_numpy()
    y = labels.to_numpy()
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(matches)

    def delta(other):
        s_other = player_frames[other].loc[base.index].to_numpy()
        s_base = base.to_numpy()
        aucs, recs = [], []
        for _ in range(1000):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(matches == m)[0] for m in pick])
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            aucs.append(roc_auc_score(yy, s_other[idx]) - roc_auc_score(yy, s_base[idx]))
            recs.append(recall_at_fpr(yy, s_other[idx]) - recall_at_fpr(yy, s_base[idx]))

        def ci(x):
            x = np.asarray(x)
            return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

        return {"delta_auc": ci(aucs), "delta_recall_fpr1": ci(recs)}

    res = {f"{m} vs mean": delta(m) for m in methods if m != "mean"}
    with open(out / "aggregation_paired.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
