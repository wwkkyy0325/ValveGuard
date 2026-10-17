"""Cross-fitted rank-quartile thresholds.

High-rank players carry a 4.8% FPR at the global 1% cutoff.  This checks
whether rank-specific cutoffs equalize FPR, while keeping the threshold
selection out of the evaluation fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def cutoff(y, s, target=0.01):
    fpr, _, thr = roc_curve(y, s)
    ok = np.where(fpr <= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.players)
    results = {}

    for mode in ["global", "rank_quartile"]:
        rows = []
        for fold in sorted(df.fold.unique()):
            tr = df[df.fold != fold]
            te = df[df.fold == fold]
            cuts = {"__all__": cutoff(tr["label"], tr["score"])}
            if mode == "rank_quartile":
                for q, g in tr.groupby("rank_q"):
                    if g["label"].nunique() == 2 and (g["label"] == 0).sum() >= 10:
                        cuts[q] = cutoff(g["label"], g["score"])
                    else:
                        cuts[q] = cuts["__all__"]
            for q, g in te.groupby("rank_q"):
                c = cuts.get(q, cuts["__all__"])
                pred = (g["score"] >= c).astype(int)
                y = g["label"].to_numpy()
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                tn = int(((pred == 0) & (y == 0)).sum())
                rows.append({
                    "fold": int(fold), "quartile": str(q),
                    "cutoff": c, "n_pos": tp + fn, "n_neg": fp + tn,
                    "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
                })
        r = pd.DataFrame(rows)
        tot_pos = int(r.n_pos.sum()); tot_neg = int(r.n_neg.sum())
        pooled_recall = float((r.recall * r.n_pos).sum() / max(tot_pos, 1))
        pooled_fpr = float((r.fpr * r.n_neg).sum() / max(tot_neg, 1))
        byq = r.groupby("quartile").apply(
            lambda g: {
                "recall": float((g.recall * g.n_pos).sum() / max(g.n_pos.sum(), 1)),
                "fpr": float((g.fpr * g.n_neg).sum() / max(g.n_neg.sum(), 1)),
                "n_pos": int(g.n_pos.sum()), "n_neg": int(g.n_neg.sum()),
            }, include_groups=False,
        ).to_dict()
        results[mode] = {
            "pooled_recall": pooled_recall, "pooled_fpr": pooled_fpr,
            "by_quartile": byq, "rows": r.to_dict(orient="records"),
        }

    with open(out / "skill_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({
        k: {"pooled_recall": v["pooled_recall"], "pooled_fpr": v["pooled_fpr"],
            "by_quartile": v["by_quartile"]}
        for k, v in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
