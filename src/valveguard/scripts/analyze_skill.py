"""Skill/rank analysis on the 160-match subset.

Questions:
  - Is the false-positive rate concentrated in high-rank players?
  - Does a rank-conditional residual (score minus expected human score at
    that rank) improve low-FPR separation?

Rank extraction uses the raw parquet `rank` column and the per-match
`CSstats_info.match_making_type`; Premier ratings and CS rank ordinals are
kept separate and converted to within-type percentiles.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import binary_metrics


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def _extract_one(job):
    match_key, raw_dir = job
    source, idx = match_key.split(":")
    folder = "with_cheater_present" if source == "cdata" else "no_cheater_present"
    base = Path(raw_dir) / folder / str(int(idx))
    try:
        d = pd.read_parquet(str(base) + ".parquet", columns=["steamid", "rank"])
        ev = json.load(open(str(base) + ".json", encoding="utf-8"))
    except Exception:
        return []
    info = ev.get("CSstats_info", [{}])
    mmt = str(info[0].get("match_making_type", "")) if info else ""
    kind = "premier" if "Premier" in mmt else "cs"
    out = []
    for player, g in d.groupby("steamid"):
        vals = g["rank"].dropna().to_numpy()
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        out.append({
            "match_key": match_key, "player_name": player,
            "rank_raw": float(np.median(vals)), "rank_type": kind,
        })
    return out


def make_folds(df: pd.DataFrame, k: int, seed: int) -> pd.Series:
    match_label = df.groupby("match_key")["label"].max()
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, keys in match_label.groupby(match_label).groups.items():
        keys = np.array(sorted(keys))
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            fold_of[key] = i % k
    return df["match_key"].map(fold_of)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(args.scores)
    matches = sorted(scores["match_key"].unique())

    ranks = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(_extract_one, [(m, args.raw) for m in matches]):
            ranks.extend(rows)
    rk = pd.DataFrame(ranks)
    df = scores.merge(rk, on=["match_key", "player_name"], how="inner")
    df = df.reset_index(drop=True)

    # Percentile within rank type, so Premier and CS ordinals are comparable.
    df["rank_pct"] = np.nan
    for kind, g in df.groupby("rank_type"):
        df.loc[g.index, "rank_pct"] = g["rank_raw"].rank(pct=True)

    q = pd.qcut(df["rank_pct"], 4, labels=False, duplicates="drop")
    qmap = {0: "Q1 low", 1: "Q2", 2: "Q3", 3: "Q4 high"}
    df["rank_q"] = q.map(qmap).fillna("Q?")
    df["fold"] = make_folds(df, args.folds, args.seed)

    results = {"n_players": int(len(df)), "by_type": df.groupby("rank_type").size().to_dict()}
    # AUC by rank quartile.
    byq = []
    for q, g in df.groupby("rank_q", observed=True):
        if g["label"].nunique() < 2 or len(g) < 20:
            continue
        byq.append({
            "quartile": str(q), "n": int(len(g)), "pos": int(g.label.sum()),
            "auc": float(roc_auc_score(g.label, g.score)),
            "recall_at_fpr1": recall_at_fpr(g.label.to_numpy(), g.score.to_numpy()),
        })
    results["by_rank_quartile"] = byq

    # FPR by quartile at a global 1% FPR cutoff (diagnostic).
    fpr, _, thr = roc_curve(df["label"], df["score"])
    ok = np.where(fpr <= 0.01)[0]
    cut = float(thr[ok[-1]]) if len(ok) else 1.0
    fpr_rows = []
    for q, g in df.groupby("rank_q", observed=True):
        y = g["label"].to_numpy()
        pred = (g["score"] >= cut).to_numpy()
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fpr_rows.append({
            "quartile": str(q), "n": int(len(g)),
            "fpr": fp / max(fp + tn, 1), "recall": tp / max(tp + fn, 1),
        })
    results["fpr_by_rank_quartile_at_global_1pct"] = fpr_rows

    # Rank-conditional human baseline, cross-fitted by match.
    df["expected_human"] = np.nan
    for fold in sorted(df.fold.unique()):
        tr = (df.fold != fold) & (df.label == 0) & df.rank_pct.notna()
        te = df.fold == fold
        if tr.sum() < 30:
            continue
        x = df.loc[tr, ["rank_pct"]].to_numpy()
        y = df.loc[tr, "score"].to_numpy()
        # Quadratic captures a saturating skill/score relationship.
        X = np.column_stack([x, x * x])
        model = LinearRegression().fit(X, y)
        Xte = df.loc[te, ["rank_pct"]].to_numpy()
        if len(Xte):
            df.loc[te, "expected_human"] = model.predict(
                np.column_stack([Xte, Xte * Xte])
            )
    df["residual"] = df["score"] - df["expected_human"]
    df["residual_pos"] = df["residual"].clip(lower=0)

    results["conditional"] = {}
    for col in ["score", "residual", "residual_pos"]:
        sub = df[df[col].notna()]
        if sub["label"].nunique() < 2:
            continue
        results["conditional"][col] = {
            "auc": float(roc_auc_score(sub.label, sub[col])),
            "recall_at_fpr1": recall_at_fpr(sub.label.to_numpy(), sub[col].to_numpy()),
        }

    df.to_parquet(out / "skill_players.parquet", index=False)
    with open(out / "skill_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
