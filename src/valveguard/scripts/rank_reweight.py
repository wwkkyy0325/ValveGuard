"""Pilot: upweight high-rank clean players during training.

Post-hoc rank thresholds did not fix the 4.8% FPR on the top rank quartile.
This tests whether making high-rank humans harder negatives during training
changes the ranking itself.  It is a 160-match subset pilot, not a final result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import binary_metrics, player_scores
from valveguard.scripts.ablation_features import META_DROP


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--index", default="data/processed/cw256/index.parquet")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--scores", required=True,
                    help="base_predictions.parquet with row/fold")
    ap.add_argument("--skill", required=True, help="skill_players.parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    index = pd.read_parquet(args.index)
    meta = index[["row", "match_key", "player_name", "kill_idx", "source", "label"]]
    df = feats.merge(meta, on="row", how="inner")
    subset_rows = pd.read_parquet(args.subset, columns=["row"])["row"].unique()
    df = df[df.row.isin(subset_rows)].copy()
    df = df[~((df.source == "cdata") & (df.label == 0))].copy()
    df = df.reset_index(drop=True)

    base = pd.read_parquet(args.scores)[["row", "fold", "score"]]
    df = df.merge(base.rename(columns={"score": "base_score"}), on="row", how="inner")
    skill = pd.read_parquet(args.skill)[["match_key", "player_name", "rank_q", "rank_pct"]]
    df = df.merge(skill, on=["match_key", "player_name"], how="left")

    weapon = [c for c in df.columns if c.startswith("weapon_")]
    outcome = ["headshot", "through_smoke", "wallbang", "midair",
               "shot_count", "victim_noise", "flashed_at_kill", "victim_health"]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    cols = [c for c in df.columns if c not in META_DROP
            and c not in {"rank_q", "rank_pct", "base_score", "fold"}
            and df[c].notna().any()
            and c not in set(weapon + outcome + motion)]

    def run(name: str, use_rank_feature: bool, weight_map: dict):
        use_cols = cols + (["rank_pct"] if use_rank_feature else [])
        oof = np.full(len(df), np.nan)
        for fold in sorted(df.fold.unique()):
            tr = df.fold != fold
            te = df.fold == fold
            Xtr = df.loc[tr, use_cols].replace([np.inf, -np.inf], np.nan)
            Xte = df.loc[te, use_cols].replace([np.inf, -np.inf], np.nan)
            ytr = df.loc[tr, "label"].to_numpy(int)
            pos = max(1, int(ytr.sum()))
            neg = max(1, len(ytr) - pos)
            sw = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg))
            if weight_map:
                q = df.loc[tr, "rank_q"].astype(str).to_numpy()
                for qq, mult in weight_map.items():
                    sw[(ytr == 0) & (q == qq)] *= mult
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.1, random_state=42 + fold,
            )
            clf.fit(Xtr, ytr, sample_weight=sw)
            oof[te.to_numpy()] = clf.predict_proba(Xte)[:, 1]
        df[f"score_{name}"] = oof
        p = player_scores(df, oof, method="mean")
        p = p.merge(
            df.groupby(["match_key", "player_name"])["rank_q"].first().reset_index(),
            on=["match_key", "player_name"], how="left",
        )
        res = {
            "player_auc": float(roc_auc_score(p.label, p.score)),
            "player_recall_fpr1": recall_at_fpr(p.label.to_numpy(), p.score.to_numpy()),
            "kill_auc": float(roc_auc_score(df.label, oof)),
        }
        fpr, _, thr = roc_curve(df.label, oof)
        ok = np.where(fpr <= 0.01)[0]
        cut = float(thr[ok[-1]]) if len(ok) else 1.0
        q4 = p[p.rank_q == "Q4 high"]
        if len(q4) and q4.label.nunique() > 1:
            pred = (q4.score >= cut).to_numpy()
            y = q4.label.to_numpy()
            fp = int(((pred == 1) & (y == 0)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            res["q4_auc"] = float(roc_auc_score(q4.label, q4.score))
            res["q4_fpr_at_global_1pct"] = fp / max(fp + tn, 1)
        return res

    results = {
        "baseline": run("base", False, {}),
        "high_rank_neg_x3": run("w3", False, {"Q4 high": 3.0}),
        "high_rank_neg_x5": run("w5", False, {"Q4 high": 5.0}),
        "rank_feature": run("rank", True, {}),
    }
    with open(out / "rank_reweight.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
