"""Feature-group ablations on the cached summary features.

The first GBDT ranked weapon group and kill-outcome flags highly.  That may
be a dataset shortcut rather than aim behaviour.  This script trains the same
model with progressively stricter feature sets and reports the cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from valveguard.data.split import assign_match_splits, player_key
from valveguard.data.tar_index import load_cache
from valveguard.metrics import binary_metrics, player_scores

META_DROP = {
    "row", "map", "weapon_idx", "label", "split", "match_key",
    "player_name", "player_key", "source", "match_idx", "player",
    "kill_idx", "augmented",
}


def train_eval(data: pd.DataFrame, cols: list[str], seed: int = 42) -> dict:
    train = data[data.split == "train"]
    val = data[data.split == "val"]
    test = data[data.split == "test"]

    def xy(d):
        return d[cols].replace([np.inf, -np.inf], np.nan), d["label"].to_numpy(int)

    Xtr, ytr = xy(train)
    Xval, yval = xy(val)
    Xte, yte = xy(test)

    pos = max(1, int(ytr.sum()))
    neg = max(1, int(len(ytr) - ytr.sum()))
    sw = np.where(ytr == 1, len(ytr) / (2.0 * pos), len(ytr) / (2.0 * neg))

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=seed, verbose=0,
    )
    clf.fit(Xtr, ytr, sample_weight=sw)

    out = {}
    for name, X, y, d in [("val", Xval, yval, val), ("test", Xte, yte, test)]:
        p = clf.predict_proba(X)[:, 1]
        m = binary_metrics(y, p)
        ps = player_scores(d, p, method="mean")
        m["player_auc_mean"] = binary_metrics(
            ps["label"].to_numpy(), ps["score"].to_numpy()
        )["auc"]
        out[name] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features", required=True,
                    help="summary_features.parquet from the baseline run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    _, meta, _ = load_cache(args.cache)
    data = feats.merge(
        meta[["row", "match_key", "player_name", "label", "source",
              "match_idx", "player", "kill_idx"]],
        on="row", how="inner",
    )
    # Strict label policy: no cdata label-0 windows.
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    data = assign_match_splits(data, seed=42)
    data["player_key"] = player_key(data)

    all_cols = [c for c in data.columns if c not in META_DROP]
    all_cols = [c for c in all_cols if data[c].notna().any()]

    weapon = [c for c in all_cols if c.startswith("weapon_")]
    outcome = [
        "headshot", "through_smoke", "wallbang", "midair",
        "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
    ]
    motion = [c for c in all_cols if c.startswith("attacker_vel")]

    sets = {
        "all": all_cols,
        "no_weapon": [c for c in all_cols if c not in weapon],
        "no_outcome": [
            c for c in all_cols if c not in set(weapon + outcome)
        ],
        "no_motion": [
            c for c in all_cols
            if c not in set(weapon + outcome + motion)
        ],
    }

    results = {}
    rows = []
    for name, cols in sets.items():
        print(f"training {name}: {len(cols)} features", flush=True)
        res = train_eval(data, cols)
        results[name] = {"n_features": len(cols), **res}
        rows.append({
            "set": name,
            "n_features": len(cols),
            "val_auc": res["val"]["auc"],
            "val_player_auc": res["val"]["player_auc_mean"],
            "test_auc": res["test"]["auc"],
            "test_player_auc": res["test"]["player_auc_mean"],
            "test_ap": res["test"]["ap"],
            "test_recall_at_fpr1": res["test"]["recall_at_fpr1"],
            "test_fpr_at_tpr95": res["test"]["fpr_at_tpr95"],
            "test_ece": res["test"]["ece"],
        })

    df = pd.DataFrame(rows).sort_values("test_player_auc", ascending=False)
    df.to_csv(out_dir / "ablation_summary.csv", index=False)
    with open(out_dir / "ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
