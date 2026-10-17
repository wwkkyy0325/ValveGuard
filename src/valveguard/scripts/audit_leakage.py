"""Cheap leakage checks on the cached summary features.

1. Metadata-only: can map + weapon + kill-outcome flags alone predict the
   label?  If yes, the dataset has a shortcut.
2. Shuffled-label: train on permuted labels; test AUC must collapse to ~0.5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.data.split import assign_match_splits
from valveguard.data.tar_index import load_cache
from valveguard.scripts.ablation_features import META_DROP, train_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    _, meta, _ = load_cache(args.cache)
    data = feats.merge(
        meta[["row", "match_key", "player_name", "label", "source",
              "match_idx", "player", "kill_idx"]],
        on="row", how="inner",
    )
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    data = assign_match_splits(data, seed=42)

    # Metadata-only feature set.
    map_dummies = pd.get_dummies(data["map"], prefix="map", dtype=float)
    weapon_cols = [c for c in data.columns if c.startswith("weapon_")]
    outcome_cols = [
        "headshot", "through_smoke", "wallbang", "midair",
        "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
    ]
    meta_cols = list(map_dummies.columns) + weapon_cols + outcome_cols
    data_meta = pd.concat([data, map_dummies], axis=1)
    meta_res = train_eval(data_meta, meta_cols)

    # Shuffled labels, same features as the full model.
    all_cols = [c for c in data.columns if c not in META_DROP]
    all_cols = [c for c in all_cols if data[c].notna().any()]
    rng = np.random.default_rng(7)
    shuffled = data.copy()
    train_mask = shuffled.split == "train"
    shuffled.loc[train_mask, "label"] = rng.permutation(
        shuffled.loc[train_mask, "label"].to_numpy()
    )
    shuffle_res = train_eval(shuffled, all_cols)

    results = {
        "metadata_only": {
            "n_features": len(meta_cols),
            "val": meta_res["val"],
            "test": meta_res["test"],
        },
        "shuffled_train_labels": {
            "n_features": len(all_cols),
            "val": shuffle_res["val"],
            "test": shuffle_res["test"],
        },
    }
    with open(out / "leakage.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
