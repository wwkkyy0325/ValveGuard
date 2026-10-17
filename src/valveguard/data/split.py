"""Match-disjoint splits.

CS2CD replaces Steam IDs with per-match pseudonyms (Player_1..Player_10),
so a true person-disjoint split is not verifiable.  The strongest honest
split is match-disjoint: every match key appears in exactly one split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_match_splits(
    meta: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    if not {"match_key", "label"} <= set(meta.columns):
        raise ValueError("meta must contain match_key and label")

    rng = np.random.default_rng(seed)
    match_label = meta.groupby("match_key", sort=False)["label"].max()

    split_map: dict[str, str] = {}
    for label, keys in match_label.groupby(match_label).groups.items():
        keys = np.array(sorted(keys))
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_test + n_val >= n:
            n_test = max(1, n // 6)
            n_val = max(1, n // 6)
        for k in keys[:n_test]:
            split_map[k] = "test"
        for k in keys[n_test:n_test + n_val]:
            split_map[k] = "val"
        for k in keys[n_test + n_val:]:
            split_map[k] = "train"

    out = meta.copy()
    out["split"] = out["match_key"].map(split_map)
    if out["split"].isna().any():
        raise RuntimeError("Some matches were not assigned a split")

    # Hard assertion: no match key crosses splits.
    cross = out.groupby("match_key")["split"].nunique()
    if int(cross.max()) != 1:
        raise AssertionError("Match leakage: a match key appears in >1 split")
    return out


def player_key(meta: pd.DataFrame) -> pd.Series:
    return meta["match_key"] + ":" + meta["player_name"].astype(str)
