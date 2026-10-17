"""Lock a stratified 20% match holdout.

The holdout is locked after the v1 experiments, so it cannot undo the
selection that produced the frozen baseline.  It bounds selection for the
v2 hypotheses: those are developed on the development 80% only, and the
final candidate is scored on the holdout once.

The split is match-disjoint (CS2CD player ids are per-match pseudonyms) and
stratified by whether the match contains a cheater.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.data.tar_index import load_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _, meta, _ = load_cache(args.cache)
    clean = meta[~((meta["source"] == "cdata") & (meta["label"] == 0))]

    match = clean.groupby("match_key").agg(
        label=("label", "max"),
        n_windows=("row", "size"),
        n_players=("player_name", "nunique"),
    )

    rng = np.random.default_rng(args.seed)
    split_of: dict[str, str] = {}
    for label, grp in match.groupby("label"):
        keys = np.array(sorted(grp.index))
        rng.shuffle(keys)
        n = max(1, int(round(len(keys) * args.frac)))
        for k in keys[:n]:
            split_of[k] = "holdout"
        for k in keys[n:]:
            split_of[k] = "dev"

    match["split"] = match.index.map(split_of)
    if match["split"].isna().any():
        raise RuntimeError("Some matches were not assigned a split")

    # Hard assertions: match-disjoint by construction, but check the frame.
    if int(match["split"].nunique()) != 2:
        raise AssertionError("Holdout split is degenerate")
    if match.groupby("split")["label"].mean().abs().diff().iloc[-1] > 0.05:
        raise AssertionError("Holdout label balance deviates by more than 5 points")

    hold_keys = sorted(match.index[match["split"] == "holdout"].tolist())
    digest = hashlib.sha256("\n".join(hold_keys).encode("utf-8")).hexdigest()

    match.reset_index().to_parquet(out / "holdout_matches.parquet", index=False)
    summary = {
        "seed": args.seed,
        "frac": args.frac,
        "sha256_holdout_matches": digest,
        "n_matches": int(len(match)),
        "n_holdout_matches": int((match["split"] == "holdout").sum()),
        "n_dev_matches": int((match["split"] == "dev").sum()),
        "holdout_windows": int(match.loc[match["split"] == "holdout", "n_windows"].sum()),
        "dev_windows": int(match.loc[match["split"] == "dev", "n_windows"].sum()),
        "holdout_positive_rate": float(
            match.loc[match["split"] == "holdout", "label"].mean()
        ),
        "dev_positive_rate": float(match.loc[match["split"] == "dev", "label"].mean()),
        "note": "post-hoc lock; not pristine with respect to v1 selection",
    }
    with open(out / "holdout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
