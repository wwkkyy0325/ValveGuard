"""Audit the cached windows before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from valveguard.data.tar_index import load_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mm, meta, names = load_cache(args.cache)
    out: dict = {
        "n_windows": int(len(meta)),
        "shape": list(mm.shape),
        "n_features": len(names),
        "label_counts": meta.groupby("label").size().to_dict(),
        "source_counts": meta.groupby("source").size().to_dict(),
        "label_by_source": {
            f"{k[0]}-{k[1]}": int(v)
            for k, v in meta.groupby(["label", "source"]).size().items()
        },
        "n_matches": int(meta["match_key"].nunique()),
        "n_player_names": int(meta["player_name"].nunique()),
        "n_match_players": int(meta[["match_key", "player_name"]].drop_duplicates().shape[0]),
        "noisy_negative_windows": int(((meta["source"] == "cdata") & (meta["label"] == 0)).sum()),
        "clean_negative_windows": int(((meta["source"] == "ncdata") & (meta["label"] == 0)).sum()),
        "augmented_counts": meta.groupby("augmented").size().to_dict(),
        "matches_per_source": meta.groupby("source")["match_key"].nunique().to_dict(),
        "windows_per_match": meta.groupby("match_key").size().describe().to_dict(),
        "windows_per_player": meta.groupby(["match_key", "player_name"]).size().describe().to_dict(),
        "duplicate_paths": int(meta["path"].duplicated().sum()),
        "feature_names": names,
    }

    # Labels must be constant within a match; otherwise the folder
    # convention is not what we think it is.
    label_per_match = meta.groupby("match_key")["label"].nunique()
    out["matches_with_mixed_labels"] = int((label_per_match > 1).sum())

    # NaN/Inf check on a sample of rows (full scan is unnecessary here).
    rng = np.random.default_rng(0)
    rows = rng.choice(len(meta), size=min(5000, len(meta)), replace=False)
    sample = np.asarray(mm[np.sort(rows)], dtype=np.float32)
    out["sample_nonfinite"] = int((~np.isfinite(sample)).sum())
    out["sample_min"] = float(np.nanmin(sample))
    out["sample_max"] = float(np.nanmax(sample))

    text = json.dumps(out, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
