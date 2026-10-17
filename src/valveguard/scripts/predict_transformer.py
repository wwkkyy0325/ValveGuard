"""Run the saved Transformer on validation and test rows.

The training script saved test predictions only.  Calibration needs the
validation scores, so this is a cheap forward pass from the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from valveguard.data.split import assign_match_splits, player_key
from valveguard.models.transformer import AimTransformer, WindowDataset, _predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--channels", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    metrics = json.load(open(model_dir / "metrics.json", encoding="utf-8"))
    index = pd.read_parquet(Path(args.cache) / "index.parquet")
    index = index[~((index["source"] == "cdata") & (index["label"] == 0))].copy()
    index = assign_match_splits(index, seed=args.seed)
    index["player_key"] = player_key(index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AimTransformer(
        n_features=metrics["n_features"], d_model=128, nhead=4,
        num_layers=4, dim_feedforward=256,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    mean = np.asarray(metrics["norm_mean"], dtype=np.float32)
    std = np.asarray(metrics["norm_std"], dtype=np.float32)
    keep_idx = metrics.get("keep_idx")
    keep_idx = None if keep_idx is None else np.asarray(keep_idx, dtype=np.int64)

    parts = []
    for split in ["val", "test"]:
        d = index[index.split == split]
        ds = WindowDataset(
            args.channels, d["row"].to_numpy(),
            d["label"].to_numpy(), mean, std, keep_idx=keep_idx,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=(args.workers > 0),
        )
        p = _predict(model, loader, device)
        part = d[["row", "match_key", "player_name", "kill_idx", "label"]].copy()
        part["split"] = split
        part["score"] = p
        parts.append(part)
        print(split, len(part))

    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {len(out)} predictions to {args.out}")


if __name__ == "__main__":
    main()
