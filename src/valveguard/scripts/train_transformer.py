"""Train the small Transformer detector."""
from __future__ import annotations

import argparse
import json

from valveguard.models.transformer import train_transformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--channels", required=True,
                    help="channels.npy from build_channels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-groups", default="",
                    help="comma-separated: weapon,outcome,motion,position")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    args = ap.parse_args()

    groups = [g.strip() for g in args.drop_groups.split(",") if g.strip()]
    results = train_transformer(
        args.cache, args.channels, args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        d_model=args.d_model, layers=args.layers, nhead=args.heads,
        workers=args.workers, limit=args.limit, seed=args.seed,
        drop_groups=groups, dropout=args.dropout, pooling=args.pooling,
    )
    print(json.dumps({
        "best_epoch": results["best_epoch"],
        "best_val_player_auc": results["best_val_player_auc"],
        "test": results["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
