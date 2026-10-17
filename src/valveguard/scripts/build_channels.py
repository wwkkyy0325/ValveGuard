"""Build the per-tick channel tensor for the sequence model."""
from __future__ import annotations

import argparse

from valveguard.data.channels import build_channel_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    names = build_channel_cache(args.cache, args.out, n_jobs=args.jobs)
    print(f"wrote {len(names)} channels to {args.out}")


if __name__ == "__main__":
    main()
