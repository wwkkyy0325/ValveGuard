"""Run the handcrafted-feature gradient-boosting baseline."""
from __future__ import annotations

import argparse
import json

from valveguard.models.baseline import run_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="stratified pilot sample; omit for full data")
    ap.add_argument("--include-noisy-negatives", action="store_true",
                    help="keep cdata label-0 windows (~55.6%% label precision)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel feature workers; default cpu_count-1")
    args = ap.parse_args()

    results = run_baseline(
        args.cache, args.out, seed=args.seed, limit=args.limit,
        clean_labels=not args.include_noisy_negatives,
        n_jobs=args.jobs,
    )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
