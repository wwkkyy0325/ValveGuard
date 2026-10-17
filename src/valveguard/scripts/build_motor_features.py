"""Build motor-control summary features for all cached windows."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.data.tar_index import load_cache
from valveguard.features.motor_model import motor_summary

_IN = None
_NAMES = None


def _init(cache_dir: str, names: list[str]) -> None:
    global _IN, _NAMES
    _IN = np.load(Path(cache_dir) / "windows.npy", mmap_mode="r")
    _NAMES = names


def _work(i: int) -> dict:
    w = pd.DataFrame(_IN[i], columns=_NAMES)
    out = motor_summary(w)
    out["row"] = int(i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mm, meta, names = load_cache(args.cache)
    n_jobs = args.jobs or max(1, (os.cpu_count() or 1) - 1)
    print(f"motor features for {len(meta)} windows with {n_jobs} workers", flush=True)

    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_jobs, initializer=_init,
                  initargs=(str(args.cache), names)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_work, range(len(meta)),
                                                    chunksize=128), 1):
            rows.append(rec)
            if i % 5000 == 0:
                print(f"motor {i}/{len(meta)}", flush=True)
    df = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns)} cols to {out}")


if __name__ == "__main__":
    main()
