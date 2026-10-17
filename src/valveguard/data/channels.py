"""Build the per-tick channel tensor used by the sequence model.

Output: channels.npy, shape (N, 256, C), float16.  Workers write disjoint
rows into a shared memmap, so no large arrays cross process boundaries.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.summarize import per_tick_channels

_IN = None
_OUT = None
_NAMES = None


def _init_worker(cache_dir: str, out_path: str, shape: tuple[int, int, int],
                 names: list[str]) -> None:
    global _IN, _OUT, _NAMES
    _IN = np.load(Path(cache_dir) / "windows.npy", mmap_mode="r")
    _OUT = np.lib.format.open_memmap(
        out_path, mode="r+", dtype=np.float16, shape=shape
    )
    _NAMES = names


def _work(i: int) -> int:
    arr, _ = per_tick_channels(pd.DataFrame(_IN[i], columns=_NAMES))
    _OUT[i] = arr.astype(np.float16)
    return int(i)


def build_channel_cache(cache_dir: str | Path, out_dir: str | Path,
                        n_jobs: int | None = None) -> list[str]:
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mm = np.load(cache_dir / "windows.npy", mmap_mode="r")
    index = pd.read_parquet(cache_dir / "index.parquet")
    with open(cache_dir / "feature_names.json", encoding="utf-8") as f:
        names = json.load(f)

    first, channel_names = per_tick_channels(
        pd.DataFrame(mm[0], columns=names)
    )
    shape = (len(index), first.shape[0], first.shape[1])
    out_path = out_dir / "channels.npy"
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=shape
    )
    out[0] = first.astype(np.float16)
    out.flush()
    del out

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)
    print(f"building channels {shape} with {n_jobs} workers", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(str(cache_dir), str(out_path), shape, names),
    ) as pool:
        done = 1
        for _ in pool.imap_unordered(_work, range(1, len(index)), chunksize=64):
            done += 1
            if done % 5000 == 0:
                print(f"channels {done}/{len(index)}", flush=True)

    with open(out_dir / "channel_names.json", "w", encoding="utf-8") as f:
        json.dump(channel_names, f, indent=2)
    with open(out_dir / "channel_meta.json", "w", encoding="utf-8") as f:
        json.dump({"shape": list(shape), "n": len(index)}, f, indent=2)
    return channel_names
