"""Segment-based movement / pre-aim features.

Rules over the pre-kill part of each 256-tick window:

  travel  -- not in the duel phase, speed > 200 units/s (fast rotation/rush)
  walk    -- not in the duel phase, 20 < speed <= 200 (silent/slow approach)
  stop    -- not in the duel phase, speed <= 20
  duel    -- last 64 ticks before the kill, or target error < 5 deg

The cache has no `is_walking` flag, so silence is approximated by the speed
band; raw demos can later replace this with the real flag.  Features describe
how much time is spent in each state, how fast the player travels, whether the
approach is near/far from the victim, and pre-aim quality before the duel.

Usage:
    PYTHONPATH=src python -m valveguard.scripts.build_segment_features \
        --cache data/processed/cw256 --features-json reports/audit.json \
        --out reports/segments --jobs 12
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.features.geometry import recover

COLUMNS = None
CACHE = None
OUT = None


def _init(cache: str, columns: list[str], out: str) -> None:
    global COLUMNS, CACHE, OUT
    COLUMNS = columns
    CACHE = np.load(Path(cache) / "windows.npy", mmap_mode="r")
    OUT = Path(out)


def _features(window: np.ndarray, df: pd.DataFrame) -> dict:
    g = recover(df)
    err = np.asarray(g["err_origin"], dtype=np.float64)
    err_yaw = np.asarray(g["err_yaw"], dtype=np.float64)
    dist = np.asarray(g["distance"], dtype=np.float64)
    speed = np.asarray(df["attacker_vel"], dtype=np.float64) * 250.0

    pre = slice(0, 224)
    duel = np.zeros(256, dtype=bool)
    duel[160:224] = True
    duel |= err < 5.0
    travel = (~duel) & (speed > 200.0)
    walk = (~duel) & (speed > 20.0) & (speed <= 200.0)
    stop = (~duel) & (speed <= 20.0)

    def frac(mask: np.ndarray, sl: slice = pre) -> float:
        m = mask[sl]
        return float(np.mean(m)) if len(m) else np.nan

    sf = speed[pre]
    out = {
        "seg_frac_travel": frac(travel),
        "seg_frac_walk": frac(walk),
        "seg_frac_stop": frac(stop),
        "seg_frac_duel": frac(duel),
        "seg_frac_silent_near": frac(walk & (dist < 600.0)),
        "seg_travel_far": frac(travel & (dist > 800.0)),
        "seg_travel_speed": float(np.mean(speed[travel])) if travel.any() else np.nan,
        "seg_walk_speed": float(np.mean(speed[walk])) if walk.any() else np.nan,
        "seg_speed_max": float(np.max(sf)) if len(sf) else np.nan,
        "seg_speed_max_time": float((223 - int(np.argmax(sf))) / 64.0) if len(sf) else np.nan,
    }

    # pre-aim phase: 1.0-2.0 s before the kill
    pa = slice(96, 160)
    out["seg_preaim_err_mean"] = float(np.mean(err[pa]))
    out["seg_preaim_err_min"] = float(np.min(err[pa]))
    out["seg_preaim_err_std"] = float(np.std(err[pa]))
    out["seg_preaim_yaw_err"] = float(np.mean(np.abs(err_yaw[pa])))
    out["seg_preaim_dist_start"] = float(dist[96])
    out["seg_preaim_dist_end"] = float(dist[159])
    out["seg_preaim_dist_slope"] = float(
        np.polyfit(np.arange(64), dist[pa], 1)[0]
    )
    out["seg_hold_before_duel"] = float(np.mean(err[pa] < 5.0))

    # transitions across the four states
    state = np.where(duel, 3, np.where(travel, 0, np.where(walk, 1, 2)))
    changes = state[1:224] != state[:223]
    out["seg_transitions"] = float(np.sum(changes))
    out["seg_fast_to_slow"] = float(
        np.sum((state[:223] == 0) & ((state[1:224] == 1) | (state[1:224] == 2)))
    )

    # duel phase
    du = slice(160, 224)
    out["seg_duel_err_mean"] = float(np.mean(err[du]))
    out["seg_duel_speed_mean"] = float(np.mean(speed[du]))
    out["seg_duel_start_err"] = float(err[160])

    # time from err<10 to err<2 inside the pre-aim + duel part
    idx10 = np.where(err[96:224] < 10.0)[0]
    idx2 = np.where(err[96:224] < 2.0)[0]
    out["seg_approach_time"] = (
        float((int(idx2[0]) - int(idx10[0])) / 64.0)
        if len(idx10) and len(idx2) and idx2[0] >= idx10[0] else np.nan
    )
    return out


def _work(job: tuple[int, int, int]) -> str:
    start, end, part = job
    rows = []
    for i in range(start, end):
        window = np.asarray(CACHE[i], dtype=np.float32)
        df = pd.DataFrame(window, columns=COLUMNS)
        try:
            feat = _features(window, df)
        except Exception:
            continue
        feat["row"] = i
        rows.append(feat)
    path = OUT / f"part_{part:04d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = json.load(open(args.features_json, encoding="utf-8"))["feature_names"]
    n = len(np.load(Path(args.cache) / "windows.npy", mmap_mode="r"))
    chunk = 4000
    jobs = [(s, min(s + chunk, n), s // chunk) for s in range(0, n, chunk)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.jobs, initializer=_init,
        initargs=(args.cache, columns, str(out)),
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_work, jobs), 1):
            if i % 5 == 0:
                print(f"parts {i}/{len(jobs)}", flush=True)

    parts = sorted(out.glob("part_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out / "segment_features.parquet", index=False)
    print(f"rows {len(df)} cols {len(df.columns)}")


if __name__ == "__main__":
    main()
