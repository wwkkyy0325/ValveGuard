"""Compute mouse-view residual features for the raw subset.

Joins the residual features to the cached window index by
(match_key, player_name, kill_idx), so they can be appended to the
existing summary features without rebuilding the windows.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.features.mouse_residual import (
    kill_ticks,
    player_residuals,
    window_features,
)

TICK_COLS = [
    "tick", "steamid", "pitch", "yaw",
    "usercmd_mouse_dx", "usercmd_mouse_dy",
    "is_alive", "X", "Y", "Z", "velocity", "active_weapon_name",
]
RAW_DIR = None
OUT_DIR = None


def _init(raw_dir: str, out_dir: str) -> None:
    global RAW_DIR, OUT_DIR
    RAW_DIR = Path(raw_dir)
    OUT_DIR = Path(out_dir)


def _work(job):
    match_key, expected = job
    source, idx = match_key.split(":")
    folder = "with_cheater_present" if source == "cdata" else "no_cheater_present"
    base = Path(RAW_DIR) / folder / str(int(idx))
    try:
        ticks = pd.read_parquet(str(base) + ".parquet", columns=TICK_COLS)
        events = json.load(open(str(base) + ".json", encoding="utf-8"))
    except Exception as exc:
        return match_key, 0, repr(exc)

    resid = player_residuals(ticks)
    rows = []
    for player, kill_idx, row in expected:
        if player not in resid:
            continue
        kills = kill_ticks(events, player)
        if kill_idx >= len(kills):
            continue
        kill_tick = kills[kill_idx]
        r = resid[player]
        feat = window_features(
            r["tick"], r["resid"], kill_tick,
            r["r2_yaw"], r["r2_pitch"], r["slope_yaw"], r["slope_pitch"],
            r["scale"], r["valid"],
        )
        feat["row"] = int(row)
        feat["match_key"] = match_key
        feat["player_name"] = player
        feat["kill_idx"] = int(kill_idx)
        rows.append(feat)

    out = Path(OUT_DIR) / f"{source}_{int(idx)}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return match_key, len(rows), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-matches", type=int, default=80)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(Path(args.cache) / "index.parquet")
    index = index[
        index["source"].isin(["cdata", "ncdata"])
        & (index["match_idx"] < args.n_matches)
    ]
    jobs = []
    for match_key, g in index.groupby("match_key"):
        expected = list(zip(g["player_name"], g["kill_idx"], g["row"]))
        jobs.append((match_key, expected))

    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(
        processes=args.workers, initializer=_init,
        initargs=(args.raw, str(parts_dir)),
    ) as pool:
        for done, res in enumerate(pool.imap_unordered(_work, jobs), 1):
            results.append(res)
            if done % 20 == 0:
                print(f"matches {done}/{len(jobs)}", flush=True)

    errs = [r for r in results if r[2] is not None]
    print(f"done {len(results)} matches, {len(errs)} errors")
    for e in errs[:5]:
        print("ERR", e)

    parts = []
    for p in sorted(parts_dir.glob("*.parquet")):
        d = pd.read_parquet(p)
        if len(d):
            parts.append(d)
    if parts:
        all_df = pd.concat(parts, ignore_index=True)
        all_df.to_parquet(out / "mouse_residual_subset.parquet", index=False)
        print(f"rows {len(all_df)} features {len(all_df.columns)}")
    else:
        print("no rows produced")


if __name__ == "__main__":
    main()
