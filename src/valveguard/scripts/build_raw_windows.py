"""Build long-context features from the raw 160-match subset.

For every kill window in the cached index, rebuild a 1024-tick window from
raw telemetry, then emit the standard near-kill summary plus macro features.
Windows without 1024 ticks of pre-kill history are skipped.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.features.macro import macro_summary
from valveguard.features.raw_window import RAW_COLUMNS, build_window
from valveguard.features.summarize import summarize

RAW_DIR = None
OUT_DIR = None
LENGTH = 1024


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
        ticks = pd.read_parquet(str(base) + ".parquet", columns=RAW_COLUMNS)
        events = json.load(open(str(base) + ".json", encoding="utf-8"))
    except Exception as exc:
        return match_key, 0, 0, repr(exc)

    map_name = events["CSstats_info"][0]["map"]
    deaths = pd.DataFrame(events["player_death"]).sort_values("tick")
    by_player = {
        p: d.sort_values("tick").set_index("tick", drop=False)
        for p, d in ticks.groupby("steamid", sort=False)
    }
    kills = {
        p: d[d["attacker_steamid"] == p]["tick"].astype(int).tolist()
        for p, d in deaths.groupby("attacker_steamid", sort=False)
    }

    kill_row = LENGTH - 33
    rows, dropped = [], 0
    for player, kill_idx, row in expected:
        if player not in by_player or player not in kills:
            dropped += 1
            continue
        if kill_idx >= len(kills[player]):
            dropped += 1
            continue
        kill_tick = kills[player][kill_idx]
        event = deaths[
            (deaths["attacker_steamid"] == player)
            & (deaths["tick"] == kill_tick)
        ].iloc[0]
        victim = event["user_steamid"]
        if victim not in by_player:
            dropped += 1
            continue
        t0 = kill_tick - (LENGTH - 33)
        if t0 < 0:
            dropped += 1
            continue
        tick_range = np.arange(t0, kill_tick + 33)
        attacker = by_player[player]
        victim_df = by_player[victim]
        if attacker["tick"].min() > t0 or victim_df["tick"].min() > t0:
            dropped += 1
            continue
        window = build_window(
            attacker, victim_df, tick_range, map_name, kill_tick, event.to_dict()
        )
        if window is None:
            dropped += 1
            continue
        feat = summarize(window, kill_row=kill_row)
        feat.update(macro_summary(window, kill_row))
        feat["row"] = int(row)
        feat["match_key"] = match_key
        feat["player_name"] = player
        feat["kill_idx"] = int(kill_idx)
        rows.append(feat)

    out = Path(OUT_DIR) / f"{source}_{int(idx)}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return match_key, len(rows), dropped, None


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
    jobs = [
        (mk, list(zip(g["player_name"], g["kill_idx"], g["row"])))
        for mk, g in index.groupby("match_key")
    ]

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

    errs = [r for r in results if r[3] is not None]
    kept = sum(r[1] for r in results)
    dropped = sum(r[2] for r in results)
    print(f"done {len(results)} matches, kept {kept} windows, dropped {dropped}, "
          f"{len(errs)} errors")
    for e in errs[:5]:
        print("ERR", e)

    parts = [pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))]
    parts = [p for p in parts if len(p)]
    if parts:
        all_df = pd.concat(parts, ignore_index=True)
        all_df.to_parquet(out / "raw_windows.parquet", index=False)
        print(f"rows {len(all_df)} columns {len(all_df.columns)}")
    else:
        print("no rows produced")


if __name__ == "__main__":
    main()
