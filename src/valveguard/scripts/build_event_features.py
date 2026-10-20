"""Build recoil/fire/mouse features for cached kill windows from raw parquet."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.features.events import EVENT_COLUMNS, event_summary

RAW_DIR = None
OUT_DIR = None
LENGTH = 256
KILL_ROW = 223


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
        ticks = pd.read_parquet(str(base) + ".parquet", columns=EVENT_COLUMNS)
        events = json.load(open(str(base) + ".json", encoding="utf-8"))
    except Exception as exc:
        return match_key, 0, repr(exc)

    deaths = pd.DataFrame(events["player_death"]).sort_values("tick")
    if deaths.empty:
        return match_key, 0, None
    by_player = {
        player: frame.sort_values("tick").set_index("tick", drop=False)
        for player, frame in ticks.groupby("steamid", sort=False)
    }
    kills = {
        player: frame[frame["attacker_steamid"] == player]["tick"].astype(int).tolist()
        for player, frame in deaths.groupby("attacker_steamid", sort=False)
    }

    rows = []
    for player, kill_idx, row in expected:
        if player not in by_player or player not in kills:
            continue
        if kill_idx >= len(kills[player]):
            continue
        kill_tick = int(kills[player][kill_idx])
        kill_rows = deaths[
            (deaths["attacker_steamid"] == player)
            & (deaths["tick"] == kill_tick)
        ]
        if kill_rows.empty:
            continue
        victim_id = kill_rows.iloc[0]["user_steamid"]
        victim = by_player.get(victim_id)
        start = kill_tick - KILL_ROW
        if start < int(ticks["tick"].min()):
            continue
        window_ticks = np.arange(start, kill_tick + (LENGTH - KILL_ROW))
        attacker = by_player[player]
        if attacker["tick"].min() > start:
            continue
        frame = attacker.reindex(window_ticks)
        if frame[["pitch", "yaw"]].isna().any().any():
            continue
        feat = event_summary(attacker, victim, window_ticks, kill_tick)
        feat["row"] = int(row)
        feat["match_key"] = match_key
        feat["player_name"] = player
        feat["kill_idx"] = int(kill_idx)
        rows.append(feat)

    out = Path(OUT_DIR) / f"{source}_{int(idx)}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return match_key, len(rows), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-matches", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

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
        (match_key, list(zip(group["player_name"], group["kill_idx"], group["row"])))
        for match_key, group in index.groupby("match_key")
    ]

    context = mp.get_context("spawn")
    results = []
    with context.Pool(
        processes=args.workers, initializer=_init,
        initargs=(args.raw, str(parts_dir)),
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(_work, jobs), 1):
            results.append(result)
            if done % 20 == 0:
                print(f"matches {done}/{len(jobs)}", flush=True)

    errors = [r for r in results if r[2] is not None]
    kept = sum(r[1] for r in results)
    print(f"done {len(results)} matches, kept {kept} windows, {len(errors)} errors")
    for error in errors[:5]:
        print("ERR", error)

    parts = [pd.read_parquet(path) for path in sorted(parts_dir.glob("*.parquet"))]
    parts = [part for part in parts if len(part)]
    if parts:
        all_rows = pd.concat(parts, ignore_index=True)
        all_rows.to_parquet(out / "event_features.parquet", index=False)
        print(f"rows {len(all_rows)} columns {len(all_rows.columns)}")
    else:
        print("no rows produced")


if __name__ == "__main__":
    main()
