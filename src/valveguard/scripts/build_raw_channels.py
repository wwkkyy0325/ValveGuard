"""Build per-tick mouse, recoil, and fire channels from raw CS2 telemetry.

The cached 43-channel tensor has no mouse input, recoil punch, or shot
timing.  This script rebuilds the same 256-tick kill windows from the raw
parquet subset and writes a second memmap aligned by cache row.  Rows whose
match is not in the raw subset stay zero and are marked in coverage.npy.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.geometry import wrap_deg

LENGTH = 256
KILL_ROW = 223
RAW_COLUMNS = [
    "tick", "steamid", "pitch", "yaw",
    "usercmd_mouse_dx", "usercmd_mouse_dy", "aim_punch_angle",
    "shots_fired", "last_shot_time",
    "is_scoped", "is_walking", "ducking", "is_airborne", "spotted",
    "approximate_spotted_by",
    "velocity_X", "velocity_Y", "velocity_Z",
]

_IN = None
_OUT = None
_COVER = None


def _residual(view: np.ndarray, mouse: np.ndarray) -> tuple[float, np.ndarray]:
    ok = np.isfinite(view) & np.isfinite(mouse)
    if ok.sum() < 10 or np.var(mouse[ok]) < 1e-9:
        return 0.0, np.zeros_like(view)
    beta = float(np.cov(view[ok], mouse[ok], bias=True)[0, 1] / np.var(mouse[ok]))
    return beta, view - beta * mouse


def _age(flags: np.ndarray, cap: float = 2.0) -> np.ndarray:
    out = np.full(len(flags), cap, dtype=float)
    first = -1
    for i, flag in enumerate(flags):
        if flag and first < 0:
            first = i
        if first >= 0:
            out[i] = min((i - first) / 64.0, cap)
    return out


def _sees(victim: pd.DataFrame, attacker_id: str) -> np.ndarray:
    values = victim["approximate_spotted_by"].to_numpy()
    return np.array(
        [attacker_id in set(x) if np.ndim(x) == 1 else False for x in values],
        dtype=float,
    )


def _window_channels(attacker: pd.DataFrame, victim: pd.DataFrame,
                     attacker_id: str, start: int, kill_tick: int) -> np.ndarray:
    ticks = np.arange(start, kill_tick + (LENGTH - KILL_ROW))
    frame = attacker.reindex(ticks)
    victim_frame = victim.reindex(ticks)
    mouse_x = frame["usercmd_mouse_dx"].to_numpy(dtype=float)
    mouse_y = frame["usercmd_mouse_dy"].to_numpy(dtype=float)
    valid = np.isfinite(mouse_x) & np.isfinite(mouse_y)
    mouse_x = np.nan_to_num(mouse_x, nan=0.0)
    mouse_y = np.nan_to_num(mouse_y, nan=0.0)

    raw_punch = frame["aim_punch_angle"].to_numpy()
    punch = np.zeros((len(raw_punch), 3), dtype=float)
    has_punch = np.array([np.ndim(x) == 1 for x in raw_punch])
    if has_punch.any():
        punch[has_punch] = np.stack(
            [np.asarray(x, dtype=float) for x in raw_punch[has_punch]]
        )
    punch = np.nan_to_num(punch, nan=0.0)
    pitch = np.asarray(frame["pitch"], dtype=float)
    yaw = np.asarray(frame["yaw"], dtype=float)
    dpitch = wrap_deg(np.diff(pitch, prepend=pitch[0]))
    dyaw = wrap_deg(np.diff(yaw, prepend=yaw[0]))
    beta_yaw, resid_yaw = _residual(dyaw, mouse_x)
    beta_pitch, resid_pitch = _residual(dpitch, mouse_y)

    shots = np.nan_to_num(frame["shots_fired"].to_numpy(dtype=float))
    shot = np.clip(np.diff(shots, prepend=shots[0]), 0.0, None)
    shot_ticks = ticks[shot > 0]
    last = np.full(len(ticks), -10_000, dtype=float)
    cursor = -10_000
    for i, tick in enumerate(ticks):
        if shot[i] > 0:
            cursor = tick
        last[i] = cursor
    age = np.where(last > -1_000, (ticks - last) / 64.0, 2.0)
    age = np.clip(age, 0.0, 2.0)

    sees = _sees(victim_frame, attacker_id)
    see_age = _age(sees > 0)
    see_onset = np.zeros(len(ticks), dtype=float)
    if sees.any():
        see_onset[int(np.argmax(sees > 0))] = 1.0
    victim_spotted = np.nan_to_num(
        victim_frame["spotted"].to_numpy(dtype=float), nan=0.0
    ) > 0
    spotted_age = _age(victim_spotted)

    out = np.column_stack([
        np.clip(mouse_x, -2000, 2000),
        np.clip(mouse_y, -2000, 2000),
        np.clip(np.hypot(mouse_x, mouse_y), 0, 3000),
        valid.astype(float),
        punch[:, 0], punch[:, 1], punch[:, 2],
        np.gradient(punch[:, 0]), np.gradient(punch[:, 1]),
        shot, age,
        np.asarray(frame["is_scoped"], dtype=float),
        np.asarray(frame["is_walking"], dtype=float),
        np.asarray(frame["ducking"], dtype=float),
        np.asarray(frame["is_airborne"], dtype=float),
        np.asarray(frame["spotted"], dtype=float),
        np.asarray(frame["velocity_X"], dtype=float) / 250.0,
        np.asarray(frame["velocity_Y"], dtype=float) / 250.0,
        np.asarray(frame["velocity_Z"], dtype=float) / 250.0,
        np.clip(resid_yaw, -30000, 30000),
        np.clip(resid_pitch, -30000, 30000),
        np.full(len(ticks), beta_yaw),
        np.full(len(ticks), beta_pitch),
        sees, see_age, see_onset, victim_spotted.astype(float), spotted_age,
    ])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _init(raw_dir: str, out_path: str, cover_path: str, shape: tuple[int, int, int]) -> None:
    global _IN, _OUT, _COVER
    _IN = Path(raw_dir)
    _OUT = np.lib.format.open_memmap(out_path, mode="r+", dtype=np.float16, shape=shape)
    _COVER = np.lib.format.open_memmap(cover_path, mode="r+", dtype=np.uint8, shape=(shape[0],))


def _work(job):
    match_key, expected = job
    source, idx = match_key.split(":")
    folder = "with_cheater_present" if source == "cdata" else "no_cheater_present"
    base = Path(_IN) / folder / str(int(idx))
    try:
        ticks = pd.read_parquet(str(base) + ".parquet", columns=RAW_COLUMNS)
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

    written = 0
    for player, kill_idx, row in expected:
        if player not in by_player or player not in kills:
            continue
        if kill_idx >= len(kills[player]):
            continue
        kill_tick = int(kills[player][kill_idx])
        kill_rows = deaths[
            (deaths["attacker_steamid"] == player) & (deaths["tick"] == kill_tick)
        ]
        if kill_rows.empty:
            continue
        victim = by_player.get(kill_rows.iloc[0]["user_steamid"])
        if victim is None:
            continue
        start = kill_tick - KILL_ROW
        attacker = by_player[player]
        if start < int(ticks["tick"].min()) or attacker["tick"].min() > start:
            continue
        channels = _window_channels(attacker, victim, player, start, kill_tick)
        if channels.shape[0] != LENGTH:
            continue
        _OUT[int(row)] = channels.astype(np.float16)
        _COVER[int(row)] = 1
        written += 1
    return match_key, written, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full_index = pd.read_parquet(Path(args.cache) / "index.parquet")
    index = full_index[~((full_index["source"] == "cdata") & (full_index["label"] == 0))].copy()
    index = index.reset_index(drop=True)

    names = [
        "mouse_dx", "mouse_dy", "mouse_speed", "mouse_valid",
        "punch_pitch", "punch_yaw", "punch_roll", "punch_dpitch", "punch_dyaw",
        "shot", "shot_age",
        "scoped", "walking", "ducking", "airborne", "spotted",
        "vel_x", "vel_y", "vel_z",
        "resid_yaw", "resid_pitch", "beta_yaw", "beta_pitch",
        "sees_victim", "see_age", "see_onset", "victim_spotted", "spotted_age",
    ]
    shape = (len(full_index), LENGTH, len(names))
    out_path = out / "channels_raw.npy"
    cover_path = out / "coverage.npy"
    np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=shape)
    np.lib.format.open_memmap(cover_path, mode="w+", dtype=np.uint8, shape=(len(full_index),))

    jobs = [
        (match, list(zip(group["player_name"], group["kill_idx"], group["row"])))
        for match, group in index.groupby("match_key", sort=False)
    ]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    print(f"building raw channels {shape} for {len(jobs)} matches", flush=True)
    ctx = mp.get_context("spawn")
    errors = 0
    with ctx.Pool(
        processes=args.workers,
        initializer=_init,
        initargs=(str(args.raw), str(out_path), str(cover_path), shape),
    ) as pool:
        done = 0
        for match, written, error in pool.imap_unordered(_work, jobs, chunksize=1):
            done += 1
            if error:
                errors += 1
            if done % 50 == 0:
                print(f"matches {done}/{len(jobs)} errors {errors}", flush=True)

    cover = np.load(cover_path, mmap_mode="r")
    covered = full_index.loc[np.asarray(cover) == 1].copy()
    covered.to_parquet(out / "covered_index.parquet", index=False)
    with open(out / "channel_names_raw.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "shape": list(shape),
            "rows": int(len(index)),
            "covered_rows": int(len(covered)),
            "covered_matches": int(covered["match_key"].nunique()),
        }, f, indent=2)
    print(f"covered {len(covered)} / {len(index)} rows", flush=True)


if __name__ == "__main__":
    main()
