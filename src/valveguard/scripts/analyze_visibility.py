"""Visibility / pre-aim case study for one player in one demo."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from demoparser2 import DemoParser

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EYE_HEIGHT = 64.0
TICK_RATE = 64.0
TICK_PROPS = [
    "tick", "steamid", "X", "Y", "Z", "pitch", "yaw", "health",
    "is_alive", "spotted", "approximate_spotted_by", "team_num", "shots_fired",
]


def _sees(spotters, steamid: str) -> bool:
    if np.ndim(spotters) != 1:
        return False
    return any(str(int(value)) == steamid for value in spotters)


def _wrap(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def parse(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parser = DemoParser(str(path))
    ticks = parser.parse_ticks(TICK_PROPS)
    deaths = parser.parse_event("player_death")
    info = parser.parse_player_info()
    names = {
        str(int(s)): str(n) for s, n in zip(info["steamid"], info["name"])
        if pd.notna(s)
    }
    deaths["attacker_steamid"] = deaths["attacker_steamid"].astype(str)
    ticks["steamid"] = ticks["steamid"].astype(str)
    return ticks, deaths, names


def visibility_frame(ticks: pd.DataFrame, steamid: str) -> pd.DataFrame:
    target = ticks[(ticks["steamid"] == steamid) & ticks["is_alive"]]
    if target.empty:
        raise SystemExit(f"{steamid} is not in the demo")
    team = float(target["team_num"].dropna().mode().iloc[0])
    enemies = ticks[(ticks["team_num"] != team) & ticks["is_alive"]]
    merged = target.merge(enemies, on="tick", suffixes=("", "_e"))
    dx = merged["X_e"] - merged["X"]
    dy = merged["Y_e"] - merged["Y"]
    dz = merged["Z_e"] - merged["Z"]
    horiz = np.hypot(dx, dy)
    expected_yaw = np.degrees(np.arctan2(dy, dx))
    expected_pitch = -np.degrees(
        np.arctan2(dz + EYE_HEIGHT, np.maximum(horiz, 1e-6))
    )
    err_yaw = _wrap(merged["yaw"] - expected_yaw)
    err_pitch = merged["pitch"] - expected_pitch
    merged["error_deg"] = np.hypot(err_yaw, err_pitch)
    merged["distance"] = np.hypot(horiz, dz)
    merged["visible"] = [
        _sees(value, steamid)
        for value in merged["approximate_spotted_by_e"].to_numpy()
    ]
    nearest = merged.loc[merged.groupby("tick")["error_deg"].idxmin()]
    nearest = nearest.sort_values("tick").reset_index(drop=True)
    nearest["enemy_speed"] = np.hypot(
        nearest["X_e"].diff(), nearest["Y_e"].diff()
    ).fillna(0.0) * TICK_RATE
    nearest["unseen_moving_near"] = (
        (nearest["error_deg"] < 5.0)
        & (~nearest["visible"])
        & (nearest["enemy_speed"] > 50.0)
    )
    return nearest


def _runs(flags: np.ndarray) -> list[int]:
    runs, current = [], 0
    for flag in flags:
        current = current + 1 if flag else 0
        if current:
            runs.append(current)
    return runs


def metrics(frame: pd.DataFrame) -> dict:
    preaim = (frame["error_deg"] < 5.0) & (~frame["visible"])
    runs = []
    current = 0
    for flag in preaim:
        current = current + 1 if flag else 0
        if current:
            runs.append(current)
    return {
        "ticks": int(len(frame)),
        "preaim_fraction": float(preaim.mean()),
        "preaim_runs_ge_8": int(sum(1 for r in runs if r >= 8)),
        "preaim_runs_ge_16": int(sum(1 for r in runs if r >= 16)),
        "longest_preaim_run": int(max(runs) if runs else 0),
        "error_visible_mean": float(frame.loc[frame["visible"], "error_deg"].mean())
        if frame["visible"].any() else None,
        "error_unseen_mean": float(frame.loc[~frame["visible"], "error_deg"].mean())
        if (~frame["visible"]).any() else None,
        "visible_fraction": float(frame["visible"].mean()),
        "unseen_moving_near_fraction": float(frame["unseen_moving_near"].mean()),
        "unseen_moving_near_runs_ge_8": int(sum(
            1 for run in _runs(frame["unseen_moving_near"].to_numpy()) if run >= 8
        )),
    }


def plot_kills(frame: pd.DataFrame, deaths: pd.DataFrame, steamid: str,
               names: dict[str, str], out: Path) -> list[dict]:
    kills = deaths[deaths["attacker_steamid"] == steamid].sort_values("tick")
    kills = kills[kills["user_steamid"].astype(str) != steamid]
    panels = len(kills)
    cols = 2
    rows = max(1, int(np.ceil(panels / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    summary = []
    for i, (_, event) in enumerate(kills.iterrows()):
        ax = axes[i // cols][i % cols]
        ax.set_visible(True)
        kill_tick = int(event["tick"])
        victim = str(event["user_steamid"])
        window = frame[(frame["tick"] >= kill_tick - 128)
                       & (frame["tick"] <= kill_tick + 32)]
        relative = (window["tick"] - kill_tick) / TICK_RATE
        error = window["error_deg"]
        ax.plot(relative, error, color="#2f6fed", linewidth=1.2)
        ax.axhline(5.0, color="#9aa0a6", linestyle=":", linewidth=1)
        ax.fill_between(relative, 0, 180, where=window["visible"],
                        color="#34a853", alpha=0.12, step="post")
        shots = window[window["shots_fired"].diff().fillna(0) > 0]
        for t in (shots["tick"] - kill_tick) / TICK_RATE:
            ax.axvline(t, color="#f9ab00", linewidth=0.8)
        ax.axvline(0, color="#d93025", linewidth=1.2)
        ax.set_xlim(-2, 0.5)
        ax.set_ylim(0, 90)
        ax.set_yticks([0, 5, 15, 30, 60, 90])
        ax.set_title(
            f"{int(i + 1)}. {event.get('weapon', '?')}  "
            f"{'HS' if bool(event.get('headshot', False)) else 'body'}  "
            f"dist {float(window['distance'].iloc[-1]) if len(window) else 0:.0f}",
            loc="left", fontsize=8,
        )
        ax.spines[["top", "right"]].set_visible(False)
        pre = window[(window["tick"] < kill_tick) & (~window["visible"])]
        seen = window[(window["tick"] < kill_tick) & window["visible"]]
        summary.append({
            "kill_tick": kill_tick,
            "victim": victim,
            "weapon": str(event.get("weapon", "")),
            "headshot": bool(event.get("headshot", False)),
            "preaim_fraction": float((pre["error_deg"] < 5).mean()) if len(pre) else None,
            "min_error_pre": float(pre["error_deg"].min()) if len(pre) else None,
            "visible_fraction_pre": float(window.loc[window["tick"] < kill_tick, "visible"].mean())
            if (window["tick"] < kill_tick).any() else None,
            "unseen_moving_near_fraction": float(pre["unseen_moving_near"].mean())
            if len(pre) else None,
            "reaction_after_visible_s": float((kill_tick - int(seen["tick"].max())) / TICK_RATE)
            if len(seen) else None,
        })
    fig.suptitle(
        f"Visibility / pre-aim: {names.get(steamid, steamid)}  "
        f"({steamid[-6:]})",
        x=0.01, ha="left", fontsize=11,
    )
    note = (
        "Blue: angular error from the crosshair to the nearest enemy head. "
        "Green shading: that enemy is visible to this player (approximate_spotted_by). "
        "Yellow lines: shots. Red line: kill. Dotted line: 5 degrees. "
        "Tracking near zero error while the shading is absent is pre-aim on an "
        "unseen enemy; short bursts are normal angle holding, long runs are not."
    )
    fig.text(0.01, 0.01, textwrap.fill(note, 150), fontsize=7, color="#5f6368")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--steamid", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ticks, deaths, names = parse(Path(args.demo))
    frame = visibility_frame(ticks, args.steamid)
    result = {
        "demo": args.demo,
        "steamid": args.steamid,
        "name": names.get(args.steamid),
        "overall": metrics(frame),
        "kills": plot_kills(frame, deaths, args.steamid, names,
                            out / "visibility.png"),
    }
    (out / "visibility.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
