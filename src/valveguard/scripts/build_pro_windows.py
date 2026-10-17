"""Build kill windows from CS2 pro demos (human negatives).

The demos come from the HLTV CS2 demo dataset (`blanchon/opencs2_dataset_demo`)
and are treated as clean human negatives.  Each demo is parsed with
demoparser2, every non-suicide kill with 256 ticks of history becomes one
window, and the standard summary features are computed with the same code used
for CS2CD.

Output is one parquet per demo plus a combined file.  Player ids are raw
SteamIDs; they are only used within a demo, never joined across demos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from valveguard.features.raw_window import build_window
from valveguard.features.summarize import summarize

TICK_PROPS = [
    "X", "Y", "Z", "pitch", "yaw", "velocity", "health",
    "active_weapon_name", "flash_duration", "shots_fired", "is_alive",
    "total_rounds_played",
]
KILL_ROW = 223


def _to_id(value) -> int | None:
    """Parse a SteamID without going through float64 (which loses precision)."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_demo(path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    parser = DemoParser(str(path))
    header = parser.parse_header()
    ticks = parser.parse_ticks(TICK_PROPS)
    events = parser.parse_event("player_death")
    map_name = header.get("map_name", "")
    ticks["steamid"] = ticks["steamid"].astype("int64")
    for col in ("attacker_steamid", "user_steamid"):
        # Int64, not a plain list: int+None would become float64 and lose
        # precision on 17-digit SteamIDs.
        events[col] = pd.array([_to_id(x) for x in events[col]], dtype="Int64")
    return map_name, ticks, events


def demo_windows(path: Path) -> pd.DataFrame:
    map_name, ticks, events = parse_demo(path)
    by_player = {
        sid: d.sort_values("tick").set_index("tick", drop=False)
        for sid, d in ticks.groupby("steamid", sort=False)
    }

    rows = []
    for _, ev in events.sort_values("tick").iterrows():
        if pd.isna(ev["attacker_steamid"]) or ev["attacker_steamid"] == ev["user_steamid"]:
            continue
        if ev["attacker_steamid"] not in by_player or ev["user_steamid"] not in by_player:
            continue
        kill_tick = int(ev["tick"])
        if kill_tick < KILL_ROW:
            continue
        kill_row = ticks[
            (ticks["steamid"] == ev["attacker_steamid"]) & (ticks["tick"] == kill_tick)
        ]
        if len(kill_row) == 0 or int(kill_row["total_rounds_played"].iloc[0]) < 1:
            continue
        tick_range = np.arange(kill_tick - KILL_ROW, kill_tick + 33)
        attacker = by_player[ev["attacker_steamid"]]
        victim = by_player[ev["user_steamid"]]
        if attacker.reindex(tick_range)[["X", "pitch", "yaw"]].isna().any().any():
            continue
        if victim.reindex(tick_range)[["X", "health"]].isna().any().any():
            continue
        window = build_window(
            attacker, victim, tick_range, map_name, kill_tick, ev.to_dict()
        )
        if window is None:
            continue
        feat = summarize(window)
        feat["player_name"] = str(int(ev["attacker_steamid"]))
        feat["match_key"] = path.stem
        feat["kill_tick"] = kill_tick
        rows.append(feat)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["kill_idx"] = out.groupby("player_name").cumcount()
    out["label"] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    demos = sorted(Path(args.demos).rglob("*.dem"))
    print(f"found {len(demos)} demos", flush=True)
    summary = []
    for i, path in enumerate(demos, 1):
        try:
            df = demo_windows(path)
        except Exception as exc:
            print(f"ERR {path.name}: {exc!r}", flush=True)
            summary.append({"demo": path.stem, "rows": 0, "error": repr(exc)})
            continue
        if len(df):
            df.to_parquet(parts_dir / f"{path.stem}.parquet", index=False)
        summary.append({"demo": path.stem, "rows": len(df), "error": None})
        print(f"{i}/{len(demos)} {path.name}: {len(df)} windows", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))]
    parts = [p for p in parts if len(p)]
    if parts:
        all_df = pd.concat(parts, ignore_index=True)
        all_df.to_parquet(out / "pro_windows.parquet", index=False)
        print(f"rows {len(all_df)} demos {all_df.match_key.nunique()} "
              f"players {all_df.groupby(['match_key','player_name']).ngroups}",
              flush=True)
    with open(out / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
