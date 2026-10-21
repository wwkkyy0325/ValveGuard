"""Build the 43-channel kill-window tensors from a CS2 demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.features.raw_window import build_window, weapon_class
from valveguard.features.summarize import per_tick_channels
from valveguard.scripts.build_pro_windows import KILL_ROW, parse_demo


def _flag(value) -> float:
    return 0.0 if pd.isna(value) else float(bool(value))


def _number(value) -> float:
    return 0.0 if pd.isna(value) else float(value)


def demo_channels(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (frame, channels, names) for every valid kill in a demo."""
    map_name, ticks, events = parse_demo(path)
    by_player = {
        sid: d.sort_values("tick").set_index("tick", drop=False)
        for sid, d in ticks.groupby("steamid", sort=False)
    }
    rows, channels, names = [], [], []
    for _, ev in events.sort_values("tick").iterrows():
        attacker_id = ev["attacker_steamid"]
        victim_id = ev["user_steamid"]
        if pd.isna(attacker_id) or pd.isna(victim_id) or attacker_id == victim_id:
            continue
        if ev["attacker_steamid"] not in by_player or ev["user_steamid"] not in by_player:
            continue
        kill_tick = int(ev["tick"])
        if kill_tick < KILL_ROW:
            continue
        kill_row = ticks[
            (ticks["steamid"] == ev["attacker_steamid"]) & (ticks["tick"] == kill_tick)
        ]
        if len(kill_row) == 0:
            continue
        warm = kill_row["is_warmup_period"].iloc[0]
        if pd.notna(warm) and bool(warm):
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
        matrix, window_names = per_tick_channels(window)
        rows.append({
            "player_name": str(int(ev["attacker_steamid"])),
            "kill_tick": kill_tick,
            "map": map_name,
            "headshot": _flag(ev.get("headshot")),
            "weapon_class": weapon_class(str(ev.get("weapon", ""))),
            "shots": float(window["attacker_shot"].iloc[:KILL_ROW].sum()),
            "through_smoke": _flag(ev.get("thrusmoke")),
            "wallbang": float(_number(ev.get("penetrated")) > 0),
            "midair": _flag(ev.get("attackerinair")),
        })
        channels.append(matrix)
        names = window_names
    if not rows:
        return pd.DataFrame(), np.zeros((0, 256, 0), dtype=np.float32), []
    return pd.DataFrame(rows), np.stack(channels).astype(np.float32), names
