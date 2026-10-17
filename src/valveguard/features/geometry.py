"""Geometry reconstruction from a CS2CD context window.

A window is 256 ticks x 44 features.  Row 223 is the kill tick (0-based);
the window covers 223 ticks before the kill and 32 ticks after it.  The
published features use normalized absolute coordinates and angles, so the
original values can be reconstructed with the map bounds used by the
dataset author (the same table as AntiCheatPT).

Everything here is vectorized and depends only on pandas/numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KILL_ROW = 223
TICK_RATE = 64.0
EYE_HEIGHT = 64.0  # CS2 player eye height in world units

# (min_x, max_x, min_y, max_y, min_z, max_z), copied from the CS2CD
# conversion code (MatchDataProcessor.map_min_max).
MAP_BOUNDS: dict[str, tuple[float, float, float, float, float, float]] = {
    "de_dust2": (-2300, 1800, -1200, 3150, -130, 400),
    "de_mirage": (-2700, 1500, -2700, 1000, -400, 150),
    "de_inferno": (-1800, 2700, -800, 3600, -100, 500),
    "de_train": (-2200, 1800, -1800, 1800, -400, 200),
    "de_nuke": (-3000, 3500, -2500, 1000, -750, 100),
    "de_ancient": (-2310, 1400, -2600, 1800, -120, 400),
    "de_vertigo": (-2700, 100, -1600, 1200, 11400, 12100),
    "de_anubis": (-2000, 1810, -1810, 3200, -150, 200),
    "cs_office": (-1800, 2400, -2200, 1300, -350, 10),
    "de_overpass": (-4000, 20, -3500, 1700, 0, 800),
    "de_basalt": (-2100, 2000, -1700, 2350, -100, 400),
    "de_edin": (500, 3700, -350, 4300, 300, 750),
    "cs_italy": (-1550, 1100, -2200, 2650, -200, 300),
    "de_thera": (600, 4300, -2600, 2200, -170, 300),
    "de_mills": (-4300, 0, -5560, -300, -100, 300),
}

MAP_COLUMNS = {
    "map_dust2": "de_dust2",
    "map_mirage": "de_mirage",
    "map_inferno": "de_inferno",
    "map_train": "de_train",
    "map_nuke": "de_nuke",
    "map_ancient": "de_ancient",
    "map_vertigo": "de_vertigo",
    "map_anubis": "de_anubis",
    "map_office": "cs_office",
    "map_overpass": "de_overpass",
    "map_basalt": "de_basalt",
    "map_edin": "de_edin",
    "map_italy": "cs_italy",
    "map_thera": "de_thera",
    "map_mills": "de_mills",
}

WEAPON_COLUMNS = [
    "attacker_weapon_knife",
    "attacker_weapon_auto_rifle",
    "attacker_weapon_semi_rifle",
    "attacker_weapon_pistol",
    "attacker_weapon_grenade",
    "attacker_weapon_smg",
    "attacker_weapon_shotgun",
]


def wrap_deg(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap degrees to [-180, 180)."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def get_map(window: pd.DataFrame) -> str:
    for col, name in MAP_COLUMNS.items():
        if col in window.columns and float(window[col].iloc[0]) > 0.5:
            return name
    raise ValueError("No map one-hot column is set in this window")


def _denorm(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + np.asarray(values, dtype=np.float32) * (hi - lo)


def recover(window: pd.DataFrame) -> dict[str, np.ndarray | str]:
    """Recover world coordinates, angles, and target-relative errors."""
    map_name = get_map(window)
    if map_name not in MAP_BOUNDS:
        raise ValueError(f"Unknown map: {map_name}")
    min_x, max_x, min_y, max_y, min_z, max_z = MAP_BOUNDS[map_name]

    n = len(window)
    if n != KILL_ROW + 33:
        raise ValueError(f"Expected 256 ticks, got {n}")

    pitch = np.asarray(window["attacker_pitch"], dtype=np.float32) * 180.0 - 90.0
    yaw = np.asarray(window["attacker_yaw"], dtype=np.float32) * 360.0 - 180.0
    # A handful of published rows contain out-of-range normalized pitch
    # (max 1.5 in a 20k-window sample). Clamp to a physically valid range.
    pitch = np.clip(pitch, -90.0, 90.0)
    yaw = wrap_deg(yaw)

    ax = _denorm(window["attacker_X"], min_x, max_x)
    ay = _denorm(window["attacker_Y"], min_y, max_y)
    az = _denorm(window["attacker_Z"], min_z, max_z)
    vx = _denorm(window["victim_X"], min_x, max_x)
    vy = _denorm(window["victim_Y"], min_y, max_y)
    vz = _denorm(window["victim_Z"], min_z, max_z)

    dx, dy, dz = vx - ax, vy - ay, vz - az
    horiz = np.hypot(dx, dy)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    expected_yaw = np.degrees(np.arctan2(dy, dx))
    # CS2's stored pitch has the opposite sign of atan2(dz, horizontal).
    # This was verified against the raw match: at the kill tick the
    # origin-relative error drops from 6.6 deg to 0.2 deg when inverted.
    expected_pitch = -np.degrees(np.arctan2(dz, np.maximum(horiz, 1e-6)))
    expected_pitch_head = -np.degrees(
        np.arctan2(dz + EYE_HEIGHT, np.maximum(horiz, 1e-6))
    )

    err_yaw = wrap_deg(yaw - expected_yaw)
    err_pitch = pitch - expected_pitch
    err_pitch_head = pitch - expected_pitch_head
    err_origin = np.hypot(err_yaw, err_pitch)
    err_head = np.hypot(err_yaw, err_pitch_head)

    # Signed angular deltas from the absolute angles.  The published
    # *_delta columns are absolute values, so we recompute them here.
    dyaw = np.diff(yaw, prepend=yaw[0])
    dpitch = np.diff(pitch, prepend=pitch[0])
    dyaw = wrap_deg(dyaw)

    t = (np.arange(n, dtype=np.float32) - KILL_ROW) / TICK_RATE

    return {
        "map": map_name,
        "pitch": pitch,
        "yaw": yaw,
        "ax": ax,
        "ay": ay,
        "az": az,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "distance": dist,
        "horiz_distance": horiz,
        "expected_yaw": expected_yaw,
        "expected_pitch": expected_pitch,
        "expected_pitch_head": expected_pitch_head,
        "err_yaw": err_yaw,
        "err_pitch": err_pitch,
        "err_pitch_head": err_pitch_head,
        "err_origin": err_origin,
        "err_head": err_head,
        "dyaw": dyaw,
        "dpitch": dpitch,
        "t": t,
    }
