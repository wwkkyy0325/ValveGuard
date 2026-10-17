"""Rebuild CS2CD-style windows from the raw match parquet.

The published 256-tick cache cannot be extended backwards, so long-context
experiments rebuild windows from the raw per-tick telemetry of the 160-match
subset.  The reconstruction was checked against the cache: tick alignment,
normalisation, victim join, velocity scaling, shot counts, and the kill flag
all match.

Only the columns used by :func:`summarize` are reconstructed faithfully.  The
published ``*_delta`` columns are left at zero because the code recomputes
signed deltas from the absolute angles anyway.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import MAP_BOUNDS, WEAPON_COLUMNS

RAW_COLUMNS = [
    "tick", "steamid", "X", "Y", "Z", "pitch", "yaw", "velocity",
    "health", "active_weapon_name", "flash_duration", "shots_fired",
    "is_alive",
]

_WEAPON_KEYWORDS: list[tuple[tuple[str, ...], int]] = [
    (("knife", "bayonet", "karambit", "zeus"), 0),
    (("ak-47", "m4a4", "m4a1", "aug", "sg 553", "galil", "famas",
      "m249", "negev"), 1),
    (("awp", "ssg 08", "g3sg1", "scar-20"), 2),
    (("glock", "usp", "p2000", "p250", "five-seven", "tec-9", "cz75",
      "desert eagle", "dual berettas", "r8 revolver"), 3),
    (("grenade", "flashbang", "molotov", "incendiary", "smoke", "decoy"), 4),
    (("mp9", "mac-10", "mp7", "mp5", "ump-45", "p90", "pp-bizon"), 5),
    (("nova", "xm1014", "sawed-off", "mag-7"), 6),
]


def weapon_class(name: str) -> int:
    low = str(name).lower()
    for keys, idx in _WEAPON_KEYWORDS:
        if any(k in low for k in keys):
            return idx
    return 3  # unknown weapons default to pistol


def _norm(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - lo) / (hi - lo)


def build_window(
    attacker: pd.DataFrame,
    victim: pd.DataFrame,
    ticks: np.ndarray,
    map_name: str,
    kill_tick: int,
    event: dict,
) -> pd.DataFrame | None:
    """Return a 44-column window or None if the ticks are incomplete."""
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = MAP_BOUNDS[map_name]
    a = attacker.reindex(ticks)
    v = victim.reindex(ticks)
    if a[["X", "Y", "Z", "pitch", "yaw"]].isna().any().any():
        return None
    if v[["X", "Y", "Z", "health"]].isna().any().any():
        return None

    n = len(ticks)
    out = pd.DataFrame(index=np.arange(n))

    out["attacker_X"] = _norm(a["X"].to_numpy(), lo_x, hi_x)
    out["attacker_Y"] = _norm(a["Y"].to_numpy(), lo_y, hi_y)
    out["attacker_Z"] = _norm(a["Z"].to_numpy(), lo_z, hi_z)
    out["attacker_vel"] = np.nan_to_num(a["velocity"].to_numpy(), nan=0.0) / 250.0
    out["attacker_pitch"] = (a["pitch"].to_numpy() + 90.0) / 180.0
    out["attacker_yaw"] = (a["yaw"].to_numpy() + 180.0) / 360.0
    for col in (
        "attacker_pitch_delta", "attacker_yaw_delta",
        "attacker_pitch_head_delta", "attacker_yaw_head_delta",
    ):
        out[col] = 0.0

    out["attacker_flashed"] = (a["flash_duration"].to_numpy() > 0).astype(float)
    shots = np.nan_to_num(a["shots_fired"].to_numpy(), nan=0.0)
    out["attacker_shot"] = np.clip(np.diff(shots, prepend=shots[0]), 0.0, None)
    out["attacker_kill"] = (ticks == kill_tick).astype(float)

    out["is_kill_headshot"] = float(bool(event.get("headshot", False)))
    out["is_kill_through_smoke"] = float(bool(event.get("thrusmoke", False)))
    out["is_kill_wallbang"] = float(int(event.get("penetrated", 0)) > 0)
    out["attacker_midair"] = float(bool(event.get("attackerinair", False)))

    wc = a["active_weapon_name"].map(weapon_class).to_numpy()
    for i, col in enumerate(WEAPON_COLUMNS):
        out[col] = (wc == i).astype(float)

    out["victim_X"] = _norm(v["X"].to_numpy(), lo_x, hi_x)
    out["victim_Y"] = _norm(v["Y"].to_numpy(), lo_y, hi_y)
    out["victim_Z"] = _norm(v["Z"].to_numpy(), lo_z, hi_z)
    out["victim_health"] = np.clip(v["health"].to_numpy() / 100.0, 0.0, 1.0)
    out["victim_noise"] = 0.0

    for col in ("map_dust2", "map_mirage", "map_inferno", "map_train",
                "map_nuke", "map_ancient", "map_vertigo", "map_anubis",
                "map_office", "map_overpass", "map_basalt", "map_edin",
                "map_italy", "map_thera", "map_mills"):
        out[col] = float(col == f"map_{map_name.replace('de_', '').replace('cs_', '')}")
    return out
