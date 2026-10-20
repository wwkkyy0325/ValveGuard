"""Recoil, fire-timing, and mouse-view features from raw per-tick telemetry.

The published 44-channel cache keeps only the view trajectory and a few kill
flags. The raw CS2CD parquet also carries ``aim_punch_angle``, the per-tick
mouse deltas, the spray counter, and the shot timestamp, which expose how a
player controls recoil and times fire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "tick", "steamid", "pitch", "yaw", "usercmd_mouse_dx", "usercmd_mouse_dy",
    "aim_punch_angle", "shots_fired", "last_shot_time", "is_scoped",
    "is_walking", "ducking", "is_airborne", "velocity", "health", "is_alive",
    "spotted",
]


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _stats(out: dict, prefix: str, values) -> None:
    array = _finite(values)
    if len(array) == 0:
        for suffix in ("mean", "std", "min", "max", "p95"):
            out[f"{prefix}_{suffix}"] = np.nan
        return
    out[f"{prefix}_mean"] = float(np.mean(array))
    out[f"{prefix}_std"] = float(np.std(array))
    out[f"{prefix}_min"] = float(np.min(array))
    out[f"{prefix}_max"] = float(np.max(array))
    out[f"{prefix}_p95"] = float(np.percentile(array, 95))


def _corr(left, right) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _mouse_residual(view: np.ndarray, mouse: np.ndarray) -> tuple[float, np.ndarray]:
    ok = np.isfinite(view) & np.isfinite(mouse)
    if ok.sum() < 10 or np.var(mouse[ok]) < 1e-9:
        return np.nan, np.empty(0, dtype=float)
    beta = float(np.cov(view[ok], mouse[ok], bias=True)[0, 1] / np.var(mouse[ok]))
    residual = view[ok] - beta * mouse[ok]
    return beta, residual


def event_summary(attacker: pd.DataFrame, ticks: np.ndarray, kill_tick: int) -> dict:
    """Summarize one 256-tick attacker window; callers must ensure full ticks."""
    frame = attacker.reindex(ticks)
    out: dict[str, float] = {}

    pitch = frame["pitch"].to_numpy(dtype=float)
    yaw = frame["yaw"].to_numpy(dtype=float)
    d_pitch = np.diff(pitch, prepend=pitch[0])
    d_yaw = np.diff(yaw, prepend=yaw[0])
    d_yaw = (d_yaw + 180.0) % 360.0 - 180.0

    raw_punch = frame["aim_punch_angle"].to_numpy()
    punch = np.zeros((len(ticks), 3), dtype=float)
    for i, value in enumerate(raw_punch):
        try:
            item = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if item.size >= 2:
            punch[i, : min(3, item.size)] = item[:3]
    punch = np.nan_to_num(punch, nan=0.0, posinf=0.0, neginf=0.0)
    punch_pitch = punch[:, 0]
    punch_yaw = punch[:, 1]
    punch_mag = np.hypot(punch_pitch, punch_yaw)
    d_punch_pitch = np.diff(punch_pitch, prepend=punch_pitch[0])
    d_punch_yaw = np.diff(punch_yaw, prepend=punch_yaw[0])

    _stats(out, "ev_punch_pitch", punch_pitch)
    _stats(out, "ev_punch_yaw", punch_yaw)
    _stats(out, "ev_punch_mag", punch_mag)
    _stats(out, "ev_punch_delta_pitch", d_punch_pitch)
    _stats(out, "ev_punch_delta_yaw", d_punch_yaw)
    out["ev_view_punch_corr_pitch"] = _corr(d_pitch[1:], -d_punch_pitch[1:])
    out["ev_view_punch_corr_yaw"] = _corr(d_yaw[1:], -d_punch_yaw[1:])
    out["ev_recoil_resid_std_pitch"] = float(np.nanstd(d_pitch + d_punch_pitch))
    out["ev_recoil_resid_std_yaw"] = float(np.nanstd(d_yaw + d_punch_yaw))

    last_shot = frame["last_shot_time"].to_numpy(dtype=float)
    shot = np.zeros(len(ticks), dtype=bool)
    with np.errstate(invalid="ignore"):
        shot[1:] = np.isfinite(last_shot[1:]) & np.isfinite(last_shot[:-1]) & (
            last_shot[1:] > last_shot[:-1] + 1e-6
        )
    shot_index = np.flatnonzero(shot)
    out["ev_shots_window"] = float(len(shot_index))
    if len(shot_index):
        times = last_shot[shot_index]
        intervals = np.diff(times)
        _stats(out, "ev_shot_interval", intervals)
        out["ev_first_shot_offset"] = float(shot_index[0])
        before_kill = shot_index[ticks[shot_index] <= kill_tick]
        out["ev_last_shot_to_kill"] = (
            float(kill_tick - ticks[before_kill[-1]]) if len(before_kill) else np.nan
        )
        out["ev_shots_before_kill"] = float(len(before_kill))
    else:
        for suffix in ("mean", "std", "min", "max", "p95"):
            out[f"ev_shot_interval_{suffix}"] = np.nan
        out["ev_first_shot_offset"] = np.nan
        out["ev_last_shot_to_kill"] = np.nan
        out["ev_shots_before_kill"] = 0.0

    spray = frame["shots_fired"].to_numpy(dtype=float)
    out["ev_spray_max"] = float(np.nanmax(spray)) if np.isfinite(spray).any() else np.nan
    rising = np.zeros(len(ticks), dtype=bool)
    if len(ticks):
        rising[0] = np.isfinite(spray[0]) and spray[0] > 0
        rising[1:] = (
            np.isfinite(spray[1:]) & np.isfinite(spray[:-1])
            & (spray[1:] > 0) & (spray[:-1] == 0)
        )
    out["ev_bursts"] = float(rising.sum())

    mouse_x = frame["usercmd_mouse_dx"].to_numpy(dtype=float)
    mouse_y = frame["usercmd_mouse_dy"].to_numpy(dtype=float)
    beta_yaw, residual_yaw = _mouse_residual(d_yaw, mouse_x)
    beta_pitch, residual_pitch = _mouse_residual(d_pitch, mouse_y)
    out["ev_mouse_view_beta_yaw"] = beta_yaw
    out["ev_mouse_view_beta_pitch"] = beta_pitch
    _stats(out, "ev_mouse_view_resid_yaw", residual_yaw)
    _stats(out, "ev_mouse_view_resid_pitch", residual_pitch)

    for column, name in [
        ("is_scoped", "scoped"), ("is_walking", "walking"),
        ("ducking", "ducking"), ("is_airborne", "airborne"),
        ("spotted", "spotted"), ("is_alive", "alive"),
    ]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        out[f"ev_{name}_frac"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan

    velocity = pd.to_numeric(frame["velocity"], errors="coerce").to_numpy(dtype=float)
    _stats(out, "ev_velocity", velocity)
    return out
