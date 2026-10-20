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
    "tick", "steamid", "X", "Y", "Z", "pitch", "yaw",
    "usercmd_mouse_dx", "usercmd_mouse_dy", "aim_punch_angle",
    "shots_fired", "last_shot_time", "is_scoped", "is_walking", "ducking",
    "is_airborne", "velocity", "health", "is_alive", "spotted",
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


def event_summary(
    attacker: pd.DataFrame,
    victim: pd.DataFrame | None,
    ticks: np.ndarray,
    kill_tick: int,
) -> dict:
    """Summarize one 256-tick attacker window; callers must ensure full ticks."""
    frame = attacker.reindex(ticks)
    out: dict[str, float] = {}
    kill_index = int(np.argmin(np.abs(ticks - kill_tick)))

    pitch = frame["pitch"].to_numpy(dtype=float)
    yaw = frame["yaw"].to_numpy(dtype=float)
    d_pitch = np.diff(pitch, prepend=pitch[0])
    d_yaw = np.diff(yaw, prepend=yaw[0])
    d_yaw = (d_yaw + 180.0) % 360.0 - 180.0

    err_pre = np.full(kill_index + 1, np.nan, dtype=float)
    err_head_pre = np.full(kill_index + 1, np.nan, dtype=float)
    victim_frame = victim.reindex(ticks) if victim is not None else None
    has_target = (
        victim_frame is not None
        and all(column in frame.columns for column in ("X", "Y", "Z"))
        and all(column in victim_frame.columns for column in ("X", "Y", "Z"))
    )
    if has_target:
        ax = frame["X"].to_numpy(dtype=float)
        ay = frame["Y"].to_numpy(dtype=float)
        az = frame["Z"].to_numpy(dtype=float)
        vx = victim_frame["X"].to_numpy(dtype=float)
        vy = victim_frame["Y"].to_numpy(dtype=float)
        vz = victim_frame["Z"].to_numpy(dtype=float)
        dx, dy, dz = vx - ax, vy - ay, vz - az
        horizontal = np.hypot(dx, dy)
        valid = (
            np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz)
            & np.isfinite(pitch) & np.isfinite(yaw)
        )
        expected_yaw = np.degrees(np.arctan2(dy, dx))
        expected_pitch = -np.degrees(
            np.arctan2(dz, np.maximum(horizontal, 1e-6))
        )
        expected_pitch_head = -np.degrees(
            np.arctan2(dz + 64.0, np.maximum(horizontal, 1e-6))
        )
        err_yaw = (yaw - expected_yaw + 180.0) % 360.0 - 180.0
        err_pitch = pitch - expected_pitch
        err_head = pitch - expected_pitch_head
        err_origin = np.hypot(err_yaw, err_pitch)
        err_origin[~valid] = np.nan
        err_head[~valid] = np.nan
        err_pre = err_origin[: kill_index + 1]
        err_head_pre = err_head[: kill_index + 1]

    finite_err = np.isfinite(err_pre)
    out["ev_err_initial"] = float(err_pre[0]) if len(err_pre) and finite_err[0] else np.nan
    out["ev_err_mean_pre"] = float(np.nanmean(err_pre)) if finite_err.any() else np.nan
    out["ev_err_std_pre"] = float(np.nanstd(err_pre)) if finite_err.any() else np.nan
    out["ev_err_min_pre"] = float(np.nanmin(err_pre)) if finite_err.any() else np.nan
    out["ev_err_at_kill"] = (
        float(err_pre[-1]) if len(err_pre) and finite_err[-1] else np.nan
    )
    out["ev_err_head_mean_pre"] = (
        float(np.nanmean(err_head_pre)) if np.isfinite(err_head_pre).any() else np.nan
    )
    for lag in (8, 16, 32, 64):
        index = kill_index - lag
        out[f"ev_err_minus_{lag}"] = (
            float(err_pre[index]) if index >= 0 and finite_err[index] else np.nan
        )

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

    acquire: dict[float, int | None] = {}
    for threshold in (3.0, 5.0, 10.0):
        hits = np.flatnonzero(np.isfinite(err_pre) & (err_pre <= threshold))
        key = int(threshold)
        if len(hits):
            first = int(hits[0])
            acquire[threshold] = first
            out[f"ev_acquire_offset_{key}"] = float(first)
            out[f"ev_acquire_to_kill_{key}"] = float(kill_index - first)
            out[f"ev_time_on_target_{key}"] = float(len(hits))
            out[f"ev_post_acquire_max_err_{key}"] = float(
                np.nanmax(err_pre[first:])
            )
        else:
            acquire[threshold] = None
            out[f"ev_acquire_offset_{key}"] = np.nan
            out[f"ev_acquire_to_kill_{key}"] = np.nan
            out[f"ev_time_on_target_{key}"] = np.nan
            out[f"ev_post_acquire_max_err_{key}"] = np.nan

    before_kill = (
        shot_index[ticks[shot_index] <= kill_tick]
        if len(shot_index) else np.array([], dtype=int)
    )
    if len(before_kill):
        first_shot = int(before_kill[0])
        out["ev_first_shot_to_kill"] = float(kill_index - first_shot)
        out["ev_error_at_first_shot"] = (
            float(err_pre[first_shot])
            if first_shot <= kill_index and np.isfinite(err_pre[first_shot])
            else np.nan
        )
        out["ev_error_drop_before_shot"] = (
            float(err_pre[first_shot] - err_pre[first_shot - 8])
            if first_shot >= 8
            and np.isfinite(err_pre[first_shot])
            and np.isfinite(err_pre[first_shot - 8])
            else np.nan
        )
        window = err_pre[: first_shot + 1]
        out["ev_time_on_target_before_first_shot_5"] = float(
            np.sum(np.isfinite(window) & (window <= 5.0))
        )
        for threshold in (3.0, 5.0, 10.0):
            first = acquire.get(threshold)
            out[f"ev_first_shot_after_acquire_{int(threshold)}"] = (
                float(first_shot - first)
                if first is not None and first_shot >= first else np.nan
            )
    else:
        out["ev_first_shot_to_kill"] = np.nan
        out["ev_error_at_first_shot"] = np.nan
        out["ev_error_drop_before_shot"] = np.nan
        out["ev_time_on_target_before_first_shot_5"] = np.nan
        for threshold in (3.0, 5.0, 10.0):
            out[f"ev_first_shot_after_acquire_{int(threshold)}"] = np.nan

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
