"""Turn a 256 x 44 CS2CD window into a flat feature vector and a per-tick
channel matrix.

The first detector uses these handcrafted features with a gradient-boosted
tree baseline.  The same functions feed the transformer later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dynamics import (
    angular_accel,
    angular_jerk,
    angular_speed,
    curvature,
    ldlj,
    lead_lag,
    moving_average,
    path_length,
    psd_slope,
    sample_entropy,
    sparc,
    straightness,
    submovement_count,
)
from .geometry import KILL_ROW, MAP_COLUMNS, TICK_RATE, WEAPON_COLUMNS, recover, wrap_deg


def _add_stats(out: dict, prefix: str, values: np.ndarray) -> None:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_std"] = np.nan
        out[f"{prefix}_min"] = np.nan
        out[f"{prefix}_max"] = np.nan
        out[f"{prefix}_p95"] = np.nan
        return
    out[f"{prefix}_mean"] = float(np.mean(v))
    out[f"{prefix}_std"] = float(np.std(v))
    out[f"{prefix}_min"] = float(np.min(v))
    out[f"{prefix}_max"] = float(np.max(v))
    out[f"{prefix}_p95"] = float(np.quantile(v, 0.95))


def summarize(window: pd.DataFrame) -> dict:
    """Return a flat dict of summary features for one window."""
    g = recover(window)
    out: dict = {"map": g["map"]}

    t = np.asarray(g["t"])
    yaw = np.asarray(g["yaw"])
    pitch = np.asarray(g["pitch"])
    dyaw = np.asarray(g["dyaw"])
    dpitch = np.asarray(g["dpitch"])
    err_head = np.asarray(g["err_head"])
    err_origin = np.asarray(g["err_origin"])
    err = err_origin  # origin-relative error aligns with CS2's pitch sign
    err_yaw = np.asarray(g["err_yaw"])
    err_pitch = np.asarray(g["err_pitch"])
    dist = np.asarray(g["distance"])
    expected_yaw = np.asarray(g["expected_yaw"])

    speed = angular_speed(dyaw, dpitch) * TICK_RATE  # deg/s
    accel = angular_accel(dyaw, dpitch)
    jerk = angular_jerk(dyaw, dpitch)

    sl64 = slice(max(0, KILL_ROW - 63), KILL_ROW + 1)
    sl128 = slice(max(0, KILL_ROW - 127), KILL_ROW + 1)
    sl16 = slice(max(0, KILL_ROW - 15), KILL_ROW + 1)

    # ---- target alignment ------------------------------------------------
    target_idx = np.where(err[sl128] < 2.0)[0]
    if len(target_idx):
        first_target = int(target_idx[0]) + sl128.start
        out["time_to_target"] = float((KILL_ROW - first_target) / TICK_RATE)
    else:
        out["time_to_target"] = np.nan

    cross = np.where(err[sl128] < 1.0)[0]
    if len(cross):
        cross_idx = int(cross[0]) + sl128.start
        after = err[cross_idx:KILL_ROW + 1]
        out["reached_target"] = 1.0
        out["overshoot"] = float(max(0.0, np.max(after) - 1.0))
        above = np.where(after > 1.0)[0]
        out["settling_time"] = (
            float((KILL_ROW - (cross_idx + int(above[-1]))) / TICK_RATE)
            if len(above) else 0.0
        )
    else:
        out["reached_target"] = 0.0
        out["overshoot"] = np.nan
        out["settling_time"] = np.nan

    out["final_err"] = float(err_origin[KILL_ROW])
    out["min_err"] = float(np.min(err_origin[sl64]))
    out["mean_err"] = float(np.mean(err_origin[sl64]))
    out["std_err"] = float(np.std(err_origin[sl64]))
    out["final_err_yaw"] = float(err_yaw[KILL_ROW])
    out["final_err_pitch"] = float(err_pitch[KILL_ROW])
    out["final_err_head"] = float(err_head[KILL_ROW])
    out["mean_err_head"] = float(np.mean(err_head[sl64]))

    # Lock: how long the crosshair sits on target with almost no movement.
    lock = (err_origin[sl16] < 1.0) & (speed[sl16] < 5.0)
    out["lock_ratio"] = float(np.mean(lock)) if len(lock) else np.nan

    # ---- dynamics ---------------------------------------------------------
    _add_stats(out, "speed_dps", speed[sl64])
    _add_stats(out, "speed_last16", speed[sl16])
    _add_stats(out, "accel", accel[sl64])
    _add_stats(out, "jerk", jerk[sl64])
    _add_stats(out, "dist", dist[sl64])
    out["dist_final"] = float(dist[KILL_ROW])
    out["dist_change_rate"] = float(
        np.polyfit(t[sl64], dist[sl64], 1)[0]
    ) if len(dist[sl64]) >= 2 else np.nan

    out["path_len"] = float(path_length(yaw[sl64], pitch[sl64]))
    out["straightness"] = float(straightness(yaw[sl64], pitch[sl64]))
    curv = curvature(yaw[sl64], pitch[sl64])
    _add_stats(out, "curvature", np.clip(curv, 0, 1e3))
    out["sparc"] = float(sparc(speed[sl64], TICK_RATE))
    out["ldlj"] = float(ldlj(yaw[sl64], pitch[sl64], TICK_RATE))
    out["psd_slope"] = float(psd_slope(speed[sl64], TICK_RATE))
    out["sample_entropy"] = float(sample_entropy(speed[KILL_ROW - 31:KILL_ROW + 1]))
    out["submovements"] = int(submovement_count(speed[sl64]))

    # ---- victim / attacker motion ----------------------------------------
    vx = np.asarray(g["vx"])
    vy = np.asarray(g["vy"])
    victim_speed = np.hypot(np.gradient(vx) * TICK_RATE, np.gradient(vy) * TICK_RATE)
    _add_stats(out, "victim_speed", victim_speed[sl64])
    attacker_vel = np.asarray(window["attacker_vel"], dtype=np.float64) * 250.0
    _add_stats(out, "attacker_vel", attacker_vel[sl64])

    exp_vel = wrap_deg(np.diff(expected_yaw, prepend=expected_yaw[0])) * TICK_RATE
    act_vel = dyaw * TICK_RATE
    lag, corr = lead_lag(act_vel[sl128], exp_vel[sl128], max_lag=12)
    out["lead_lag_ticks"] = float(lag)
    out["lead_lag_corr"] = float(corr)

    # ---- kill / context ---------------------------------------------------
    kill = window.iloc[KILL_ROW]
    out["shot_count"] = float(np.sum(window["attacker_shot"].to_numpy()[sl64]))
    out["flashed_at_kill"] = float(kill.get("attacker_flashed", 0.0))
    out["headshot"] = float(kill.get("is_kill_headshot", 0.0))
    out["through_smoke"] = float(kill.get("is_kill_through_smoke", 0.0))
    out["wallbang"] = float(kill.get("is_kill_wallbang", 0.0))
    out["midair"] = float(kill.get("attacker_midair", 0.0))
    out["victim_health"] = float(kill.get("victim_health", 0.0))
    out["victim_noise"] = float(kill.get("victim_noise", 0.0))

    weapon = 0
    for i, col in enumerate(WEAPON_COLUMNS):
        val = float(kill.get(col, 0.0))
        out[f"weapon_{i}"] = val
        if val > 0.5:
            weapon = i
    out["weapon_idx"] = weapon

    return out


def per_tick_channels(window: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build a (256, C) matrix for sequence models.

    Raw non-map features are kept as-is (they are already normalized).
    The published *_delta columns are absolute values, and the published
    pitch-head delta uses the wrong pitch sign; they are dropped and
    replaced by signed, corrected channels.
    """
    g = recover(window)
    raw_drop = {
        "attacker_pitch_delta",
        "attacker_yaw_delta",
        "attacker_pitch_head_delta",
        "attacker_yaw_head_delta",
    }
    cols = [
        c for c in window.columns
        if c not in MAP_COLUMNS and c not in raw_drop
    ]
    out = window[cols].to_numpy(dtype=np.float32)

    yaw = np.asarray(g["yaw"], dtype=np.float32)
    pitch = np.asarray(g["pitch"], dtype=np.float32)
    dyaw = np.asarray(g["dyaw"], dtype=np.float32)
    dpitch = np.asarray(g["dpitch"], dtype=np.float32)
    expected_yaw = np.asarray(g["expected_yaw"], dtype=np.float32)

    speed = angular_speed(dyaw, dpitch) * TICK_RATE
    accel = angular_accel(dyaw, dpitch)
    jerk = angular_jerk(dyaw, dpitch)
    exp_vel = wrap_deg(np.diff(expected_yaw, prepend=expected_yaw[0])) * TICK_RATE
    act_vel = dyaw * TICK_RATE
    dist = np.asarray(g["distance"], dtype=np.float32)
    victim_speed = np.hypot(
        np.gradient(np.asarray(g["vx"])) * TICK_RATE,
        np.gradient(np.asarray(g["vy"])) * TICK_RATE,
    )
    attacker_speed = np.asarray(window["attacker_vel"], dtype=np.float32) * 250.0

    derived = np.column_stack([
        dyaw * TICK_RATE,
        dpitch * TICK_RATE,
        speed,
        accel,
        jerk,
        np.asarray(g["err_yaw"]),
        np.asarray(g["err_pitch"]),
        np.asarray(g["err_pitch_head"]),
        np.asarray(g["err_origin"]),
        np.asarray(g["err_head"]),
        dist / 4000.0,
        np.asarray(g["distance"]) / 4000.0,
        exp_vel,
        act_vel,
        victim_speed,
        attacker_speed,
        np.gradient(dist) * TICK_RATE / 1000.0,
        np.asarray(g["t"]) / 4.0,
    ]).astype(np.float32)
    # Jerk can overflow float16 in the channel cache; keep values finite
    # and bounded before the caller casts to float16.
    derived = np.nan_to_num(derived, nan=0.0, posinf=0.0, neginf=0.0)
    derived = np.clip(derived, -30000.0, 30000.0)

    names = cols + [
        "dyaw_dps", "dpitch_dps", "speed_dps", "accel_dps2", "jerk_dps3",
        "err_yaw", "err_pitch", "err_pitch_head", "err_origin", "err_head",
        "dist_norm", "dist",
        "expected_yaw_vel", "actual_yaw_vel", "victim_speed", "attacker_speed",
        "dist_rate", "t_norm",
    ]
    return np.concatenate([out, derived], axis=1), names
