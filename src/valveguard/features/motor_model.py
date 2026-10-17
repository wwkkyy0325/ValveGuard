"""Motor-control summary features.

These are deliberately parametric and cheap: an exponential error-decay
time constant, an AR(1)/OU velocity coefficient, FFT band energies, a
constant-velocity target-tracking residual, and a few profile ratios.
They add information that the generic smoothness statistics do not encode.
"""
from __future__ import annotations

import numpy as np

from .geometry import KILL_ROW, TICK_RATE, recover

SL = slice(KILL_ROW - 63, KILL_ROW + 1)


def _linear_r2(x, y):
    if len(x) < 5 or np.std(x) < 1e-12:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(a), float(b), 1.0 - ss_res / max(ss_tot, 1e-12)


def motor_summary(window) -> dict:
    g = recover(window)
    err = np.asarray(g["err_origin"], dtype=np.float64)
    yaw = np.asarray(g["yaw"], dtype=np.float64)
    pitch = np.asarray(g["pitch"], dtype=np.float64)
    dyaw = np.asarray(g["dyaw"], dtype=np.float64)
    dpitch = np.asarray(g["dpitch"], dtype=np.float64)
    yaw_vel = dyaw * TICK_RATE
    pitch_vel = dpitch * TICK_RATE
    speed = np.hypot(yaw_vel, pitch_vel)
    t = np.arange(64, dtype=np.float64) / TICK_RATE  # 0..1 s inside the slice

    out = {}
    e = err[SL]
    v = yaw_vel[SL]
    s = speed[SL]

    # Exponential error decay: log(e - floor) = a + b*t; tau = -1/b.
    floor = float(np.min(e))
    y = np.log(np.maximum(e - floor, 1e-4))
    slope, intercept, r2 = _linear_r2(t, y)
    out["mm_decay_slope"] = slope
    out["mm_decay_r2"] = r2
    out["mm_tau"] = float(-1.0 / slope) if np.isfinite(slope) and slope < -1e-6 else np.nan
    out["mm_decay_floor"] = floor
    out["mm_decay_amp"] = float(np.exp(intercept)) if np.isfinite(intercept) else np.nan

    # AR(1) / OU on yaw velocity.
    x, yv = v[:-1], v[1:]
    phi, intercept, r2 = _linear_r2(x, yv)
    resid = yv - (phi * x + intercept) if np.isfinite(phi) else np.array([np.nan])
    out["mm_ar1_phi"] = phi
    out["mm_ar1_r2"] = r2
    out["mm_ar1_sigma"] = float(np.nanstd(resid))
    if np.isfinite(phi) and 0.0 < phi < 1.0:
        out["mm_ar1_halflife"] = float(-np.log(2.0) / np.log(phi))
    else:
        out["mm_ar1_halflife"] = np.nan

    # FFT band energies of yaw velocity.
    z = v - np.mean(v)
    power = np.abs(np.fft.rfft(z)) ** 2
    freq = np.fft.rfftfreq(len(z), d=1.0 / TICK_RATE)
    total = float(np.sum(power)) + 1e-12
    for lo, hi in [(1, 4), (4, 8), (8, 16), (16, 32)]:
        m = (freq >= lo) & (freq < hi)
        out[f"mm_band_{lo}_{hi}"] = float(np.sum(power[m]) / total)

    # Constant-velocity target tracking residual.
    vyaw = np.asarray(g["expected_yaw"], dtype=np.float64)
    exp_vel = np.diff(vyaw, prepend=vyaw[0]) * TICK_RATE
    track = yaw_vel - exp_vel
    out["mm_track_mean"] = float(np.mean(track[SL]))
    out["mm_track_std"] = float(np.std(track[SL]))
    out["mm_track_p95"] = float(np.quantile(np.abs(track[SL]), 0.95))

    # Fitts-like and profile ratios.
    dist = float(g["distance"][KILL_ROW])
    final_err = float(err[KILL_ROW])
    out["mm_final_err_norm"] = final_err / max(dist, 1.0)
    out["mm_log_dist"] = float(np.log1p(dist))
    peak = float(np.max(s)) + 1e-9
    mean = float(np.mean(s)) + 1e-9
    out["mm_peak_over_mean"] = peak / mean
    out["mm_time_to_peak"] = float(np.argmax(s)) / TICK_RATE
    out["mm_speed_skew"] = float(
        np.mean(((s - np.mean(s)) / (np.std(s) + 1e-9)) ** 3)
    )
    out["mm_speed_kurt"] = float(
        np.mean(((s - np.mean(s)) / (np.std(s) + 1e-9)) ** 4)
    )
    out["mm_yaw_vel_std"] = float(np.std(v))
    out["mm_pitch_vel_std"] = float(np.std(pitch_vel[SL]))

    return out
