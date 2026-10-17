"""Long-context features for the 1024-tick context pilot.

``summarize`` only looks at the last 128 ticks before the kill.  The macro
features here describe the whole pre-kill window: target acquisition count,
fraction of time on target, and multi-scale smoothness/distance statistics.
They are meaningless on a 256-tick window (there the last 128 ticks are half
the context); the H2 pilot computes them on 1024 ticks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dynamics import angular_speed
from .geometry import TICK_RATE, recover


def _stats(out: dict, prefix: str, values: np.ndarray) -> None:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        for s in ("mean", "std", "p95"):
            out[f"{prefix}_{s}"] = np.nan
        return
    out[f"{prefix}_mean"] = float(np.mean(v))
    out[f"{prefix}_std"] = float(np.std(v))
    out[f"{prefix}_p95"] = float(np.quantile(v, 0.95))


def macro_summary(window: pd.DataFrame, kill_row: int) -> dict:
    g = recover(window, kill_row)
    yaw = np.asarray(g["yaw"])
    pitch = np.asarray(g["pitch"])
    dyaw = np.asarray(g["dyaw"])
    dpitch = np.asarray(g["dpitch"])
    err = np.asarray(g["err_origin"])
    dist = np.asarray(g["distance"])
    speed = angular_speed(dyaw, dpitch) * TICK_RATE

    pre = slice(0, kill_row + 1)
    mid = slice(max(0, kill_row - 511), kill_row + 1)
    near = slice(max(0, kill_row - 255), kill_row + 1)
    out: dict = {}

    for name, sl in (("full", pre), ("mid", mid), ("near768", near)):
        e = err[sl]
        s = speed[sl]
        d = dist[sl]
        out[f"macro_err_{name}_mean"] = float(np.mean(e))
        out[f"macro_err_{name}_p95"] = float(np.quantile(e, 0.95))
        out[f"macro_on_target_{name}"] = float(np.mean(e < 2.0))
        out[f"macro_on_target5_{name}"] = float(np.mean(e < 5.0))
        out[f"macro_speed_{name}_mean"] = float(np.mean(s))
        _stats(out, f"macro_speed_{name}", s)
        out[f"macro_dist_{name}_mean"] = float(np.mean(d))
        out[f"macro_dist_{name}_std"] = float(np.std(d))

    on = err[pre] < 2.0
    starts = int(np.sum(on[1:] & ~on[:-1]))
    if on.size and on[0]:
        starts += 1
    out["macro_acquisitions"] = float(starts)
    out["macro_acq_rate"] = float(starts / max(1e-6, kill_row / TICK_RATE))

    # Longest continuous stretch on target before the kill.
    longest = run = 0
    for flag in on:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    out["macro_longest_lock_ticks"] = float(longest)

    out["macro_speed_max"] = float(np.max(speed[pre])) if kill_row else np.nan
    out["macro_speed_max_time"] = float(
        (kill_row - int(np.argmax(speed[pre]))) / TICK_RATE
    ) if kill_row else np.nan

    # Total angular path over the long window relative to the last 256 ticks:
    # a large ratio means the player moved around a lot before the near-aim
    # segment, which a pure near-window model cannot see.
    def path(sl: slice) -> float:
        dy = np.diff(yaw[sl], prepend=yaw[sl.start])
        dp = np.diff(pitch[sl], prepend=pitch[sl.start])
        return float(np.sum(np.hypot(dy, dp)))

    p_full, p_near = path(pre), path(near)
    out["macro_path_full"] = p_full
    out["macro_path_ratio"] = float(p_full / max(p_near, 1e-6))
    out["macro_yaw_iqr"] = float(np.subtract(*np.percentile(yaw[pre], [75, 25])))
    out["macro_pitch_iqr"] = float(np.subtract(*np.percentile(pitch[pre], [75, 25])))

    first = np.where(err[pre] < 2.0)[0]
    out["macro_first_target_s"] = (
        float((kill_row - int(first[0])) / TICK_RATE) if len(first) else np.nan
    )
    return out
