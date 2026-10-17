"""Mouse-view consistency residual for a raw CS2CD match.

The raw parquet contains per-tick `usercmd_mouse_dx/dy` and the resulting
view angles.  A human player's yaw/pitch deltas are explained by the mouse
deltas up to a per-player sensitivity slope.  An internal aimbot that writes
view angles directly leaves a large residual.

The fit is a fast robust least-squares (iterative outlier rejection), not a
full Huber solver, so 160 matches can be processed in minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK_RATE = 64.0
KILL_ROW = 223


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def _fit_line(x: np.ndarray, y: np.ndarray):
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    denom = float(np.sum((x - xm) ** 2))
    if denom < 1e-9:
        return 0.0, ym
    a = float(np.sum((x - xm) * (y - ym)) / denom)
    return a, ym - a * xm


def robust_fit(x: np.ndarray, y: np.ndarray, iters: int = 3):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 100 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0, 0.0, 0.0, np.nan
    a, b = _fit_line(x, y)
    for _ in range(iters):
        r = y - (a * x + b)
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-6
        keep = np.abs(r - med) < 3.0 * 1.4826 * mad
        if keep.sum() < 100:
            break
        a, b = _fit_line(x[keep], y[keep])
    r = y - (a * x + b)
    ss_res = float(np.sum(r * r))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    scale = float(np.median(np.abs(r - np.median(r))) + 1e-3)
    return float(a), float(b), float(r2), scale


def player_residuals(ticks: pd.DataFrame) -> dict:
    """Fit per-player slopes and return residual arrays keyed by player."""
    out = {}
    for player, d in ticks.groupby("steamid", sort=False):
        d = d.sort_values("tick")
        yaw = d["yaw"].to_numpy(dtype=np.float64)
        pitch = d["pitch"].to_numpy(dtype=np.float64)
        dx = d["usercmd_mouse_dx"].to_numpy(dtype=np.float64)
        dy = d["usercmd_mouse_dy"].to_numpy(dtype=np.float64)
        alive = d["is_alive"].to_numpy(dtype=bool)
        dyaw = _wrap(np.diff(yaw, prepend=yaw[0]))
        dpitch = np.diff(pitch, prepend=pitch[0])
        good = alive & np.isfinite(dx) & np.isfinite(dy)
        good &= np.isfinite(yaw) & np.isfinite(pitch)
        good &= (np.abs(dyaw) < 30) & (np.abs(dpitch) < 30)

        # Fit only where the mouse actually moved.  The many exact-zero
        # ticks would otherwise dominate the robust residual scale and
        # collapse the slope toward zero.
        move_y = good & (np.abs(dx) >= 1.0)
        move_p = good & (np.abs(dy) >= 1.0)
        if move_y.sum() < 100:
            move_y = good
        if move_p.sum() < 100:
            move_p = good

        a, b, r2y, _ = robust_fit(dx[move_y], dyaw[move_y])
        c, e, r2p, _ = robust_fit(dy[move_p], dpitch[move_p])
        res_y = dyaw - (a * dx + b)
        res_p = dpitch - (c * dy + e)
        resid = np.sqrt(res_y * res_y + res_p * res_p)
        used = good & (move_y | move_p)
        scale = float(np.nanmedian(resid[used]) + 1e-3) if used.any() else np.nan
        scale = max(scale, 0.01) if np.isfinite(scale) else np.nan
        valid = bool(
            np.isfinite(scale)
            and r2y > 0.2 and r2p > 0.2
            and move_y.sum() >= 100 and move_p.sum() >= 100
        )
        out[player] = {
            "tick": d["tick"].to_numpy(),
            "resid": np.clip(resid / max(scale, 1e-6), 0.0, 100.0),
            "r2_yaw": r2y,
            "r2_pitch": r2p,
            "slope_yaw": a,
            "slope_pitch": c,
            "scale": scale,
            "valid": valid,
        }
    return out


def window_features(tick_arr: np.ndarray, resid: np.ndarray, kill_tick: int,
                    r2y: float, r2p: float, ay: float, ap: float,
                    scale: float, valid: bool = True) -> dict:
    tick_arr = np.asarray(tick_arr)
    r = np.asarray(resid, dtype=np.float64)
    if len(tick_arr) != len(r):
        raise ValueError("tick and residual arrays must align")

    idx = int(np.searchsorted(tick_arr, kill_tick))
    if idx >= len(tick_arr):
        idx = len(tick_arr) - 1
    start = idx - KILL_ROW
    end = start + 256
    left = max(0, -start)
    right = max(0, end - len(r))
    win = r[max(0, start):min(len(r), end)]
    if left or right:
        win = np.pad(win, (left, right), mode="edge")
    win = win[:256]
    last64 = win[KILL_ROW - 63:KILL_ROW + 1]
    last8 = win[KILL_ROW - 7:KILL_ROW + 1]
    out = {
        "mr_valid": float(valid),
        "mr_r2_yaw": r2y,
        "mr_r2_pitch": r2p,
        "mr_slope_yaw": ay,
        "mr_slope_pitch": ap,
        "mr_scale": scale,
        "mr_shot": float(win[KILL_ROW]) if np.isfinite(win[KILL_ROW]) else np.nan,
    }

    def add(prefix, arr):
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            for k in ("mean", "p95", "max", "frac3"):
                out[f"{prefix}_{k}"] = np.nan
            return
        out[f"{prefix}_mean"] = float(np.mean(a))
        out[f"{prefix}_p95"] = float(np.quantile(a, 0.95))
        out[f"{prefix}_max"] = float(np.max(a))
        out[f"{prefix}_frac3"] = float(np.mean(a > 3.0))

    add("mr_last64", last64)
    add("mr_last8", last8)
    add("mr_win", win)
    if not valid:
        for k in list(out):
            if k.startswith(("mr_last64_", "mr_last8_", "mr_win_")) or k == "mr_shot":
                out[k] = np.nan
    return out


def kill_ticks(events: dict, player: str) -> list[int]:
    d = pd.DataFrame(events["player_death"])
    d = d[d["attacker_steamid"] == player]
    return [int(x) for x in d["tick"].tolist()]
