"""Motor-control and trajectory-shape features.

All functions are pure numpy and operate on a single window.  They are
intentionally simple: the first detector must be reproducible and cheap
before adding anything heavier.
"""
from __future__ import annotations

import numpy as np

TICK_RATE = 64.0


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, kernel, mode="valid")


def angular_speed(dyaw: np.ndarray, dpitch: np.ndarray) -> np.ndarray:
    """Degrees per tick."""
    return np.hypot(np.asarray(dyaw), np.asarray(dpitch))


def angular_accel(dyaw: np.ndarray, dpitch: np.ndarray) -> np.ndarray:
    """Degrees per tick^2."""
    w = angular_speed(dyaw, dpitch)
    return np.gradient(w, 1.0 / TICK_RATE)


def angular_jerk(dyaw: np.ndarray, dpitch: np.ndarray) -> np.ndarray:
    """Degrees per tick^3."""
    w = angular_speed(dyaw, dpitch)
    a = np.gradient(w, 1.0 / TICK_RATE)
    return np.gradient(a, 1.0 / TICK_RATE)


def submovement_count(speed: np.ndarray, min_ratio: float = 0.15,
                      min_distance: int = 3) -> int:
    """Count local maxima in the speed profile.

    A human flick usually has a primary peak and one or more corrective
    peaks; a perfect snap often has one peak and no correction.
    """
    s = moving_average(speed, 5)
    if len(s) < 3 or float(np.max(s)) <= 0:
        return 0
    thresh = min_ratio * float(np.max(s))
    peaks: list[int] = []
    for i in range(1, len(s) - 1):
        if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] >= thresh:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
            elif s[i] > s[peaks[-1]]:
                peaks[-1] = i
    return len(peaks)


def path_length(yaw: np.ndarray, pitch: np.ndarray) -> float:
    dy = np.diff(yaw)
    dp = np.diff(pitch)
    return float(np.sum(np.hypot(dy, dp)))


def straightness(yaw: np.ndarray, pitch: np.ndarray) -> float:
    length = path_length(yaw, pitch)
    if length <= 1e-9:
        return 0.0
    disp = float(np.hypot(yaw[-1] - yaw[0], pitch[-1] - pitch[0]))
    return disp / length


def curvature(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Discrete curvature of the 2D angular path."""
    x = np.asarray(yaw, dtype=np.float64)
    y = np.asarray(pitch, dtype=np.float64)
    if len(x) < 3:
        return np.zeros_like(x)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    out = np.zeros_like(x)
    mask = denom > 1e-9
    out[mask] = np.abs(dx[mask] * ddy[mask] - dy[mask] * ddx[mask]) / denom[mask]
    return out


def sparc(speed: np.ndarray, fs: float = TICK_RATE) -> float:
    """Spectral arc length (negative; larger is smoother)."""
    v = moving_average(speed, 5)
    v = v - np.mean(v)
    if len(v) < 8 or np.max(np.abs(v)) < 1e-9:
        return 0.0
    mag = np.abs(np.fft.rfft(v))
    freq = np.fft.rfftfreq(len(v), d=1.0 / fs)
    keep = (freq >= 0.5) & (freq <= 20.0)
    if keep.sum() < 3:
        return 0.0
    mag = mag[keep]
    freq = freq[keep]
    mag = mag / (np.max(mag) + 1e-12)
    dw = np.diff(freq)
    dmag = np.diff(mag)
    integrand = np.sqrt((1.0 / (freq[:-1] + 1e-6)) ** 2 + (dmag / (dw + 1e-9)) ** 2)
    return float(-np.sum(integrand * dw))


def ldlj(yaw: np.ndarray, pitch: np.ndarray, fs: float = TICK_RATE) -> float:
    """Log dimensionless jerk (negative; larger magnitude is smoother)."""
    x = np.asarray(yaw, dtype=np.float64)
    y = np.asarray(pitch, dtype=np.float64)
    n = len(x)
    if n < 8:
        return 0.0
    dt = 1.0 / fs
    duration = (n - 1) * dt
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if disp < 1e-6 or duration < 1e-6:
        return 0.0
    jx = np.gradient(np.gradient(np.gradient(x, dt), dt), dt)
    jy = np.gradient(np.gradient(np.gradient(y, dt), dt), dt)
    integral = float(np.sum(jx * jx + jy * jy) * dt)
    if integral <= 1e-12:
        return 0.0
    return float(-np.log((duration ** 5 / disp ** 2) * integral + 1e-12))


def psd_slope(x: np.ndarray, fs: float = TICK_RATE) -> float:
    """Slope of log power vs log frequency in [0.5, 20] Hz."""
    v = np.asarray(x, dtype=np.float64)
    v = v - np.mean(v)
    if len(v) < 16 or np.std(v) < 1e-9:
        return 0.0
    power = np.abs(np.fft.rfft(v)) ** 2
    freq = np.fft.rfftfreq(len(v), d=1.0 / fs)
    keep = (freq >= 0.5) & (freq <= 20.0) & (power > 0)
    if keep.sum() < 3:
        return 0.0
    slope = np.polyfit(np.log10(freq[keep]), np.log10(power[keep] + 1e-12), 1)[0]
    return float(slope)


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < m + 2 or np.std(x) < 1e-12:
        return 0.0
    r = r_ratio * float(np.std(x))

    def count(mm: int) -> int:
        total = 0
        for i in range(n - mm):
            for j in range(i + 1, n - mm):
                if np.max(np.abs(x[i:i + mm] - x[j:j + mm])) <= r:
                    total += 1
        return total

    b = count(m)
    a = count(m + 1)
    if b == 0:
        return 0.0
    return float(-np.log((a + 1e-12) / (b + 1e-12)))


def lead_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 12) -> tuple[int, float]:
    """Best lag and normalized correlation between two 1D signals.

    Positive lag means `a` lags `b`.  Used to compare the victim's angular
    motion with the attacker's actual yaw velocity.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < max_lag + 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0, 0.0
    best_lag, best_corr = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x, y = a[lag:], b[:-lag]
        elif lag < 0:
            x, y = a[:lag], b[-lag:]
        else:
            x, y = a, b
        if len(x) < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if abs(c) > abs(best_corr):
            best_lag, best_corr = lag, c
    return best_lag, best_corr
