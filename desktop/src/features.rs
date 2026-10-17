//! Port of the Python aim features (geometry + dynamics + summary).
//!
//! The Python pipeline normalizes world coordinates into a 256x44 window and
//! denormalizes them again.  Rust receives world coordinates directly, so the
//! normalization round-trip is skipped; all feature math is identical.

pub const KILL_ROW: usize = 223;
pub const LEN: usize = 256;
pub const TICK_RATE: f64 = 64.0;

pub struct Window {
    pub ax: [f64; LEN],
    pub ay: [f64; LEN],
    pub az: [f64; LEN],
    pub pitch: [f64; LEN],
    pub yaw: [f64; LEN],
    pub vx: [f64; LEN],
    pub vy: [f64; LEN],
    pub vz: [f64; LEN],
    pub vhealth: [f64; LEN],
}

fn wrap_deg(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}

fn gradient(x: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }
    if n == 1 {
        return out;
    }
    out[0] = (x[1] - x[0]) / dt;
    out[n - 1] = (x[n - 1] - x[n - 2]) / dt;
    for i in 1..n - 1 {
        out[i] = (x[i + 1] - x[i - 1]) / (2.0 * dt);
    }
    out
}

fn diff_prepend(x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for i in 1..x.len() {
        out[i] = x[i] - x[i - 1];
    }
    out
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_pop(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
    }
}

fn stats(v: &[f64]) -> [f64; 5] {
    let mut vals: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if vals.is_empty() {
        return [f64::NAN; 5];
    }
    let m = mean(&vals);
    let sd = std_pop(&vals);
    let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p95 = quantile(&mut vals, 0.95);
    [m, sd, mn, mx, p95]
}

fn moving_average(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    if k <= 1 || n < k {
        return x.to_vec();
    }
    let pad = k / 2;
    let mut padded = Vec::with_capacity(n + 2 * pad);
    padded.extend(std::iter::repeat_n(x[0], pad));
    padded.extend_from_slice(x);
    padded.extend(std::iter::repeat_n(x[n - 1], pad));
    let mut out = Vec::with_capacity(n);
    let inv = 1.0 / k as f64;
    for i in 0..n {
        out.push(padded[i..i + k].iter().sum::<f64>() * inv);
    }
    out
}

fn path_length(yaw: &[f64], pitch: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 1..yaw.len() {
        s += (yaw[i] - yaw[i - 1]).hypot(pitch[i] - pitch[i - 1]);
    }
    s
}

fn straightness(yaw: &[f64], pitch: &[f64]) -> f64 {
    let len = path_length(yaw, pitch);
    if len <= 1e-9 {
        return 0.0;
    }
    let disp = (yaw[yaw.len() - 1] - yaw[0]).hypot(pitch[pitch.len() - 1] - pitch[0]);
    disp / len
}

fn curvature(yaw: &[f64], pitch: &[f64]) -> Vec<f64> {
    let n = yaw.len();
    let mut out = vec![0.0; n];
    if n < 3 {
        return out;
    }
    let dx = gradient(yaw, 1.0);
    let dy = gradient(pitch, 1.0);
    let ddx = gradient(&dx, 1.0);
    let ddy = gradient(&dy, 1.0);
    for i in 0..n {
        let denom = (dx[i] * dx[i] + dy[i] * dy[i]).powf(1.5);
        if denom > 1e-9 {
            out[i] = (dx[i] * ddy[i] - dy[i] * ddx[i]).abs() / denom;
        }
    }
    out
}

fn real_dft_mag(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let half = n / 2;
    let mut out = Vec::with_capacity(half + 1);
    for k in 0..=half {
        let mut re = 0.0;
        let mut im = 0.0;
        for (t, &v) in x.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        out.push((re * re + im * im).sqrt());
    }
    out
}

fn linear_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..x.len() {
        num += (x[i] - mx) * (y[i] - my);
        den += (x[i] - mx) * (x[i] - mx);
    }
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

fn sparc(speed: &[f64]) -> f64 {
    let fs = TICK_RATE;
    let mut v = moving_average(speed, 5);
    let m = mean(&v);
    for x in &mut v { *x -= m; }
    if v.len() < 8 || v.iter().map(|x| x.abs()).fold(0.0, f64::max) < 1e-9 {
        return 0.0;
    }
    let mag = real_dft_mag(&v);
    let n = v.len();
    let mut freqs = Vec::new();
    let mut mags = Vec::new();
    for (k, &mgt) in mag.iter().enumerate() {
        let f = k as f64 * fs / n as f64;
        if (0.5..=20.0).contains(&f) {
            freqs.push(f);
            mags.push(mgt);
        }
    }
    if freqs.len() < 3 {
        return 0.0;
    }
    let mx = mags.iter().copied().fold(0.0, f64::max) + 1e-12;
    for m in &mut mags { *m /= mx; }
    let mut total = 0.0;
    for i in 0..freqs.len() - 1 {
        let dw = freqs[i + 1] - freqs[i];
        let dmag = mags[i + 1] - mags[i];
        let integrand = ((1.0 / (freqs[i] + 1e-6)).powi(2) + (dmag / (dw + 1e-9)).powi(2)).sqrt();
        total += integrand * dw;
    }
    -total
}

fn ldlj(yaw: &[f64], pitch: &[f64]) -> f64 {
    let fs = TICK_RATE;
    let n = yaw.len();
    if n < 8 {
        return 0.0;
    }
    let dt = 1.0 / fs;
    let duration = (n - 1) as f64 * dt;
    let disp = (yaw[n - 1] - yaw[0]).hypot(pitch[n - 1] - pitch[0]);
    if disp < 1e-6 || duration < 1e-6 {
        return 0.0;
    }
    let g1x = gradient(yaw, dt);
    let g2x = gradient(&g1x, dt);
    let jx = gradient(&g2x, dt);
    let g1y = gradient(pitch, dt);
    let g2y = gradient(&g1y, dt);
    let jy = gradient(&g2y, dt);
    let integral: f64 = jx.iter().zip(jy.iter()).map(|(a, b)| a * a + b * b).sum::<f64>() * dt;
    if integral <= 1e-12 {
        return 0.0;
    }
    -(((duration.powi(5) / (disp * disp)) * integral) + 1e-12).ln()
}

fn psd_slope(x: &[f64]) -> f64 {
    let fs = TICK_RATE;
    let n = x.len();
    if n < 16 {
        return 0.0;
    }
    let m = mean(x);
    let v: Vec<f64> = x.iter().map(|a| a - m).collect();
    if std_pop(&v) < 1e-9 {
        return 0.0;
    }
    let mag = real_dft_mag(&v);
    let mut lf = Vec::new();
    let mut lp = Vec::new();
    for (k, &mgt) in mag.iter().enumerate() {
        let f = k as f64 * fs / n as f64;
        let p = mgt * mgt;
        if (0.5..=20.0).contains(&f) && p > 0.0 {
            lf.push(f.log10());
            lp.push((p + 1e-12).log10());
        }
    }
    if lf.len() < 3 {
        return 0.0;
    }
    linear_slope(&lf, &lp)
}

fn sample_entropy(x: &[f64], m: usize, r_ratio: f64) -> f64 {
    let n = x.len();
    if n < m + 2 || std_pop(x) < 1e-12 {
        return 0.0;
    }
    let r = r_ratio * std_pop(x);
    let count = |mm: usize| -> usize {
        let mut total = 0;
        for i in 0..n.saturating_sub(mm) {
            for j in i + 1..n.saturating_sub(mm) {
                let mut ok = true;
                for k in 0..mm {
                    if (x[i + k] - x[j + k]).abs() > r {
                        ok = false;
                        break;
                    }
                }
                if ok { total += 1; }
            }
        }
        total
    };
    let b = count(m);
    if b == 0 {
        return 0.0;
    }
    let a = count(m + 1);
    -(((a as f64 + 1e-12) / (b as f64 + 1e-12)).ln())
}

fn submovements(speed: &[f64]) -> f64 {
    let s = moving_average(speed, 5);
    if s.len() < 3 {
        return 0.0;
    }
    let mx = s.iter().copied().fold(0.0, f64::max);
    if mx <= 0.0 {
        return 0.0;
    }
    let thresh = 0.15 * mx;
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..s.len() - 1 {
        if s[i] >= s[i - 1] && s[i] > s[i + 1] && s[i] >= thresh {
            if peaks.is_empty() || i - peaks[peaks.len() - 1] >= 3 {
                peaks.push(i);
            } else if s[i] > s[peaks[peaks.len() - 1]] {
                let last = peaks.len() - 1;
                peaks[last] = i;
            }
        }
    }
    peaks.len() as f64
}

fn corr(a: &[f64], b: &[f64]) -> Option<f64> {
    let ma = mean(a);
    let mb = mean(b);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da < 1e-18 || db < 1e-18 {
        return None;
    }
    Some(num / (da.sqrt() * db.sqrt()))
}

fn lead_lag(a: &[f64], b: &[f64], max_lag: i32) -> (f64, f64) {
    if a.len() < (max_lag as usize) + 4 || std_pop(a) < 1e-9 || std_pop(b) < 1e-9 {
        return (0.0, 0.0);
    }
    let mut best_lag: f64 = 0.0;
    let mut best_corr: f64 = 0.0;
    for lag in -max_lag..=max_lag {
        let (x, y): (&[f64], &[f64]) = if lag > 0 {
            (&a[lag as usize..], &b[..b.len() - lag as usize])
        } else if lag < 0 {
            let l = (-lag) as usize;
            (&a[..a.len() - l], &b[l..])
        } else {
            (a, b)
        };
        if x.len() < 8 || std_pop(x) < 1e-9 || std_pop(y) < 1e-9 {
            continue;
        }
        if let Some(c) = corr(x, y) {
            if c.abs() > best_corr.abs() {
                best_lag = lag as f64;
                best_corr = c;
            }
        }
    }
    (best_lag, best_corr)
}

pub fn summarize(w: &Window) -> Vec<f64> {
    let n = LEN;
    let mut pitch = [0.0; LEN];
    let mut yaw = [0.0; LEN];
    for i in 0..n {
        pitch[i] = w.pitch[i].clamp(-90.0, 90.0);
        yaw[i] = wrap_deg(w.yaw[i]);
    }
    let mut dist = [0.0; LEN];
    let mut expected_yaw = [0.0; LEN];
    let mut err_yaw = [0.0; LEN];
    let mut err_pitch = [0.0; LEN];
    let mut err_head = [0.0; LEN];
    let mut err = [0.0; LEN];
    for i in 0..n {
        let dx = w.vx[i] - w.ax[i];
        let dy = w.vy[i] - w.ay[i];
        let dz = w.vz[i] - w.az[i];
        let horiz = dx.hypot(dy);
        dist[i] = (dx * dx + dy * dy + dz * dz).sqrt();
        // atan2(dy, dx)
        let ey = dy.atan2(dx).to_degrees();
        expected_yaw[i] = ey;
        let ep = -dz.atan2(horiz.max(1e-6)).to_degrees();
        let eph = -(dz + 64.0).atan2(horiz.max(1e-6)).to_degrees();
        err_yaw[i] = wrap_deg(yaw[i] - ey);
        err_pitch[i] = pitch[i] - ep;
        // err_head is the 2D error to the head, not the pitch component alone.
        err_head[i] = err_yaw[i].hypot(pitch[i] - eph);
        err[i] = err_yaw[i].hypot(err_pitch[i]);
    }
    let dyaw = diff_prepend(&yaw);
    let dpitch = diff_prepend(&pitch);
    let mut dyaw_w = [0.0; LEN];
    for i in 0..n { dyaw_w[i] = wrap_deg(dyaw[i]); }
    let mut pertick = [0.0; LEN];
    let mut speed = [0.0; LEN];
    for i in 0..n {
        pertick[i] = dyaw_w[i].hypot(dpitch[i]);
        speed[i] = pertick[i] * TICK_RATE;
    }
    // Python derives accel/jerk from degrees-per-tick, not from deg/s.
    let accel = gradient(&pertick, 1.0 / TICK_RATE);
    let jerk = gradient(&accel, 1.0 / TICK_RATE);

    let sl64 = 160..=KILL_ROW;
    let sl128 = 96..=KILL_ROW;
    let sl16 = 208..=KILL_ROW;

    let mut out = vec![f64::NAN; 59];

    // target alignment
    let mut ttt = f64::NAN;
    for i in sl128.clone() {
        if err[i] < 2.0 {
            ttt = (KILL_ROW as f64 - i as f64) / TICK_RATE;
            break;
        }
    }
    out[0] = ttt;
    let mut reached = 0.0;
    let mut overshoot = f64::NAN;
    let mut settling = f64::NAN;
    let mut cross_idx: Option<usize> = None;
    for i in sl128.clone() {
        if err[i] < 1.0 { cross_idx = Some(i); break; }
    }
    if let Some(ci) = cross_idx {
        reached = 1.0;
        let mut mx: f64 = 0.0;
        for i in ci..=KILL_ROW { mx = mx.max(err[i]); }
        overshoot = (mx - 1.0).max(0.0);
        let mut last_above: Option<usize> = None;
        for i in ci..=KILL_ROW { if err[i] > 1.0 { last_above = Some(i); } }
        settling = match last_above {
            Some(idx) => (KILL_ROW as f64 - idx as f64) / TICK_RATE,
            None => 0.0,
        };
    }
    out[1] = reached;
    out[2] = overshoot;
    out[3] = settling;
    out[4] = err[KILL_ROW];
    let e64: Vec<f64> = sl64.clone().map(|i| err[i]).collect();
    out[5] = e64.iter().copied().fold(f64::INFINITY, f64::min);
    out[6] = mean(&e64);
    out[7] = std_pop(&e64);
    out[8] = err_yaw[KILL_ROW];
    out[9] = err_pitch[KILL_ROW];
    out[10] = err_head[KILL_ROW];
    let eh64: Vec<f64> = sl64.clone().map(|i| err_head[i]).collect();
    out[11] = mean(&eh64);
    let mut lock = 0.0;
    let mut lock_n = 0.0;
    for i in sl16.clone() {
        lock_n += 1.0;
        if err[i] < 1.0 && speed[i] < 5.0 { lock += 1.0; }
    }
    out[12] = lock / lock_n;

    let s64: Vec<f64> = sl64.clone().map(|i| speed[i]).collect();
    let s16: Vec<f64> = sl16.clone().map(|i| speed[i]).collect();
    let a64: Vec<f64> = sl64.clone().map(|i| accel[i]).collect();
    let j64: Vec<f64> = sl64.clone().map(|i| jerk[i]).collect();
    let d64: Vec<f64> = sl64.clone().map(|i| dist[i]).collect();
    let st64 = stats(&s64); for k in 0..5 { out[13 + k] = st64[k]; }
    let st16 = stats(&s16); for k in 0..5 { out[18 + k] = st16[k]; }
    let at = stats(&a64); for k in 0..5 { out[23 + k] = at[k]; }
    let jt = stats(&j64); for k in 0..5 { out[28 + k] = jt[k]; }
    let dt = stats(&d64); for k in 0..5 { out[33 + k] = dt[k]; }
    out[38] = dist[KILL_ROW];
    let t64: Vec<f64> = sl64.clone().map(|i| (i as f64 - KILL_ROW as f64) / TICK_RATE).collect();
    out[39] = linear_slope(&t64, &d64);

    let y64: Vec<f64> = sl64.clone().map(|i| yaw[i]).collect();
    let p64: Vec<f64> = sl64.clone().map(|i| pitch[i]).collect();
    out[40] = path_length(&y64, &p64);
    out[41] = straightness(&y64, &p64);
    let mut curv: Vec<f64> = curvature(&y64, &p64);
    for c in &mut curv { *c = c.clamp(0.0, 1000.0); }
    let ct = stats(&curv); for k in 0..5 { out[42 + k] = ct[k]; }
    out[47] = sparc(&s64);
    out[48] = ldlj(&y64, &p64);
    out[49] = psd_slope(&s64);
    let se: Vec<f64> = (192..=223).map(|i| speed[i]).collect();
    out[50] = sample_entropy(&se, 2, 0.2);
    out[51] = submovements(&s64);

    let gvx = gradient(&w.vx, 1.0);
    let gvy = gradient(&w.vy, 1.0);
    let vs: Vec<f64> = sl64.clone().map(|i| (gvx[i] * TICK_RATE).hypot(gvy[i] * TICK_RATE)).collect();
    let vt = stats(&vs); for k in 0..5 { out[52 + k] = vt[k]; }

    let exp_vel: Vec<f64> = sl128.clone().map(|i| {
        let d = if i == 0 { 0.0 } else { expected_yaw[i] - expected_yaw[i - 1] };
        wrap_deg(d) * TICK_RATE
    }).collect();
    let act_vel: Vec<f64> = sl128.map(|i| dyaw_w[i] * TICK_RATE).collect();
    let (lag, cc) = lead_lag(&act_vel, &exp_vel, 12);
    out[57] = lag;
    out[58] = cc;
    out
}
