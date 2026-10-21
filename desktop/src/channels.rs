//! The 29 aim-only channels used by the ONNX champion.

use anyhow::{Result, bail};

use crate::features::{KILL_ROW, LEN, TICK_RATE, Window};

pub const CHANNELS: usize = 29;

const EYE_HEIGHT: f64 = 64.0;

fn map_bounds(map: &str) -> Option<[f64; 6]> {
    Some(match map {
        "de_dust2" => [-2300.0, 1800.0, -1200.0, 3150.0, -130.0, 400.0],
        "de_mirage" => [-2700.0, 1500.0, -2700.0, 1000.0, -400.0, 150.0],
        "de_inferno" => [-1800.0, 2700.0, -800.0, 3600.0, -100.0, 500.0],
        "de_train" => [-2200.0, 1800.0, -1800.0, 1800.0, -400.0, 200.0],
        "de_nuke" => [-3000.0, 3500.0, -2500.0, 1000.0, -750.0, 100.0],
        "de_ancient" => [-2310.0, 1400.0, -2600.0, 1800.0, -120.0, 400.0],
        "de_vertigo" => [-2700.0, 100.0, -1600.0, 1200.0, 11400.0, 12100.0],
        "de_anubis" => [-2000.0, 1810.0, -1810.0, 3200.0, -150.0, 200.0],
        "cs_office" => [-1800.0, 2400.0, -2200.0, 1300.0, -350.0, 10.0],
        "de_overpass" => [-4000.0, 20.0, -3500.0, 1700.0, 0.0, 800.0],
        "de_basalt" => [-2100.0, 2000.0, -1700.0, 2350.0, -100.0, 400.0],
        "de_edin" => [500.0, 3700.0, -350.0, 4300.0, 300.0, 750.0],
        "cs_italy" => [-1550.0, 1100.0, -2200.0, 2650.0, -200.0, 300.0],
        "de_thera" => [600.0, 4300.0, -2600.0, 2200.0, -170.0, 300.0],
        "de_mills" => [-4300.0, 0.0, -5560.0, -300.0, -100.0, 300.0],
        _ => return None,
    })
}

fn wrap_deg(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}

fn gradient(x: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    if n < 2 {
        return out;
    }
    out[0] = (x[1] - x[0]) / dt;
    out[n - 1] = (x[n - 1] - x[n - 2]) / dt;
    for i in 1..n - 1 {
        out[i] = (x[i + 1] - x[i - 1]) / (2.0 * dt);
    }
    out
}

fn norm(value: f64, lo: f64, hi: f64) -> f64 {
    if (hi - lo).abs() < 1e-9 {
        0.0
    } else {
        (value - lo) / (hi - lo)
    }
}

fn clean(value: f64) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(-30000.0, 30000.0) as f32
}

/// Build one `[LEN, 29]` channel matrix for a kill window.
pub fn build(window: &Window, map: &str) -> Result<Vec<[f32; CHANNELS]>> {
    let Some([min_x, max_x, min_y, max_y, min_z, max_z]) = map_bounds(map) else {
        bail!("unknown map: {map}");
    };
    let n = LEN;
    let mut pitch = [0.0f64; LEN];
    let mut yaw = [0.0f64; LEN];
    let mut dist = [0.0f64; LEN];
    let mut expected_yaw = [0.0f64; LEN];
    let mut expected_pitch = [0.0f64; LEN];
    let mut expected_pitch_head = [0.0f64; LEN];
    let mut err_yaw = [0.0f64; LEN];
    let mut err_pitch = [0.0f64; LEN];
    let mut err_pitch_head = [0.0f64; LEN];
    let mut err_origin = [0.0f64; LEN];
    let mut err_head = [0.0f64; LEN];
    for i in 0..n {
        pitch[i] = window.pitch[i].clamp(-90.0, 90.0);
        yaw[i] = wrap_deg(window.yaw[i]);
        let dx = window.vx[i] - window.ax[i];
        let dy = window.vy[i] - window.ay[i];
        let dz = window.vz[i] - window.az[i];
        let horiz = (dx * dx + dy * dy).sqrt();
        dist[i] = (dx * dx + dy * dy + dz * dz).sqrt();
        expected_yaw[i] = dy.atan2(dx).to_degrees();
        expected_pitch[i] = -(dz).atan2(horiz.max(1e-6)).to_degrees();
        expected_pitch_head[i] = -(dz + EYE_HEIGHT).atan2(horiz.max(1e-6)).to_degrees();
        err_yaw[i] = wrap_deg(yaw[i] - expected_yaw[i]);
        err_pitch[i] = pitch[i] - expected_pitch[i];
        err_pitch_head[i] = pitch[i] - expected_pitch_head[i];
        err_origin[i] = err_yaw[i].hypot(err_pitch[i]);
        err_head[i] = err_yaw[i].hypot(err_pitch_head[i]);
    }
    let mut dyaw = vec![0.0f64; n];
    let mut dpitch = vec![0.0f64; n];
    let mut pertick = vec![0.0f64; n];
    for i in 0..n {
        if i > 0 {
            dyaw[i] = wrap_deg(yaw[i] - yaw[i - 1]);
            dpitch[i] = pitch[i] - pitch[i - 1];
        }
        pertick[i] = dyaw[i].hypot(dpitch[i]);
    }
    let speed: Vec<f64> = pertick.iter().map(|v| v * TICK_RATE).collect();
    let accel = gradient(&pertick, 1.0 / TICK_RATE);
    let jerk = gradient(&accel, 1.0 / TICK_RATE);
    let victim_speed: Vec<f64> = {
        let gx = gradient(&window.vx, 1.0);
        let gy = gradient(&window.vy, 1.0);
        (0..n)
            .map(|i| (gx[i] * TICK_RATE).hypot(gy[i] * TICK_RATE))
            .collect()
    };
    let dist_rate: Vec<f64> = gradient(&dist, 1.0)
        .into_iter()
        .map(|v| v * TICK_RATE / 1000.0)
        .collect();
    let mut expected_yaw_vel = vec![0.0f64; n];
    for i in 1..n {
        expected_yaw_vel[i] = wrap_deg(expected_yaw[i] - expected_yaw[i - 1]) * TICK_RATE;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let attacker_speed = window.avel[i];
        out.push([
            clean(norm(window.ax[i], min_x, max_x)),
            clean(norm(window.ay[i], min_y, max_y)),
            clean(norm(window.az[i], min_z, max_z)),
            clean(window.avel[i] / 250.0),
            clean((pitch[i] + 90.0) / 180.0),
            clean((yaw[i] + 180.0) / 360.0),
            clean(norm(window.vx[i], min_x, max_x)),
            clean(norm(window.vy[i], min_y, max_y)),
            clean(norm(window.vz[i], min_z, max_z)),
            clean((window.vhealth[i] / 100.0).clamp(0.0, 1.0)),
            0.0,
            clean(dyaw[i] * TICK_RATE),
            clean(dpitch[i] * TICK_RATE),
            clean(speed[i]),
            clean(accel[i]),
            clean(jerk[i]),
            clean(err_yaw[i]),
            clean(err_pitch[i]),
            clean(err_pitch_head[i]),
            clean(err_origin[i]),
            clean(err_head[i]),
            clean(dist[i] / 4000.0),
            clean(dist[i] / 4000.0),
            clean(expected_yaw_vel[i]),
            clean(dyaw[i] * TICK_RATE),
            clean(victim_speed[i]),
            clean(attacker_speed),
            clean(dist_rate[i]),
            clean((i as f64 - KILL_ROW as f64) / TICK_RATE / 4.0),
        ]);
    }
    Ok(out)
}
