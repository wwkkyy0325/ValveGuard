use valveguard_core::{demo, model};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::path::PathBuf;

fn arg(name: &str) -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn main() -> Result<()> {
    let input = arg("--demo").context("usage: --demo <zip|dem> --model <json> [--out <json>]")?;
    let model_path = arg("--model").context("missing --model")?;
    let out = arg("--out");
    let dump = arg("--dump-features");
    let dump_window = arg("--dump-window").and_then(|p| {
        let s = p.to_string_lossy().to_string();
        let (a, b) = s.split_once(':')?;
        Some((a.parse::<u64>().ok()?, b.parse::<i32>().ok()?))
    });
    let dump_window_out = arg("--dump-window-out");

    let model = model::Model::load(&model_path)?;
    let analysis = demo::analyze(&input, &model, dump_window)?;

    if let (Some(path), Some(dw)) = (dump_window_out, &analysis.debug_window) {
        let obj = json!({
            "ax": dw.ax, "ay": dw.ay, "az": dw.az,
            "pitch": dw.pitch, "yaw": dw.yaw,
            "vx": dw.vx, "vy": dw.vy, "vz": dw.vz,
        });
        std::fs::write(&path, serde_json::to_string(&obj)?)
            .with_context(|| format!("write {}", path.display()))?;
    }

    if let Some(path) = dump {
        let rows: Vec<Value> = analysis
            .kill_features
            .iter()
            .map(|k| {
                json!({
                    "steamid": k.steamid.to_string(),
                    "kill_tick": k.kill_tick,
                    "features": k.features.iter().map(|v| num(*v)).collect::<Vec<_>>(),
                })
            })
            .collect();
        std::fs::write(&path, serde_json::to_string(&rows)?)
            .with_context(|| format!("write {}", path.display()))?;
    }

    let players: Vec<Value> = analysis
        .rows
        .iter()
        .map(|p| {
            json!({
                "steamid": p.steamid.to_string(),
                "name": p.name,
                "kills": p.kills,
                "mean_score": p.mean,
                "max_kill_score": p.max,
                "shrunk_score": p.shrunk,
                "flagged": p.flagged,
                "reliable": p.kills >= 3,
            })
        })
        .collect();
    let result = json!({
        "demo": input.to_string_lossy(),
        "map": analysis.map,
        "model_version": model.version,
        "threshold": model.threshold,
        "prior": model.prior,
        "k0": model.k0,
        "kills_total": analysis.kills_total,
        "kills_used": analysis.kills_used,
        "players": players,
    });
    if let Some(path) = out {
        std::fs::write(&path, serde_json::to_string_pretty(&result)?)
            .with_context(|| format!("write {}", path.display()))?;
    }
    println!(
        "map={} kills_used={}/{} players={} flagged={}",
        result["map"],
        result["kills_used"],
        result["kills_total"],
        result["players"].as_array().map(|a| a.len()).unwrap_or(0),
        result["players"]
            .as_array()
            .map(|a| a.iter().filter(|p| p["flagged"] == true).count())
            .unwrap_or(0),
    );
    Ok(())
}
