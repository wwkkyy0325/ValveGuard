//! Pure-Rust demo scoring for the Tauri UI.

use anyhow::{Context, Result, bail};
use awpy::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use crate::channels;
use crate::demo::{build_window, in_live_round, prepare_demo};
use crate::features::KILL_ROW;
use crate::onnx::Ensemble;

#[derive(Deserialize)]
struct Manifest {
    version: String,
    threshold: Threshold,
    aggregation: Aggregation,
    reference: Option<Reference>,
}

#[derive(Deserialize)]
struct Threshold {
    value: f64,
}

#[derive(Deserialize)]
struct Aggregation {
    k0: f64,
    prior: f64,
}

#[derive(Deserialize)]
struct Reference {
    player_scores: Option<String>,
}

struct KillRow {
    steamid: u64,
    tick: i32,
    headshot: bool,
    weapon: String,
    smoke: bool,
    wallbang: bool,
    midair: bool,
    score: f64,
}

struct PlayerResult {
    steamid: u64,
    name: String,
    tier: &'static str,
    score: Option<f64>,
    percentile: Option<f64>,
    kills: usize,
    mean: Option<f64>,
    max: Option<f64>,
    reliable: bool,
    reasons: Vec<(String, String)>,
    behavior: Value,
    kills_json: Vec<Value>,
}

fn weapon_class(name: &str) -> &'static str {
    let low = name.to_lowercase();
    let groups: [(&[&str], &str); 7] = [
        (&["knife", "bayonet", "karambit", "zeus"], "knife"),
        (
            &[
                "ak-47", "m4a4", "m4a1", "aug", "sg 553", "galil", "famas", "m249", "negev",
            ],
            "rifle",
        ),
        (&["awp", "ssg 08", "g3sg1", "scar-20"], "sniper"),
        (
            &[
                "glock",
                "usp",
                "p2000",
                "p250",
                "five-seven",
                "tec-9",
                "cz75",
                "desert eagle",
                "dual berettas",
                "r8 revolver",
            ],
            "pistol",
        ),
        (
            &[
                "grenade",
                "flashbang",
                "molotov",
                "incendiary",
                "smoke",
                "decoy",
            ],
            "grenade",
        ),
        (
            &["mp9", "mac-10", "mp7", "mp5", "ump-45", "p90", "pp-bizon"],
            "smg",
        ),
        (&["nova", "xm1014", "sawed-off", "mag-7"], "shotgun"),
    ];
    for (keys, class) in groups {
        if keys.iter().any(|key| low.contains(key)) {
            return class;
        }
    }
    "pistol"
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    }
}

fn behavior_json(kills: &[&KillRow]) -> (Value, f64, Option<f64>, f64, f64, f64, f64, usize) {
    let n = kills.len();
    let mut ticks: Vec<i32> = kills.iter().map(|kill| kill.tick).collect();
    ticks.sort_unstable();
    let headshots = kills.iter().filter(|kill| kill.headshot).count();
    let hs_rate = headshots as f64 / n as f64;
    // awpy does not expose shot events cheaply, so shots/kill is left unknown
    // rather than fabricated. The CNN does not use this field.
    let shots_per_kill: Option<f64> = None;
    let mut max_multi = 1usize;
    for tick in &ticks {
        let count = ticks
            .iter()
            .filter(|other| **other >= *tick && **other <= *tick + 256)
            .count();
        max_multi = max_multi.max(count);
    }
    let gaps: Vec<i32> = ticks.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let burst = if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().filter(|gap| **gap <= 256).count() as f64 / gaps.len() as f64
    };
    let median_gap = if gaps.is_empty() {
        None
    } else {
        let mut sorted = gaps.clone();
        sorted.sort_unstable();
        Some(sorted[sorted.len() / 2] as f64 / 64.0)
    };
    let sniper = kills
        .iter()
        .filter(|kill| weapon_class(&kill.weapon) == "sniper")
        .count() as f64
        / n as f64;
    let wall = kills.iter().filter(|kill| kill.wallbang).count() as f64 / n as f64;
    let smoke = kills.iter().filter(|kill| kill.smoke).count() as f64 / n as f64;
    let value = json!({
        "headshot_rate": hs_rate,
        "shots_per_kill": shots_per_kill,
        "max_multikill_4s": max_multi,
        "burst_fraction": burst,
        "median_kill_gap_s": median_gap,
        "sniper_share": sniper,
        "wallbang_rate": wall,
        "smoke_rate": smoke,
    });
    (
        value,
        hs_rate,
        shots_per_kill,
        sniper,
        wall,
        smoke,
        burst,
        max_multi,
    )
}

fn reasons_for(
    n: usize,
    hs: f64,
    shots_per_kill: Option<f64>,
    sniper: f64,
    wall: f64,
    smoke: f64,
    multi: usize,
    burst: f64,
) -> Vec<(String, String)> {
    let mut reasons = Vec::new();
    if n >= 3 {
        if hs >= 0.80 {
            reasons.push(("HS".to_string(), format!("{:.0}%", hs * 100.0)));
        }
        if let Some(shots) = shots_per_kill {
            if shots <= 1.5 {
                reasons.push(("EFF".to_string(), format!("{shots:.2} shots/kill")));
            }
        }
        if sniper >= 0.50 && hs >= 0.80 {
            reasons.push((
                "SNIPER_HS".to_string(),
                format!("{:.0}% sniper", sniper * 100.0),
            ));
        }
        if wall >= 0.20 {
            reasons.push(("WALL".to_string(), format!("{:.0}% wallbang", wall * 100.0)));
        }
        if smoke >= 0.20 {
            reasons.push(("SMOKE".to_string(), format!("{:.0}% smoke", smoke * 100.0)));
        }
    }
    if multi >= 3 {
        reasons.push(("MULTI4".to_string(), format!("{multi} kills in 4s")));
    }
    if n >= 4 && burst >= 0.30 {
        reasons.push(("BURST".to_string(), format!("{:.0}% burst", burst * 100.0)));
    }
    reasons
}

pub fn score_demo(input: &Path, onnx_path: &Path) -> Result<Value> {
    let manifest_path = onnx_path
        .parent()
        .context("model has no parent directory")?
        .join("manifest.json");
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .context("parse manifest")?;
    let human_scores: Vec<f64> = manifest
        .reference
        .as_ref()
        .and_then(|reference| reference.player_scores.as_ref())
        .and_then(|name| std::fs::read_to_string(onnx_path.parent()?.join(name)).ok())
        .and_then(|text| serde_json::from_str::<Vec<f64>>(&text).ok())
        .unwrap_or_default();
    let mut sorted_scores = human_scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = quantile(&sorted_scores, 0.95);
    let k0 = manifest.aggregation.k0;
    let prior = manifest.aggregation.prior;
    let threshold = manifest.threshold.value;

    let (demo_path, temp) = prepare_demo(input)?;
    let result = (|| -> Result<Value> {
        let started = Instant::now();
        let parser = Parser::from_file(&demo_path).context("open demo")?;
        if std::env::var("VG_DEBUG").is_ok() {
            eprintln!("open {:?}", started.elapsed());
        }
        let header = parser.file_header().context("demo header")?;
        let map = header.map_name.unwrap_or_default();
        let kills = parser.kills().context("kills")?;
        if std::env::var("VG_DEBUG").is_ok() {
            eprintln!("kills {:?} count {}", started.elapsed(), kills.len());
        }
        let players = parser.players().unwrap_or_default();
        if std::env::var("VG_DEBUG").is_ok() {
            for player in &players {
                eprintln!(
                    "roster {:?} {:?} {:?}",
                    player.steamid, player.name, player.side
                );
            }
        }
        let rounds = parser.rounds().unwrap_or_default();
        let valid: Vec<&awpy::Kill> = kills
            .iter()
            .filter(|kill| {
                kill.tick >= KILL_ROW as i32
                    && in_live_round(&rounds, kill.tick)
                    && kill.attacker_steamid.is_some()
                    && kill.victim_steamid.is_some()
                    && kill.attacker_steamid != kill.victim_steamid
            })
            .collect();
        let mut ticks = HashSet::new();
        let mut min_tick = i32::MAX;
        let mut max_tick = i32::MIN;
        for kill in &valid {
            let start = kill.tick - KILL_ROW as i32;
            for tick in start..=kill.tick + 32 {
                ticks.insert(tick);
            }
            min_tick = min_tick.min(start);
            max_tick = max_tick.max(kill.tick + 32);
        }
        if ticks.is_empty() {
            bail!("no usable kills in {}", input.display());
        }
        let states = parser
            .snapshots_query(None, &ticks, min_tick, max_tick)
            .context("snapshots")?;
        if std::env::var("VG_DEBUG").is_ok() {
            eprintln!("snapshots {:?} states {}", started.elapsed(), states.len());
        }
        let lookup: HashMap<(u64, i32), &awpy::PlayerState> = states
            .iter()
            .filter_map(|state| state.steamid.map(|id| ((id, state.tick), state)))
            .collect();
        let mut names: HashMap<u64, String> = HashMap::new();
        let mut remember = |id: u64, name: Option<&String>| {
            if let Some(name) = name {
                if !name.is_empty() {
                    names.insert(id, name.clone());
                }
            }
            names.entry(id).or_insert_with(|| {
                let text = id.to_string();
                format!("玩家 {}", &text[text.len().saturating_sub(6)..])
            });
        };
        for player in &players {
            if let Some(id) = player.steamid.filter(|id| *id > 0) {
                if matches!(
                    player.side.as_deref(),
                    Some("terrorist" | "counter-terrorist")
                ) {
                    remember(id, player.name.as_ref());
                }
            }
        }
        for state in &states {
            if let Some(id) = state.steamid.filter(|id| *id > 0) {
                if matches!(state.side, Some("terrorist" | "counter-terrorist")) {
                    remember(id, state.name.as_ref());
                }
            }
        }
        for kill in &valid {
            if let Some(id) = kill.attacker_steamid {
                remember(id, kill.attacker_name.as_ref());
            }
            if let Some(id) = kill.victim_steamid {
                remember(id, kill.victim_name.as_ref());
            }
        }

        let mut windows = Vec::new();
        let mut rows = Vec::new();
        for kill in &valid {
            let Some(window) = build_window(kill, &lookup) else {
                continue;
            };
            let channel_window = channels::build(&window, &map)?;
            let attacker = kill.attacker_steamid.unwrap();
            let midair = lookup
                .get(&(attacker, kill.tick))
                .map(|state| state.is_jumping)
                .unwrap_or(false);
            windows.push(channel_window);
            rows.push(KillRow {
                steamid: attacker,
                tick: kill.tick,
                headshot: kill.headshot,
                weapon: kill.weapon.clone(),
                smoke: kill.thrusmoke,
                wallbang: kill.penetrated > 0,
                midair,
                score: 0.0,
            });
        }
        if windows.is_empty() {
            bail!("no complete kill windows in {}", input.display());
        }
        if std::env::var("VG_DEBUG").is_ok() {
            eprintln!("windows {:?} count {}", started.elapsed(), windows.len());
        }
        let mut ensemble = Ensemble::load(onnx_path)?;
        let scores = ensemble.score(&windows)?;
        if std::env::var("VG_DEBUG").is_ok() {
            eprintln!("scored {:?}", started.elapsed());
        }
        if scores.len() != rows.len() {
            bail!(
                "model returned {} scores for {} windows",
                scores.len(),
                rows.len()
            );
        }
        for (row, score) in rows.iter_mut().zip(scores) {
            row.score = score as f64;
        }

        let mut by_player: HashMap<u64, Vec<usize>> = HashMap::new();
        for (index, row) in rows.iter().enumerate() {
            by_player.entry(row.steamid).or_default().push(index);
        }
        let mut results = Vec::new();
        for (steamid, indices) in &by_player {
            let count = indices.len();
            let sum: f64 = indices.iter().map(|index| rows[*index].score).sum();
            let max = indices
                .iter()
                .map(|index| rows[*index].score)
                .fold(0.0, f64::max);
            let mean = sum / count as f64;
            let score = (sum + k0 * prior) / (count as f64 + k0);
            let percentile = if sorted_scores.is_empty() {
                0.0
            } else {
                sorted_scores.partition_point(|value| *value < score) as f64
                    / sorted_scores.len() as f64
                    * 100.0
            };
            let kill_refs: Vec<&KillRow> = indices.iter().map(|index| &rows[*index]).collect();
            let (behavior, hs, spk, sniper, wall, smoke, burst, multi) = behavior_json(&kill_refs);
            let reasons = reasons_for(count, hs, spk, sniper, wall, smoke, multi, burst);
            let tier = if score >= threshold {
                "flag"
            } else if score >= p95 {
                "review"
            } else {
                "below"
            };
            let mut kills_json: Vec<Value> = kill_refs
                .iter()
                .map(|kill| {
                    json!({
                        "tick": kill.tick,
                        "score": kill.score,
                        "headshot": if kill.headshot { 1 } else { 0 },
                        "weapon": weapon_class(&kill.weapon),
                        "shots": null,
                        "smoke": if kill.smoke { 1 } else { 0 },
                        "wallbang": if kill.wallbang { 1 } else { 0 },
                        "midair": if kill.midair { 1 } else { 0 },
                    })
                })
                .collect();
            kills_json.sort_by_key(|value| value["tick"].as_i64().unwrap_or(0));
            results.push(PlayerResult {
                steamid: *steamid,
                name: names
                    .get(steamid)
                    .cloned()
                    .unwrap_or_else(|| steamid.to_string()),
                tier,
                score: Some(score),
                percentile: Some(percentile),
                kills: count,
                mean: Some(mean),
                max: Some(max),
                reliable: count >= 3,
                reasons,
                behavior,
                kills_json,
            });
        }
        let known: HashSet<u64> = results.iter().map(|player| player.steamid).collect();
        for (steamid, name) in &names {
            if known.contains(steamid) {
                continue;
            }
            results.push(PlayerResult {
                steamid: *steamid,
                name: name.clone(),
                tier: "insufficient",
                score: None,
                percentile: None,
                kills: 0,
                mean: None,
                max: None,
                reliable: false,
                reasons: Vec::new(),
                behavior: json!({
                    "headshot_rate": null,
                    "shots_per_kill": null,
                    "max_multikill_4s": 0,
                    "burst_fraction": null,
                    "median_kill_gap_s": null,
                    "sniper_share": null,
                    "wallbang_rate": null,
                    "smoke_rate": null,
                }),
                kills_json: Vec::new(),
            });
        }
        let tier_rank = |tier: &str| match tier {
            "flag" => 0,
            "review" => 1,
            "below" => 2,
            _ => 3,
        };
        results.sort_by(|a, b| {
            tier_rank(a.tier)
                .cmp(&tier_rank(b.tier))
                .then_with(|| {
                    if a.tier == "below" || a.tier == "insufficient" {
                        std::cmp::Ordering::Equal
                    } else {
                        b.reasons.len().cmp(&a.reasons.len())
                    }
                })
                .then_with(|| {
                    b.score
                        .unwrap_or(0.0)
                        .partial_cmp(&a.score.unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.steamid.cmp(&b.steamid))
        });
        let players_json: Vec<Value> = results
            .iter()
            .enumerate()
            .map(|(index, player)| {
                json!({
                    "rank": index + 1,
                    "steamid": player.steamid.to_string(),
                    "name": player.name,
                    "tier": player.tier,
                    "score": player.score,
                    "percentile": player.percentile,
                    "kill_count": player.kills,
                    "mean": player.mean,
                    "max": player.max,
                    "reliable": player.reliable,
                    "reasons": player.reasons.iter().map(|(code, detail)| json!({
                        "code": code, "detail": detail
                    })).collect::<Vec<_>>(),
                    "behavior": player.behavior,
                    "kills": player.kills_json,
                })
            })
            .collect();
        let flagged = results
            .iter()
            .filter(|player| player.tier == "flag")
            .count();
        let review = results
            .iter()
            .filter(|player| player.tier == "review")
            .count();
        Ok(json!({
            "demo": input.file_name().and_then(|name| name.to_str()).unwrap_or("demo"),
            "map": map,
            "model": manifest.version,
            "threshold": threshold,
            "p95": p95,
            "summary": {
                "players": players_json.len(),
                "flagged": flagged,
                "review": review,
                "kills": rows.len(),
            },
            "players": players_json,
        }))
    })();
    if let Some(dir) = temp {
        let _ = std::fs::remove_dir_all(dir);
    }
    result
}
