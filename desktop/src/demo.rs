//! Demo ingestion: zip/dem -> kills -> windows -> scores.

use anyhow::{Context, Result, bail};
use awpy::{Kill, Parser, PlayerState, Round};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::features::{KILL_ROW, LEN, Window, summarize};
use crate::model::Model;

pub struct PlayerRow {
    pub steamid: u64,
    pub name: String,
    pub kills: usize,
    pub mean: f64,
    pub max: f64,
    pub shrunk: f64,
    pub flagged: bool,
}

pub struct KillFeature {
    pub steamid: u64,
    pub kill_tick: i32,
    pub features: Vec<f64>,
}

pub struct DebugWindow {
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
    pub az: Vec<f64>,
    pub pitch: Vec<f64>,
    pub yaw: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vz: Vec<f64>,
}

pub struct Analysis {
    pub map: String,
    pub kills_total: usize,
    pub kills_used: usize,
    pub rows: Vec<PlayerRow>,
    pub kill_features: Vec<KillFeature>,
    pub debug_window: Option<DebugWindow>,
}

pub fn prepare_demo(input: &Path) -> Result<(PathBuf, Option<PathBuf>)> {
    let is_zip = input
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("zip"));
    if !is_zip {
        return Ok((input.to_path_buf(), None));
    }
    let dir = std::env::temp_dir().join(format!("valveguard_{}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let file = std::fs::File::open(input)?;
    let mut archive = zip::ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(name) = entry.enclosed_name() else {
            continue;
        };
        if name
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("dem"))
        {
            let dest = dir.join("match.dem");
            let mut out = std::fs::File::create(&dest)?;
            std::io::copy(&mut entry, &mut out)?;
            return Ok((dest, Some(dir)));
        }
    }
    bail!("no .dem inside {}", input.display());
}

pub fn in_live_round(rounds: &[Round], tick: i32) -> bool {
    if rounds.is_empty() {
        return true;
    }
    rounds.iter().any(|r| {
        if r.round_num < 1 {
            return false;
        }
        let start = r.freeze_end_tick.or(r.start_tick).unwrap_or(r.end_tick);
        tick >= start && tick <= r.end_tick
    })
}

pub fn a_id_unwrap(k: &Kill) -> u64 {
    k.attacker_steamid.unwrap()
}

pub fn build_window(kill: &Kill, lookup: &HashMap<(u64, i32), &PlayerState>) -> Option<Window> {
    let a_id = kill.attacker_steamid?;
    let v_id = kill.victim_steamid?;
    let t0 = kill.tick - KILL_ROW as i32;
    if t0 < 0 {
        return None;
    }
    let mut w = Window {
        ax: [0.0; LEN],
        ay: [0.0; LEN],
        az: [0.0; LEN],
        pitch: [0.0; LEN],
        yaw: [0.0; LEN],
        vx: [0.0; LEN],
        vy: [0.0; LEN],
        vz: [0.0; LEN],
        vhealth: [0.0; LEN],
        avel: [0.0; LEN],
    };
    for i in 0..LEN {
        let tick = t0 + i as i32;
        let a = lookup.get(&(a_id, tick))?;
        let v = lookup.get(&(v_id, tick))?;
        w.ax[i] = a.x? as f64;
        w.ay[i] = a.y? as f64;
        w.az[i] = a.z? as f64;
        w.pitch[i] = a.pitch as f64;
        w.yaw[i] = a.yaw as f64;
        w.vx[i] = v.x? as f64;
        w.vy[i] = v.y? as f64;
        w.vz[i] = v.z? as f64;
        w.vhealth[i] = v.health as f64;
        w.avel[i] = a.velocity.unwrap_or(0.0) as f64;
    }
    Some(w)
}

pub fn analyze(input: &Path, model: &Model, debug: Option<(u64, i32)>) -> Result<Analysis> {
    let (dem, temp) = prepare_demo(input)?;
    let parser = Parser::from_file(&dem).context("open demo")?;
    let header = parser.file_header().context("demo header")?;
    let map = header.map_name.unwrap_or_default();
    let players = parser.players().context("players")?;
    let kills = parser.kills().context("kills")?;
    let rounds = parser.rounds().unwrap_or_default();
    if std::env::var("VG_DEBUG").is_ok() {
        for r in &rounds {
            eprintln!(
                "round {} start={:?} freeze={:?} end={} official={:?}",
                r.round_num, r.start_tick, r.freeze_end_tick, r.end_tick, r.official_end_tick
            );
        }
    }
    let total = kills.len();

    let valid: Vec<&Kill> = kills
        .iter()
        .filter(|k| {
            k.tick >= KILL_ROW as i32
                && in_live_round(&rounds, k.tick)
                && k.attacker_steamid.is_some()
                && k.victim_steamid.is_some()
                && k.attacker_steamid != k.victim_steamid
        })
        .collect();

    let mut ticks: HashSet<i32> = HashSet::new();
    let mut min_tick = i32::MAX;
    let mut max_tick = i32::MIN;
    for k in &valid {
        let t0 = k.tick - KILL_ROW as i32;
        for t in t0..=k.tick + 32 {
            ticks.insert(t);
        }
        min_tick = min_tick.min(t0);
        max_tick = max_tick.max(k.tick + 32);
    }
    if ticks.is_empty() {
        bail!("no usable kills in {}", input.display());
    }
    let states = parser
        .snapshots_query(None, &ticks, min_tick, max_tick)
        .context("snapshots")?;
    let mut lookup: HashMap<(u64, i32), &PlayerState> = HashMap::with_capacity(states.len());
    for s in &states {
        if let Some(sid) = s.steamid {
            lookup.insert((sid, s.tick), s);
        }
    }

    let names: HashMap<u64, (String, Option<String>)> = players
        .iter()
        .filter_map(|p| {
            p.steamid
                .map(|s| (s, (p.name.clone().unwrap_or_default(), p.side.clone())))
        })
        .collect();

    if model.features.len() != 59 {
        bail!(
            "model expects {} features, extractor produces 59",
            model.features.len()
        );
    }
    let mut agg: HashMap<u64, (usize, f64, f64)> = HashMap::new();
    let mut kill_features: Vec<KillFeature> = Vec::new();
    let mut debug_window: Option<DebugWindow> = None;
    let mut used = 0usize;
    for k in &valid {
        let Some(w) = build_window(k, &lookup) else {
            continue;
        };
        if debug == Some((a_id_unwrap(k), k.tick)) {
            debug_window = Some(DebugWindow {
                ax: w.ax.to_vec(),
                ay: w.ay.to_vec(),
                az: w.az.to_vec(),
                pitch: w.pitch.to_vec(),
                yaw: w.yaw.to_vec(),
                vx: w.vx.to_vec(),
                vy: w.vy.to_vec(),
                vz: w.vz.to_vec(),
            });
        }
        let feats = summarize(&w);
        let score = model.score(&feats);
        let a_id = k.attacker_steamid.unwrap();
        kill_features.push(KillFeature {
            steamid: a_id,
            kill_tick: k.tick,
            features: feats,
        });
        let e = agg.entry(a_id).or_insert((0, 0.0, 0.0));
        e.0 += 1;
        e.1 += score;
        e.2 = e.2.max(score);
        used += 1;
    }

    let mut rows: Vec<PlayerRow> = agg
        .into_iter()
        .map(|(steamid, (n, sum, max))| {
            let mean = sum / n as f64;
            let shrunk = (sum + model.k0 * model.prior) / (n as f64 + model.k0);
            let name = names
                .get(&steamid)
                .map(|(n, _)| n.clone())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| steamid.to_string());
            PlayerRow {
                steamid,
                name,
                kills: n,
                mean,
                max,
                shrunk,
                flagged: shrunk >= model.threshold,
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.shrunk
            .partial_cmp(&a.shrunk)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(dir) = temp {
        let _ = std::fs::remove_dir_all(dir);
    }
    Ok(Analysis {
        map,
        kills_total: total,
        kills_used: used,
        rows,
        kill_features,
        debug_window,
    })
}
