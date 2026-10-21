# Windows Application Plan

Status: Tauri desktop review app implemented 2026-10-20 (Rust shell,
WebView2 frontend, awpy + channels + ort). See `review-app.md`.

## Target input

The app accepts a `.dem` file or a `.zip` containing one `.dem`.

Validated on 2026-10-18:

- The zip contains one standard `.dem` (189 MB uncompressed).
- The Rust crate `awpy` 3.0 parses the header, roster, and 145 kill events.
- A broadcast spectator entry (`steamid 0`, no kills) is present; it is
  filtered and must never appear in the player table.
- Player names are UTF-8 and parse correctly.
- The Python reference pipeline (parse -> windows -> features -> score)
  finishes in **6 s** on this demo.

The app accepts `.zip` and `.dem`; for zip, extract to a temp directory with
path-traversal checks and locate the single `.dem`.

## End-to-end flow

```
zip/dem
  -> extract .dem (zip crate)
  -> parse header, players, kills (awpy)
  -> filter warm-up, suicides, spectators (steamid 0), short-history kills
  -> build (256,44) windows, kill row 223
  -> extract 59 features
  -> score with the JSON tree ensemble
  -> aggregate per player: n, mean, shrunk, flagged, reliable
  -> UI table + review cards + export
```

Players with no valid kill windows are omitted. The spectator entry
(`steamid = 0`, no kills) must never appear in the table.

## Review UX

Player table (sortable):

- nickname, steamid, side, kills, mean score, shrunk score, flagged, reliable;
- filter: flagged / reliable / all.

Per-kill review card:

- start tick = kill_tick - 223, kill tick, end tick = kill_tick + 32;
- victim, weapon, headshot, through-smoke, wallbang, midair;
- kill score and key feature percentiles (final error, time to target,
  straightness, submovements, lock ratio);
- a 2D preview: yaw/pitch trajectory, target-error curve, top-down path.

Export: CSV/JSON of the player table and the selected kills.

## Review segments

A demo is a state log, not a video. The app maps every kill to a tick-range
segment `(demo, start_tick, kill_tick, end_tick, player)`; review uses 2D
previews rendered from tick data, so CS2 does not need to be running.

## Rust/Tauri architecture

Tauri v2, Rust backend, web frontend (React/Svelte/vanilla).

| Concern | Choice |
|---|---|
| Demo parsing | `awpy` 3.0 crate (validated on a local CS2 demo) |
| Zip handling | `zip` crate, extract to temp |
| Model | JSON tree ensemble, Rust traversal |
| UI | Tauri webview; WebView2 is preinstalled on Windows 10/11 |
| Charts | frontend canvas/SVG from the kill window data |

Model export: `export_model_json.py` produces a ~931 KB JSON with 300 trees.
The Rust traversal matches sklearn exactly (max abs diff 3.3e-16 on 200 real
rows with NaNs), so no Python is needed at inference time.

## Feature port (main remaining work)

Port to Rust, in this order, with a Python-Rust golden comparison after each:

1. map bounds and geometry recovery (denormalize, angles, target error);
2. dynamics (speed/accel/jerk, curvature, straightness, submovements, SPARC,
   LDLJ, PSD slope, entropy, lead-lag);
3. window summary (the 59 features);
4. validation: same demo, same kills, max feature difference < 1e-6.

Effort estimate: 3-5 days for the feature port, 3-5 days for the Tauri UI.

## Alternatives (historical)

- **Python sidecar**: Tauri + bundled PyInstaller worker reuses the exact
  Python pipeline immediately; larger package (~150-250 MB) and slower start.
- **Pure Rust**: ~10-20 MB app, single process, but requires the feature port.

Pure Rust was chosen and implemented: `desktop/src/channels.rs` builds the 29
aim channels and `desktop/src/onnx.rs` runs the ONNX ensemble through `ort`.
The Python sidecar was removed.

## Rust core status (2026-10-18)

Implemented in `desktop/`:

- `.zip` / `.dem` input, temp extraction, `awpy` parsing, per-tick snapshots;
- port of the 59 aim features;
- JSON tree-ensemble scoring;
- player aggregation and JSON output; player names preserved.

Validated on a local Dust2 demo:

- 129 of 145 kills used; 10 players; 0 flagged;
- **3.9 s** end to end, including zip extraction;
- Python vs Rust: 129/129 kills matched, kill-score max diff **0.0**,
  player-score max diff **4.6e-7**, identical kill counts and flags.

A real bug surfaced during the parity check: `total_rounds_played` is 0 for
the whole first live round, so the Python warm-up filter was dropping
first-round kills. Both pipelines now use explicit round/warm-up logic; the
pro-demo results are re-run with the fix.

Remaining work: Tauri UI (Chinese), 2D previews.

## Decisions

1. **Pure Rust** for the desktop app.
2. v1 review = tick ranges + 2D preview; video capture and game launch are out of scope.
3. Chinese UI by default.
4. Single demo first; batch queue after the table/review flow is stable.
