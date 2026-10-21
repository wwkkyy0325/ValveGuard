# Tauri Desktop Review App

Status: working (2026-10-20). Rust + WebView2 desktop window; no browser
and no local HTTP server in production.

## Run

```bash
cd desktop-ui
npm install
npm run tauri dev
```

Dev frontend: `http://127.0.0.1:47183` (fixed, strict port; chosen high to
avoid common dev ports). Production Tauri uses its custom protocol, so no
port is exposed to the browser.

Requirements: Node 20+, Rust, and `artifacts/valveguard_cnn_v1_onnx`.
The Rust core parses the demo with `awpy`, builds the 29 aim channels, and
runs the ONNX ensemble through `ort`. No Python is needed at runtime.

## Layout

The window has no in-app title bar and is dark-only: there is no light
theme and no theme switch. The native Windows title bar is themed dark via
`"theme": "Dark"` and the window background is set to the same near-black,
so launch does not flash white. A fixed 288 px sidebar holds the processing
history, the `添加 demo` button, the loaded model, and the app version; the
content region shows one screen at a time.

- Empty state: one large dashed drop target filling the content region.
  Click it, click `添加 demo`, or drop a file anywhere on the window.
- Working state: filename plus the current stage (parse, windows, model,
  aggregation).
- Failed state: the parser or scorer error and a retry button.
- Results and player detail: the existing table/detail screens, scoped to
  the selected record.

## Input

- Drag a `.dem` or a `.zip` containing one `.dem` into the window. The
  native Tauri drag-drop event supplies the path and starts analysis.
- Or click the drop target / `添加 demo` for the native file dialog.
- No file is uploaded; the Rust core reads the path directly.
- Players with zero valid kill windows are still listed as `无窗口`
  (insufficient evidence) instead of disappearing from the 10-player roster.

## Processing records

Records live under `Documents/ValveGuard/records.json` (created on the
first successful analysis), the same Documents-folder convention Verse
uses. Each entry stores the demo path, status, and the full result payload,
so a record opens instantly after a restart without re-parsing the demo.

- The newest record is first; the file keeps at most 50 entries.
- `load_records` runs on startup and restores the list; no record is
  selected automatically, so the first page stays the drop target.
- `save_record` replaces an entry with the same id and writes the file
  atomically enough for a desktop history (read, merge, write).
- Failed analyses are stored too, with the parser error, and can be retried
  from the failure screen.

## Screens

- Four summary cards: map, player count, FLAG count, REVIEW count.
- The overview page is the match chart alone; there is no separate player
  table.
- Match chart: one interactive row per player, sorted high to low, with
  rank, avatar, name, tier, gradient bar, and the human p95/p99 chips and
  shaded bands. Bars animate in. A formal legend under the chart spells out
  FLAG / REVIEW / BELOW with their score ranges; tiers are always written in
  uppercase in the UI.
- Hovering a row dims the rest, brightens that bar, and opens a card with
  score, percentile, window count, headshot rate, mean, maximum, and the
  behavior reasons.
- Clicking a row (or pressing Enter/Space on a focused row) opens that
  player's detail page.
- Player detail summary card: avatar, name, tier, percentile, evidence
  status, a risk-score gauge, and a six-cell metric strip (score,
  percentile, window mean, window maximum, headshot rate, threshold).
- Player chart: per-kill score line with axes, grid, p95/p99 lines.
  Moving inside the plot snaps to the nearest window and shows a vertical
  guide, a ring on the point, and a card with index, score, tick, weapon,
  and flags. Window count, mean, and maximum sit beside the title; a legend
  under the chart explains the dot colors and threshold bands.
- "判定依据" is a structured label/value table: tier and rule, risk score,
  human p95/p99, percentile, window count, window mean/maximum, and reason
  chips.
- Behavior cards: HS%, shots/kill, 4s multikill, burst, kill gap, sniper
  share, wallbang, smoke.
- Kill table: percentage column widths so every column keeps an even
  share of the panel, headers aligned with their cells, tick and score
  right-aligned, booleans centered, score shown as a colored number
  (no bar), and check marks for headshot/smoke/wallbang/midair. A summary
  shows how many windows fall in the review and high-risk bands.
- `shots/kill` is shown as `—` in the Rust path: `awpy` shot-enrichment is
  too slow for interactive use, so the app does not fabricate it and the
  efficiency reason is disabled.

The layout follows Verse's neutral, card-based, dense-table style, while
the frontend stays dependency-light (vanilla HTML/CSS/JS) so the Tauri bundle
does not pull a component framework.

## Architecture

```
desktop-ui/                 Vite frontend (HTML/CSS/JS, Chinese UI)
  src/main.js               records, screens, invoke("health"),
                            invoke("score_demo"), invoke("load_records"),
                            invoke("save_record"), drag-drop listener
  src-tauri/src/main.rs     Tauri commands health / score_demo /
                            load_records / save_record
  src-tauri/tauri.conf.json window, devUrl 127.0.0.1:47183, icons
desktop/src/                Rust core (shared with the CLI)
  demo.rs                   awpy parsing and kill windows
  channels.rs               256 x 29 aim channels
  onnx.rs                   ort session and ensemble inference
  app.rs                    aggregation, reasons, UI JSON
```

Rust commands:

| Command | Result |
|---|---|
| `health` | model version, threshold, runtime |
| `score_demo(path)` | full analysis payload or an error string |
| `load_records()` | saved processing history (`[]` when absent) |
| `save_record(record)` | upsert one record under Documents/ValveGuard |

The core reads the ONNX bundle manifest and `human_player_scores.json` for
the threshold, prior, k0, and percentiles; scoring uses the same 15-model
graph as the Python artifact.

## Runtime notes

- Release scoring on a 392 MB demo: about 4.5 s (1.2 s events, 2.3 s
  snapshots/roster, 0.8 s ONNX). Debug builds are much slower and are not the
  deployment path.
- 27 of the 29 channels match the Python pipeline at 1e-5. `attacker_vel`
  and `attacker_speed` differ because `awpy` exposes `m_vecVelocity` while
  the original pipeline used a different velocity property; player scores
  differ by roughly 0.01-0.02 on the validation demo and the ranking is
  near-identical, but it is not bit-exact. Do not claim numeric parity with
  the Python artifact.
- `shots/kill` is unavailable because `awpy` shot enrichment is too slow for
  interactive use; the efficiency reason is disabled.
- Some demos accepted by `demoparser2` fail in `awpy` with a file-info
  header error. The app surfaces the parser error instead of hiding it.

## Packaging

`npm run tauri build` builds the WebView2 app. The current build expects
`artifacts/valveguard_cnn_v1_onnx` next to the repo during development; the
release bundle will add the ONNX files as Tauri resources. No Python sidecar
is involved.
