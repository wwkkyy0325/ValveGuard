# ValveGuard Desktop — Rust core

Pure-Rust demo parser and scorer. No Python at runtime.

## Build

```bash
cargo build --release
```

## Run

```bash
./target/release/valveguard-desktop \
  --demo "D:/5EDemocache/g161-...zip" \
  --model ../artifacts/valveguard_gbdt_v1.model.json \
  --out result.json
```

Output: per-player `kills`, `mean_score`, `max_kill_score`, `shrunk_score`,
`flagged`, `reliable`.

Debug options:

```bash
# dump the 59-feature vectors for Python/Rust parity checks
--dump-features features.json
# dump a single 256-tick window for debugging
--dump-window <steamid>:<tick> --dump-window-out window.json
```

## What is implemented

- `.zip` / `.dem` input; the zip is extracted to a temp directory with
  path-traversal checks.
- Parsing via the `awpy` 3.0 crate: header, players, kills, rounds, per-tick
  player states.
- Warm-up filtering via round intervals (the first live round is kept).
- Port of the 59 aim features from `valveguard/features`.
- JSON tree-ensemble scoring (300 trees, 931 KB).
- Player aggregation with empirical-Bayes shrinkage.

## Validation (5E dust2 sample, 2026-10-18)

- 129 of 145 kills used (short-history and warm-up kills dropped).
- 10 players, 0 flagged.
- **3.9 s** end to end on CPU, including zip extraction.
- Python vs Rust: 129/129 kills matched; kill-score max diff **0.0**;
  player-score max diff **4.6e-7**; identical kill counts and flags.

## Not yet implemented

- Tauri UI and Chinese interface.
- 2D trajectory/error previews.
- "Open in CS2" launch and OBS background capture.
