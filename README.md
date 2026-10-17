# ValveGuard

CS2 aim-assist detection from kill-centred view trajectories. Detector-first;
the conditional trajectory generator is deferred.

## Data

Primary dataset: [`CS2CD/Context_window_256`](https://huggingface.co/datasets/CS2CD/Context_window_256)
(CC-BY-4.0).

- `context_windows_256.tar` (9.4 GB), one parquet per kill window.
- Each window is `256 x 44`: 223 ticks before the kill, the kill tick at row
  223, and 32 ticks after it.
- Label is the directory (`cheater/`, `not_cheater/`).
- Filename: `cdata-cheater-file_{match}-Player_{k}-kill_{i}.parquet`; `_augN`
  files are augmented copies.
- `file_{match}` maps to `with_cheater_present/{match}` (or
  `no_cheater_present/{match}`) in the raw CS2CD dataset. To inspect the
  original event, download only that match's parquet + JSON; the full 48.9 GB
  raw dataset is not needed.
- `Player_{k}` is a per-match pseudonym, not a stable Steam identity. Splits are
  match-disjoint; a true person-disjoint claim is not possible on this dataset.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu124 torch

# Build a float16 memmap cache + index.
PYTHONPATH=src .venv/Scripts/python -m valveguard.scripts.build_cache \
  --tar data/raw/context_windows_256.tar --out data/processed/cw256

# Audit labels, matches, shapes, non-finite values.
PYTHONPATH=src .venv/Scripts/python -m valveguard.scripts.audit \
  --cache data/processed/cw256 --out reports/audit.json

# Handcrafted-feature baseline.
PYTHONPATH=src .venv/Scripts/python -m valveguard.scripts.run_baseline \
  --cache data/processed/cw256 --out reports/baseline

# Tests.
python -I -m pytest -q tests
```

## Layout

```
src/valveguard/
  data/tar_index.py     tar -> memmap cache + metadata
  data/split.py         match-disjoint splits
  features/geometry.py  angles, world positions, target-relative errors
  features/dynamics.py  smoothness / submovement / spectral metrics
  features/summarize.py flat features and per-tick channels
  models/baseline.py    HistGradientBoosting baseline
  metrics.py            AUC/AP/FPR/ECE
  scripts/              CLIs
tests/                  geometry, features, split
docs/llmwiki/           design, detailed model spec, task plan, changelog
```

## Findings so far

- Window mapping is exact: row 223 is the kill tick; recovered pitch, yaw, and
  distance match the raw parquet (sample: distance 1470.8 units).
- CS2's stored pitch has the opposite sign of `atan2(dz, horizontal)`. Correcting
  it reduces the sample kill-tick target error from 6.38 deg to 0.24 deg. The
  published `attacker_pitch_head_delta` uses the uncorrected sign; the pipeline
  drops the published absolute deltas and recomputes signed, corrected channels.
- `not_cheater` negatives come from clean matches (`ncdata`); non-cheater
  players inside cheater matches are not used as negatives (their label
  precision is only ~55.6% according to the dataset card).
- The dataset does not contain raw mouse deltas. Mouse–view consistency
  (`dyaw ~ usercmd_mouse_dx`) is available only if a small raw parquet subset is
  fetched later.
- Elite human aim is not mistaken for aim assist: on 1,020 kill windows from
  seven professional HLTV demos (70 player-maps), the champion flags 0/70
  players and 0.29% of kills at its 1% FPR operating point. Pros score below
  average CS2CD humans despite being more accurate.
- The excluded `cdata` label-0 windows score like clean humans (5.5% estimated
  cheater admixture by first-moment decomposition), so the dataset card's
  55.6% precision is not corroborated at the window level. Details:
  `docs/llmwiki/label-noise-audit.md`.

## Limitations

- 64 Hz view angles, not 1000 Hz raw mouse input.
- Match-disjoint evaluation only; per-match pseudonyms prevent a verifiable
  person-disjoint split and any cross-match person linkage. Only within-match
  player series are available (median 13 kills per player-match).
- No target visibility/raycast in v1; FOV, distance, and smoke/wallbang flags
  are proxies.
- Reported metrics must include the low-FPR operating point; accuracy alone is
  not a deployment criterion.

## Status (v2 checkpoint, 2026-10-17)

Detector v1 remains the champion; the v2 hypotheses did not beat it:

- Champion: aim-only GBDT, 5-fold match-disjoint OOF:
  - player AUC 0.9331, player recall@FPR=1% 0.1735, kill AUC 0.8497.
- Locked 20% match holdout (158 matches): champion trained on the dev 80%
  scores player AUC 0.9384, recall@FPR=1% 0.3091 (CI 0.187–0.523).
- H1 43-channel Transformer on dev 80%: 0.9251 / 0.2297; paired ΔAUC
  [-0.0153, 0.0057], Δrecall [-0.0071, 0.1809] — not promoted.
- H2 1024-tick context: no gain (ΔAUC [-0.0055, 0.0078]) — rejected.
- H3 rank-group DRO: no change — rejected.
- Professional demos (7 HLTV demos, 1,020 windows, 70 players): 0/70 players
  flagged at the 1% operating point; adding them as negatives changes nothing.
- Q4 false positives were understated by a threshold error. At the correct
  player-level 1% cut the champion's Q4 FPR is 19.05%; empirical-Bayes
  shrinkage of player scores by kill count (k0=5) reduces it to 4.76% and
  raises recall@FPR=1% from 0.1735 to 0.2288 (dev80: 13.16% -> 5.26%,
  0.1475 -> 0.2386). The holdout was already spent, so this is dev-validated
  only.
- Reports: `reports/REPORT.md` (sections 11-14); figures: `reports/figures/`.
