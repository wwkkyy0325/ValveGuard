# ValveGuard

CS2 aim-assist detection from kill-centred view trajectories. The repository
contains the detector, the packaged scoring artifact, and a dataset-backed
demo report; it does not include game-client or overlay tooling.

## Paper

Final bilingual write-up with formulas, figures, and a one-match end-to-end
worked example:

- [中文版：基于击杀窗口视线轨迹的 CS2 辅助瞄准检测](docs/paper/valveguard-paper-zh.md)
- [English: Detecting Aim Assist in CS2 from Kill-Centred View Trajectories](docs/paper/valveguard-paper-en.md)

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

# Score two CS2CD matches with the packaged CNN artifact and write a report.
# The artifact bundle is local; rebuild it from the saved fold models.
PYTHONPATH=src python -m valveguard.scripts.demo_report_dataset \
  --match cdata:0 --match ncdata:36 --out reports/demo_dataset

# Optional: score an arbitrary CS2 demo through the same artifact.
PYTHONPATH=src python -m valveguard.scripts.demo_report \
  --demo match.dem --artifact artifacts/valveguard_cnn_v1 \
  --out reports/demo_reports

# Portable ONNX bundle (15 fold models, normalization, checksums).
pip install -r requirements-onnx.txt
PYTHONPATH=src python -m valveguard.scripts.export_onnx \
  --artifact artifacts/valveguard_cnn_v1 \
  --out artifacts/valveguard_cnn_v1_onnx --zip
PYTHONPATH=src python -m valveguard.scripts.demo_report_dataset \
  --artifact artifacts/valveguard_cnn_v1_onnx \
  --match cdata:0 --out reports/demo_onnx
```

Model card: `docs/llmwiki/model-card.md`.
ONNX bundle: `docs/llmwiki/onnx-distribution.md`.

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
- Elite human aim is not mistaken for aim assist: on 1,075 kill windows from
  seven professional HLTV demos (70 player-maps), the champion flags 0/70
  players and 0.28% of kills at its 1% FPR operating point. Pros score below
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

## Status (2026-10-20)

Champion: aim-only 29-channel wide CNN, 3 seeds x 5 fixed match folds,
hard-negative BCE alpha=5. OOF: player AUC **0.9502**, recall@FPR=1%
**0.3619**, kill AUC **0.9015**, Q4 FPR **3.11%**, Q4 recall **39.1%**.
Paired bootstrap vs the GBDT baseline: Δrecall +0.1223 [0.0476, 0.1988],
ΔAUC +0.0165 [0.0120, 0.0211].

- Dataset demo: `cdata:0` (Mirage) flags the labelled cheater `Player_3` at
  0.709 (human p99.8) with four behavior reasons; the second labelled cheater
  lands in REVIEW at 0.399. `ncdata:36` (Dust2, clean) has 0 FLAG/REVIEW.
- Demo scores are not held-out metrics: the demonstration matches overlap the
  artifact fold models. Use the fixed-fold table in `experiments.md` for
  held-out numbers.
- Professional demos (7 demos, 1,075 windows, 70 players): 0/70 players
  flagged at the 1% operating point.
- Reports: `reports/REPORT.md`; artifact: `artifacts/valveguard_cnn_v1`
  (local, not tracked).
