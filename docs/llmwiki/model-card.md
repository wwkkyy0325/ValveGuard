# Model Card — ValveGuard Aim-Assist Detector v1

Status: deployment-oriented artifact, 2026-10-20. Research model, not an
automatic ban system.

## Model

- Estimator: wide attention 1D CNN over 256 ticks and 29 aim channels
  (4 conv blocks at 96/192/256/256, time attention, binary head).
- Training data: 79,533 clean CS2CD windows from 793 matches (ncdata label 0
  plus cheater windows; cdata label-0 excluded). CS2 view data at 64 Hz.
- Fixed match folds from `reports/cv_gbdt/cv_predictions.parquet`; every
  seed uses the same five folds.
- Ensemble: 3 seeds (42/43/44) x 5 folds = 15 models, averaged. Negative
  windows use hard-negative BCE weight `1 + 5 * p^2`.
- Artifact: `artifacts/valveguard_cnn_v1` (local, not tracked): manifest,
  channel names, per-seed normalization, fold models, human reference
  distributions, SHA256 checksums. The loader verifies checksums and selects
  channels by name.
- Portable ONNX bundle: `export_onnx.py` writes a self-contained graph
  (15 models + normalization + sigmoid average) with checksums. PyTorch vs
  ONNX parity is 2.58e-8 on random inputs; see `onnx-distribution.md`.
- Player score: empirical-Bayes shrinkage of the kill-score mean,
  `(sum + 5 * prior) / (n + 5)`, prior 0.3018746, threshold **0.608027**
  (OOF human player p99, target FPR 1%).

## Input contract

Two supported paths:

1. **CS2CD cache (default demo path).** `(N, 256, 43)` float16 channels plus
   `channel_names.json`. The artifact selects the 29 aim channels by name and
   applies per-seed median/IQR normalization. Kill row is 223.
2. **Arbitrary CS2 demo.** Parse the demo, build the same 256x43 windows, and
   score them. Required per-tick fields at 64 Hz:
   - attacker/victim `X`, `Y`, `Z`, `pitch`, `yaw`, `health`;
   - `active_weapon_name`, `flash_duration`, `shots_fired`, `is_alive`;
   - `total_rounds_played` (warm-up exclusion);
   - `player_death` events with attacker, victim, tick, headshot, thrusmoke,
     penetrated, attackerinair.

Each kill with 223 ticks of history becomes a window; fewer than 223 ticks
means the kill is skipped. No labels or raw mouse input are required.

## Output contract

```json
{
  "source": "CS2CD",
  "match": "cdata:0",
  "map": "mirage",
  "artifact": "valveguard_cnn_v1",
  "threshold": 0.608027,
  "players": [
    {"player": "Player_3", "dataset_label": 1, "tier": "flag",
     "review_rank": 1, "kills": 29, "headshot_rate": 0.9655,
     "shots_per_kill": 1.862, "max_multikill_4s": 4,
     "burst_fraction": 0.3571, "shrunk_score": 0.708932,
     "percentile": 99.8, "flagged": true, "reliable": true,
     "review_reasons": ["HS>=80%", "smoke>=20%", "4s multikill>=3",
                        "burst>=30%"]}
  ],
  "kills": [{"player": "Player_3", "kill_tick": 3496, "score": 0.61}]
}
```

- `shrunk_score` is the decision score; `flagged` is
  `shrunk_score >= threshold`.
- `tier` is `flag`, `review` (human p95-p99 band), or `below`.
- `reliable` is `kills >= 3`. Treat `flagged && !reliable` as insufficient
  evidence, not as a detection.
- `review_reasons` and `review_rank` are review context only. They never
  change the risk score or the flag.
- Scores are risk scores, not calibrated probabilities (player ECE ~0.18).

## Metrics (fixed match folds)

| Model | Player AUC | Recall@FPR=1% | Q4 FPR | Q4 recall | Kill AUC |
|---|---|---|---|---|---|
| Champion: 29ch wide CNN, 3 seeds, hard-neg | **0.9502** | **0.3619** | 3.11% | **39.1%** | **0.9015** |
| Mean 29ch ensemble, no hard-neg | 0.9500 | 0.3121 | 2.59% | 33.9% | -- |

- Champion vs GBDT baseline: Δrecall **+0.1223 [0.0476, 0.1988]**,
  ΔAUC **+0.0165 [0.0120, 0.0211]**, ΔQ4 FPR +0.0030 [-0.0090, 0.0161].
- Champion vs the mean ensemble: Δrecall **+0.0459 [0.0048, 0.0883]**,
  ΔAUC +0.0002 [-0.0014, 0.0016], ΔQ4 FPR +0.0048 [-0.0057, 0.0158].
- Professional demos (7 demos, 1,075 windows, 70 players): 0/70 players
  flagged at the 1% operating point.
- Q4 FPR at the correct player-level 1% cut is 3.11%.

## Demo report

`demo_report_dataset.py` scores one or more CS2CD matches and writes
`result.json`, `review_queue.csv`, and three charts per match, plus a
combined queue at the output root.

- `cdata:0` (Mirage, 78 windows): 1 FLAG (`Player_3`, 0.709, p99.8, four
  behavior reasons). The second labelled cheater is REVIEW at 0.399.
- `ncdata:36` (Dust2, 232 windows): 0 FLAG / 0 REVIEW; top score 0.227
  (p67.5).
- These matches overlap the artifact fold models, so the demo is a path and
  presentation check, not a held-out evaluation.

## Intended use

- Post-match review and triage on CS2 demos or CS2CD windows, with a human
  analyst.
- Batch scoring; per-player risk ranking and a review queue.

## Out of scope

- Automatic bans or real-time in-game enforcement.
- Cross-demo player identity or ban-evasion detection.
- CS:GO (Source 1) demos; other games.
- Judging players with fewer than three kills (insufficient evidence).
- Maps outside `MAP_BOUNDS`; unknown-map demos are rejected.

## Limitations

- 64 Hz view angles, not 1000 Hz mouse input; silent/low-FOV aim may be
  invisible.
- Match-disjoint training only; CS2CD player ids are per-match pseudonyms.
- Labels are VAC-based with limited recall; the excluded cdata label-0 windows
  look human at the window level (5.5% estimated admixture), so label noise is
  present but not dominant.
- Professional players are not flagged, but professional demos are not a
  random sample of the ranked population.
- Behavior context is descriptive. In `cdata:0` a clean player with two
  reasons ranks above a labelled cheater inside the REVIEW band.
- The demonstration matches are in-sample for the fold ensemble; do not quote
  their scores as generalization metrics.

## Retraining policy

- **Batch retraining, not incremental.** The full ensemble retrains on the
  training box; incremental updates risk catastrophic forgetting.
- Retrain monthly or after a CS2 patch, on the same clean label policy.
- Before promotion: fixed-fold metrics, pro-demo specificity (0 players
  flagged), and a score-distribution drift check against the previous model.
- Version the artifact; keep the previous version for rollback.

## Reproduction

```bash
# Dataset demo: two matches, combined review queue.
PYTHONPATH=src python -m valveguard.scripts.demo_report_dataset \
  --match cdata:0 --match ncdata:36 --out reports/demo_dataset

# Arbitrary CS2 demo through the same artifact.
PYTHONPATH=src python -m valveguard.scripts.demo_report \
  --demo match.dem --artifact artifacts/valveguard_cnn_v1 \
  --out reports/demo_reports
```

Training, ablations, and bootstrap protocol: `experiments.md` section 20.
