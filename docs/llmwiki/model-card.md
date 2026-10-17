# Model Card — ValveGuard Aim-Assist Detector v1

Status: deployment-oriented artifact, 2026-10-18. Research model, not an
automatic ban system.

## Model

- Estimator: `HistGradientBoostingClassifier` (300 iterations, 31 leaves,
  learning rate 0.05, L2 1.0), 59 handcrafted aim features.
- Training data: all 79,533 clean CS2CD windows from 793 matches (ncdata label
  0 plus cheater windows; cdata label-0 excluded). CS2 view data at 64 Hz.
- Artifact: `artifacts/valveguard_gbdt_v1.joblib` (1.2 MB) plus a JSON sidecar
  with version, feature order, threshold, prior, and k0.
- Player score: empirical-Bayes shrinkage of the kill-score mean,
  `(sum + 5 * prior) / (n + 5)`, prior 0.290, threshold 0.7916 at the training
  player-level 1% FPR.

## Input contract

One CS2 (Source 2) demo per inference call. Required per-tick fields at 64 Hz:

- attacker/victim `X`, `Y`, `Z`, `pitch`, `yaw`, `health`;
- `active_weapon_name`, `flash_duration`, `shots_fired`, `is_alive`;
- `total_rounds_played` (warm-up exclusion);
- `player_death` events with attacker, victim, tick, headshot, thrusmoke,
  penetrated, attackerinair;
- map name in the demo header, one of the 15 maps in `MAP_BOUNDS`.

Each kill with 223 ticks of history becomes a 256x44 window; fewer than 223
ticks means the kill is skipped. No labels or raw mouse input are required.

## Output contract

```json
{
  "demo": "match.dem",
  "map": "de_mirage",
  "model_version": "valveguard-gbdt-2026-10-18",
  "threshold": 0.7916,
  "prior": 0.2901,
  "k0": 5.0,
  "players": [
    {"steamid": "...", "name": "...", "kills": 18,
     "mean_score": 0.42, "max_kill_score": 0.81,
     "shrunk_score": 0.40, "flagged": false, "reliable": true}
  ]
}
```

- `shrunk_score` is the decision score; `flagged` is `shrunk_score >= threshold`.
- `reliable` is `kills >= 3`. Treat `flagged && !reliable` as insufficient
  evidence for action, not as a detection.
- Scores are risk scores, not calibrated probabilities (player ECE ~0.18).

## Metrics (frozen evaluation)

| Setting | Player AUC | Recall@FPR=1% | Notes |
|---|---|---|---|
| 5-fold OOF, champion | 0.9331 | 0.1735 | headline |
| dev 80%, 5-fold | 0.9294 | 0.1475 | development baseline |
| locked holdout, trained on dev | 0.9384 | 0.3091 | one-shot, CI 0.187-0.523 |
| dev 80% + shrinkage | 0.9316 | 0.2386 | player-scoring rule |
| professional demos | -- | 0/70 players flagged | 1,020 kills, 7 demos |

Rank-stratified Q4 FPR after replication: **3.37%** at the player-level 1%
cut (386 clean Q4 players), **2.59%** with shrinkage.

## Intended use

- Post-match review and triage on CS2 demos, with a human analyst.
- Batch scoring of demos; per-player risk ranking within a demo.

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
- The holdout was already used for the frozen evaluation; the shrinkage rule
  is dev-validated only.

## Retraining policy

- **Batch retraining, not incremental.** The full model retrains in 1-2 minutes
  on CPU; incremental GBDT updates risk catastrophic forgetting for no benefit.
- Retrain monthly or after a CS2 patch, on the same clean label policy.
- Before promotion: dev 5-fold metrics, pro-demo specificity (0 players
  flagged), and a score-distribution drift check against the previous model.
- Version the artifact; keep the previous version for rollback.

## Reproduction

```bash
# Build the artifact (all clean windows)
PYTHONPATH=src python -m valveguard.scripts.train_final \
  --features reports/baseline_clean/summary_features.parquet \
  --cache data/processed/cw256 \
  --out artifacts/valveguard_gbdt_v1.joblib

# Score one demo (~19 s for a 24-round map on CPU)
PYTHONPATH=src python -m valveguard.scripts.score_demo \
  --demo match.dem --model artifacts/valveguard_gbdt_v1.joblib \
  --out result.json
```

After `pip install -e .`, the console entry points are `valveguard-train` and
`valveguard-score`.
