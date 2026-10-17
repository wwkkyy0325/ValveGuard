# Label-Noise Audit

Status: completed 2026-10-18 on the development 80% only. The locked holdout
was not used.

## Question

Two label questions remained open:

1. The 11,174 `cdata` label-0 windows (non-cheater players inside cheater
   matches) were excluded because the dataset card estimates ~55.6% label
   precision. Do those windows actually look cheater-like?
2. After player-score shrinkage, two Q4 clean players remain above the
   player-level 1% threshold. Are they likely undetected cheaters (label
   noise) or unusual human styles?

## Data and groups (dev 80%)

| Group | Definition | Windows | Players |
|---|---|---|---|
| clean | `ncdata`, label 0 (no cheater in the match) | 49,164 | 3,668 |
| uncertain | `cdata`, label 0 (non-cheater in a cheater match) | 9,137 | 1,171 |
| cheater | `cdata`, label 1 | 14,746 | 1,010 |

## Method

- Out-of-match scores: for fold k, fit the champion on the clean rows of the
  other four folds and score every window in fold k, including the uncertain
  rows that were never trained on.
- Thresholds: kill-level and player-level 1% FPR on the clean group.
- Three estimates of the uncertain group's cheater fraction:
  - threshold excess (fraction above the clean 1% cut),
  - first-moment decomposition
    `pi = (mean_unc - mean_clean) / (mean_cheat - mean_clean)`,
  - KDE likelihood ratio with equal priors, and a nearest-neighbour vote in
    standardised feature space.

## Results

| Group | Kill mean | Kill median | Kill p95 | Kills above clean 1% | Player mean | Players above clean 1% |
|---|---|---|---|---|---|---|
| clean | 0.304 | 0.241 | 0.751 | 1.00% | 0.309 | 1.01% |
| uncertain | 0.323 | 0.274 | 0.734 | **0.34%** | 0.341 | **0.68%** |
| cheater | 0.657 | 0.720 | 0.952 | 15.50% | 0.655 | 14.75% |

Estimates of the uncertain cheater fraction:

| Method | Estimate |
|---|---|
| First-moment decomposition | **5.5%** |
| KDE likelihood ratio | 33.6% (unstable; see below) |
| Nearest neighbour closer to cheater | 27.2% |

The uncertain group's mean score is 0.019 above clean but its high-score tail
is *below* clean (0.34% vs 1.00% above the 1% cut). Median nearest-neighbour
distance is 2.49 to humans and 2.73 to cheaters.

The KDE estimate is inconsistent with the mean decomposition and the tail:
a 33.6% admixture of the cheater score distribution would raise the mean to
about 0.42, not 0.323. It is driven by the low-score overlap where the
cheater density is broad, so it is reported as an unstable shape-based
cross-check, not as the headline.

## Interpretation

- At the window level, the excluded `cdata` label-0 windows look human, not
  cheater-like. There is no evidence of a large admixture of *detectable*
  cheaters.
- The dataset card's 55.6% precision is not corroborated by behaviour on these
  windows. It may refer to player-level label assignment (which player in the
  match is the cheater) rather than to the behaviour of the other players.
- The model's cheater recall is only 15.5% of cheater windows at this
  operating point, so absence of evidence is weak evidence of absence: a
  mislabeled cheater whose aim is not detected would look like a human here.
- Practical consequence: the excluded windows are probably safe to reuse as
  negatives, but the gain would be small and they are not needed for the
  current claim.

## Remaining Q4 false positives after shrinkage

| Player | Kills | Shrunk score | Mean kill score | Max kill | Closer to cheater | Mean straightness | Mean final err | Mean speed |
|---|---|---|---|---|---|---|---|---|
| ncdata:66 Player_1 | 18 | 0.687 | 0.795 | 0.971 | 66.7% | 0.784 | 5.78 | 21.7 |
| ncdata:79 Player_6 | 9 | 0.692 | 0.910 | 0.964 | 77.8% | 0.945 | 2.06 | 8.3 |

Both players have a majority of kills whose nearest neighbour is a cheater
window. Player_6 is extreme: very straight (0.945 vs human median 0.61), very
accurate (2.06 deg), slow while aiming (8.3 vs 18.0). This is consistent with
either a very disciplined low-sensitivity style or an undetected smooth-aim
cheat. The 64 Hz view data cannot separate the two; raw mouse input would be
needed.

## Caveats

- Single-score model; the NN vote is not a calibrated probability.
- The uncertain group's players may be systematically weaker (the cheater
  takes the kills), which would bias scores downward and hide an admixture.
- Dev 80% only; no holdout claim.

## Reproduce

```bash
PYTHONPATH=src python -m valveguard.scripts.analyze_label_noise \
  --features reports/baseline_clean/summary_features.parquet \
  --cache data/processed/cw256 \
  --holdout reports/holdout/holdout_matches.parquet \
  --folds reports/cv_gbdt/cv_predictions.parquet \
  --out reports/label_noise

PYTHONPATH=src python -m valveguard.scripts.audit_q4_fps \
  --scored reports/label_noise/scored_windows.parquet \
  --pred reports/cv_gbdt_dev80/cv_predictions.parquet \
  --skill reports/skill/skill_players.parquet \
  --features reports/baseline_clean/summary_features.parquet \
  --out reports/q4_fp_audit
```
