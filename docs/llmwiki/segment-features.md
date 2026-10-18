# Movement / Pre-Aim Segment Features

Status: exploratory, dev-80 only, 2026-10-18. Not promoted.

## Question

The champion uses only view-derived aim features. Does adding movement and
pre-aim context — travel, silent approach, angle clearing, duel — improve
player-level detection at the low-FPR operating point?

## Segmentation (rule-based, pilot)

Per tick in the 223 pre-kill ticks:

| State | Rule |
|---|---|
| duel | last 64 ticks before the kill, or target error < 5 deg |
| travel ("赶路") | not duel, speed > 200 units/s |
| walk / silent approach ("静步") | not duel, 20 < speed <= 200 |
| stop | not duel, speed <= 20 |

The cache has no `is_walking` flag, so silence is approximated by the speed
band; raw demos can supply the real flag later.

## Features (24)

- state fractions: travel, walk, stop, duel;
- silent-near fraction (walk and victim distance < 600);
- travel-far fraction (travel and distance > 800);
- mean speed by travel/walk/duel, max speed and its timing;
- pre-aim phase (1.0-2.0 s before the kill): error mean/min/std, yaw error,
  distance start/end/slope, fraction of ticks with error < 5 deg;
- state transition counts, fast-to-slow transitions;
- duel error/speed, error at the duel start;
- approach time from error < 10 deg to < 2 deg.

## Results (dev 80, 5-fold, paired vs the 59-feature champion)

| Feature set | Player AUC | Recall@FPR=1% | Kill AUC | Q4 FPR (mean) | Q4 FPR (shrunk) |
|---|---|---|---|---|---|
| base (59) | 0.9294 | **0.1475** | 0.8472 | 2.63% | 5.26% |
| base + segments (83) | **0.9311** | 0.1337 | **0.8677** | 5.26% | 10.53% |
| segments only (24) | 0.9041 | 0.1050 | 0.8097 | -- | -- |

Paired base+segments vs base:

- ΔAUC **[-0.0015, +0.0054]** (crosses zero);
- Δrecall@FPR=1% **[-0.0663, +0.0526]** (crosses zero, point estimate lower).

With shrinkage (k0=5): base AUC 0.9316 / recall 0.2386; base+segments
AUC 0.9334 / recall 0.2079.

## Interpretation

- Segments alone carry real signal (AUC 0.904 from 24 features), so movement
  and pre-aim context is not noise.
- Added to the champion, they raise kill-level AUC by ~0.02 but do **not**
  improve player-level low-FPR recall, and they double Q4 FPR.
- Decision: **not promoted**. The finite gain is at the kill level, which is
  useful for ranking kills in a review UI, not for flagging players.

## Caveats

- Coarse rules: no real `is_walking`, no visibility, duel is a time/error
  proxy.
- One dev-only experiment; the holdout is spent.
- A more faithful silent/travel state needs raw `is_walking` (160-match
  subset or new demos) and map visibility; revisit only if a concrete
  hypothesis justifies it.

## Reproduce

```bash
PYTHONPATH=src python -m valveguard.scripts.build_segment_features \
  --cache data/processed/cw256 --features-json reports/audit.json \
  --out reports/segments --jobs 12

# merge with the base summary, then run the same dev-80 folds
python -m valveguard.scripts.run_cv_gbdt \
  --cache data/processed/cw256 \
  --features reports/segments/base_plus_segments.parquet \
  --out reports/cv_seg_plus \
  --holdout reports/holdout/holdout_matches.parquet \
  --folds-from reports/cv_gbdt/cv_predictions.parquet
```
