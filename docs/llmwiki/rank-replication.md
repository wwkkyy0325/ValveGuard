# Rank-Stratified FPR Replication

Status: completed 2026-10-18 on the development 80% only. No holdout use.

## Question

The rank-stratified analysis used the 160-match raw subset (42-38 clean Q4
players) and produced Q4 false-positive rates from 4.76% (old threshold) to
19.05% (corrected threshold). Is that effect stable on a larger sample, and
what is its real magnitude?

## Data

- Ranks come from the per-match JSON `rank_update` events (last `rank_new` per
  player), not from the 48.9 GB raw parquet. 635 JSON files were downloaded
  for all dev-80 matches (~1.3 GB).
- 4,761 ranked player rows across 616 matches; after merging with the cached
  windows, **3,429 of 4,678 players (73.3%)** have a rank.
- Trajectories and scores are the existing dev-80 5-fold OOF predictions.

## Method

- Rank type from `CSstats_info.match_making_type` (Premier vs CS ordinal);
  percentile within type; quartiles over all ranked players.
- Operating point: player-level 1% FPR on **all** dev-80 human players
  (threshold 0.8217).
- Shrinkage: `(sum + 5 * human prior) / (n + 5)`, prior from the other folds.

## Results

| Quartile | Players | Clean | AUC (mean) | FPR (mean) | AUC (shrunk) | FPR (shrunk) |
|---|---|---|---|---|---|---|
| Q1 low | 858 | 774 | 0.926 | 0.26% | 0.930 | 0.52% |
| Q2 | 857 | 737 | 0.878 | 1.22% | 0.875 | 1.22% |
| Q3 | 865 | 664 | 0.927 | 1.05% | 0.928 | 1.36% |
| Q4 high | 849 | 386 | 0.864 | **3.37%** | 0.879 | **2.59%** |

Cheater share rises sharply with rank: Q1 9.8%, Q2 14.0%, Q3 23.2%,
Q4 54.5%.

### Sample-size comparison

| Analysis | Quartile basis | Clean Q4 | Q4 FPR |
|---|---|---|---|
| 160-match subset, parquet rank, subset threshold | within subset | 42 | 7.14% |
| 160-match subset, parquet rank, all-dev threshold | within subset | 38 | 13.16% |
| full OOF, skill table, all-OOF threshold | within subset | 42 | 19.05% |
| **all dev80, JSON rank, all-dev threshold** | full dev80 | **386** | **3.37%** |
| same + shrinkage | full dev80 | 386 | **2.59%** |

The large-sample estimate is **3.37%** (2.59% shrunk). The earlier 19.05%
estimate is a small-sample number: with 42 clean players and a true rate of
3.4%, the expected count is 1.4 false positives; observing 8 is a rare but
possible draw (binomial p < 1%). The 160-match subset is also harder than the
full dev set even under global quartiles (11.1% at the same threshold), so it
is not representative.

## Interpretation

- The Q4 effect is **real but modest**: 3.37% FPR at a 1% target (3.4x).
  It is not the 19% blocker the subset suggested.
- Shrinkage still helps Q4 ranking (AUC 0.864 -> 0.879) and reduces Q4 FPR to
  2.59%.
- The dominant earlier problem was estimation variance from a small clean-Q4
  sample, not a difference in rank definitions (parquet vs JSON ranks give
  similar subset results).

## Caveats

- JSON `rank_new` is close to, but not identical with, the parquet `rank`
  median used in the original 160-match analysis.
- 73% rank coverage; unranked players are excluded from the quartile table but
  included in the threshold population.
- Dev 80% only; the holdout is not re-used.

## Reproduce

```bash
scripts/download_cs2cd_json.py data/raw/cs2cd_json_list.txt data/raw/cs2cd_full_json
PYTHONPATH=src python -m valveguard.scripts.rank_replication \
  --json-dir data/raw/cs2cd_full_json \
  --pred reports/cv_gbdt_dev80/cv_predictions.parquet \
  --out reports/rank_replication
```
