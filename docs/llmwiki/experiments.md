# ValveGuard — Experiment Protocol and Queue

Status: active (2026-10-17)

## 1. Champion / challenger protocol

- **Fixed folds**: the 5 match-disjoint folds in `reports/cv_gbdt/cv_predictions.parquet`; every variant uses the same folds.
- **Primary metric**: player-level recall at FPR = 1% (with actual FPR reported).
- **Secondary**: player AUC, AP, kill AUC, calibration, high-rank FPR.
- **Decision rule**: a challenger replaces the champion only if
  1. player recall@FPR=1% improves, and
  2. player AUC does not materially degrade (paired match-grouped bootstrap; no test-set tuning), or
  3. AUC improves with the same low-FPR recall.
  If the two are in tension, prefer higher low-FPR recall for the anti-cheat use case and record the trade-off.
- **Complexity penalty**: if metrics are tied, keep the simpler model.
- **No post-hoc calibration/threshold tricks** that improve ECE but hurt ranking; they are recorded as negative results.
- Every run writes OOF predictions, metrics, CI, and a one-line decision to `experiments/registry.csv`.

Current champion: **aim-only GBDT**, 5-fold OOF:
- player AUC 0.9331, player recall@FPR1 0.1735, kill AUC 0.8497.
- High-rank Q4 FPR **19.05%** at the player-level 1% cut (corrected
  2026-10-18; the older 4.76% applied a kill-level threshold to player means).
  Player-score shrinkage reduces it to 4.76% (section 12).

## 2. Verified ablations (existing)

| Ablation | Result | Decision |
|---|---|---|
| weapon removal | player AUC 0.9443 (single split) | weapon is not the signal; excluded from base |
| outcome removal | 0.9370 | outcome flags help but are confounded; excluded from base |
| motion removal | 0.9359 | modest; excluded from base for a cleaner claim |
| signed vs abs deltas | signed features used | keep |
| pitch-sign correction | kill-tick error 6.38 -> 0.24 deg | keep |
| metadata-only | player AUC 0.8873 | dataset shortcut; never use as headline |
| shuffled labels | 0.4944 | no label leakage |
| map holdout | player AUC 0.846–0.956 | report as generalization |
| mouse–view residual | full set trades AUC for recall; tail set marginal | not adopted |
| HMM / top-k / noisy-OR aggregation | HMM low-FPR gain at AUC cost; noisy-OR over-triggers | mean stays champion |

## 3. Experiment queue

### A. Model family and hyperplane

- E01 logistic regression (L2, balanced) on standardized features — **done: player AUC 0.8935, recall 0.0949; worse.**
- E02 ExtraTrees — **done: 0.9274 / 0.1315; worse.**
- E03 GBDT hyperparameters (leaves 15/63, learning rate, L2) — running.
- E04 linear SVM / calibrated linear SVM (margin analysis).
- E05 one-class SVM / autoencoder trained on humans only.
- E06 RBF kernel approximation (Nystroem) + logistic — running.
- E07 polynomial interactions on top-k features — running.

### B. Dimensionality: low -> high and high -> low

- E08 PCA scree / intrinsic dimension of the human feature cloud; participation ratio.
- E09 PCA-whitened Mahalanobis distance to the human cohort.
- E10 supervised dimensionality reduction (LDA / PLS) and reconstruction error.
- E11 random Fourier features / spline bases; compare to exact RBF on a subsample.
- E12 class-conditional subspaces: angle between human/cheater PCA subspaces; principal angles.

### C. Tail and decision theory

- E13 extreme-value tail: fit a generalized Pareto distribution to human player scores; convert scores to tail p-values; evaluate low-FPR recall and calibration.
- E14 split/cross-conformal prediction at the player level for finite-sample FPR control.
- E15 Bayesian hierarchical aggregation: beta-binomial / empirical-Bayes player posterior; compare with mean/HMM.
- E16 cost-sensitive threshold optimization with a false-positive cost model.

### D. Motor-control / trajectory likelihood

- E17 Fitts / minimum-jerk / OU process fit per window; likelihood-ratio features.
- E18 spectral and wavelet features; DFA; multiscale entropy.
- E19 target-motion coupling: lead-lag, Kalman residual, predictive-lead features.

### E. Architecture

- E20 Transformer variants: d_model 64/128/192, layers 2/4/6, heads 2/4/8, CLS vs mean/attention pooling, dropout, pre/post-LN.
- E21 multi-scale: 64-tick micro crop, 1024-tick macro; hierarchical fusion.
- E22 hybrid: GBDT summary features concatenated to the Transformer CLS.
- E23 two-stream attacker/victim cross-attention.
- E24 multi-task auxiliary heads (time-to-target, overshoot, smoothness).

### F. Skill and distribution shift

- E25 high-rank human negatives (pro/faceit demos) in training.
- E26 group DRO / worst-group loss across rank quartiles.
- E27 rank-adversarial representation (remove rank-predictive directions) without using rank as a shortcut.
- E28 map/adversarial domain generalization.

## 4. Mathematical analysis to run alongside

- **Bayes error estimate** in feature space (human vs cheater k-NN overlap) to see whether the low-FPR gap is data-limited.
- **Intrinsic dimension** (PCA scree, participation ratio) to decide whether linear/hyperplane models are sufficient.
- **Principal angles** between human/cheater subspaces; if they overlap heavily, nonlinear/tail models are needed.
- **Margin distribution** of linear SVM; fraction of players within the margin; expected FPR at a target recall.
- **EVT tail index** of human scores; if the tail is heavy, Gaussian calibration is invalid.
- **Conformal guarantee**: finite-sample FPR control under exchangeability; report the empirical coverage.
- **Mutual information / distance correlation** between rank and score; if strong, rank-aware training is mandatory.

## 5. Registry

Results are written to `experiments/<batch>/results.csv` and OOF predictions to
`experiments/<batch>/oof/<variant>.parquet`. A challenger is promoted only by
editing the champion block in this file and recording the decision in
`changelog.md`.

## 6. Batch 1 status

Running: 5 feature sets × {GBDT, logistic, PCA-logistic, Nystroem-logistic, ExtraTrees}, plus two GBDT hyperparameter variants and a top-20 polynomial logistic variant.

## 7. Pause and Monday continuation (2026-10-17)

Experiments are paused until Monday 2026-10-20. The repository is frozen at
the detector-v1 baseline plus completed audits/figures.

Frozen champion: aim-only GBDT, 5-fold OOF player AUC 0.9331, recall@FPR=1%
0.1735. No batch-1 or motor-feature challenger passed the paired-bootstrap
decision rule.

Monday protocol (see `next-session.md` for commands and timebox):

1. Lock a 20% match holdout; never use it for selection.
2. Re-run the GBDT baseline on the development 80% with nested CV.
3. Test only three pre-registered hypotheses: H1 43-channel Transformer,
   H2 1024-tick context, H3 high-rank human negatives / group DRO.
4. Holm-correct the three tests; promote only if Δrecall@FPR=1% CI lower bound
   > 0 and ΔAUC is non-inferior.
5. Evaluate the final candidate once on the locked holdout, write the report,
   and stop. Target 3–4 hours; do not exceed 5.

Hardware: RTX 5070 Ti 16 GB. The larger VRAM permits 1024-token Transformer
training, but the search space stays limited to the three hypotheses.

## 8. Full 43-channel Transformer result

5-fold match-disjoint: player AUC 0.9288, AP 0.7882, recall@FPR=1% 0.2350,
kill AUC 0.8774. Paired vs aim-only GBDT: ΔAUC [-0.0125, 0.0039], Δrecall
[-0.0146, 0.1847]. Not promoted under the decision rule. It is the leading
H1 candidate for Monday, but it must be re-run on the locked holdout and
with a fair channel set.

## 9. v2 dev-80 results (2026-10-17, RTX 3060)

Holdout: 158 matches / 15,623 windows locked with seed 20261020; dev 635
matches / 63,910 windows. Post-hoc lock, documented in
`reports/holdout/holdout_summary.json`.

| Run | Player AUC | Recall@FPR1 | Kill AUC | Decision |
|---|---|---|---|---|
| GBDT on dev 80% (5-fold) | 0.9294 | 0.1475 | 0.8472 | reference |
| H1 43-channel Transformer (dev 80%) | 0.9251 | 0.2297 | 0.8704 | rejected: ΔAUC [-0.0153, 0.0057], Δrecall [-0.0071, 0.1809] |
| H2 near (subset) | 0.8822 | 0.0374 | 0.8241 | reference |
| H2 near+macro (1024) | 0.8830 | 0.0349 | 0.8290 | rejected: ΔAUC [-0.0055, 0.0078] |
| H2 macro only | 0.8240 | 0.0050 | 0.7176 | rejected: ΔAUC [-0.084, -0.031] |
| H3 baseline (rank subset) | 0.8771 | 0.0374 | 0.8155 | reference |
| H3 DRO reweighting | 0.8800 | 0.0299 | 0.8184 | rejected: Q4 FPR unchanged |
| Holdout, GBDT trained on dev | 0.9384 | 0.3091 | 0.8474 | final estimate |

- H1 (43-channel Transformer on dev 80%) did not clear the rule: the recall
  point estimate is +0.082, but the CI crosses zero and AUC is slightly lower.
- No hypothesis passed; the aim-only GBDT remains champion.
- H2 and H3 are pilots on the 160-match rank/raw subset; their absolute metrics
  are not comparable to the full-dataset numbers, only within-row.
- H2 reconstruction sanity check: rebuilt near features vs published cached
  features on the same subset, ΔAUC [-0.0067, 0.0040].

## 10. Within-player self-baseline (exploratory, dev-80 only)

| Set | Player AUC | Recall@FPR1 | Kill AUC | Decision |
|---|---|---|---|---|
| raw | 0.9383 | 0.1851 | 0.8475 | reference |
| raw+selfz | 0.9269 | 0.2160 | 0.9210 | exploratory |
| raw+selfz+count | 0.9276 | 0.2447 | 0.9224 | shortcut, not aim |
| selfz only | 0.9028 | 0.1021 | 0.8456 | worse |

- Paired raw+selfz vs raw: ΔAUC [-0.0162, -0.0065], Δrecall [-0.0394, +0.0963].
- The kill-level signal is real (shuffled-label audit 0.50; unchanged after
  dropping <3-kill players) but does not reach player-level recall; max/top-k
  aggregation fails too.
- Q4 FPR unchanged at 3.85% (2/52 clean); Q4 AUC 0.726 -> 0.764.
- Not promoted. No holdout reuse; a future confirmation needs a new split or
  external data.

## 11. Professional demos (2026-10-18)

Data: 7 HLTV pro demos (`blanchon/opencs2_dataset_demo`), 1,075 kill windows,
70 player-maps. Treated as clean human negatives.

| Operating point | Pro kill FPR | Pro player FPR |
|---|---|---|
| 0.1% | 0/1,075 | 0/70 |
| 1% | 0.28% (3/1,075) | 0/70 |
| 5% | 2.70% | 0/70 |

- Pro mean score 0.258 vs CS2CD humans 0.293 and cheaters 0.705.
- Pro-negative augmentation (dev-80 5-fold, pro only in training):
  ΔAUC [-0.0019, +0.0014], Δrecall [-0.0098, +0.0480], Q4 FPR unchanged.
- Conclusion: elite human aim is not confused with aim assist; adding pro
  negatives does not move the detector. The Q4 FPR is not a generic
  high-skill effect.

## 12. Player-score shrinkage and the Q4 threshold fix (2026-10-18)

- **Correction**: the earlier Q4 FPR applied a kill-level threshold to player
  means. At a player-level 1% threshold the champion's Q4 FPR is 19.05%
  (8/42, full OOF) or 13.16% (5/38, dev80), not 4.76%.
- 5 of the 8 false positives had one or two kills; Q4 clean players have a
  median of 9.5 kills but a 25th percentile of 2.
- Empirical-Bayes shrinkage `(sum + k0*prior)/(n+k0)`, prior = training-fold
  human mean, k0=5:
  - full OOF: AUC 0.9339, recall@FPR=1% 0.2288, Q4 FPR 4.76%, Q4 AUC 0.797.
  - dev80: AUC 0.9316, recall 0.2386, Q4 FPR 5.26%, Q4 AUC 0.791.
  - paired dev80: ΔAUC [-0.0015, +0.0058], Δrecall [+0.0068, +0.1404].
- The k0 grid [1..20] is a plateau (AUC 0.928-0.936, recall 0.21-0.25,
  Q4 FPR 4.8-7.9%), so the result is not knife-edge on k0.
- Status: adopted as the player-level scoring rule for reporting; dev-validated
  only. The holdout was already spent and is not re-used.

## 13. Rank-stratified FPR replication (all dev 80, JSON ranks)

- Ranks from per-match JSON `rank_update` (635 files); 3,429/4,678 players
  (73.3%) ranked. Same OOF scores and player-level 1% threshold (0.8217).
- Q4 (386 clean players): FPR **3.37%**, AUC 0.864; shrunk k0=5: FPR
  **2.59%**, AUC 0.879.
- Q1/Q2/Q3 FPR: 0.26% / 1.22% / 1.05% (shrunk 0.52% / 1.22% / 1.36%).
- The earlier 19.05% and 13.16% were small-sample estimates on 42/38 clean Q4
  players; the 160-match subset is also harder than the full dev set (11.1% at
  the same threshold).
- Conclusion: the Q4 effect is real but modest (~3.4x the 1% target); the
  19% figure is superseded. Details: `rank-replication.md`.

## 14. Movement / pre-aim segment features (dev-80 pilot)

- 24 rule-based features: travel (speed > 200), walk/silent (20-200), stop,
  duel (last 64 ticks or error < 5), pre-aim error/distance, transitions.
- base + segments (83): player AUC 0.9311, recall@1% 0.1337, kill AUC 0.8677;
  Q4 FPR 5.26% mean / 10.53% shrunk.
- segments only (24): player AUC 0.9041, recall 0.1050, kill AUC 0.8097.
- Paired base+seg vs base: ΔAUC [-0.0015, +0.0054], Δrecall [-0.0663, +0.0526].
- Decision: **not promoted**. Kill-level AUC rises ~0.02 but player-level
  recall does not improve and Q4 FPR doubles. Details: `segment-features.md`.
