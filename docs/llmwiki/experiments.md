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
- High-rank Q4 FPR 4.76% (release blocker).

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
