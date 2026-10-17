# ValveGuard — Detector v1 Report

Status: research checkpoint, 2026-10-17. Not deployment-ready.

## 1. Data

- CS2CD `Context_window_256`: 90,707 unaugmented kill windows, 794 matches.
- Labels: 18,423 cheater kills / 72,284 non-cheater kills.
- 278 matches contain both labels. 11,174 negative windows come from cheater-present matches and have ~55.6% label precision; they are excluded. Clean training/evaluation uses 79,533 windows.
- CS2CD player IDs are per-match pseudonyms (`Player_k`), so the split is match-disjoint; true person-disjoint evaluation is not verifiable on this dataset.
- Raw subset for mouse-view residual: 80 cheater + 80 clean matches (8.7 GB).

## 2. Method

- 256-tick windows, kill at row 223.
- Reconstructed world positions and angles from the 44 published features using the dataset's map bounds.
- Corrected the published pitch sign: CS2 pitch is opposite to `atan2(dz, horizontal)`. At the sample kill tick the origin-relative error drops from 6.38 deg to 0.24 deg.
- Aim-only feature set: signed angular dynamics, target-relative errors, distance, victim motion, motor-control metrics (SPARC, LDLJ, PSD slope, entropy, submovements), and weapon/kill outcomes only in ablations.
- Models:
  - GBDT (`HistGradientBoostingClassifier`) on handcrafted window summaries.
  - 0.59M-parameter Transformer on 43 per-tick channels (relative-only version uses 20 channels).
  - Player aggregation: mean, top-3, max, HMM, noisy-OR.
- Evaluation: 5-fold match-disjoint CV, player-level and kill-level AUC/AP, recall at 1% FPR, grouped bootstrap CI.

## 3. Headline results (5-fold OOF)

| Metric | Aim-only GBDT | Relative-only Transformer |
|---|---|---|
| kill AUC | 0.8497 | **0.8628** |
| kill AP | 0.6589 | **0.6746** |
| kill recall@FPR=1% | **0.1572** | 0.1307 |
| player AUC | **0.9331** | 0.9292 |
| player AP | **0.7682** | 0.7407 |
| player recall@FPR=1% | **0.1735** | 0.0911 |
| player FPR@TPR=95% | 0.2917 | **0.2730** |
| player ECE | 0.1819 | **0.1263** |

Bootstrap 95% CI (Transformer): player AUC [0.9150, 0.9423]; player recall@FPR=1% [0.0538, 0.2134].

Single-split results (player AUC ~0.95, recall ~0.44) were optimistic and are not the headline.

### Full 43-channel Transformer (exploratory)

The full-channel Transformer (weapon, outcome, motion, and absolute position included) was run after the aim-only/relative-only versions:

| Metric | Full 43-channel Transformer |
|---|---|
| player AUC | 0.9288 |
| player AP | 0.7882 |
| player recall@FPR=1% | **0.2350** |
| player FPR@TPR=95% | 0.3400 |
| player ECE | 0.1154 |
| kill AUC | 0.8774 |
| kill recall@FPR=1% | 0.2236 |

Paired match-grouped bootstrap vs the aim-only GBDT:
- ΔAUC **[-0.0125, 0.0039]**
- Δrecall@FPR=1% **[-0.0146, 0.1847]**

The point estimate for low-FPR recall is higher (+0.072), but the CI crosses zero and the model uses context/outcome channels that the metadata-only audit showed to be confounded. It is therefore recorded as an exploratory challenger, not the frozen champion. The clean champion remains the aim-only GBDT.

## 4. Aggregation and cascade

| Aggregator | Player AUC | Recall@FPR=1% | ECE |
|---|---|---|---|
| mean | **0.9367** | 0.1183 | 0.1095 |
| HMM | 0.9187 | **0.1518** | **0.0812** |
| top-3 | 0.9122 | 0.1058 | 0.1652 |
| noisy-OR | 0.8529 | 0.2008 | 0.5764 |
| max | 0.8972 | 0.1128 | 0.2320 |

Cross-fitted player threshold: mean recall 0.140 at FPR 1.22%; HMM 0.146 at FPR 1.24%. Per-fold recall ranges ~0.025–0.427. The low-FPR estimate is unstable.

## 5. Residual and subgroup audits

- Metadata-only model (map + weapon + kill outcome): kill AUC 0.8775, player AUC 0.8873. This is a strong dataset shortcut; headline numbers must not be attributed to aim alone.
- Shuffled-label audit: kill AUC 0.4944, player AUC 0.4499. No label leakage.
- Map holdout (aim-only GBDT): player AUC 0.846–0.956 across maps.
- Per-weapon FPR at the global 1% cutoff: semi-auto rifle 4.8%, grenade (n=56) 4.4%, auto rifle 0.12%.
- Per-weapon isotonic calibration reduces ECE (0.033 → 0.009) but collapses recall@FPR=1% (0.131 → 0.069). Calibration is not the low-FPR fix.
- Mouse-view residual on 16,357 windows from 160 matches:
  - base player AUC 0.8833, recall@FPR=1% 0.0249;
  - +4 tail residual features: 0.8817 / 0.0648;
  - +all residual features: 0.7740 / 0.1222;
  - residual only: 0.6784 / 0.0374.
  The full residual set trades AUC for low-FPR recall; the subset is small and CIs overlap.

## 6. Skill / rank strata

875 players from the 160-match subset (477 CS rank, 398 Premier):

| Rank quartile | n | positives | Player AUC | FPR at global 1% |
|---|---|---|---|---|
| Q1 low | 251 | 38 | 0.911 | 0.00% |
| Q2 | 191 | 45 | 0.895 | 0.68% |
| Q3 | 214 | 58 | 0.828 | 0.64% |
| Q4 high | 219 | 177 | 0.712 | **4.76%** |

Failed post-hoc fixes:

- Rank-conditional residual: AUC 0.846 vs raw 0.882; recall@FPR=1% 0.006 vs 0.016.
- Cross-fitted rank-quartile thresholds: Q4 FPR stays 4.76%; Q4 recall 0.
- Rank as a feature: overall recall rises to 0.229, but Q4 FPR rises to 14.3% — a shortcut.
- Upweighting Q4 clean negatives ×3/×5: no meaningful improvement.

Conclusion: high-rank false positives are a ranking/data problem, not a calibration or threshold problem.

## 7. Figures

- `figures/fig01_player_roc.png` — player ROC (GBDT / Transformer / ensemble).
- `figures/fig02_risk_coverage.png` — recall vs FPR in the low-FPR region.
- `figures/fig03_calibration.png` — calibration deciles and score distribution.
- `figures/fig04_subgroup_fpr.png` — per-weapon FPR before/after calibration.
- `figures/fig05_fold_variability.png` — fold-to-fold AUC/recall variability.
- `figures/fig06_residual_compare.png` — mouse-residual feature comparison.
- `figures/fig07_map_holdout.png` — held-out-map player AUC.
- `figures/fig08_trajectory_deviation.png` — human error/speed bands vs sample kills.
- `figures/fig09_feature_percentiles.png` — sample cheater kill feature percentiles.
- `figures/fig10_rank_strata.png` — AUC and FPR by rank quartile.

## 8. Limitations

- 64 Hz view angles, not 1000 Hz raw mouse input.
- Match-disjoint only; per-match pseudonyms.
- No target visibility/raycast; FOV/distance/smoke are proxies.
- Labels: VAC-based positives have low recall; cdata negatives excluded.
- Small high-rank negative sample (42 clean players in Q4); CI is wide.
- No true cross-dataset transfer yet.
- Not deployment-ready: high-rank FPR and low-FPR recall are blockers.

## 9. Recommended next steps

1. Freeze detector v1 and use this report as the honest baseline.
2. If performance must improve, the credible levers are:
   - high-skill human negatives (pro demos, no cheat labels needed),
   - 1024-tick context,
   - rank-aware training that does not use rank as a shortcut (group DRO / adversarial rank removal),
   - more labeled cheater matches.
3. Do not add more post-hoc calibration, rank thresholds, or residual features; the audits show they trade away ranking.

## 10. Reproduce

```bash
# cache + audit + GBDT
python -m valveguard.scripts.build_cache --tar data/raw/context_windows_256.tar --out data/processed/cw256
python -m valveguard.scripts.audit --cache data/processed/cw256 --out reports/audit.json
python -m valveguard.scripts.run_baseline --cache data/processed/cw256 --out reports/baseline_clean --jobs 14

# 5-fold GBDT and Transformer
python -m valveguard.scripts.run_cv_gbdt --cache data/processed/cw256 --features reports/baseline_clean/summary_features.parquet --out reports/cv_gbdt
python -m valveguard.scripts.build_channels --cache data/processed/cw256 --out data/processed/cw256_channels --jobs 14
python -m valveguard.scripts.run_cv_transformer --cache data/processed/cw256 --channels data/processed/cw256_channels/channels.npy --out reports/cv_transformer --folds 5

# cascade / residual / skill analyses
python -m valveguard.scripts.analyze_cascade_residuals --gbdt reports/cv_gbdt/cv_predictions.parquet --transformer reports/cv_transformer/cv_predictions.parquet --features reports/baseline_clean/summary_features.parquet --cache data/processed/cw256 --out reports/cascade
python -m valveguard.scripts.analyze_skill --scores reports/mouse_residual_eval2/base_player_scores.parquet --raw data/raw/cs2cd_subset --out reports/skill
python -m valveguard.scripts.make_figures
```
