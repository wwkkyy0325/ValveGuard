# Changelog

## 2026-10-17

- Initialized the project knowledge base.
- Researched prior work and datasets: GAN-Aimbots (generation + detection, ViZDoom), XGuardian (CS2 pitch/yaw detection, USENIX Security '26), AntiCheatPT + CS2CD (795 CS2 matches, 48.9 GB, CC-BY-4.0), Same-Player Verification, HAWK, Kaggle CS:GO cheating dataset, RekaCS2-10k, OpenCS2, PureSkill.gg.
- Drafted `design.md`: two-model architecture (discriminative detector + conditional generator), shared preprocessing, skill-aware one-sided suspicion scoring, evaluation protocol, chart deliverables, risks, open decisions.
- Inspected raw CS2CD samples: confirmed per-tick `usercmd_mouse_dx/dy`, `usercmd_input_history`, `rank`, `cheaters` labels, and `player_death`/`weapon_fire` events. Mouse–view robust regression reaches R^2 0.75–0.89 on normal players in the sample; a labeled cheater shows a lower fit, motivating a physical-consistency feature.
- Added `model-design.md`: multi-scale 64/256/1024/match window pyramid, hierarchical bottom-up/top-down detector, HMM aggregation, smoothness/motor-control metrics, attacker–victim cross-attention, skill-conditioned likelihood ratio, strict person-disjoint protocol, leakage audits, reproducibility and compute plan.
- Rewrote the phased task plan to match the raw-CS2CD pipeline and the stricter evaluation protocol.


## 2026-10-17 (implementation start)

- Created project `.venv` with torch 2.6.0+cu124; CUDA is available on the RTX 3060 Laptop 6 GB.
- Selected `CS2CD/Context_window_256` (9.4 GB tar, 256 x 44 per kill) for the detector-first phase; no raw `.dem` or 48.9 GB dataset download.
- Verified one window maps to the raw match: `cdata-cheater-file_0-Player_3-kill_0` -> row 223 is the kill tick; recovered pitch/yaw/distance match the raw parquet exactly.
- Found and documented a pitch-sign bug in the published geometry: CS2 pitch is opposite to `atan2(dz, horizontal)`. Inverting it reduces the sample kill-tick target error from 6.38 deg to 0.24 deg (mean absolute error over the window from 6.64 deg to 2.59 deg).
- Implemented: tar index/cache builder (float16 memmap), match-disjoint split, geometry reconstruction, motor-control feature extraction, GBDT baseline, audit script, and tests.
- Tests: 4 pass (geometry, angle wrap, summary/channels, split disjointness).
- Pilot cache built from a truncated tar (15 windows); full cache awaits the 9.4 GB download.

## 2026-10-17 (detector v1 results)

- Full cache: 90,707 unaugmented windows; 794 matches; labels 18,423 cheater / 72,284 non-cheater.
- Label audit: 278 matches mix cheater/non-cheater windows; 11,174 negatives come from `cdata` (cheater-present) matches with ~55.6% label precision. The clean run excludes them, keeping 79,533 windows.
- GBDT (clean labels, match-disjoint): test kill AUC 0.9006, AP 0.7596; player-level mean AUC 0.9430; recall@FPR=1% 0.2626 kill / 0.3967 player; FPR@TPR=95% 0.4963 kill / 0.1864 player; ECE 0.138.
- Feature ablations: no weapon 0.9443 player AUC; no outcome flags 0.9370; no motion 0.9359. Aim features carry the signal; weapon is not the shortcut.
- Transformer (43 channels, d=128, 4 layers, 0.59M params, best epoch 9): test kill AUC 0.8970, AP 0.7639; player AUC 0.9406; recall@FPR=1% 0.2769 kill / 0.4293 player; ECE 0.132.
- Calibrated probability ensemble (GBDT + Transformer): test kill AUC 0.9114, AP 0.7899; player AUC 0.9497; recall@FPR=1% 0.4457 player; FPR@TPR=95% 0.2009 player. The transductive rank ensemble reaches 0.9518 AUC but is diagnostic only.
- Low-FPR kill-level detection remains the weak point; player-level aggregation over kills is the stronger deployment unit.
- Caveats: single split/seed, no subgroup-by-rank analysis yet, no aim-only Transformer yet, test numbers used for model comparison; next runs must select on validation only.

## 2026-10-17 (ablations and leakage audit)

- Aim-only Transformer (26 channels; dropped weapon, kill-outcome, attacker-motion): test player AUC 0.9469, kill AUC 0.8909, AP 0.7441, recall@FPR=1% 0.2777, ECE 0.1120. It matches/beats the full model at player level and is the more defensible primary model.
- Metadata-only GBDT (map one-hot + weapon group + kill-outcome flags only): test kill AUC 0.8775, player AUC 0.8873, recall@FPR=1% 0.2788. This is a strong dataset shortcut/confound; the full model's headline number must not be attributed to aim alone.
- Shuffled-label audit: test kill AUC 0.4944, player AUC 0.4499; no label leakage.
- Next: relative-only Transformer (also drops absolute positions) to test whether aim dynamics alone survive; then map-matched/map-holdout evaluation.

## 2026-10-17 (5-fold CV and honest estimate)

- Relative-only Transformer (20 channels; no absolute positions, weapon, outcome, or attacker motion): test player AUC 0.9480, kill AUC 0.8872, AP 0.7478, recall@FPR=1% 0.2614, ECE 0.0999. Aim dynamics alone remain predictive at player level.
- Calibrated ensemble of GBDT + relative-only Transformer: test player AUC 0.9528, AP 0.8632, recall@FPR=1% 0.4402; kill AUC 0.9124, AP 0.7957. Transductive rank ensemble reaches 0.9534 (diagnostic only).
- Match-grouped bootstrap on the single split: player AUC 0.9528 [0.9254, 0.9747]; recall@FPR=1% 0.4402 [0.2419, 0.6605]. The low-FPR recall interval is wide (119 test matches, 184 positive players).
- 5-fold match-disjoint CV, aim-only GBDT (59 features): OOF kill AUC 0.8497, AP 0.6589, recall@FPR=1% 0.1572; OOF player AUC 0.9331, AP 0.7682, recall@FPR=1% 0.1735, CI [0.1046, 0.2656]. The single-split numbers were optimistic; the CV estimate is the honest primary result.
- Metadata-only, shuffled-label, per-map, and map-holdout audits are recorded in `reports/`. Map-holdout aim-only GBDT player AUC ranges from 0.846 (de_anubis) to 0.956 (de_dust2); map-specific variance is material.
- Next: 5-fold CV for the relative-only Transformer, calibration/risk-coverage, and charts. Do not report the single-split 0.95 as the headline.

## 2026-10-17 (5-fold Transformer CV complete)

- 5-fold match-disjoint CV, relative-only Transformer (20 channels): OOF kill AUC 0.8628, AP 0.6746, recall@FPR=1% 0.1307, ECE 0.1110; OOF player AUC 0.9292, AP 0.7407, recall@FPR=1% 0.0911, FPR@TPR=95% 0.2730, ECE 0.1263.
- Match-grouped bootstrap: player AUC 95% CI [0.9150, 0.9423]; recall@FPR=1% 95% CI [0.0538, 0.2134].
- Comparison with aim-only GBDT 5-fold CV: player AUC 0.9331 vs 0.9292, player recall@FPR=1% 0.1735 vs 0.0911. The Transformer has higher kill-level AUC but worse player-level low-FPR recall.
- Conclusion: the single-split 0.95 player AUC / 0.44 recall was optimistic. The honest OOF estimate is ~0.93 player AUC and ~0.09-0.17 recall at 1% FPR. The aim signal is real but low-FPR ranking is the bottleneck.
- Primary model decision for now: keep the aim-only GBDT as the low-FPR baseline; the Transformer is an ablation/ensemble member, not a clear winner.
- Next candidates: player-level aggregation beyond mean, calibration/risk-coverage, feature engineering for low-FPR ranking, 1024 context, skill stratification, raw-mouse residual subset.

## 2026-10-17 (cascade, calibration, residual audits)

- Player aggregators on OOF scores: mean AUC 0.9367 / recall@FPR=1% 0.1183; HMM 0.9187 / 0.1518 / ECE 0.081; top-3 0.9122 / 0.1058; noisy-OR 0.8529 / 0.2008 but Brier 0.49 (over-triggers); max 0.8972 / 0.1128.
- Cross-fitted player threshold (cutoff chosen on other folds): mean recall 0.1401 at FPR 0.0122; HMM 0.1455 at FPR 0.0124. Per-fold recall ranges ~0.025–0.427, so the low-FPR estimate is unstable even after cross-fitting. The global-threshold diagnostic was optimistic.
- Per-weapon isotonic calibration: ECE improves (kill 0.033 -> 0.009, player 0.110 -> 0.078) but recall@FPR=1% collapses (kill 0.131 -> 0.069, player 0.118 -> 0.098). Weapon-2 FPR only drops 4.8% -> 3.7%; weapon-4 worsens. Calibration is not the low-FPR fix.
- Calibration residuals by decile: mid bins under-predicted by 2–3 points; top decile over-predicted by ~10 points.
- Conclusion: the bottleneck is low-FPR ranking, not calibration or the mean aggregator. Next: mouse–view residual features from a raw subset, then re-evaluate. Downloaded subset: 80 cheater + 80 clean matches (parquet + JSON).

## 2026-10-17 (mouse-view residual pilot)

- Downloaded 80 cheater + 80 clean raw matches (8.7 GB) and built mouse-view residual features for 16,357 windows. Fit uses per-player robust yaw/pitch slopes on non-zero mouse movement; invalid fits are flagged.
- 5-fold match-disjoint CV on the 160-match subset (base = aim-only summary features, 59):
  - base: player AUC 0.8833, recall@FPR=1% 0.0249; kill AUC 0.8211.
  - base + all 19 residual features: player AUC 0.7740, recall@FPR=1% 0.1222; kill AUC 0.7874. The full set trades AUC for low-FPR recall.
  - base + 4 tail residual features: player AUC 0.8817, recall@FPR=1% 0.0648.
  - residual only: player AUC 0.6784, recall@FPR=1% 0.0374.
  - Bootstrap CIs overlap; the subset is small (160 matches) and its base AUC is lower than the full 793-match CV.
- Conclusion: mouse-view residual is not a clear win at 64 Hz. Do not download the full 48.9 GB raw dataset for it. Keep the tail variant only as an optional ablation. The next real lever is more data/longer context or skill-conditioned evaluation, not residual features.

## 2026-10-17 (skill/rank analysis and figures)

- Skill pilot on 875 players (160 raw matches): rank quartiles Q1 low -> Q4 high.
  - Player AUC falls 0.911 -> 0.895 -> 0.828 -> 0.712.
  - At the global 1% cutoff, Q4-high FPR is 4.76%; Q1/Q2/Q3 are 0%, 0.68%, 0.64%.
  - Two FPs out of 42 Q4 clean players; the sample is small, but the direction is consistent with the user's concern.
- Post-hoc skill corrections failed:
  - rank-conditional residual (score minus expected human score at rank): AUC 0.846 vs raw 0.882; recall@FPR1 0.006 vs 0.016.
  - Cross-fitted rank-quartile thresholds: pooled recall 0.050 vs 0.013, FPR 1.08% vs 0.72%, but Q4 FPR stays 4.76%, Q4 recall 0.
  - Adding rank as a feature: overall recall@FPR1 jumps to 0.229 but Q4 FPR rises to 14.3% — a shortcut, not skill conditioning.
  - Upweighting Q4 clean negatives x3/x5: no meaningful improvement (Q4 FPR 7.1%/4.8%).
- Conclusion: high-rank false positives are a ranking/data problem, not a calibration or threshold problem. Post-hoc skill weighting should not be shipped.
- Generated 10 figures in `reports/figures/`: ROC, risk-coverage, calibration, subgroup FPR, fold variability, residual comparison, map holdout, trajectory deviation, feature percentiles, rank strata.

## 2026-10-17 (systematic experiment harness and math analysis)

- Added `docs/llmwiki/experiments.md`: fixed folds, champion/challenger rule, experiment queue, and mathematical analysis plan.
- Added `run_cpu_experiments.py`: one runner for 5 feature sets x {GBDT, logistic, PCA-logistic, Nystroem-RBF-logistic, ExtraTrees}, plus GBDT hyperparameters and a top-20 polynomial-logistic variant. All use the same 5 match-disjoint folds.
- Pilot results: GBDT baseline reproduces 0.9331 player AUC / 0.1735 recall@FPR1; logistic 0.8935 / 0.0949; ExtraTrees 0.9274 / 0.1315. GBDT remains champion.
- Math analysis (`reports/math/math_analysis.json`):
  - PCA participation ratio 9.61 (effective dimension ~10 for 59 features).
  - Principal angles between human/cheater top-10 subspaces: first five 3.6–10 deg, later 19–25 deg, last two 73/89.8 deg; substantial overlap in the leading directions.
  - PCA-whitened Mahalanobis one-class score: player AUC 0.714, recall@FPR1 0.026; not competitive alone.
  - EVT/GPD tail p-values are monotone transforms of the same score: AUC 0.897, recall@FPR1 0.1735 (no ranking gain).
  - Cross-fitted conformal p-values: AUC 0.932, recall@FPR1 0.160; gives FPR control but does not improve ranking.
- Conclusion: low-FPR is a ranking problem; tail/calibration/conformal methods cannot create separation. Nonlinear models are needed (GBDT > logistic), but the effective dimension and subspace overlap show the headroom is in features/data, not a new decision rule.
- Batch 1 (full grid) is running in the background.

## 2026-10-17 (batch 1 complete, no champion change)

- Batch 1: 28 variants (5 feature sets x 5 model families + GBDT hyperparameters + top-20 polynomial logistic), same 5 match-disjoint folds.
- Best recall@FPR=1%: base_outcome__gbdt 0.2171 (AUC 0.9338); base_context__gbdt 0.2047; base__gbdt63 0.1805; baseline base__gbdt 0.1735.
- Paired match-grouped bootstrap vs baseline:
  - base__gbdt63: dAUC +0.0006 [-0.0006, 0.0018], dRecall +0.0027 [-0.0257, 0.0283].
  - base_motion__gbdt: dAUC +0.0010 [-0.0017, 0.0036], dRecall +0.0048 [-0.0386, 0.0537].
  - base_outcome__gbdt: dAUC +0.0009 [-0.0031, 0.0046], dRecall +0.0384 [-0.0374, 0.1102].
  - base_context__gbdt: dAUC -0.0014 [-0.0055, 0.0027], dRecall +0.0399 [-0.0235, 0.1102].
  - base_context__logreg_rff: dAUC -0.0106 [-0.0159, -0.0054] (significantly worse).
- Decision: **no challenger passes**. The apparent 0.217 is outcome-driven and its CI crosses zero; the clean champion remains base__gbdt (player AUC 0.9331, recall@FPR1 0.1735).
- Interpretation: handcrafted feature/model space is near its ceiling. The next batch must add new information (target-coupling/OU/Fitts features, 1024 context, high-rank negatives) or a better architecture, not more hyperparameter search.

## 2026-10-17 (batch 2: motor-control features)

- Added 20 parametric motor-control features per window: exponential error-decay slope/tau/R2, AR(1)/OU velocity phi/R2/sigma/half-life, four FFT band energies, constant-velocity target-tracking residual, Fitts/profile ratios, velocity skew/kurtosis.
- 5-fold GBDT with base+motor: player AUC 0.9325, recall@FPR1 0.1805 (baseline 0.9331 / 0.1735).
- Paired match-grouped bootstrap vs baseline: dAUC -0.0005 [-0.0027, 0.0016]; dRecall +0.0052 [-0.0239, 0.0314].
- gbdt63+motor: 0.9323 / 0.1899; dRecall +0.0067 [-0.0235, 0.0367].
- Decision: not promoted. The parametric motor features add no significant signal beyond the existing smoothness/dynamics features.
- Started the full 43-channel Transformer 5-fold run (`reports/cv_transformer_full`) as the architecture baseline; architecture screening will follow (d_model, layers, dropout, pooling).

## 2026-10-17 (freeze before Monday)

- Experiments paused until Monday 2026-10-20. Frozen champion: aim-only GBDT, 5-fold OOF player AUC 0.9331 / recall@FPR1 0.1735.
- Batch 1 (28 variants) and motor-control features produced no challenger passing the paired-bootstrap rule; all recall deltas had CIs crossing zero.
- Added `docs/llmwiki/next-session.md` with the Monday holdout/nested-CV protocol, three pre-registered hypotheses, RTX 5070 Ti notes, timebox, stop rules, and commands.
- Git repository initialized and detector-v1 snapshot committed after the full 43-channel Transformer run completed.

## 2026-10-17 (full 43-channel Transformer 5-fold)

- Full 43-channel Transformer, 5-fold match-disjoint: player AUC 0.9288, AP 0.7882, recall@FPR=1% 0.2350, FPR@TPR=95% 0.3400, ECE 0.1154; kill AUC 0.8774, kill recall@FPR=1% 0.2236.
- Paired match-grouped bootstrap vs aim-only GBDT: ΔAUC [-0.0125, 0.0039], Δrecall@FPR=1% [-0.0146, 0.1847].
- Decision: not promoted. The recall point estimate is higher, but the CI crosses zero and the model uses context/outcome/position channels that the metadata-only audit showed to be confounded. It remains H1 for Monday under the locked-holdout protocol.

## 2026-10-17 (freeze commit)

- Commit `ddcabfb` on `master`; annotated tag `v0.1.0-detector-baseline`.
- Working tree clean. The repository contains code, tests, docs, reports, figures, and experiment summaries; raw data, caches, virtualenv, OOF parquet files, and model checkpoints are ignored.
- Monday continuation starts from this tag; see `next-session.md`.

## 2026-10-17 (resumed on the RTX 3060: holdout lock and dev-80 baseline)

- Resumed before Monday at the user's request; only work that does not need the
  5070 Ti is being run.
- Added `scripts/lock_holdout.py`. Locked a stratified 20% match holdout:
  158 matches / 15,623 windows; dev 635 matches / 63,910 windows. Positive rate
  0.3987 (holdout) vs 0.3984 (dev). Seed 20261020; holdout match digest
  `01ff51744460a145bb18c56c052673d2372d754b7f97b6b7c8ecea005322520e`.
- The lock is post-hoc: v1 selection already used every match, so the holdout
  bounds v2 selection but cannot retroactively cleanse the frozen baseline. This
  is recorded in `reports/holdout/holdout_summary.json`.
- Re-ran the aim-only GBDT on dev 80% with the original match folds and every
  holdout match removed from training. Player AUC 0.9294 (frozen 0.9331),
  recall@FPR=1% 0.1475 (frozen 0.1735), kill AUC 0.8472. The drop is consistent
  with 20% fewer training matches and the recall CI [0.096, 0.252] covers the
  frozen estimate; the baseline is stable, not inflated.
- Added `--holdout` / `--folds-from` options to the GBDT and Transformer CV
  scripts so every v2 run uses the same dev split and fold map.

## 2026-10-17 (H2 pilot: 1024-tick context)

- Rebuilt 1024-tick windows from raw telemetry for the 160-match subset:
  16,185 of 16,357 windows kept; 172 dropped for missing pre-kill history.
  Reconstruction validated against the cached 256 windows (tick alignment,
  normalisation, victim join, velocity/250, shot deltas, kill flag, weapon
  class) and, as a like-for-like check, the rebuilt near features match the
  published cached features: player AUC 0.8822 vs 0.8810, paired ΔAUC
  [-0.0067, 0.0040].
- H2 comparison on the same matches and folds:
  - near (59 features): player AUC 0.8822, recall@FPR=1% 0.0374.
  - near + 37 macro features (96): 0.8830 / 0.0349.
  - macro only (37): 0.8240 / 0.0050.
  - paired near+macro vs near: ΔAUC [-0.0055, 0.0078], Δrecall [-0.043, 0.038].
  - paired macro vs near: ΔAUC [-0.084, -0.031].
- Decision: **H2 rejected**. Context beyond the near-kill window adds no
  ranking signal on this subset; the near window carries the signal. The
  macro-only arm is significantly worse. This is a subset pilot (160 matches),
  so the upper bound on any missed gain is about +0.008 AUC.
- Added `features/raw_window.py`, `features/macro.py`,
  `scripts/build_raw_windows.py`, and `scripts/run_h2_pilot.py`.
  `geometry.recover` and `summarize.summarize` now take a `kill_row` argument
  (default unchanged).

## 2026-10-17 (H3 pilot: rank-group DRO reweighting)

- Added `scripts/run_h3_dro.py`. Pilot on the 160-match rank subset: 14,558
  windows / 1,123 players; groups Q1-Q4 plus unknown; 4 adaptive reweighting
  iterations with inner 2-fold loss estimates per outer fold.
- Baseline vs DRO-style reweighting:
  - player AUC 0.8771 -> 0.8800; recall@FPR=1% 0.0374 -> 0.0299.
  - Q4 player AUC 0.6749 -> 0.6645; Q4 FPR at the global 1% cut 4.76% -> 4.76%.
  - Weights stayed in 0.93-1.11: the group losses are nearly identical, so the
    data does not support a worst-group reweighting signal.
- Decision: **H3 rejected**. Adaptive group reweighting does not move Q4 FPR;
  this confirms the earlier fixed-upweighting result. The blocker is feature
  overlap between top humans and cheaters, not the loss weighting.

## 2026-10-17 (H1 Transformer on dev 80% and final holdout evaluation)

- H1 43-channel Transformer, 5-fold match-disjoint on the dev 80% (same folds
  as the GBDT): player AUC 0.9251, AP 0.7773, recall@FPR=1% 0.2297
  (CI 0.146-0.357), ECE 0.1091, kill AUC 0.8704.
- Paired vs the dev-80 GBDT: ΔAUC [-0.0153, 0.0057], Δrecall [-0.0071, 0.1809].
  Decision: **not promoted** (same pattern as the all-data run).
- Added `scripts/compare_oof.py` for paired match-grouped bootstrap between two
  OOF prediction files.
- Added `scripts/run_holdout_eval.py`. Champion fitted on dev 80% and scored
  once on the locked holdout: player AUC 0.9384 (CI 0.912-0.958), AP 0.8354,
  recall@FPR=1% 0.3091 (CI 0.187-0.523), kill AUC 0.8474.
- v2 conclusion: no pre-registered hypothesis passed. The frozen aim-only GBDT
  stands; Q4 FPR 4.76% remains the blocker and points to missing information
  rather than model capacity.
- The first H1 attempt (25-epoch, workers=4) crashed in a Windows DataLoader
  spawn (`pickle data was truncated`) under CPU contention; rerun with
  batch-size 48 and workers=2 completed cleanly.

## 2026-10-17 (identity and temporal scope audit)

- Verified that the raw demos expose no cross-match player identity: every
  player field is `Player_k`, and `player_connect` / `player_disconnect` /
  `rank_update` carry no name, XUID, or network id. `player_name` is a
  per-match pseudonym (`Player_1` appears in 539-645 matches).
- Cross-match (multi-game) person analysis is therefore impossible on CS2CD.
- Within-match series are rich: 5,868 player-match groups, median 13 kills,
  97% with >=2 kills and 67% with >=10, so per-player temporal analysis inside
  a demo is supported.
- Recorded the limitation in `REPORT.md`, `README.md`, and `design.md`.

## 2026-10-17 (v2 commit)

- Commit `3726052` on `master`: v2 holdout protocol, H1/H2/H3 results, the
  1024-tick reconstruction, the paired comparison utility, and the identity
  audit. 37 files, +3,930/-50.
- Large artifacts (OOF parquet, model checkpoints, raw/processed data) remain
  ignored; the commit contains only code, docs, and small result files.

## 2026-10-17 (within-player self-baseline diagnostic)

- Added `scripts/run_self_baseline.py`: per (match, player) robust z-scores
  (median/MAD over that player's kills) added to the champion features.
  Dev-80 only; the holdout was already used for the champion's final score.
- Unfiltered dev-80 (all players, same folds):
  - raw: player AUC 0.9302, recall@FPR=1% 0.1515, kill AUC 0.8478.
  - raw+selfz: 0.9188 / 0.1881 / **0.9201**.
  - selfz only: 0.8932 / 0.1139 / 0.8447.
- The all-zero z rows for players with <3 kills could signal kill count, so a
  filtered run dropped those players (63,532 rows / 4,422 players):
  - raw: 0.9383 / 0.1851 / 0.8475.
  - raw+selfz: 0.9269 / 0.2160 / 0.9210.
  - raw+selfz+count: 0.9276 / 0.2447 / 0.9224.
- Paired (filtered) raw+selfz vs raw: ΔAUC [-0.0162, -0.0065],
  Δrecall@FPR=1% [-0.0394, +0.0963]. The kill-level gain is large and robust,
  but the player-level recall gain is not significant and AUC drops
  significantly.
- Kill-count features add nothing significant (ΔAUC [-0.001, +0.003],
  Δrecall [-0.014, +0.070]) and are a performance shortcut, not aim.
- Shuffled-label audit (match-level permutation): player AUC 0.495-0.504,
  kill AUC 0.497-0.501. No label leakage.
- Q4 (filtered, dev-80): raw AUC 0.726 / FPR 3.85%; raw+selfz 0.764 / 3.85%;
  raw+selfz+count 0.767 / 3.85%. Q4 ranking improves slightly, FPR is
  unchanged (2 false positives out of 52 clean Q4 players).
- Aggregation check on raw+selfz scores: mean is still best; max/top2/top3/top5
  do not improve recall and lose AUC (Δrecall CIs all cross zero or are
  negative). The kill-level signal does not convert to player-level decisions.
- Decision: **not promoted**. The stated target (Q4 false positives) is not
  moved; the result is recorded as an exploratory diagnostic.

## 2026-10-17 (self-baseline commit)

- Commit `b788912` on `master`: within-player self-baseline experiment,
  count/shortcut ablation, shuffled-label audit, aggregation check, and the
  recorded negative result. 21 files, +848.
- Working tree clean at commit; parquet OOF files remain ignored.

## 2026-10-18 (professional demos: specificity and negative augmentation)

- Data: 7 professional CS2 demos from `blanchon/opencs2_dataset_demo`
  (HLTV matches, CC-BY-4.0), downloaded through the HF mirror with `hf_xet`;
  parsed with demoparser2. 1,020 kill windows / 70 player-maps / 7 maps.
- Added `scripts/download_pro_demos.py`, `scripts/build_pro_windows.py`,
  `scripts/eval_pro_fpr.py`, `scripts/run_pro_augment.py`. Pro windows are
  rebuilt with the same `raw_window`/`summarize` code as CS2CD; feature
  distributions are comparable (e.g. mean final error 3.19 deg vs 4.07 for
  CS2CD humans, 4.40 for cheaters).
- Specificity of the champion (trained on CS2CD dev 80%), thresholds set on
  CS2CD human scores:
  - 1% operating point: pro kill FPR **0.29% (3/1,020)**, pro player FPR
    **0% (0/70)**.
  - 0.1% point: 0/1,020 kills, 0/70 players.
  - 5% point: 2.55% kills, 0/70 players.
  - Mean score: pro 0.253, CS2CD humans 0.293, CS2CD cheaters 0.705.
- Interpretation: elite human aim is **not** mistaken for aim assist; pros
  score below average CS2CD humans despite being more accurate. The Q4 false
  positives are therefore not a generic "high skill" effect.
- Adding the 1,020 pro windows as extra negatives in every training fold
  changes nothing: paired ΔAUC [-0.0010, +0.0017], Δrecall@FPR=1%
  [-0.0215, +0.0290]; Q4 FPR unchanged (10.5% on the dev-80 threshold),
  Q4 AUC 0.685 -> 0.677.
- Decision: **high-skill-negative hypothesis rejected**, but the specificity
  result is recorded as a positive validity check.

## 2026-10-18 (pro-demo commit)

- Commit `1877510` on `master`: pro-demo download/build pipeline, specificity
  evaluation, pro-negative augmentation pilot, and documentation. 14 files,
  +682/-3.
- Demos and processed windows stay outside the repository; only code and small
  result files are committed.

## 2026-10-18 (Q4 threshold correction and player-score shrinkage)

- **Correction**: the previously reported Q4 FPR (4.76%) applied a kill-level
  threshold to player mean scores. Player-level FPR must use a player-level
  threshold. Corrected: champion full OOF 19.05% (8/42), dev80 13.16% (5/38);
  the mouse-residual rank-strata table is 7.14% (3/42).
- Root cause of most false positives: low evidence. 5 of 8 champion Q4 false
  positives had one or two kills; the Q4 clean group has median 9.5 kills and
  25th percentile 2.
- Added `scripts/analyze_player_shrinkage.py`: empirical-Bayes shrinkage
  `(sum + k0*prior)/(n+k0)` with the prior from other training folds.
  - k0=5, full OOF: AUC 0.9339, recall@FPR=1% 0.2288, Q4 FPR 4.76%,
    Q4 AUC 0.797.
  - k0=5, dev80: AUC 0.9316, recall 0.2386, Q4 FPR 5.26%, Q4 AUC 0.791.
  - Paired dev80: ΔAUC [-0.0015, +0.0058], Δrecall [+0.0068, +0.1404].
- The k0 grid 1-20 is a broad plateau; the result is not knife-edge.
- The two remaining Q4 false positives have extremely straight trajectories
  (0.88 and 0.98 vs human median 0.61), consistent with unusual style or
  undetected cheating; 64 Hz view data cannot distinguish them.
- Status: shrinkage adopted as the player-level scoring rule for reporting,
  dev-validated only. The holdout was already spent, so it is not re-used and
  no holdout claim is made.

## 2026-10-18 (Q4 correction commits)

- Commit `105eb62`: player-level FPR threshold fix, empirical-Bayes shrinkage,
  and the corrected Q4 numbers. 13 files.
- Commit `d3eea64`: reconciled the corrected Q4 FPR across `REPORT.md`, the
  experiment log, and the session plan so no living document repeats the old
  4.76% figure without a note.
- Working tree clean; no holdout re-evaluation was performed.

## 2026-10-18 (label-noise audit)

- Added `scripts/analyze_label_noise.py` and `scripts/audit_q4_fps.py`, plus
  `docs/llmwiki/label-noise-audit.md`.
- Out-of-match scoring on dev 80 (no holdout use), groups:
  - clean: 49,164 windows / 3,668 players, mean score 0.304.
  - uncertain (cdata label 0): 9,137 / 1,171, mean 0.323, but only 0.34% of
    kills and 0.68% of players above the clean 1% cut.
  - cheater: 14,746 / 1,010, mean 0.657, 15.5% of kills above the cut.
- Estimated uncertain cheater fraction: 5.5% by first-moment decomposition;
  the KDE likelihood ratio gives 33.6% but is inconsistent with the mean and
  tail, so it is reported as an unstable cross-check.
- Interpretation: the excluded windows look human at the window level; the
  dataset card's 55.6% precision is not corroborated by behaviour. The model's
  15.5% cheater recall means this is weak evidence of absence.
- Residual Q4 false positives after shrinkage: two players with 67% and 78% of
  kills closer to cheater windows; the second is extremely straight (0.945),
  accurate (2.06 deg) and slow (8.3). Unusual style or undetected smooth aim;
  64 Hz cannot separate them.

## 2026-10-18 (label-noise commit)

- Commit `f74d46c`: label-noise audit scripts/results, Q4 false-positive
  per-kill inspection, the dedicated audit document, and the knowledge-base
  index. 14 files, +665.
- Working tree clean; scored-window parquet remains ignored.

## 2026-10-18 (rank-stratified FPR replication)

- Added `scripts/download_cs2cd_json.py` and `scripts/rank_replication.py`.
  Downloaded the per-match JSON for all 635 dev-80 matches (~1.3 GB) and
  recovered ranks from `rank_update` events, avoiding the 48.9 GB parquet set.
- Coverage: 3,429/4,678 players (73.3%) ranked.
- Replication on all dev80, player-level 1% threshold 0.8217:
  - Q1 0.26%, Q2 1.22%, Q3 1.05%, Q4 **3.37%** (386 clean Q4 players).
  - Shrinkage k0=5: Q1 0.52%, Q2 1.22%, Q3 1.36%, Q4 **2.59%**.
  - Q4 AUC 0.864 -> 0.879 with shrinkage.
- The earlier 19.05% and 13.16% Q4 FPR figures came from 42/38 clean Q4
  players and are superseded as population estimates. At a true rate near
  3.4%, 8 false positives out of 42 is a rare but possible draw; the
  160-match subset is also harder than the full dev set (11.1% at the same
  threshold).
- Conclusion: the Q4 effect is real but modest (~3.4x the 1% target), not a
  19% blocker. This is the number to carry into engineering.

## 2026-10-18 (replication commit)

- Commit `a20126e`: rank replication on all dev-80 matches, JSON-rank
  downloader, replication document, and superseding Q4 numbers. 11 files,
  +405/-16.
- Working tree clean; JSON files and rank player parquet stay outside the
  repository.

## 2026-10-18 (deployment artifact and scoring CLI)

- Added `scripts/train_final.py`: fits the deployment GBDT on all 79,533
  clean windows / 793 matches and saves a 1.2 MB joblib artifact with the
  feature order, player-level 1% threshold (0.7916), human prior (0.290) and
  shrinkage k0=5.
- Added `scripts/score_demo.py`: demo -> windows -> features -> kill scores ->
  per-player shrunk scores with `flagged` and `reliable` fields. Tested on a
  professional demo: 10 players, 0 flagged, 18.7 s wall clock on CPU.
- Added `docs/llmwiki/model-card.md` (input/output contract, metrics,
  limitations, batch-retraining policy) and console entry points in
  `pyproject.toml` (`valveguard-train`, `valveguard-score`).
- Decision: **no incremental training**. Full retraining is 1-2 minutes and
  avoids catastrophic forgetting; increments apply to data ingestion and
  versioned batch retraining, not to model fitting.
- Artifacts are gitignored; they are rebuilt with one command.

## 2026-10-18 (model I/O contract)

- Added `docs/llmwiki/io-contract.md`: exact shapes and dtypes for
  demo -> window (256x44) -> features (59,) -> kill score (N,) -> player
  table; window column order, kill row 223, 64 Hz, versioning rules, and the
  Windows application boundary.
- Clarified that the raw model is a 59-feature tabular GBDT; the 256x44 window
  is an upstream extractor artifact, not the model input.
- ONNX export attempted; `skl2onnx` 1.20 fails on this HGB model (boolean
  attribute in TreeEnsembleClassifier). The raw artifact stays joblib, which
  is sufficient for a Python/PyInstaller Windows app.

## 2026-10-18 (I/O contract commit)

- Commit `cd69f31`: frozen model I/O contract of 3 files.
- Working tree clean; the joblib artifact and experimental ONNX deps stay
  outside version control.

## 2026-10-18 (Windows app feasibility: 5E demo, Rust parser, Rust model)

- Validated the target input on a real 5E file:
  `D:\5EDemocache\g161-...zip` contains one 189 MB `.dem`. The Python
  reference pipeline parses 10 players (Chinese UTF-8 names) / 145 kills and
  scores the demo in 6 s.
- Rust parser spike: the `awpy` 3.0 crate parsed the same demo (map de_dust2,
  145 kills, 11 roster entries including a GOTV spectator), so a pure-Rust
  Tauri backend is viable without Python.
- Model export: added `scripts/export_model_json.py`; the 300-tree ensemble is
  931 KB JSON. A Rust traversal matched sklearn on 200 real rows with NaNs:
  max abs diff 3.3e-16. ONNX is not needed.
- Added `docs/llmwiki/windows-app.md`: zip->dem handling, player/kill review
  UX, tick-range video mapping, CS2 playback/capture options, Rust architecture
  and feature-port estimate.
- Note: demos contain no video. Clips require CS2 playback + capture
  (OBS/FFmpeg) or HLAE; the app always provides 2D trajectory previews and
  tick ranges for review.

## 2026-10-18 (Rust feasibility commit)

- Commit `41b7936`: Rust parser/model spikes, portable JSON export, 5E sample
  result, and the Windows application plan. 6 files, +342/-2.
- Working tree clean; artifacts, demos, and zips stay outside the repository.

## 2026-10-18 (pure-Rust core and warm-up correction)

- Implemented `desktop/` (pure Rust): `.zip`/`.dem` -> `awpy` parse ->
  256-tick windows -> 59 features -> JSON tree ensemble -> player table with
  shrinkage. No Python at runtime.
- Validated on the 5E dust2 sample: 129/145 kills used, 10 players, 0 flagged,
  **3.9 s** end to end including zip extraction.
- Parity vs Python: 129/129 kills matched; kill-score max diff **0.0**;
  player-score max diff **4.6e-7**; identical kill counts and flags.
- Found and fixed a real warm-up bug: `total_rounds_played` is 0 for the whole
  first live round, so the Python filter dropped first-round kills. Both
  pipelines now use explicit warm-up/round logic.
- Pro windows rebuilt with the fix: **1,075** (was 1,020). Pro FPR at 1% is
  **0.28% (3/1,075)**, 0/70 players; at 5% it is 2.70%. Pro-negative
  augmentation: ΔAUC [-0.0019, +0.0014], Δrecall [-0.0098, +0.0480];
  Q4 unchanged. Conclusions hold.
- Added `desktop/README.md`; updated `windows-app.md` and all pro-demo numbers
  in the living docs.

## 2026-10-18 (Rust core commit)

- Commit `f3815de`: pure-Rust desktop core, parity validation, warm-up fix,
  and refreshed pro-demo results. 22 files, +2,177/-100.
- Working tree clean; `desktop/target`, demos, zips, and model artifacts stay
  outside the repository.

## 2026-10-18 (movement / pre-aim segment pilot)

- Added `scripts/build_segment_features.py`: 24 rule-based features for
  travel / walk-silent / stop / duel, pre-aim error and distance, and state
  transitions. The cache has no `is_walking`, so silence is a speed band.
- Dev-80 5-fold: base 0.9294/0.1475, base+segments 0.9311/0.1337 (kill AUC
  0.8472 -> 0.8677), segments only 0.9041/0.1050.
- Paired base+segments vs base: ΔAUC [-0.0015, +0.0054], Δrecall
  [-0.0663, +0.0526]. Q4 FPR 2.63% -> 5.26% (mean), 5.26% -> 10.53% shrunk.
- Decision: **not promoted**; kill-level gain only. Recorded in
  `segment-features.md` and `experiments.md` section 14.

## 2026-10-18 (segment pilot commit)

- Commit `1f1a550`: movement/pre-aim segment features, dev-80 comparison,
  negative result, and the 5E review summaries. 16 files, +1,907.
- Working tree clean; feature parquets and OOF files remain ignored.

## 2026-10-20 (detector/generator split)

- Split the repository into the detector-only `ValveGuard` and the sibling
  `AimForge` project.
- Moved `src/valveguard/generator`, the generator/demo scripts, generator
  docs, generator reports, and `data/raw/aim_maps` to `AimForge`.
- Detector imports were re-checked; no detector code referenced the generator
  package. The KB index, design note, model-design scope, and root README now
  state the split.

## 2026-10-20 (git history split)

- Rewrote the detector history to drop all generator and demo paths:
  `src/valveguard/generator`, the generator scripts, generator docs,
  `reports/generator*`, `reports/cs2_*`, `reports/vizdoom*`, and
  `_vizdoom.ini`. Pack size fell from 9.7 MiB to 1.2 MiB.
- Report videos were never tracked (`reports/**/*.mp4|mov` was already
  ignored), so the rewrite removed metadata and images only.
- Archived the pre-rewrite history outside the repository as
  `../ValveGuard-history-backup-20261020.bundle`.
- Initialized `AimForge` as its own repository with one import commit. Its
  videos, workshop maps, and caches stay on disk but are gitignored.

## 2026-10-20 (generation and CS-operation cleanup)

- Removed the last generator and CS-operation material from this repository:
  the CS2 video/HLAE capture section in the app plan, the "open in CS2"/OBS
  capture items, and the generator design section in `model-design.md`.
- Moved the dated generator, CS2 demo, and ViZDoom changelog entries to
  `AimForge/docs/changelog.md`.
- Renamed the task doc to `tasks/aimbot-detection.md`.
- Demo parsing/scoring and 2D review previews stay in scope; game launch,
  camera scripting, and capture are out of scope.

## 2026-10-20 (remote training workspace)

- Set up `E:\ValveGuardWork` on `DESKTOP-BC3OC2K` (192.168.3.143): Python
  3.12.10, venv, editable install, PyTorch 2.9.0+cu128, `cw256` and
  `cw256_channels` caches, and three 5E demos.
- Verified the full clean-label baseline on 79,533 windows (test player AUC
  0.9430, kill AUC 0.9006) and 5E demo scoring (10 players, 0 flagged, 3.9 s).
- Started the 100+100 CS2CD raw subset download through the local proxy
  `192.168.3.129:7890`; `hf-mirror.com` is the fallback when the proxy is off.
- Layout, commands, and cleanup are recorded in `remote-training.md`.

## 2026-10-20 (kill-information weighting)

- Re-aggregated the existing OOF kill scores with context weights (distance,
  victim speed/health, smoke, wallbang, flash); no retraining.
- 500-replicate paired match-grouped bootstrap: every variant loses AUC with a
  CI below zero; recall@FPR=1% and Q4 FPR changes have CIs crossing zero.
  Best point estimate (`no_smoke_wall`): recall 0.2296 -> 0.2444, Q4 FPR
  0.0259 -> 0.0233, AUC 0.9338 -> 0.9321.
- Decision: not promoted. Recorded in `experiments.md` section 15; artifact
  under `reports/reliability/`.

## 2026-10-20 (raw event feature pilot)

- Added `features/events.py` and `build_event_features.py`: 63 features from
  `aim_punch_angle`, fire timing, mouse-view residual, scope/movement/spotted.
- Built event features for 100 cheater-present + 100 clean raw matches:
  20,252 kill windows.
- 5-fold match-disjoint pilot: kill AUC 0.8276 -> 0.8459 (Δ +0.0183,
  bootstrap CI [0.0106, 0.0258]); player AUC 0.8668 -> 0.8731 (CI crosses
  zero); player recall@FPR1 0.1567 -> 0.1781 (CI crosses zero).
- Group ablation: aim-only (recoil+fire+mouse) adds +0.0100 kill AUC and
  +0.0150 recall, player AUC flat; context (spotted/walking/ducking/velocity)
  gives +0.0117 player AUC but -0.0215 recall; recoil alone adds almost
  nothing. The top permutation feature is `ev_spotted_frac`, so the pilot is
  not yet a clean aim-assist claim.
- Decision: pilot only. Kill-level signal is real; player-level needs more
  matches. A 200+200 subset is downloading.
- Recorded in `experiments.md` section 16; artifacts under
  `reports/events_pilot/`.

## 2026-10-20 (400-match event replication)

- Downloaded 200 cheater-present + 200 clean raw matches (800 files, ~22.9 GB)
  through the local proxy; built 63 event features for 41,888 windows.
- Clean-label 5-fold replication (35,814 windows, 2,704 players):
  - all events vs base: kill AUC 0.8702 -> 0.8904 (Δ +0.0202, CI
    [0.0145, 0.0258]); player AUC 0.9144 -> 0.9188 (Δ +0.0048, CI
    [0.0009, 0.0087]); player recall 0.1505 -> 0.1840 (CI crosses zero).
  - aim-only (recoil+fire+mouse): kill AUC 0.8847 (Δ +0.0145, CI
    [0.0090, 0.0197]); player AUC and recall CIs cross zero.
- Shortcut audit: cheater kills have much lower spray length
  (5.45 -> 1.61), less recoil-yaw variation, lower spotted fraction, and
  higher scope/velocity; part aim, part role/play style.
- Layer check: base Q4 FPR 2.78% / Q4 recall 14.6%; all events 3.33% / 17.9%.
  Event features trade Q4 precision for recall; sniper kills carry ~5% FPR.
- Decision: kill-level signal replicated; player-level low-FPR recall still
  not significant. Not promoted. Recorded in `experiments.md` section 17;
  artifacts under `reports/events_pilot_400/`.
- Started the full 500+500 raw download (all available cheater and clean
  matches, ~45 GB) for the full-cohort event run.
- Full raw download complete: 317 cheater-present + 431 clean matches
  (1,496 files, ~43.5 GB).
- Built full-cohort event features: 78,202 windows over 701 matches; 93
  missing raw files were skipped. Full pilot and aim-only pilot started.

## 2026-10-20 (full-cohort event replication)

- 700 matches / 67,067 clean windows / 4,947 players (1,284 cheater players).
- base: kill AUC 0.8794, player AUC 0.9284, recall@FPR1 0.2087.
- all events: kill AUC 0.9005, player AUC 0.9339, recall 0.2111.
- aim-only: kill AUC 0.8937, player AUC 0.9295, recall 0.1908.
- Bootstrap: all vs base Δkill +0.0212 [0.0171, 0.0253], Δplayer AUC
  +0.0057 [0.0030, 0.0084], Δrecall +0.0072 [-0.0353, 0.0493].
- Rank/weapon layers: Q4 FPR 3.14% -> 2.86% (all), Q4 recall 24.5% -> 21.4%;
  sniper FPR ~5.6-6.0%; auto rifle FPR ~0%; pistol recall 4.0% -> 9.7%.
- Group ablation: fire is the best single group (recall 0.2157), context adds
  the most player AUC (+0.0043), aim-only lowers recall to 0.1908, recoil and
  mouse are negligible.
- Decision: not promoted. Event features improve ranking, not low-FPR recall;
  Q4 improves slightly but Q2/Q3 regress. Recorded in `experiments.md`
  section 18; artifacts under `reports/events_pilot_full/`.

## 2026-10-20 (target acquisition and first-shot features)

- Added 26 target-acquisition features: target error trajectory, time to
  enter 3/5/10 degrees, time on target, first-shot-to-kill, error at first
  shot, acquire-to-first-shot reaction.
- Full cohort (700 matches / 67,067 windows): base recall 0.2087; +acquire
  0.2150; +fire 0.2173; +aim_all 0.1986; +all 0.2095. Kill AUC and player
  AUC improve, but the primary low-FPR recall gain is not significant
  (+0.0052, CI [-0.0361, +0.0492]).
- Q4 FPR: base 3.14%, all 2.86%, aim_all 2.29%; overall recall falls in
  aim_all. Sniper FPR stays ~5.6%.
- Decision: not promoted. Summary-feature engineering is exhausted; next is
  a GPU sequence model over the raw channels. Recorded in `experiments.md`
  section 19; artifacts under `reports/events_pilot_full_acq/`.

## 2026-10-20 (GPU 1D CNN on 43 channels)

- Added `train_channel_cnn.py`: 3-conv 1D CNN on 256 x 43, 5-fold, 8 epochs,
  RTX 5070 Ti. Robust median/IQR normalization handles inf in jerk/speed.
- CNN vs GBDT on the same 79,533 OOF rows:
  - kill AUC 0.8497 -> **0.9046**; player AUC 0.9338 -> **0.9467**;
    recall@FPR1 0.2296 -> **0.2872**; Q4 FPR 2.59% -> 3.11%; Q4 recall
    22.9% -> 30.0%.
  - Bootstrap: Δkill AUC **+0.0550** [0.0505, 0.0598]; Δplayer AUC
    **+0.0127** [0.0084, 0.0174]; Δrecall +0.0486 [-0.0108, +0.1119];
    ΔQ4 FPR +0.0077 [-0.0065, +0.0224].
- Decision: strong challenger, not promoted yet. Recall is near-significant
  and Q4 FPR regresses. Seeds 43/44 are training in parallel for an ensemble.
- Recorded in `experiments.md` section 20; artifacts under `reports/cnn43/`.

## 2026-10-20 (CNN 3-seed ensemble)

- Seeds 42/43/44 ensemble: kill AUC 0.9089, player AUC 0.9475, recall@FPR1
  0.2794, Q4 FPR 3.11%, Q4 recall 31.1%.
- Bootstrap vs GBDT: Δkill **+0.0592** [0.0546, 0.0638]; Δplayer AUC
  **+0.0137** [0.0090, 0.0187]; Δrecall +0.0535 [-0.0056, +0.1174];
  ΔQ4 FPR +0.0079 [-0.0064, +0.0224].
- Ensemble improves AUC/stability but recall CI still crosses zero and Q4 FPR
  remains worse. Not promoted. Next: event-channel CNN and Q4-aware stage.
- Recorded in `experiments.md` section 20; artifacts under
  `reports/cnn43_ensemble/`.

## 2026-10-20 (wide attention CNN)

- Wider 4-conv + attention pooling, 10 epochs, 5-fold, 43 channels:
  kill AUC 0.9038, player AUC 0.9488, recall@FPR1 0.2911, Q4 FPR 3.37%,
  Q4 recall 31.3%.
- Bootstrap vs GBDT: Δkill **+0.0542** [0.0489, 0.0594]; Δplayer AUC
  **+0.0151** [0.0104, 0.0200]; Δrecall **+0.0744** **[0.0107, 0.1432]**;
  ΔQ4 FPR +0.0057 [-0.0084, +0.0205].
- First sequence model with low-FPR recall CI entirely above zero. Promotion
  candidate; Q4 point regression is the trade-off. Wide seeds 43/44 training
  for an ensemble before the champion decision.
- Recorded in `experiments.md` section 20; artifacts under `reports/cnn43_wide/`.

## 2026-10-20 (wide CNN 3-seed ensemble)

- Wide seeds 42/43/44 recall 0.2911 / 0.2981 / 0.3051; ensemble kill AUC
  0.9086, player AUC **0.9501**, recall **0.2965**, Q4 FPR 3.11%, Q4 recall
  33.5%.
- Bootstrap vs GBDT: Δkill **+0.0590** [0.0538, 0.0639]; Δplayer AUC
  **+0.0164** [0.0117, 0.0214]; Δrecall **+0.0822** **[0.0135, 0.1532]**;
  ΔQ4 FPR +0.0057 [-0.0091, +0.0209].
- Promotion candidate. Running the aim-only 29-channel ablation to rule out
  metadata/outcome shortcuts before the champion decision.
- Recorded in `experiments.md` section 20; artifacts under
  `reports/cnn43_wide_ensemble/`.

## 2026-10-20 (aim-only wide CNN)

- Dropped weapon, kill outcome, kill flags, flash, and shot; 29 aim-only
  channels. Wide CNN, 5-fold, 10 epochs.
- New champion candidate: kill AUC 0.8979, player AUC 0.9474, recall@FPR1
  **0.3261**, Q4 FPR **2.59%** (same as GBDT), Q4 recall **35.0%**.
- Bootstrap vs GBDT: Δkill **+0.0483** [0.0434, 0.0534]; Δplayer AUC
  **+0.0136** [0.0087, 0.0188]; Δrecall **+0.0930** **[0.0276, 0.1618]**;
  ΔQ4 FPR **0.0000** [-0.0090, +0.0174].
- The metadata/outcome channels were hurting low-FPR ranking; the gain is
  aim-specific. Aim seeds 43/44 training for the final ensemble.
- Recorded in `experiments.md` section 20; artifacts under
  `reports/cnn43_aim_wide/`.

## 2026-10-20 (aim-only ensemble, top-k aggregation, raw channels)

- Aim-only wide seeds 42/43/44 completed. Ensemble: kill AUC 0.9113,
  player AUC 0.9489, recall@FPR1 0.2872, Q4 FPR 3.37%, Q4 recall 31.5%.
- Bootstrap vs GBDT: Δkill **+0.0616** [0.0569, 0.0663]; Δplayer AUC
  **+0.0152** [0.0105, 0.0202]; Δrecall **+0.0720** **[0.0095, 0.1397]**;
  ΔQ4 FPR +0.0057 [-0.0074, +0.0190].
- Corrected the single-seed aim-only claim: the same-Q4 result was seed
  variance and does not replicate. Aim-only remains the preferred
  representation for shortcut safety, not a metric win over full-channel.
- Added `--topk` to `compare_cnn_gbdt.py`. Top-5 player aggregation gives
  recall 0.3401, Q4 FPR 3.11%, player AUC 0.9322; it improves CNN recall
  in all five folds but costs 1.7 pp player AUC. Recorded as an optional
  screening mode, not the default.
- Added `build_raw_channels.py`: 23 per-tick mouse/recoil/shot/stance
  channels aligned to the cached kill windows, for the next CNN iteration.
- Recorded in `experiments.md` section 20 and `remote-training.md`;
  artifacts under `reports/cnn43_aim_wide_ensemble/` and
  `reports/cnn43_aim_wide_ensemble_top5/`.

## 2026-10-20 (raw mouse/recoil/fire channels: negative)

- Built 23 per-tick raw channels for 67,449 covered rows and merged them
  with the 43-channel cache (`cw256_channels_aug`, 66 channels).
- 66 vs 43 on the same rows: kill AUC **+0.0037 [0.0004, 0.0070]**;
  player AUC +0.0011 [-0.0014, 0.0034]; recall +0.0183
  [-0.0262, 0.0609]; Q4 FPR flat.
- Aim-only ablations: 29 + 23 raw vs 29 gives kill **+0.0051
  [0.0017, 0.0087]** but player AUC -0.0016 and recall +0.0011, both CIs
  crossing zero; 29 + 8 compact raw is flat.
- Decision: not adopted. No full raw-channel pipeline build; keep the
  29-channel aim-only representation.
- Added `--channel-index` to `train_channel_cnn.py` for subset ablations.
- Artifacts under `reports/cnn66*/`, `reports/cnn37*/`,
  `reports/cnn52*/`; recorded in `experiments.md` section 20.

## 2026-10-20 (deploy ensemble, provenance correction, demo scoring)

- Found a provenance error: the earlier "aim-only ensemble" mixed the
  29-channel seed 42 with 43-channel seeds 43/44. Its recall 0.2872 and
  the "no gain" conclusion are invalid; artifacts renamed
  `reports/*_mixed_superseded`.
- Retrained all three aim-only seeds on the full cohort and saved 5 fold
  models each (`runs/cnn29_full_wide_seed42/43/44`). Corrected ensemble:
  kill AUC 0.9027, player AUC **0.9480**, recall **0.3284**, Q4 FPR
  **3.11%**, Q4 recall 36.5%.
- Bootstrap vs GBDT: Δkill **+0.0530** [0.0482, 0.0579]; Δplayer AUC
  **+0.0141** [0.0094, 0.0190]; Δrecall **+0.0879** **[0.0262, 0.1516]**;
  ΔQ4 FPR +0.0024 [-0.0101, +0.0154]. The single-seed recall gain
  replicates; the single-seed 2.59% Q4 claim does not.
- Corrected top-5 aggregation: recall 0.3518, Q4 FPR 2.85%, player AUC
  0.9255; optional screening mode only.
- Added `--save-models` to `train_channel_cnn.py` and `score_demo_cnn.py`.
  Deploy threshold 0.712435 from the OOF human 99th percentile.
- Scored the 5E Dust2 demo: 10 players, 0 flagged, top shrunk score
  0.2866 (human p95 0.4692). Feature/latency sanity check passed.

## 2026-10-20 (longitudinal dataset proposal)

- Added `docs/llmwiki/longitudinal-dataset.md`: cohort sizes, stable-id and
  ban-date requirements, dense telemetry, time-forward evaluation, and a
  5E feasibility pilot (20 banned + 40 control players x 10 matches).
- One player's history is scoped as a case study/demo, not training data.

## 2026-10-20 (player-view fusion: negative, stable-id audit)

- Stable-id audit: CS2CD player ids are per-match pseudonyms (`Player_1..10`),
  so person-disjoint evaluation and cross-match player models are impossible;
  match-disjoint remains the strongest split.
- Post-hoc player fusion on the corrected aim ensemble:
  - mean: AUC 0.9480, recall 0.3284, Q4 FPR 3.37%, Q4 recall 37.2% (local
    threshold convention).
  - top-5: recall 0.3518 but AUC 0.9255; Δrecall +0.0268 [-0.0181, 0.0824]
    vs mean, ΔAUC -0.0228 [-0.0286, -0.0174].
  - rank fusion: recall 0.3323, AUC 0.9419, Δrecall +0.0177 [-0.0200, 0.0624].
  - match-relative: AUC 0.7932, recall 0.1626. Dispersion: AUC 0.7807,
    recall 0.0296. Both harmful.
- Decision: no post-hoc fusion. Mean aggregation stays; the next experiment is
  a player-level MIL model trained on player labels.
- Added `src/valveguard/scripts/analyze_player_fusion.py`; artifacts under
  `reports/player_fusion/`.

## 2026-10-20 (fixed-fold retrain, authoritative ensemble)

- Found that the GBDT used stratified match folds while the CNN used a
  fresh GroupKFold. Added `--folds-from` to both trainers and retrained the
  aim-only ensemble on the `cv_predictions` fold map.
- Fixed-fold ensemble: kill AUC 0.9021, player AUC **0.9500**, recall
  **0.3121**, Q4 FPR **2.59%** (same as GBDT), Q4 recall **33.9%**.
- Bootstrap vs GBDT: Δkill **+0.0524** [0.0476, 0.0573]; ΔAUC **+0.0163**
  [0.0120, 0.0207]; Δrecall **+0.0824 [0.0208, 0.1490]**; ΔQ4 FPR +0.0021
  [-0.0098, +0.0162]. These supersede the earlier 0.3284/3.11% numbers.
- Top-5 on fixed folds: recall 0.3230, AUC 0.9277, Q4 FPR 3.37%; mean stays
  the default.
- Re-scored the 5E demo with the fixed-fold models and threshold 0.714925:
  10 players, 0 flagged, top shrunk 0.2769 (human p95 0.4717).
- Artifacts under `reports/cnn29_fixed_deploy*/` and
  `runs/cnn29_fixed_wide_seed42/43/44`.

## 2026-10-20 (player-level MIL: negative)

- Trained attention-pooled MIL on player labels with the fixed folds.
  Frozen encoder: AUC 0.9478, recall 0.3175, Q4 FPR 5.44%; Δrecall +0.0148
  [-0.0765, 0.1151], ΔQ4 **+0.0232 [0.0023, 0.0455]**.
  Warm start + unfrozen: AUC 0.9329, recall 0.2856; ΔAUC **-0.0172**.
- Decision: not adopted. 5,868 player labels are too few and the attention
  pooling adds variance; keep window BCE + mean. Next: low-FPR loss.
- Added `train_player_mil.py`, `compare_player_scores.py`; artifacts under
  `reports/mil29*/`.

## 2026-10-20 (hard-negative BCE: new champion)

- Added `--hard-neg` to the window trainer: negative windows weighted by
  `1 + alpha * p^2`, targeting the top of the human score distribution.
- Alpha probes (fixed folds, seed 42): alpha=2 recall 0.3346, alpha=5
  0.3432, alpha=10 0.3479; alpha=5 kept as the least aggressive near-peak.
- 3-seed alpha=5 ensemble: kill AUC 0.9015, player AUC **0.9502**, recall
  **0.3619**, Q4 FPR **3.11%**, Q4 recall **39.1%**.
- Bootstrap vs GBDT: Δrecall **+0.1223 [0.0476, 0.1988]**; ΔAUC
  **+0.0165**; ΔQ4 FPR +0.0030 [-0.0090, 0.0161].
- Bootstrap vs previous mean ensemble: Δrecall **+0.0459 [0.0048,
  0.0883]**, ΔAUC +0.0002, ΔQ4 FPR +0.0048 [-0.0057, 0.0158].
- New champion. Export models for all three seeds are being written;
  artifacts under `reports/cnn29_hn5_ensemble*/`.

## 2026-10-20 (hn5 demo scoring)

- Saved five fold models for hn5 seeds 42/43 and reused seed 44; rescored
  the 5E Dust2 demo with the hn5 ensemble and threshold 0.608027.
- 10 players, 0 flagged, top shrunk 0.2374 vs human mean 0.218 / p95 0.3973.

## 2026-10-20 (visibility reaction channels: negative)

- Extended `build_raw_channels.py` with five channels from
  `approximate_spotted_by`: sees-victim, time-since-seen, onset flag,
  victim-spotted, spotted-age. Rebuilt the 28-channel raw cache and the
  71-channel combined cache.
- 34ch (aim + reaction) vs 29ch on the same 67,449 rows with hn5:
  kill AUC **+0.0074 [0.0048, 0.0101]**, player AUC **+0.0040
  [0.0019, 0.0060]**, recall -0.0019 [-0.0394, 0.0366], Q4 FPR +0.0049
  [-0.0048, 0.0165].
- Decision: not adopted. Kill-level signal only; the primary low-FPR recall
  does not move. Kept for longitudinal work.

## 2026-10-20 (champion artifact and demo report)

- Packaged the hn5 champion into `artifacts/valveguard_cnn_v1/`: manifest,
  29 channel names, human reference distributions, 15 fold models, per-seed
  normalization, checksums, README.
- Added `valveguard/models/artifact.py` (`load_detector`, `score_channels`,
  `aggregate`) with checksum and channel-name validation.
- Added `demo_report.py`: accepts one `.dem` or a directory, writes
  `result.json`, `players.png`, and `kill_windows.png`.
- 5E Dust2 demo: 10 players, 0 flagged, top shrunk 0.2374 (human p72);
  charts show each player against the human player distribution and each
  player's kill windows against the human window distribution.
- Directory drop-in tested with one demo.

## 2026-10-20 (banned-player demo validation)

- Scored a banned-player 5E demo (7 kills, all headshots; identifiers
  omitted from the repository).
  Champion gives shrunk 0.414 (p96), below the 0.608 p99 flag; max window
  0.853 vs human window p99 0.863. Genuine strict-threshold false negative.
- Added a p95-p99 review tier (amber) to `demo_report.py`, player names,
  and a lobby-gap line. The banned player is surfaced as REVIEW, +0.115
  over the second player.
- Case recorded in `experiments.md`; next candidate is player-level
  outcome/behavior fusion (headshot rate, burst multikills).

## 2026-10-20 (visibility/pre-aim case study: inconclusive/negative)

- Added `analyze_visibility.py`: nearest-enemy head error, approximate
  visibility, unseen-moving tracking, reaction time, per-kill timelines.
- Banned player vs strong human control: pre-aim 8.6% vs 10.6%,
  unseen-moving 5.9% vs 7.8%, longest run 305 ticks both. No separation.
- Decision: simple pre-aim metrics are confounded by normal angle holding
  and approximate visibility; not adopted. A visibility model needs
  raycast and a control cohort.
- Next recommended: player-level outcome/behavior fusion (headshot rate,
  burst multikills) validated on CS2CD OOF.

## 2026-10-20 (XGUARDIAN processed data audit: confounded)

- Downloaded the XGUARDIAN processed CSVs (527 MB) and built a 96-tick
  window baseline on 28 summary features.
- Test window AUC 0.702, player AUC 0.858, recall@1% 3.1% / 0.8%; aim-only
  window AUC 0.620.
- Found an activity confound: cheater player-matches average 22.1 windows
  vs 4.6 for humans; window count alone gives player AUC 0.9705. Within
  count bins the model AUC is 0.44-0.68.
- CS2CD has no such confound: window count alone is AUC 0.499.
- The processed angle is a coarse 2D direction, not pitch/yaw. Raw demos
  (585 GB) would be required for a fair external test.
- Decision: do not use the processed CSV; prefer a HAWK data request or a
  raw XGUARDIAN sample with all players relabeled.
