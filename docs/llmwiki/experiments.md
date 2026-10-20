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

Champion: **aim-only 29-channel wide CNN, 3-seed ensemble, hard-negative
BCE (alpha=5)** on the fixed `cv_predictions` match folds — player AUC
0.9502, recall@FPR1 0.3619, Q4 FPR 3.11%, Q4 recall 39.1% (section 20).
Δrecall vs GBDT +0.1223 [0.0476, 0.1988]; vs the previous mean ensemble
+0.0459 [0.0048, 0.0883]. Fold models are saved for demo scoring.

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

## 15. Kill-information weighting (2026-10-20)

Hypothesis: low-information kills (close range, stationary or already-damaged
victim, smoke/wallbang/flash) add noise to the player mean. Re-aggregate the
existing OOF kill scores with context weights; no retraining.

- Data: `reports/cv_gbdt/cv_predictions.parquet` (79,533 clean windows),
  `reports/baseline_clean/summary_features.parquet`, and the rank replication
  table (386 clean Q4 players, 463 cheater Q4 players).
- Aggregation: `(Σ w_i s_i + k0 prior) / (Σ w_i + k0)`, `k0=5`, prior = human
  OOF mean; same rule as the champion, only `w` changes.
- Weights use context only (distance, victim speed/health, smoke, wallbang,
  flash), never the label or the kill score.
- Paired match-grouped bootstrap, 500 replicates, vs uniform weights.

| Variant | Player AUC | Recall@FPR1 | Q4 FPR | ΔAUC 95% CI | ΔRecall 95% CI | ΔQ4 FPR 95% CI |
|---|---|---|---|---|---|---|
| uniform | **0.9338** | 0.2296 | 0.0259 | reference | reference | reference |
| no_smoke_wall | 0.9321 | 0.2444 | **0.0233** | [-0.0028, -0.0007] | [-0.0371, +0.0458] | [-0.0120, +0.0064] |
| difficulty_soft | 0.9329 | 0.2428 | 0.0285 | [-0.0014, -0.0002] | [-0.0116, +0.0349] | [-0.0083, +0.0071] |
| distance | 0.9289 | 0.2490 | 0.0285 | [-0.0070, -0.0028] | [-0.0228, +0.0522] | [-0.0100, +0.0079] |
| info | 0.9260 | 0.2607 | 0.0311 | [-0.0106, -0.0054] | [-0.0290, +0.0702] | [-0.0107, +0.0115] |
| difficulty | 0.9313 | 0.2405 | 0.0311 | [-0.0037, -0.0012] | [-0.0159, +0.0534] | [-0.0108, +0.0106] |

- `no_smoke_wall` has the best point estimates (recall +0.0148, Q4 FPR
  -0.0026), but the recall and Q4 CIs cross zero and every variant loses AUC
  with a CI entirely below zero.
- Decision: **not promoted**. Hand-designed kill reliability does not clear the
  decision rule. The aggregation rule is not the main bottleneck; the next
  aggregation candidate is a learned attention/MIL pooling over kills, and the
  next feature candidate is the event stream (fire timing, hit offset).
- Artifact: `reports/reliability/metrics.json`, `player_scores.csv`; script
  `src/valveguard/scripts/analyze_reliability_weighting.py`.

## 16. Raw event features: recoil / fire / mouse (2026-10-20 pilot)

- Raw CS2CD parquet: 100 cheater-present + 100 clean matches (`with_cheater_present`,
  `no_cheater_present`), joined to the cached kill index. 20,252 valid
  256-tick windows, 1,353 players, 466 cheater players.
- 63 features from `aim_punch_angle`, `last_shot_time` / `shots_fired`,
  `usercmd_mouse_dx/dy`, scope/walk/duck/air/spotted, and velocity.
- Base = 79 summary features; 5-fold match-disjoint GroupKFold; same GBDT and
  balanced sample weights for both arms.

| Arm | Kill AUC | Player AUC | Recall@FPR1 |
|---|---|---|---|
| base | 0.8276 | 0.8668 | 0.1567 |
| base + events | **0.8459** | **0.8731** | **0.1781** |

- 500-replicate match-grouped paired bootstrap (events - base):
  - Δkill AUC **+0.0183** [0.0106, 0.0258] — significant.
  - Δplayer AUC +0.0064 [-0.0005, +0.0126] — borderline.
  - Δplayer recall@FPR1 +0.0318 [-0.0362, +0.0866] — not significant on
    1,353 players.
- Group ablation (same folds; base is the reference):

| Arm | Kill AUC | Player AUC | Recall@FPR1 |
|---|---|---|---|
| base | 0.8276 | 0.8668 | 0.1567 |
| + recoil | 0.8280 | 0.8638 | 0.1738 |
| + fire | 0.8348 | 0.8663 | 0.1695 |
| + mouse | 0.8311 | 0.8643 | 0.1502 |
| + aim (recoil+fire+mouse) | 0.8376 | 0.8674 | 0.1717 |
| + context | 0.8401 | **0.8785** | 0.1352 |
| + all | **0.8459** | 0.8731 | **0.1781** |

- Permutation importance puts `ev_spotted_frac` (0.022), `ev_spray_max`
  (0.017), and `ev_mouse_view_beta_pitch` (0.015) on top.
- Aim-only events add +0.0100 kill AUC and +0.0150 player recall over base,
  but player AUC is flat (+0.0006). The **context** group
  (spotted/walking/ducking/airborne/velocity) gives the largest player AUC
  (+0.0117) but a lower recall (-0.0215). Recoil alone adds almost nothing.
- Caveat: `spotted` is visibility/team context, not aim. The event gain is
  therefore not yet a clean aim-assist claim; the next run must separate
  aim-specific features from movement/visibility context and check whether
  the context gain survives a map/side/rank control.
- Decision: **pilot only, not promoted**. The kill-level signal is real; the
  player-level estimate needs more matches. A 200+200 raw subset is downloading
  for the next run; the full raw dataset (794 matches) is the eventual target.
- Artifact: `reports/events_pilot/metrics.json`, `bootstrap.json`,
  `ablation.json`, `importance.csv`, `oof.parquet`, `event_features.parquet`;
  scripts `build_event_features.py`, `run_event_pilot.py`,
  `bootstrap_event_pilot.py`, `run_event_ablation.py`, `event_importance.py`,
  `features/events.py`.

## 17. Raw event features: 400-match replication (2026-10-20)

- Data: 200 cheater-present + 200 clean matches; 41,888 valid 256-tick windows
  from 400 matches; the clean-label filter leaves 35,814 windows, 2,704 players,
  837 cheater players.
- Same 63 event features, same 79 base features, same 5-fold GroupKFold and
  GBDT as section 16.

| Arm | Kill AUC | Player AUC | Recall@FPR1 |
|---|---|---|---|
| base | 0.8702 | 0.9144 | 0.1505 |
| + recoil | 0.8735 | 0.9145 | 0.1541 |
| + fire | 0.8829 | 0.9158 | 0.1565 |
| + mouse | 0.8718 | 0.9136 | 0.1601 |
| + context | 0.8807 | 0.9190 | 0.1720 |
| + aim (recoil+fire+mouse) | 0.8847 | 0.9154 | **0.1852** |
| + all | **0.8904** | **0.9188** | 0.1840 |

- 500-replicate match-grouped bootstrap:
  - all vs base: Δkill AUC **+0.0202** [0.0145, 0.0258]; Δplayer AUC
    **+0.0048** [0.0009, 0.0087]; Δplayer recall +0.0314
    [-0.0183, +0.0903].
  - aim vs base: Δkill AUC **+0.0145** [0.0090, 0.0197]; Δplayer AUC
    +0.0014 [-0.0025, +0.0058]; Δplayer recall +0.0122
    [-0.0301, +0.0540].
- The kill-level gain replicates and is significant. The all-feature player
  AUC gain is now significant, but the aim-only player AUC/recall gains are
  not. Low-FPR recall remains the bottleneck.
- Shortcut audit (35,814 windows; SMD cheater - human):
  - `ev_spray_max` -0.86 (5.45 -> 1.61), `ev_punch_delta_yaw_std` -0.78,
    `ev_spotted_frac` -0.76 (0.163 -> 0.055), `ev_punch_yaw_max` -0.70,
    `ev_shots_window` -0.70 (8.42 -> 3.45), `ev_walking_frac` -0.66,
    `ev_scoped_frac` +0.60, `ev_velocity_max` +0.56.
  - Cheater kills in this subset use fewer shots, have less recoil-yaw
    movement, are spotted less, walk less, scope more, and move faster. Some
    of this is aim-relevant (fewer shots to kill), but much is role/play-style
    context. A weapon/rank/side-controlled run is required before calling
    these aim features.
- 400-match rank/weapon layers: base Q4 FPR 2.78% / Q4 recall 14.6%; all events
  3.33% / 17.9%. Event features trade Q4 precision for recall. Sniper kills
  carry ~5% FPR in both arms; auto-rifle FPR is near zero. The Q4 regression is
  a blocker for promotion.
- Decision: **replicated at kill level, not promoted at player level**. The
  full raw dataset and a context-controlled design are the next requirements.
- Artifact: `reports/events_pilot_400/` (`metrics.json`, `bootstrap.json`,
  `ablation.json`, `aim_metrics.json`, `aim_bootstrap.json`,
  `event_shortcuts.csv`, `layers.json`, `base_layers/layers.json`,
  `oof.parquet`, `event_features.parquet`).

## 18. Full-cohort event replication (2026-10-20)

- Full raw cohort: 317 cheater-present + 431 clean matches; 78,202 event
  windows built; clean-label fold input 67,067 windows over 700 matches,
  4,947 players, 1,284 cheater players.
- Same 79 base + 63 event features, 5-fold GroupKFold, GBDT.

| Arm | Kill AUC | Player AUC | Recall@FPR1 |
|---|---|---|---|
| base | 0.8794 | 0.9284 | 0.2087 |
| + recoil | 0.8818 | 0.9282 | 0.2126 |
| + fire | 0.8906 | 0.9298 | **0.2157** |
| + mouse | 0.8829 | 0.9292 | 0.1994 |
| + context | 0.8891 | **0.9327** | 0.2111 |
| + aim (recoil+fire+mouse) | 0.8937 | 0.9295 | 0.1908 |
| + all events | **0.9005** | **0.9339** | 0.2111 |

- 500-replicate match-grouped bootstrap:
  - all vs base: Δkill AUC **+0.0212** [0.0171, 0.0253]; Δplayer AUC
    **+0.0057** [0.0030, 0.0084]; Δplayer recall +0.0072
    [-0.0353, +0.0493].
  - aim vs base: Δkill AUC **+0.0144** [0.0107, 0.0183]; Δplayer AUC
    +0.0013 [-0.0014, +0.0037]; Δplayer recall -0.0024
    [-0.0404, +0.0363].
- Group read: **fire** is the best single group (recall 0.2087 -> 0.2157,
  player AUC +0.0014); **context** gives the largest player AUC (+0.0043);
  **aim-only lowers recall** (0.1908); recoil/mouse are negligible. The
  all-feature player AUC gain comes from fire + context, not recoil.
- The kill and player AUC gains replicate on the full cohort; the low-FPR
  recall gain does **not**. Aim-only features are a kill-level signal, not a
  player-level one.
- Rank/weapon layers (full cohort):
  - Q4 FPR: base 3.14%, all 2.86%, aim 2.86%; Q4 recall 24.5% / 21.4% / 20.8%.
    Q4 FPR improves slightly, but Q2/Q3 FPR worsens slightly (Q2 0.89% ->
    1.24%, Q3 0.75% -> 0.94%).
  - Sniper: FPR 6.0% / 5.6% / 5.7%, recall 25.5% / 26.5% / 25.9%.
    Auto rifle: FPR ~0%, recall 0.6% -> 3.1% (all). Pistol: FPR 0.26% ->
    0.54%, recall 4.0% -> 9.7%.
  - The sniper group remains the dominant false-positive source; auto-rifle
    and pistol gains are large in relative terms but small in absolute recall.
- Decision: **not promoted**. Event features improve ranking but not the
  primary low-FPR recall; Q4 FPR improves only slightly and Q2/Q3 regress.
  The next candidates are target-acquisition/first-shot sequence features and
  a GPU sequence model; GBDT cannot use the 5070 Ti.
- Artifact: `reports/events_pilot_full/` (`metrics.json`, `bootstrap.json`,
  `aim_metrics.json`, `aim_bootstrap.json`, `layers_base.json`,
  `layers_all.json`, `layers_aim.json`, `oof.parquet`,
  `event_features.parquet`).

## 19. Target-acquisition and first-shot features (2026-10-20)

- Added 26 features from raw attacker/victim positions: target error at
  0/-8/-16/-32/-64 ticks, time to enter 3/5/10 degrees, time on target,
  post-acquire error, first-shot-to-kill, error at first shot, and
  acquire-to-first-shot reaction.
- Full cohort: 700 matches / 67,067 clean windows / 4,947 players; 92 event
  features total, 79 base features.

| Arm | Kill AUC | Player AUC | Recall@FPR1 |
|---|---|---|---|
| base | 0.8794 | 0.9284 | 0.2087 |
| + acquire | 0.8809 | 0.9286 | 0.2150 |
| + fire | 0.8912 | 0.9297 | **0.2173** |
| + aim (recoil+fire+mouse) | 0.8942 | 0.9302 | 0.1931 |
| + aim_all (+ acquisition) | 0.8950 | 0.9306 | 0.1986 |
| + all | **0.9005** | **0.9334** | 0.2095 |

- Bootstrap (500 match-grouped):
  - all vs base: Δkill AUC **+0.0212** [0.0171, 0.0253]; Δplayer AUC
    **+0.0052** [0.0026, 0.0078]; Δplayer recall +0.0052
    [-0.0361, +0.0492].
  - aim_all vs base: Δkill AUC **+0.0157** [0.0120, 0.0195]; Δplayer AUC
    +0.0024 [-0.0001, +0.0049]; Δplayer recall +0.0034
    [-0.0390, +0.0432].
- Acquisition alone gives recall +0.0063 over base but no player-AUC gain;
  adding it to aim raises that arm from 0.1931 to 0.1986 but it remains below
  base. Fire timing is still the best single group (0.2173).
- Rank/weapon layers: Q4 FPR base 3.14% / all 2.86% / aim_all **2.29%**;
  Q4 recall 24.5% / 21.9% / 22.7%; overall recall 20.9% / 21.0% / 19.9%.
  Sniper FPR remains ~5.6-6.0%; auto rifle FPR ~0; pistol recall 4.0% ->
  9.1% (all) / 6.1% (aim_all).
- Decision: **not promoted**. Acquisition is a thin kill-level signal and does
  not fix low-FPR player recall; aim_all trades overall recall for a lower Q4
  FPR. Summary-feature engineering is now exhausted. The next step is a GPU
  sequence model over the 256-tick recoil/mouse/fire/error channels.
- Artifact: `reports/events_pilot_full_acq/` (`metrics.json`,
  `bootstrap.json`, `aim_all_metrics.json`, `aim_all_bootstrap.json`,
  `ablation.json`, `layers_base.json`, `layers_all.json`,
  `layers_aim_all.json`, `oof.parquet`, `event_features.parquet`).

## 20. GPU 1D CNN on the 43-channel windows (2026-10-20)

- The summary GBDT has exhausted its feature space. First sequence baseline:
  a 3-conv 1D CNN over the existing 43-channel, 256-tick cache.
- Training: 5-fold match-disjoint GroupKFold, 8 epochs, batch 512, AdamW
  lr 1e-3, balanced positive weight, AMP, RTX 5070 Ti.
- Heavy-tailed channels (jerk, speed) contain inf; the CNN uses robust
  median/IQR normalization and clips normalized inputs to [-10, 10].

| Model | Kill AUC | Player AUC | Recall@FPR1 | Q4 FPR | Q4 recall |
|---|---|---|---|---|---|
| GBDT champion | 0.8497 | 0.9338 | 0.2296 | **2.59%** | 22.9% |
| 43-channel CNN | **0.9046** | **0.9467** | **0.2872** | 3.11% | **30.0%** |

- 500-replicate match-grouped paired bootstrap (CNN - GBDT):
  - Δkill AUC **+0.0550** [0.0505, 0.0598] — large and significant.
  - Δplayer AUC **+0.0127** [0.0084, 0.0174] — significant.
  - Δplayer recall@FPR1 +0.0486 [-0.0108, +0.1119] — point estimate large,
    CI still crosses zero.
  - ΔQ4 FPR +0.0077 [-0.0065, +0.0224] — CNN Q4 FPR is worse in the point
    estimate, not significant.
- Per-fold player recall ranges 0.239-0.380, so the remaining recall
  uncertainty is mostly model variance. A multi-seed ensemble is the next
  test; seeds 43 and 44 are training in parallel.
- Decision: **strong challenger, not promoted yet**. The sequence model
  clearly beats the summary GBDT on kill and player AUC; the primary recall
  gain is close but not significant, and Q4 FPR regresses.
- Artifact: `reports/cnn43/metrics.json`, `comparison.json`, `oof.parquet`;
  script `src/valveguard/scripts/train_channel_cnn.py`,
  `compare_cnn_gbdt.py`.

### 3-seed ensemble (seeds 42/43/44)

- Individual seed player recall: 0.2872 / 0.2669 / 0.2786; player AUC
  0.9446 / 0.9458 / 0.9430.
- Ensemble point: kill AUC 0.9089, player AUC **0.9475**, recall 0.2794,
  Q4 FPR 3.11%, Q4 recall 31.1%.
- 500-replicate paired bootstrap vs GBDT:
  - Δkill AUC **+0.0592** [0.0546, 0.0638].
  - Δplayer AUC **+0.0137** [0.0090, 0.0187].
  - Δplayer recall +0.0535 **[-0.0056, +0.1174]**.
  - ΔQ4 FPR +0.0079 [-0.0064, +0.0224].
- The ensemble improves AUC and seed stability but does not move the recall CI
  past zero; Q4 FPR stays worse than GBDT. Not promoted.
- Next: event-channel CNN (recoil / mouse / punch / target error / first-shot
  sequence) and a larger model, then a Q4-aware calibration or second stage.

### Wide attention CNN (seed 42)

- Wider 4-conv model (96/192/256/256) with attention pooling, 10 epochs,
  5-fold, same 43-channel input.
- Point: kill AUC 0.9038, player AUC **0.9488**, recall@FPR1 **0.2911**,
  Q4 FPR 3.37%, Q4 recall 31.3%.
- 500-replicate paired bootstrap vs GBDT:
  - Δkill AUC **+0.0542** [0.0489, 0.0594].
  - Δplayer AUC **+0.0151** [0.0104, 0.0200].
  - Δplayer recall **+0.0744** **[0.0107, 0.1432]** — the first sequence
    model whose low-FPR recall CI is entirely above zero.
  - ΔQ4 FPR +0.0057 [-0.0084, +0.0205] — point regression, not significant.
- Decision: **promotion candidate**. It passes the recall/AUC rule; the Q4
  point estimate is worse, so the trade-off is recorded and a wide-seed
  ensemble plus Q4 mitigation is required before replacing the champion.
- Artifact: `reports/cnn43/wide_metrics.json`,
  `reports/cnn43_wide/comparison.json`, `wide_oof.parquet`.

### Wide 3-seed ensemble

- Wide seed player recall: 0.2911 / 0.2981 / 0.3051; player AUC 0.9488 /
  0.9471 / 0.9477.
- Ensemble: kill AUC 0.9086, player AUC **0.9501**, recall **0.2965**,
  Q4 FPR 3.11%, Q4 recall 33.5%.
- 500-replicate paired bootstrap vs GBDT:
  - Δkill AUC **+0.0590** [0.0538, 0.0639].
  - Δplayer AUC **+0.0164** [0.0117, 0.0214].
  - Δplayer recall **+0.0822** **[0.0135, 0.1532]**.
  - ΔQ4 FPR +0.0057 [-0.0091, +0.0209].
- The ensemble stabilises the recall gain; it remains a **promotion
  candidate**. The Q4 FPR point estimate is 0.5 pp worse but Q4 recall is
  33.5% vs 22.9%, so the trade-off favours recall under the project rule.
- Critical check in progress: train the same wide CNN on the **aim-only
  29-channel subset** (drop weapon, kill outcome, kill flags, flash, shot) to
  rule out a metadata/outcome shortcut before promotion.
- Artifact: `reports/cnn43_wide_ensemble/comparison.json`,
  `reports/cnn43_wide/seed43_metrics.json`, `seed44_metrics.json`.

### Aim-only 29-channel wide CNN

- Channel subset: attacker position/view/velocity, victim position/health/
  noise, angular dynamics, target errors, distance, speeds, t_norm. Dropped
  weapon one-hot, kill flags, kill outcome, flash, and shot.
- This is the fair comparison to the aim-only GBDT.

Single seed (seed 42):

| Model | Kill AUC | Player AUC | Recall@FPR1 | Q4 FPR | Q4 recall |
|---|---|---|---|---|---|
| aim-only GBDT | 0.8497 | 0.9338 | 0.2296 | 2.59% | 22.9% |
| aim-only wide CNN | **0.8979** | **0.9474** | **0.3261** | **2.59%** | **35.0%** |

- 500-replicate paired bootstrap vs GBDT:
  - Δkill AUC **+0.0483** [0.0434, 0.0534].
  - Δplayer AUC **+0.0136** [0.0087, 0.0188].
  - Δplayer recall **+0.0930** **[0.0276, 0.1618]**.
  - ΔQ4 FPR **0.0000** [-0.0090, +0.0174].
- The single seed was not a free Q4 improvement; the ensemble settles it.

### Aim-only 3-seed ensemble (fixed folds)

- Fold audit: the GBDT uses stratified match folds from
  `reports/cv_gbdt/cv_predictions.parquet`; the first CNN runs used a
  fresh `GroupKFold`, so the two were not paired. `--folds-from` was
  added and the CNN ensemble was retrained on the GBDT fold map.
- Runs: `runs/cnn29_fixed_wide_seed42/43/44`, all 29-channel, 10 epochs,
  5 fixed folds, models saved.
- Seed player recall (training-script metric): 0.2864 / 0.2887 / 0.3253;
  player AUC 0.9479 / 0.9462 / 0.9464; kill AUC 0.8983 / 0.8971 / 0.8969.
- Ensemble on the fixed folds: kill AUC **0.9021**, player AUC
  **0.9500**, recall **0.3121**, Q4 FPR **2.59%**, Q4 recall **33.9%**.
- 500-replicate paired bootstrap vs GBDT on the same folds:
  - Δkill AUC **+0.0524** [0.0476, 0.0573].
  - Δplayer AUC **+0.0163** [0.0120, 0.0207].
  - Δplayer recall **+0.0824** **[0.0208, 0.1490]**.
  - ΔQ4 FPR **+0.0021** [-0.0098, +0.0162].
- The Q4 point estimate now matches GBDT exactly (10/386) while Q4 recall
  rises from 22.9% to 33.9%. The earlier 0.3284/3.11% numbers came from
  a different GroupKFold split and are superseded.
- **Provenance note**: an even earlier "aim-only ensemble" mixed the
  29-channel seed 42 with 43-channel seeds 43/44 and reported recall
  0.2872. Those artifacts were superseded and later removed.
- Decision: **champion-elect**. Δrecall and ΔAUC are both significant and
  Q4 FPR is unchanged. Deployment artifact: 15 fold models + normalization
  under `runs/cnn29_fixed_wide_seed*`.

### Player aggregation: top-k windows

- Replacing the shrunk mean with the mean of the top-k window scores
  targets cheaters who toggle and have only a few blatant kills.

| Aggregation | Player AUC | Recall@FPR1 | Q4 FPR | Q4 recall |
|---|---|---|---|---|
| mean (current) | **0.9500** | 0.3121 | **2.59%** | 33.9% |
| top-5 | 0.9277 | **0.3230** | 3.37% | **36.1%** |

- With top-5 applied to both models: CNN recall 0.3230 vs GBDT 0.1860;
  Δrecall **+0.1146** **[0.0578, 0.1787]**, Δplayer AUC +0.0423
  [0.0347, 0.0495], ΔQ4 FPR +0.0028 [-0.0120, +0.0163].
- Top-5 buys +1.1 pp recall but costs 2.2 pp AUC and 0.8 pp Q4 FPR.
  This repeats the earlier top-k result (section 2).
- Decision: **not the default**. Mean aggregation stays the headline
  and the deployed score.
- Artifact: `reports/cnn29_fixed_deploy/`,
  `reports/cnn29_fixed_deploy_compare/`, `reports/cnn29_fixed_deploy_top5/`;
  script `src/valveguard/scripts/compare_cnn_gbdt.py` (`--topk`).

### Raw mouse / recoil / fire channels (negative)

- Added 23 per-tick raw channels (mouse dx/dy/speed/valid, recoil punch and
  its derivative, shot and shot age, stance, spotted, velocity components,
  mouse-view regression residuals and betas) for the 67,449 windows whose
  raw match is local (`build_raw_channels.py`).
- 43 + 23 = 66 channels vs 43 on the same rows, wide CNN, seed 42:
  - kill AUC +0.0037 **[0.0004, 0.0070]**.
  - player AUC +0.0011 [-0.0014, 0.0034].
  - recall +0.0183 [-0.0262, 0.0609].
  - Q4 FPR -0.0011 [-0.0136, 0.0101].
- Aim-only combinations on the same rows:

| Channels | Kill AUC | Player AUC | Recall@FPR1 |
|---|---|---|---|
| 29 aim | 0.8907 | **0.9397** | **0.2825** |
| 29 + 8 compact raw | 0.8908 | 0.9376 | 0.2825 |
| 29 + 23 raw | **0.8958** | 0.9387 | 0.2770 |

- 52 vs 29: Δkill **+0.0051 [0.0017, 0.0087]**; Δplayer AUC -0.0016
  [-0.0045, 0.0012]; Δrecall +0.0011 [-0.0465, 0.0445].
- 37 vs 29: Δkill +0.0001 [-0.0030, 0.0033]; Δrecall +0.0107
  [-0.0318, 0.0521].
- The raw stream carries a small kill-level signal but nothing at the
  player level, consistent with the summary event ablation (section 18).
  Mouse-view residual and recoil control are already captured by the
  derived aim channels well enough for this cohort.
- Decision: **not adopted**. Keep the 29-channel aim-only representation;
  do not spend a full raw pipeline build on these channels.
- Artifact: `reports/cnn66/`, `reports/cnn43_subset/`,
  `reports/cnn29_subset/`, `reports/cnn37_subset/`, `reports/cnn52_subset/`,
  `reports/cnn66_vs_43/`, `reports/cnn52_vs_29/`, `reports/cnn37_vs_29/`;
  scripts `build_raw_channels.py`, `merge_channel_caches.py`.

### Dataset demo report (sanity check)

- `demo_report_dataset.py` reads the cached 256-tick CS2CD windows for one
  match, selects the 29 aim channels by name, applies the saved
  normalization, and averages the 15 fold models across three seeds. No raw
  demo file is needed, and player names are the dataset pseudonyms.
- `cdata:0` (Mirage, 78 kill windows, 10 players): **1 FLAG** at the 0.608027
  human-p99 threshold. `Player_3` (dataset label 1) scores 0.709, human
  percentile 99.8, with four behavior reasons. The second labelled cheater,
  `Player_4`, scores 0.399 (p95.1) and lands in REVIEW but is not flagged.
- `ncdata:36` (Dust2, 232 kill windows, 10 players): **0 FLAG / 0 REVIEW**;
  top shrunk score 0.227 (p67.5). All ten players are below the human p95.
- This is a path and presentation check, not a held-out metric: the artifact
  fold models overlap the demonstration matches. Held-out numbers are the
  fixed-fold table above.
- Artifact: `reports/demo_dataset/cdata_0/`, `reports/demo_dataset/ncdata_36/`,
  combined `reports/demo_dataset/review_queue.csv` and `summary.json`;
  script `src/valveguard/scripts/demo_report_dataset.py`.

### Player-level MIL (negative)

- Trained the wide encoder with attention pooling directly on the 5,868
  player labels, five fixed match folds. Two variants: warm start from the
  window fold model with the encoder unfrozen, and a frozen-encoder
  attention/head probe.
- Compared with the window ensemble + mean (AUC 0.9500, recall 0.3121,
  Q4 FPR 2.59%):

| Variant | Player AUC | Recall@FPR1 | Q4 FPR | Q4 recall |
|---|---|---|---|---|
| frozen encoder | 0.9478 | 0.3175 | 5.44% | 41.0% |
| warm start, unfrozen | 0.9329 | 0.2856 | 3.63% | 36.5% |

- Frozen vs mean: ΔAUC -0.0024 [-0.0066, +0.0021]; Δrecall +0.0148
  [-0.0765, +0.1151]; ΔQ4 FPR **+0.0232 [0.0023, 0.0455]**.
- Warm vs mean: ΔAUC **-0.0172 [-0.0231, -0.0108]**; Δrecall -0.0316
  [-0.1147, +0.0511]; ΔQ4 FPR +0.0090 [-0.0126, +0.0283].
- Player-level supervision has 5,868 examples against 79,533 window labels;
  attention pooling adds variance and, when unfrozen, degrades the window
  representation. The mean of window scores remains stronger.
- Decision: **not adopted**. Keep window training + mean aggregation; the
  next lever is the window loss, not the pooling.
- Artifact: `reports/mil29/`, `reports/mil29_frozen_vs_ensemble/`,
  `reports/mil29_warm_vs_ensemble/`; script
  `src/valveguard/scripts/train_player_mil.py`.

### Hard-negative weighting (new champion)

- The low-FPR metric cares about the top of the human score distribution.
  Negative windows are weighted by `1 + alpha * p^2` during BCE, so
  confidently-human windows carry more loss. `alpha=5` was selected after
  `alpha=2` and `alpha=10` probes; all arms use the fixed folds.
- Single-seed recall (training-script metric): 0.3183 / 0.3416 / 0.3440
  for seeds 42/43/44; `alpha=10` seed 42 gives 0.3479, a negligible gain
  over `alpha=5`, so `alpha=5` is kept as the less aggressive setting.
- 3-seed `alpha=5` ensemble vs GBDT:
  - kill AUC **0.9015**, player AUC **0.9502**, recall **0.3619**,
    Q4 FPR **3.11%**, Q4 recall **39.1%**.
  - Δkill **+0.0519** [0.0470, 0.0568].
  - Δplayer AUC **+0.0165** [0.0120, 0.0211].
  - Δrecall **+0.1223** **[0.0476, 0.1988]**.
  - ΔQ4 FPR +0.0030 [-0.0090, +0.0161].
- Against the previous 3-seed mean ensemble (recall 0.3121, AUC 0.9500):
  Δrecall **+0.0459 [0.0048, 0.0883]**, ΔAUC +0.0002
  [-0.0014, +0.0016], ΔQ4 FPR +0.0048 [-0.0057, +0.0158].
- Decision: **new champion**. The recall gain is significant, AUC is
  unchanged, and the Q4 FPR increase is 2 extra false positives out of
  386 Q4 humans (CI crosses zero) for +11 Q4 true positives.
- Artifact: `reports/cnn29_hn5_ensemble/`,
  `reports/cnn29_hn5_ensemble_vs_gbdt/`,
  `reports/cnn29_hn5_ensemble_vs_base/`; flag `--hard-neg 5` in
  `train_channel_cnn.py`.

### Visibility-onset reaction channels (negative)

- Rebuilt the raw channel cache with five visibility channels from
  `approximate_spotted_by`: attacker-sees-victim, time since first seen,
  onset flag, victim team-spotted, and time since victim was spotted.
- On the 67,449 raw-covered rows, aim 29 + reaction 5 = 34 channels vs the
  29-channel baseline, both with hard-negative BCE alpha=5, fixed folds:
  - 34ch: kill AUC 0.8976, player AUC 0.9461, recall 0.3136, Q4 FPR 3.13%.
  - 29ch: kill AUC 0.8901, player AUC 0.9420, recall 0.3315, Q4 FPR 2.56%.
- Paired bootstrap:
  - Δkill **+0.0074** [0.0048, 0.0101].
  - Δplayer AUC **+0.0040** [0.0019, 0.0060].
  - Δrecall -0.0019 [-0.0394, +0.0366].
  - ΔQ4 FPR +0.0049 [-0.0048, +0.0165].
- First-visibility reaction adds a real kill-level signal but does not move
  player-level low-FPR recall on this cohort. The derived angular error and
  acquisition channels already capture most of it.
- Decision: **not adopted**. `raw_channels_react` is kept for future
  longitudinal work where per-attacker visibility can be measured more
  precisely.
- Artifact: `reports/cnn34_react/`, `reports/cnn29_subset_hn5/`,
  `reports/cnn34_vs_29/`; code in `build_raw_channels.py`.

### XGUARDIAN processed data audit (not usable for fair validation)

- Downloaded the three processed CSVs from the XGUARDIAN artifact
  (`Train/Validation/Test_cleaned.csv`, 527 MB total). Each row is a
  96-tick elimination window; labels are window-level `isCheater`.
- Split sizes: 10,334 / 10,569 / 11,068 windows; 2,385 / 2,760 / 2,900
  positive windows; 971 / 944 / 988 matches.
- HistGradientBoosting on 28 summary features (angle change, speed,
  acceleration, distance, firing, stance, path):
  - test window AUC 0.702, player AUC 0.858, recall@1% 3.1% / 0.8%.
  - aim-only window AUC 0.620; distance-only 0.589.
- **Activity confound**: cheater player-matches have 22.1 windows on
  average vs 4.6 for humans. Using window count alone as the score gives
  player AUC **0.9705**, higher than any feature model. Within window-count
  bins the model AUC falls to 0.44-0.68; sampling five windows per player
  gives window AUC 0.743.
- CS2CD control: window count alone gives player AUC 0.499 and recall
  6.8% at 1% FPR, and the champion stays at AUC 0.88-0.96 within count
  bins. The confound is specific to the XGUARDIAN processed sample.
- The processed `angle` is a coarse 2D direction derived from positions,
  not the paper's pitch/yaw trajectory. Raw demos (585 GB, 13 parts) would
  be needed to extract all ten players and the full aim channels.
- Decision: **do not use the processed CSV for external validation**.
  Prefer requesting the HAWK demo datasets, or download a raw XGUARDIAN
  part and label all players in the match rather than only the tracked one.
- Artifact: `reports/xguardian_external/metrics.json`; script
  `src/valveguard/scripts/evaluate_xguardian.py`.

### XGUARDIAN raw prefix audit (not worth the full download)

- Downloaded the first 2 GB of raw part 1 and decompressed the zstd
  prefix with a 1 GB window. The 2 GB yielded 21 demo zips; 20 were
  complete and 13 demos were unpacked.
- Only 9 of the 20 demos appear in the processed label CSVs. They contain
  8 labeled humans and 1 labeled cheater. Raw demos do not carry
  per-player labels; labels come from the processed tracked-player rows.
- Champion scores on the 9 labeled demos:
  - labeled cheater: 52 kills, 69% headshots, score 0.304, p89.5,
    rank 4/10, **not flagged**.
  - 8 labeled humans: 0.235-0.371, p71-94, none flagged.
  - no false positives, one false negative; one positive is anecdotal.
- Visibility check on the labeled cheater: pre-aim 5.4%, unseen-moving
  4.1%, longest run 220 ticks, lower than the earlier banned-demo case and
  its control. No clear wallhack signature.
- The raw archive adds full telemetry for the tracked players, not more
  labels. The same activity confound applies (cheater player-matches have
  22 windows vs 4.6 for humans), so downloading 48 GB or 585 GB would not
  produce a fair external validation set.
- Decision: **stop the XGUARDIAN raw download**. Prefer the HAWK data
  request or a directly collected labeled cohort.
- Code fixes from this audit: missing event fields (`headshot`,
  `penetrated`, `is_warmup_period`, NULL victim ids) no longer crash demo
  parsing.

### Player-level outcome/behavior fusion (negative for the primary metric)

- Built player-level features from kill outcomes: headshot rate, shots per
  kill, kill-per-shot efficiency, 4-second multikill bursts, burst share,
  minimum/median kill gap, weapon mix, sniper headshot rate, smoke/wallbang/
  midair rates, and full-health kill rate. Absolute kill count is included
  but is not predictive (AUC 0.499).
- Univariate player AUC: kill-per-shot 0.895, headshot rate 0.806, sniper
  share 0.808, wallbang rate 0.788, burst share 0.679, max 4s multikill
  0.628, median gap 0.244 (inverse).
- OOF HistGradientBoosting behavior model on the fixed folds:

| Score | Player AUC | Recall@FPR1 | Q4 FPR | Q4 recall |
|---|---|---|---|---|
| aim champion | **0.9500** | **0.3121** | **2.59%** | 36.1% |
| behavior only | 0.9196 | 0.1572 | 2.85% | 18.4% |
| rank fusion 0.75 champion | 0.9519 | 0.2887 | 3.11% | 33.9% |
| rank fusion 0.50 | 0.9472 | 0.2545 | 3.63% | 29.4% |

- 500-replicate match-grouped bootstrap for the best fusion vs champion:
  ΔAUC -0.0049 [-0.0181, +0.0032]; Δrecall -0.0554 [-0.1465, +0.0072];
  ΔQ4 FPR +0.0051 [-0.0105, +0.0236].
- The behavior score has high global AUC but poor low-FPR recall: its top
  1% tail is not cheat-specific. Of 884 champion misses, behavior catches
  70, but the union raises FPR.
- High-kill/high-headshot subgroup (>=15 kills, >=60% HS): champion recall
  67.4%, behavior 27.5%. The champion already catches most of these; the
  fusion would not fix the remaining high-kill misses.
- Decision: **not adopted as the default score**. Headshot rate and burst
  patterns are useful as a human-review queue alongside the aim score, not
  as a ranking replacement.
- Artifact: `reports/behavior_fusion/metrics.json`, `univariate.csv`;
  scripts `analyze_behavior_fusion.py`, `build_kill_ticks.py`.

### Demo report behavior context (review only)

- The demo report now shows per-player HS%, shots per kill, maximum
  4-second multikill, burst share, median kill gap, and sniper share.
  These columns are descriptive; the risk score, threshold, and flag are
  unchanged.
- On `cdata:0` the flagged `Player_3` has HS 96.6%, 1.9 shots per kill,
  4 kills inside four seconds, and 35.7% burst share, so the context shows
  why the flag is worth review. The second labelled cheater, `Player_4`,
  has HS 42.1%, 1.3 shots per kill, and no burst; only the efficiency rule
  fires, and the player stays in REVIEW.
- The clean match shows ordinary human values (HS 15-46%, 1.7-6.0 shots per
  kill, at most 2 in four seconds). One clean player still triggers two
  reasons, which is why reasons order the queue and never change the flag.
- Artifact: `reports/demo_dataset/*/players.png`, `behavior.png`,
  `kill_windows.png`, `result.json`.

### Review queue ordering (review-only)

- Added `review_queue.csv` and a review priority order to the demo report.
  The flag rule is unchanged: FLAG is still aim score >= human p99.
- Order: FLAG first; within REVIEW, more behavior reasons first, then aim
  score; below-threshold players last by aim score. Behavior reasons never
  move a player across tiers.
- Reason rules: HS >= 80%, <= 1.5 shots/kill, 4-second multikill >= 3,
  burst >= 30%, sniper headshots, wallbang/smoke >= 20%.
- `cdata:0`: `Player_3` is FLAG with 4 reasons (HS, smoke, 4s multikill,
  burst) and queue rank 1. `Player_2` (dataset label 0) is REVIEW with 2
  reasons and ranks above the second labelled cheater, `Player_4`, who has
  1 reason. Behavior ordering is not a label and can put a clean player
  ahead inside the review band.
- `ncdata:36`: no FLAG or REVIEW; all players sit below the human p95 and
  the queue is ordered by aim score, with one diffuse 4s-multikill reason.
- Artifact: `review_queue.csv`, `behavior.png`, `result.json`; the
  `behavior.png` table uses short reason codes.

### Batch review queue

- Directory mode now writes a combined `review_queue.csv` at the output
  root with `queue_rank`, `demo`, per-demo rank, tier, score, reason count,
  and full reasons, plus `summary.json`.
- Ordering is identical to the single-demo queue: FLAG first, REVIEW by
  reason count then aim score, below-threshold players by aim score.
- Two-match check (`cdata:0` + `ncdata:36`): the flagged cheater ranks
  first; the two REVIEW players follow; all below-threshold players across
  both matches are ordered by aim score, not by behavior reasons.
- Artifact: `reports/demo_dataset/review_queue.csv`, `summary.json` (CS2CD
  pseudonyms only).

### Regression tests for the demo path

- `python -m pytest -q tests` -> 11 passed.
- Covered: artifact load and deterministic scoring, missing-channel and
  checksum rejection, behavior context aggregation, review reason rules,
  single-demo review ordering, and batch queue merging.
- Artifact tests skip automatically when `artifacts/valveguard_cnn_v1` is
  absent, so the repository tests do not depend on the local binary bundle.

### ONNX runtime bundle

- `export_onnx.py` wraps the 15 fold models with their per-seed
  median/IQR normalization and sigmoid averaging, then exports one
  batch-dynamic graph (opset 17) taking `(N, 256, 29)` and returning
  `(N,)` kill scores.
- PyTorch vs ONNX parity: random channels max abs difference `2.58e-8`;
  `cdata:0` real windows `1.27e-7`; per-kill scores in the demo report
  differ by at most `1.0e-6`, with identical players, tiers, and flags.
- Bundle: 31.5 MB `model.onnx`, 28 MB zip, manifest with SHA256 checksums,
  channel names, human reference distributions, and README.
- `load_detector` selects the ONNX loader when the bundle manifest says
  `format: onnx`, so the existing demo scripts need only
  `--artifact artifacts/valveguard_cnn_v1_onnx`.
- CPU timing for 78 windows: 401 ms PyTorch vs 520 ms ONNX Runtime. The
  ONNX path is for portable deployment, not speed on this machine.
- Tests: `tests/test_onnx_artifact.py` covers load, parity, aggregation,
  and missing-channel rejection. Full suite: 15 passed.
