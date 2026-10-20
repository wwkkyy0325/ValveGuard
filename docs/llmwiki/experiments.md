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
