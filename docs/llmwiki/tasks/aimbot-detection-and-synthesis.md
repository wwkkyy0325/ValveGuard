# Task: CS2 Aim-Assist Detector (detector-first)

Status: in progress (2026-10-17)
Detailed spec: [`../model-design.md`](../model-design.md)

## Success Criteria

- Detect aim-assist from 256-tick kill windows on CS2CD.
- Player-level ROC-AUC >= 0.85 as the first gate, then target >= 0.93 if the sequence model supports it.
- Report FPR@95TPR, recall@FPR=1%, calibration, and match-disjoint splits.
- No claim of person-disjoint evaluation; CS2CD IDs are per-match pseudonyms.
- One-command reproduction; fixed seed; split assertions.

## Phase 0 — Environment and Data

- [x] Create project `.venv`; install torch 2.6.0+cu124 and data stack.
- [x] Verify CUDA: RTX 3060 Laptop, 6 GB.
- [x] Inspect raw CS2CD sample: `usercmd_mouse_dx/dy`, labels, `rank`, events.
- [x] Verify window mapping: `file_0/Player_3/kill_0` -> raw match, row 223 = kill tick.
- [x] Verify pitch/yaw/position reconstruction from the 44 features.
- [x] Quantify the pitch-sign bug: kill-tick error 6.38 deg -> 0.24 deg after inversion.
- [ ] Download the full 9.4 GB `context_windows_256.tar`.
- [ ] Build the float16 memmap cache and index.
- [ ] Audit label/source/match counts and mixed-label matches.

## Phase 1 — Features

- [x] Geometry: signed deltas, target-relative errors, distance, lead-lag.
- [x] Dynamics: speed/accel/jerk, SPARC, LDLJ, PSD slope, entropy, submovements, curvature.
- [x] Per-tick channel matrix and flat summary features.
- [x] Drop the published `abs` deltas and mis-signed pitch-head delta.
- [ ] Validate distributions by weapon/distance/map on the full data.
- [ ] Confirm no infinite values and document missing-value policy.

## Phase 2 — Split and Baseline

- [x] Match-disjoint split implementation with hard assertions.
- [x] Assign the full-data split (seed 42, 70/15/15).
- [x] Train `HistGradientBoostingClassifier` on summary features.
- [x] Report AUC/AP/F1/FPR@95TPR/recall@FPR=1%/ECE; player-level mean and max.
- [x] Gate: test player-level AUC >= 0.85 (0.943); passed.

## Phase 3 — Sequence Model

- [x] Build the 256-tick channel tensors (float16 memmap).
- [x] Train a 4-layer Transformer (d=64/128, 4 heads, FF=256) with AMP.
- [x] Compare with the GBDT on the same split; no test-set tuning.
- [~] Gate: Transformer is tied with GBDT (0.9406 vs 0.9430 player AUC); calibrated ensemble improves to 0.9497. Keep both.

## Phase 4 — Minimal Ablations

- [x] Aim-only / relative-only Transformer and metadata-only / shuffled-label audits.
- [x] Single-split match-grouped bootstrap CI.
- [x] 5-fold match-disjoint GBDT CV (honest primary estimate).
- [x] 5-fold CV for the relative-only Transformer (player AUC 0.9292; recall@FPR1 0.0911; GBDT is better at low FPR).
- [~] Map-holdout evaluation done for GBDT; Transformer map-holdout pending.

- [ ] Signed vs raw deltas.
- [ ] Corrected vs published target errors.
- [ ] Smoothness features on/off.
- [ ] Kill/event flags on/off.
- [ ] Player aggregation mean vs max.
- [ ] Metadata-only audit (map/weapon/kill flags) and shuffled-label audit.

## Phase 5 — Only If Stable

- [ ] 64-tick micro crop; 1024-tick macro only if the 256 model leaves clear macro errors.
- [ ] Skill stratification using `rank` (requires per-match JSON).
- [ ] One-sided skill-conditioned likelihood ratio.
- [ ] HMM aggregation over kills.
- [ ] Raw 100-match subset for mouse–view residual.

## Deferred

- [ ] Conditional trajectory generator.
- [ ] Map raycast / true visibility.
- [ ] Cross-dataset transfer.
- [ ] True person-disjoint evaluation (needs stable IDs).

## Phase 5 — Paused (2026-10-17)

- [x] Detector v1 frozen: aim-only GBDT, 5-fold player AUC 0.9331 / recall@FPR1 0.1735.
- [x] Batch 1 (28 variants) and motor-control features evaluated; no significant challenger.
- [x] Audits: metadata-only, shuffled labels, map holdout, calibration, subgroup FPR, rank strata.
- [x] Ten figures and `reports/REPORT.md`.
- [ ] Monday: lock 20% holdout and re-run baseline on the development 80%.
- [ ] Monday: H1 full 43-channel Transformer nested CV.
- [ ] Monday: H2 1024-tick context (GBDT first, Transformer only if GBDT passes).
- [ ] Monday: H3 high-rank negatives / group DRO pilot.
- [ ] Monday: final holdout evaluation, report, stop.

## Phase 6 — Resumed on the RTX 3060 (2026-10-17)

- [x] Lock a stratified 20% match holdout (158 matches / 15,623 windows).
- [x] Re-run the aim-only GBDT on the development 80% with the original folds
      and holdout matches removed from training (player AUC 0.9294,
      recall@FPR=1% 0.1475).
- [x] H1: full 43-channel Transformer on the development 80%
      (0.9251 / 0.2297; Δrecall CI crosses zero; not promoted).
- [x] H2 pilot: 1024-tick context on the 160-match raw subset
      (ΔAUC [-0.0055, 0.0078]; rejected).
- [x] H3: high-rank group-DRO pilot (Q4 FPR 4.76% unchanged; rejected).
- [x] Final holdout evaluation (player AUC 0.9384, recall@FPR=1% 0.3091).
- [x] Within-player self-baseline diagnostic (dev-80 only): kill AUC
      0.847 -> 0.921, but no player-level recall gain and no Q4 FPR change;
      not promoted.
- [x] Professional-demo specificity: 7 HLTV demos, 1,020 windows, 70 players;
      0/70 players flagged at 1% FPR; pro-negative augmentation has no effect.
- [x] Q4 false-positive audit: fixed the player-level threshold, found the
      low-kill concentration, and added empirical-Bayes player shrinkage
      (dev80 recall 0.1475 -> 0.2386, Q4 FPR 13.2% -> 5.3%).
- [x] Label-noise audit: out-of-match scoring of the excluded cdata label-0
      windows (5.5% estimated cheater admixture, human-like tail) and a
      per-kill audit of the two residual Q4 false positives.
- [x] Rank-stratified FPR replication on all dev-80 matches via JSON ranks:
      Q4 FPR 3.37% (386 clean), 2.59% shrunk; supersedes the 19% subset
      estimate.
