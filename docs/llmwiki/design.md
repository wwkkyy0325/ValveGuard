# ValveGuard — Design & Research Notes

Status: detector-first (2026-10-17)

> Detailed, implementation-ready model specification: [`model-design.md`](model-design.md). It supersedes the model sketch below where they differ. Key updates: raw CS2CD is the primary dataset because it contains `usercmd_mouse_dx/dy`; the 96-tick window is replaced by a 64/256/1024/match multi-scale pyramid; CS2CD player IDs are per-match pseudonyms, so evaluation is match-disjoint (a true person-disjoint check needs a stable-ID dataset).
>
> **Scope change:** the conditional generator is deferred. Phase 1 builds and validates the detector only. The detector's feature learning and human-likelihood calibration will make the later generator easier.

## 1. Goal

Build a CS2 aim-analysis system with two related capabilities:

1. **Detection**: given a player's crosshair/view-angle trajectory around a kill, estimate whether the movement is consistent with human behavior or with aim assistance.
2. **Synthesis**: given a start aim point and an end aim point (plus context), generate an intermediate trajectory that is statistically human-like.

A single conditional generative model can serve both goals: sample from it for synthesis, and use its likelihood (or a supervised score) for detection. A dedicated discriminative detector is still expected to be stronger when labeled cheat data is available, so the working plan is a **two-model system with a shared preprocessing layer**.

Non-goals (for now):

- Real-time client-side anti-cheat.
- Detecting cheat types that leave no aiming artifact (e.g. wallhack-only, information sharing).
- Automatic bans. Output is a suspicion score plus evidence for human review.

## 2. Key Constraint: What "Mouse Trajectory" Means from a Demo

CS2 demos expose per-tick player state (`pitch`, `yaw`, positions, velocities) via `demoparser2` / `awpy2`, not raw 1000 Hz mouse input. Derived data is therefore **view-angle trajectory at tick rate (64 or 128 Hz)**, which is a scaled projection of mouse movement (scale depends on the player's sensitivity). Consequences:

- Coarse timing signals (reaction time, snap duration, overshoot, correction count, curvature) are observable.
- High-frequency human tremor and raw input jitter are not observable.
- A deployment that needs raw mouse telemetry must collect it client-side; this is out of scope for the demo-only v1.

The public CS2 datasets listed below share the same limitation, so training and evaluation are consistent with demo-only deployment.

## 3. Data Sources

| Source | Content | Labels | Size / Access | Notes |
|---|---|---|---|---|
| **CS2CD raw (primary)** | 795 CS2 matches; per-tick pitch/yaw **and raw mouse deltas** (`usercmd_mouse_dx/dy`), positions, weapon, economy, rank | 317 matches with manually reviewed cheaters; 478 matches with no VAC-banned player | ~48.9 GB, CC-BY-4.0, Hugging Face | Primary dataset. Non-cheater labels inside cheater matches have only ~55.6% precision and must be excluded. |
| Kaggle `emstatsl/csgo-cheating-dataset` | View angles around engagements: `(players, 30, 192, 5)` = `AttackerDeltaYaw`, `AttackerDeltaPitch`, `CrosshairToVictimYaw`, `CrosshairToVictimPitch`, `Firing` | 10,000 clean, 2,000 cheaters | ~1.4 GB, public | CS:GO-era, precomputed windows, provenance/label quality unverified. Cross-domain baseline only. |
| XGuardian (USENIX Security '26) | CS2, real 5EPlay matches; elimination windows with pitch/yaw; raw demos released across 13 Zenodo records | Manually re-verified cheater/normal per match | ~585 GB raw + 22 GB artifact; GPL-3.0-or-later | State of the art for CS2 aim-assist detection, but too large for the current disk. Use only a subset for external validation. |
| RekaCS2-10k / OpenCS2 | Pro HLTV demos, rendered first-person video + per-tick parquet (`mouse_x_delta`, `mouse_y_delta`, yaw, pitch, position, actions) | None (all human/pro) | Reka: 10k+ hours, CC BY-NC 4.0, sample subset on HF. OpenCS2: ~11 TB total, tick sidecars much smaller, CC-BY-4.0 | Pro-only human reference distribution and generator training. Skill mismatch with average players. |
| PureSkill.gg CS2 dataset | CS2 demos via Steam/FACEIT APIs, mixed skill | None | Application/request, documented ethics | Potential non-pro human baseline. |
| HLTV raw demos | Pro matches | None | Scraping/terms risk | Upstream source for the rendered datasets. |
| Self-collected | Own/community demos; optional private-server sessions with open-source cheats | Controllable | Small | Ethical/legal review required; keep offline, never on live servers. |

Practical note: Kaggle and Zenodo access from mainland China may be unreliable. Data acquisition is a Phase 0 gate; alternatives are a mirror, a user-provided download, or starting with the Kaggle dataset.

## 4. System Architecture

```
CS2 demo / dataset
        │
        ▼
 Preprocessing (shared)
   - kill event extraction
   - elimination window (m ticks before, n after)
   - pitch/yaw → screen coordinates
   - target-relative offsets (victim position)
   - per-player, per-match normalization
        │
        ├──────────────► Detection model A (discriminative)
        │                   input: window features
        │                   output: P(aim-assist), feature attributions
        │
        ├──────────────► Generative model B (conditional)
        │                   input: start aim, target aim, context, skill
        │                   output: trajectory samples + likelihood
        │
        └──────────────► Skill estimator
                            input: non-kill behavior, rank/rating if available
                            output: skill embedding / cohort
                                   │
                                   ▼
                    Skill-aware calibration & aggregation
                    - one-sided "superhuman" score
                    - match/player-level aggregation
                    - charts + evidence report
```

### 4.1 Detection model A

Input per elimination window: a sequence of T ticks with features such as

- `Δyaw`, `Δpitch` per tick; angular velocity and acceleration.
- Angular offset from crosshair to victim (yaw/pitch components, distance).
- Path efficiency, curvature, overshoot, number of direction reversals.
- Time from target entering the angular FOV to firing (proxy reaction time).
- Firing / hit indicator, weapon class, target distance, player speed.

Model: small 1D-CNN + GRU (XGuardian's INSPECTOR design) or a TCN/Transformer encoder. Window-level logit, aggregated to player level by an attention/logistic head. Compare with an unsupervised baseline (autoencoder / normalizing flow on human windows).

Training protocol:

- Split by **player**, not by match, to prevent identity leakage.
- Report both in-domain and cross-dataset performance.
- Control for skill: report FPR on held-out pro and amateur cohorts separately.

### 4.2 Skill-aware suspicion

The reference population is not "average human" but "human with this player's skill". Two-stage approach:

1. Estimate a skill embedding from behavior that is not the target trajectory (crosshair placement, idle aim stability, reaction time on easy targets, movement/utility statistics) and/or external rank.
2. Model the conditional human distribution `p(features | skill, context)`. Score the **upper tail** (better-than-expected direction) rather than generic deviation:

```
superhuman_score = -log P(feature >= observed | skill, context)
```

A weaker-than-expected deviation should not increase suspicion; it is used to update the skill estimate and can be mildly exculpatory. A trajectory matching a pro is normal for a pro but highly suspicious for a low-skill player. This is the user's asymmetric weighting idea in a statistically explicit form.

Implementation options: quantile regression, conditional normalizing flow, or per-skill-cohort empirical CDFs. Calibrate the final score on labeled data and report reliability diagrams.

### 4.3 Generative model B

Condition: start aim point, target aim point, target distance/width, weapon, time budget, skill embedding. Output: sequence of `(Δyaw, Δpitch)` at fixed dt, with exact endpoint conditioning.

Candidate architectures, in increasing complexity:

1. Parametric baseline: minimum-jerk / Fitts-law submovement model + signal-dependent noise + corrective submovements.
2. Conditional VAE or autoregressive mixture-density network (easy likelihood, stable training).
3. Conditional flow matching / diffusion over trajectories (best multimodality and realism; heavier).

Endpoint conditioning: project the final sample onto the target tolerance, or use inpainting/guidance during sampling.

Two uses:

- **Synthesis**: sample multiple human-like paths for a given endpoint pair.
- **Detection aid**: compute trajectory likelihood / percentile under the human model, and combine with model A.

### 4.4 Aggregation and decision

Window score → match score (attention or calibrated logistic regression) → player risk. Output must include:

- score and calibrated probability,
- top contributing features (SHAP),
- trajectory overlays against human percentile bands,
- an explicit "insufficient evidence" state.

## 5. Evaluation

Detection:

- ROC-AUC, PR-AUC, F1, FPR at 95% TPR.
- Player-level and match-level metrics; per-kill metrics as diagnostics.
- Skill-stratified FPR (pro / amateur / low-skill).
- Cross-dataset generalization (Kaggle ↔ XGuardian).
- Adversarial test: humanized/smoothed aimbot trajectories.

Generation:

- Endpoint error (degrees).
- Time-to-target, peak velocity, velocity profile, curvature, submovement count, jerk distributions vs human.
- Discriminator balanced accuracy (a strong detector should be near chance on generated data; GAN-Aimbots reported ~86% for its best detector on a simpler setting).
- Distribution distance (e.g. Fréchet distance on summary features).

Calibration:

- Reliability diagram, ECE.
- Sensitivity of the suspicion threshold to the skill estimator.

## 6. Charts / Deliverables

1. Target-relative 2D trajectory over human density contours (pro and same-skill cohort).
2. Per-tick angular speed / acceleration / jerk vs human percentile envelope.
3. Per-kill feature attribution (SHAP) bar chart.
4. Player radar of feature percentiles vs cohort.
5. Match timeline of suspicion scores with kill markers.
6. Population scatter: skill estimate vs superhuman score (normal / pro / cheat).
7. Model evaluation: ROC/PR, calibration, cross-dataset heatmap.
8. Generator validation: human vs generated distributions and discriminator ROC.

## 7. Main Risks

- **Label quality**: Kaggle labels are unverified; XGuardian labels are manually verified but access/licensing is unconfirmed.
- **Leakage**: splitting by match leaks player identity; split by player.
- **Skill confound**: models may learn "bad player" instead of "cheater"; skill conditioning and matched evaluation are mandatory.
- **Demo artifacts**: tick rate, server settings, and dataset preprocessing can be learned as shortcuts; normalize and test cross-dataset.
- **Target visibility**: demos do not encode whether the victim was visible. Target-relative features use geometry only; wallhack-related pre-aim analysis needs a visibility proxy.
- **Raw mouse unavailability**: high-frequency signals are lost; state this in every accuracy claim.
- **Licensing**: Reka is CC BY-NC 4.0 (non-commercial); HLTV demos are bound by tournament terms; XGuardian license TBD.
- **False positives**: pro players are the hardest negatives; any deployment requires human review.

## 8. References

- GAN-Aimbots: Kanervisto et al., "GAN-Aimbots: Using Machine Learning for Cheating in First Person Shooters", IEEE Transactions on Games, 2022. https://arxiv.org/abs/2205.07060 — conditional GAN that generates human-like aim; also evaluates detectors. ViZDoom, not CS2.
- XGuardian: Zhang et al., "XGuardian: Towards Generalized, Explainable and More Effective Server-side Anti-cheat in First-Person Shooter Games", USENIX Security '26. https://www.usenix.org/conference/usenixsecurity26/presentation/zhang-jiayi — CS2 detection from pitch/yaw elimination windows; 0.889 accuracy, 0.062 FPR on their CS2 split.
- HAWK: "Identify As A Human Does: A Pathfinder of Next-Generation Anti-Cheat Framework for First-Person Shooter Games". https://arxiv.org/abs/2409.14830
- Kaggle CS:GO cheating dataset: https://www.kaggle.com/datasets/emstatsl/csgo-cheating-dataset
- RekaCS2-10k: https://reka.ai/labs/research/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset
- OpenCS2: https://huggingface.co/datasets/blanchon/opencs2_dataset
- PureSkill.gg CS2 dataset: https://docs.pureskill.gg/datascience/adx/cs2/csds/
- demoparser2 / awpy2: https://github.com/LaihoE/demoparser , https://github.com/pnxenopoulos/awpy
- Endpoint-conditioned trajectory generation (desktop cursor): https://github.com/loong-coder/lstm_cvae_mouse_trajectory
- Submovement-based mouse input cheating detection (patent): https://patents.google.com/patent/US20230244329A1/en

## 9. Decisions and Remaining Questions

Resolved:

1. Input scope: server-side demo, but using CS2CD's raw `usercmd_mouse_dx/dy` at 64 Hz (not 1000 Hz client telemetry). Demo-first.
2. Primary dataset: CS2CD raw, CC-BY-4.0, ~48.9 GB. Proxy is available; disk has ~104 GB free. XGuardian raw (585 GB) is out of scope.
3. Compute: RTX 3060 6 GB. The detector must stay in the 3–6 M parameter range; use AMP and patchified macro windows.
4. Window: 96 ticks is too short alone. Use the 64/256/1024/match pyramid.
5. Skill labels: `rank` (Premier/CS rank) is present per tick and `CSstats_info.avg_rank` per match. Use for conditioning and stratified evaluation; estimate behaviorally when missing.
6. Primary reference papers: XGuardian (detection), AntiCheatPT/CS2CD (dataset + baseline), GAN-Aimbots (generation + detection), Same-Player Verification (style/disjoint protocol).

Remaining:

1. Synthesis endpoints: use 2D view angles (yaw, pitch) plus target angular size; confirm whether the user also wants screen-pixel endpoints.
2. Responsible release: detector/code open; generator weights restricted or released with a dual-use statement. Confirm the intended release policy.
3. Target visibility: no raycast in v1; use a geometry/FOV/smoke proxy and state the limitation. Is a map-raycast module in scope for v2?

## 10. Current State (v2, 2026-10-17)

- Champion: aim-only GBDT on 59 window features; 5-fold match-disjoint OOF
  player AUC 0.9331, recall@FPR=1% 0.1735, kill AUC 0.8497.
- Locked 20% match holdout (158 matches); the champion fitted on the dev 80%
  scores player AUC 0.9384, recall@FPR=1% 0.3091 on it.
- Rejected v2 hypotheses: 43-channel Transformer (recall gain not significant,
  context channels confounded), 1024-tick context (no gain), rank-group DRO
  (no change to Q4 FPR).
- Exploratory within-player self-baseline: kill-level AUC rises 0.847 -> 0.921
  (leakage audit 0.50), but player-level recall gain is not significant, AUC
  drops, and Q4 FPR is unchanged. Not promoted; recorded as a diagnostic.
- Professional-demo check (7 HLTV demos, 1,020 windows, 70 players): the
  champion flags 0/70 pro players at the 1% operating point and 0.29% of pro
  kills. Pros score below average CS2CD humans despite being more accurate.
  Adding pro windows as training negatives changes nothing.
- Q4 FPR was understated by a threshold error. On the larger rank replication
  (all dev-80 matches, 386 clean Q4 players) the corrected player-level 1% Q4
  FPR is **3.37%**; the earlier 19.05% was a 42-player small-sample estimate,
  not the population rate.
- Empirical-Bayes shrinkage of the player score by kill count (k0=5) reduces
  Q4 FPR to **2.59%** on the replication and raises recall@FPR=1% to 0.2288
  (dev80 0.2386), with AUC non-inferior. It is a dev-validated player-scoring
  rule; the holdout was spent, so it awaits external confirmation.
- The remaining Q4 false positives have extremely straight trajectories; with
  64 Hz view data they cannot be separated from silent aim without raw mouse
  input or better labels.
- Label-noise audit: the excluded `cdata` label-0 windows score like clean
  humans (first-moment estimate 5.5% cheater admixture; 0.34% above the clean
  1% cut). The dataset card's 55.6% precision is not corroborated at the
  window level. The two residual Q4 false positives have 67-78% of kills
  closer to cheater windows in feature space; the 64 Hz data cannot say
  whether they are unusual styles or undetected cheats.
  Details: `docs/llmwiki/label-noise-audit.md`.
- The conditional generator remains deferred.

## 11. Player Identity and Temporal Scope (verified 2026-10-17)

- **No cross-match identity.** Every player field in the demos is already
  replaced by `Player_1..Player_10`; `player_connect`, `player_disconnect`,
  `rank_update`, and the tick table contain no name, XUID, or network id.
  `Player_k` is regenerated per demo: `Player_1` alone appears in 539-645
  different matches. Linking a person across matches is not possible, and
  attempting it from timing/rank fingerprints would be unreliable and outside
  the dataset's anonymisation contract.
- **Within-match series are rich.** 5,868 player-match groups with kills;
  median 13 kills per player-match, 97% have >=2 and 67% have >=10 (max 92).
  The cache already supports per-player aggregation (mean/top-k/max/HMM) and
  per-player mouse-view residual fits.
- **Consequence.** "Same person across games" analyses (habit drift, style
  change across matches) cannot be run on CS2CD. Within-match temporal work is
  possible: per-player self-baselines, kill-order sequences, and style-change
  detection inside a demo. Cross-match claims require an external dataset with
  stable ids.
