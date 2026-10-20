# ValveGuard — Detailed Model Design (v2)

Status: detector-first (2026-10-17)

> **Scope split (2026-10-20):** this repository is detector-only. The conditional generator was split into the sibling project `AimForge`; section 8 is kept as design history.

This document supersedes the model sketch in `design.md`. It is written to be implementation-ready and paper-grade: every modeling choice has a stated motivation, a verification step, and a leakage/robustness check.

## 0. What Changed After Inspecting the Data

1. **Primary dataset: CS2CD raw**, not the precomputed context windows. The raw parquet contains per-tick `usercmd_mouse_dx` / `usercmd_mouse_dy` (raw mouse deltas), `usercmd_input_history` (fractional-tick view angles), `pitch`, `yaw`, `aim_punch_angle`, `spotted`, `flash_duration`, `velocity`, weapon state, buttons, economy, and rank. This enables signals that pitch/yaw-only papers cannot use.
2. **Mouse–view consistency is a first-class feature.** In a sample match, robust regression gives `R² = 0.75–0.89` for `dyaw ~ mouse_dx` and `0.63–0.85` for `dpitch ~ mouse_dy` for normal players; the one labeled cheater in that match had `R² = 0.554` for yaw. An internal aimbot that writes view angles without corresponding mouse input produces a large residual. This is a model-external physical consistency check, not a learned pattern.
3. **Single-window detection is insufficient.** Use a multi-scale window pyramid plus a whole-match context stream and a hierarchical player-level model. See section 3.
4. **Label policy must be stricter than the source papers.** In CS2CD, non-cheater labels inside cheater-present matches have only ~55.6% precision. They must not be used as negatives.
5. **Split honesty.** CS2CD pseudonymizes players per match, so true person-disjoint evaluation is impossible on this dataset. Use match-disjoint splits and state the limitation. The AntiCheatPT split is match-keyed, which is appropriate given the anonymization; do not claim person-disjoint on CS2CD.

## 1. Data

### 1.1 Primary dataset: CS2CD raw

- Hugging Face: `CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection`
- License: CC-BY-4.0
- 795 matches; 478 with no VAC-banned player, 317 with at least one manually reviewed cheater.
- ~48.9 GB; one parquet (ticks) + one JSON (events) per match.
- Tick rate: 64 Hz in CS2 matchmaking.
- Per-tick rows: one row per player per tick (10 players).
- Key fields (names as published):

| Group | Fields |
|---|---|
| View | `pitch`, `yaw`, `usercmd_viewangle_x/y/z`, `usercmd_input_history` |
| Raw mouse | `usercmd_mouse_dx`, `usercmd_mouse_dy` |
| Aim state | `aim_punch_angle`, `aim_punch_angle_vel`, `shots_fired`, `spotted`, `is_scoped`, `flash_duration` |
| Position | `X`, `Y`, `Z`, `velocity`, `velocity_X/Y/Z`, `is_airborne`, `duck_amount` |
| Weapon | `active_weapon_name`, `total_ammo_left`, `is_in_reload`, `zoom_lvl`, `weapon_mode` |
| Movement | `usercmd_forward_move`, `usercmd_left_move`, `buttons`, `in_crouch`, `is_walking` |
| Match context | `round_start_time`, `game_time`, `is_bomb_planted`, `total_rounds_played`, `team_score_*`, `balance`, `round_start_equip_value` |
| Skill | `rank` (per tick), `CSstats_info.avg_rank`, `match_making_type` |

Labels:

- `events["cheaters"]` is a list of `{steamid: "Player_k"}` for the match. It identifies the manually reviewed cheater(s).
- `no_cheater_present` matches have no `cheaters` key.
- `no_cheater_present` negatives have ~97.2% precision on a 50-demo audit.
- Non-cheater players inside `with_cheater_present` matches have only ~55.6% precision; **exclude them from the negative class**.
- VAC-ban positives have high precision but low recall (24.7% in the authors' audit); the dataset's manual review improves this, but subtle cheats remain missing. Report this as a limitation.

### 1.1.1 Detector-first data without downloading raw demos

For the first detector we do not need the 48.9 GB raw CS2CD. Use the precomputed kill windows:

- Hugging Face: `CS2CD/Context_window_256`, 9.4 GB tar, CC-BY-4.0.
- One parquet per kill window, shape `256 x 44`.
- Directory `cheater/` or `not_cheater/` is the label.
- Filename format: `cdata-cheater-file_{match_idx}-Player_{k}-kill_{i}.parquet` (or `ncdata-notcheater-...`).
- Mapping: `match_idx` is the same index used by the raw dataset folders (`with_cheater_present/{match_idx}` or `no_cheater_present/{match_idx}`); `Player_{k}` is the per-match pseudonym; `kill_{i}` is that player's kill index. To inspect the original kill event, download only that match's `{match_idx}.parquet` + `.json` (tens of MB), not the full dataset.
- It is **not CSV**. The dataset card's `.csv` mention is stale; the published files are parquet.
- It does **not** contain raw mouse deltas. If we need the mouse–view residual, we must fetch a small subset of raw matches later (e.g. 100 matches, ~4–6 GB) and rebuild windows ourselves.
- The precomputed features are the AntiCheatPT features: `attacker_*`, `victim_*`, map one-hot, weapon groups, kill flags. They use `abs` deltas and absolute normalized coordinates; our own pipeline will improve on this once raw matches are added.

The offline index we will build:

```
window_index.parquet:
  path, label, match_file, player, kill_idx, split, n_ticks
```

This index is the "annotated table" for training. A CSV export is possible for inspection, but training uses parquet.

### 1.2 Secondary datasets

| Dataset | Use | Caveat |
|---|---|---|
| `CS2CD/Context_window_256` / `_1024` | Reproduce AntiCheatPT; sanity-check our pipeline | Precomputed features use absolute values (`abs`) and absolute coordinates; no mouse deltas; not the primary data |
| Kaggle `emstatsl/csgo-cheating-dataset` | Cross-domain baseline (CS:GO) | Unverified labels, precomputed windows, no demos |
| XGuardian Zenodo artifact (22 GB) / raw (585 GB) | External validation on a different platform/label source | Raw is too large for the current disk; use only a subset if needed |
| Same-player verification datasets (DPER/DPRO, if available) | Player-style encoder pretraining / consistency baseline | Not cheat labels; person-disjoint protocol is the useful part |
| HLTV/OpenCS2/Reka pro data | High-skill human reference | No cheat labels; large; optional |

### 1.3 Splits (important limitation)

CS2CD anonymizes each match independently: `steamid_to_player = {steamid: f'Player_{i+1}'}` is built per demo. **`Player_k` is a per-match pseudonym, not a stable player identity across matches.** Therefore:

- True person-disjoint evaluation is **not verifiable** on CS2CD.
- Use **match-disjoint 5-fold** (no match in two splits), stratified by cheater presence.
- Do not claim person-disjoint; state the limitation explicitly.
- A true person-disjoint check requires a dataset with stable IDs (e.g., XGuardian/5EPlay if IDs persist, self-collected demos with Steam IDs, or the same-player-verification DPER/DPRO datasets).
- Within a match, player-level aggregation and the HMM are still valid; cross-match self-baseline is not available from CS2CD alone.
- The test fold is locked: no hyperparameter, threshold, or feature selection on it.
- Report both kill-level and player-level metrics; player-level is primary.

## 2. Preprocessing

### 2.1 Tick parsing and time base

- Sort by `(steamid, tick)`; verify one row per player per tick.
- Actual `dt = (tick - tick_prev) / tick_rate`, with `tick_rate = 64`; verify by checking the modal delta. If a match is 128-tick, normalize.
- `usercmd_input_history` entries carry `player_tick_count + player_tick_fraction`; use them for fractional timing jitter, not as a high-rate signal (in the sample, the effective rate is ~64–68 Hz, not 1000 Hz).
- Resample view angles to a uniform 64 Hz grid for the main model; keep `dt` and fractional timestamps as auxiliary features.

### 2.2 Angle handling

- `wrap_pi(x) = ((x + 180) mod 360) - 180`.
- Signed deltas: `dyaw = wrap_pi(yaw_t - yaw_{t-1})`, `dpitch = pitch_t - pitch_{t-1}`. Never take `abs`; the direction is signal.
- Angular speed: `sqrt((dyaw/dt)^2 + (dpitch/dt)^2)` in deg/s.
- Acceleration and jerk: finite differences with `dt`, clipped to physical bounds and flagged if clipped.
- Avoid discontinuities at ±180 and ±90; mark wrap ticks and exclude them from residual fitting.

### 2.3 Mouse–view consistency residual

For each player and match, fit robust linear models on alive, non-wrap ticks:

```
dyaw_t   ≈ a_y * mouse_dx_t + b_y
dpitch_t ≈ a_p * mouse_dy_t + b_p
```

Use Huber regression or RANSAC. The slope absorbs the player's sensitivity/DPI. Then:

```
r_yaw,t   = dyaw_t - (a_y * mouse_dx_t + b_y)
r_pitch,t = dpitch_t - (a_p * mouse_dy_t + b_p)
r_t       = sqrt(r_yaw,t^2 + r_pitch,t^2) / robust_scale
```

Features per window: mean/median/p95/max of `|r|`, fraction of ticks with `|r|` above a per-player threshold, residual entropy, and residual during the 32 ticks before the shot. Also fit the model on non-kill ticks and apply to kill windows to avoid contamination.

Caveats:

- External cheats that move the OS mouse (`SendInput`) will produce consistent deltas; the residual is then not diagnostic.
- Some cheats write view angles only during the snap; the residual spikes only there. Use a short-window max, not the whole-window mean.
- Sensitivity can change mid-match; fit per round or use a robust rolling estimate.
- Verify on clean players that the residual distribution is stable across maps, weapons, and ranks.

### 2.4 Target-relative geometry

For attacker `a` and victim `v` at tick `t`:

- Relative position `d = p_v - p_a`; distance `||d||`.
- Target bearing `atan2(d_y, d_x)`; target elevation `atan2(d_z, ||d_xy||)`.
- Angular error: `yaw_err = wrap_pi(yaw_target - yaw_a)`, `pitch_err = pitch_target - pitch_a`; angular distance.
- Radial/tangential decomposition of the error derivative.
- Relative velocity and its radial/tangential components.
- Victim state: velocity, health, `spotted`, `flash_duration`, `is_alive`, `noise` proxy (velocity ≥ 140 or shot in window).
- Visibility proxy: target within FOV, distance below a weapon-dependent range, no smoke flag, and not behind known cover. A true raycast against map geometry is optional v2; do not claim visibility without it.

### 2.5 Feature normalization

- Angles: divide by 180/90; deltas by 45/90; speeds by 500 deg/s; positions by map bounds; distance by map diagonal.
- Do not use absolute world coordinates as raw inputs to the main model; use map-normalized coordinates plus relative geometry. Absolute coordinates caused the AntiCheatPT authors to augment positions to avoid overfitting to kill locations.
- Per-player robust scaling for mouse deltas and residuals; per-match scaling for economy.
- No future information: all rolling statistics are causal within a window; the whole-match style encoder uses only the match being scored (offline) and never crosses splits.

## 3. Multi-Scale Windows and Why 96 Ticks Is Too Short

Evidence:

- XGuardian: 96 ticks = 1.5 s (64 before, 32 after). It captures the final engagement but not the approach or pre-aim.
- AntiCheatPT: 256 ticks = 4 s (224 before, 32 after); their own future work says to increase the context window.
- They also released a 1024-tick = 16 s version (896 before, 128 after). This is strong evidence the field is moving to longer context.
- Human motor timescales span 100–300 ms submovements, 150–300 ms reaction, 200–600 ms flicks, 0.5–3 s crosshair placement, and 5–30 s round positioning.

Design: a **window pyramid**, all aligned at the kill tick:

| Branch | Length | Composition | Captures |
|---|---|---|---|
| Micro | 64 ticks (1 s) | 48 before + 16 after | snap dynamics, overshoot, submovement structure, mouse residual at the shot |
| Meso | 256 ticks (4 s) | 224 before + 32 after | reaction, pre-aim, engagement, target coupling |
| Macro | 1024 ticks (16 s) | 896 before + 128 after | approach, crosshair placement, round phase, prior fights |
| Match | whole match, binned | 1 s bins / kill set | player style, non-kill aim baseline, consistency, toggling |

Do not feed 1024 raw tokens to every branch. Use patchification (patch 8 or 16) for macro and a low-rate context stream for the match. The micro branch stays dense.

Ablate each branch. Report the single-scale 96/256 baseline and the full pyramid separately; do not hide the contribution of extra context.

## 4. Model A: Detector Architecture

### 4.1 Input representation

Per tick, a feature vector of ~60–80 values:

- signed view deltas, angular speed/accel/jerk, wrap flags;
- mouse deltas and mouse–view residuals;
- target-relative error and derivatives;
- attacker and victim kinematics;
- weapon/zoom/ammo/reload, flash, aim punch, spotted, airborne/duck;
- event flags: shot, kill, headshot, smoke, wallbang, midair;
- relative geometry and visibility proxy;
- round phase/economy/score (causal, current tick only).

Per window, auxiliary scalar features:

- motor-control metrics (section 5);
- residual summaries;
- distance, weapon, target speed, engagement duration;
- player style embedding from the match stream (section 6).

### 4.2 Encoders

- **Stem**: `Linear(F -> 128) + LayerNorm + GELU`; for micro/meso add depthwise 1D convolutions with kernels 3/5 and dilations 1/2/4.
- **Micro branch**: 1D CNN + GRU (hidden 128); 64 tokens.
- **Meso branch**: Transformer encoder, `d=128`, 4 heads, 4 layers, FF 256, dropout 0.1; 256 tokens or 64 patched tokens (patch 4).
- **Macro branch**: patchify 16 -> 64 tokens; Transformer encoder, same width.
- **Victim stream**: encode victim kinematics and visibility tokens at the meso scale; cross-attention with attacker tokens (attacker Q, victim K/V), one layer.
- **Style conditioning**: player style vector (128-d) -> FiLM `(gamma, beta)` on the fused kill CLS.
- **Context conditioning**: round phase/economy/score/weapon embeddings -> FiLM or additive tokens.
- **Scale fusion**: self-attention over `[CLS_micro, CLS_meso, CLS_macro]` plus scale embeddings; gated fusion.
- **Heads**:
  - kill-level cheat logit;
  - player-level logit via attention pooling over kill embeddings + style;
  - auxiliary regression heads: time-to-target, overshoot, smoothness, reaction time (masked, Huber loss).
- **Parameter budget**: ~3–6 M; fits 6 GB with AMP, batch 32–64 at 1024 tokens.

### 4.3 Loss

```
L = BCE_kill + λ1 * BCE_player + λ2 * Huber(aux) + λ3 * InfoNCE_style + λ4 * L_residual
```

- `BCE_kill`: attacker-level label (is the attacker a verified cheater in this match).
- `BCE_player`: player-level label; use all kills of the player in the match.
- `InfoNCE_style`: pull human kill embeddings toward the player's non-kill style embedding; push cheater kills away. Use a temperature of 0.1 and a gradient-reversal player-ID adversary with a small weight to avoid learning identity instead of cheating.
- `L_residual`: optional auxiliary prediction of mouse–view residual summary; helps the model use it.
- Class weights: use `pos_weight` or focal loss; do not tune the decision threshold on the test set.

### 4.4 Skill-aware, one-sided suspicion

Do not score generic anomaly. Score the likelihood ratio against a skill-conditioned human model:

```
s(x | skill) = log p_cheat(x) - log p_human(x | skill, context)
```

- Estimate skill from `rank` when present (Premier rating or CS rank); otherwise from non-aim behavior (crosshair placement, idle stability, reaction on easy targets, movement, economy).
- Model `p_human(x | skill)` with a conditional normalizing flow or per-skill-cohort empirical CDFs / quantile regression.
- The likelihood ratio naturally handles direction: a weaker-than-expected trajectory yields a negative score (exculpatory); a stronger-than-expected one yields a positive score.
- For features where both tails are suspicious (e.g., jerk too low or too high), use a two-sided density but keep the learned ratio; do not hand-code "more is worse".
- Pro/high-rank data is the hardest negative. Report FPR separately for high-rank players; a model that flags pros is not deployable.

## 5. Smoothness and Motor-Control Modeling

### 5.1 Metrics

Per window, compute:

- normalized jerk and log dimensionless jerk (LDLJ);
- SPARC (spectral arc length);
- number of velocity peaks / submovements (peak detection on smoothed speed);
- total absolute curvature, curvature entropy;
- straightness index `|displacement| / path_length`;
- overshoot ratio, settling time, endpoint error;
- velocity-profile skewness, kurtosis, time-to-peak;
- PSD slope (1–30 Hz), sample entropy, detrended fluctuation analysis;
- `lock_ratio`: fraction of ticks with angular error < 0.5° and speed < 5°/s;
- post-peak correction count and sign changes;
- lead-lag cross-correlation between victim lateral velocity and attacker yaw velocity;
- mouse–view residual summaries.

### 5.2 Why "too smooth" alone is not enough

- Elite humans can be extremely smooth; smoothness must be conditioned on distance, target speed, weapon, and skill.
- A humanized cheat can add noise or a minimum-jerk profile. The discriminator must rely on the **joint** pattern: fast + accurate + smooth + zero reaction + no correction + no mouse residual.
- A long-distance flick that is perfectly straight and ends exactly on the target with no overshoot is more suspicious than a smooth micro-correction.
- Internal aimbots often produce a large mouse–view residual even when the trajectory looks smooth. External aimbots may not, but their target coupling and reaction profile remain unusual.
- Always report smoothness ablations; if the model without smoothness performs the same, the feature is not contributing.

## 6. Bidirectional / Hierarchical Analysis

### 6.1 Bottom-up

- tick -> window (micro/meso/macro) -> kill embedding -> player embedding -> match risk.
- Attention pooling at each level; report attention weights as evidence.

### 6.2 Top-down

- Player style and skill condition the interpretation of each kill: `p(kill | player, skill, context)`.
- A kill that is normal for a pro but impossible for a low-rank player is suspicious; the reverse is exculpatory.
- Use FiLM conditioning and a prior on the player's kill-score distribution.

### 6.3 Single kill vs all kills

Model the player's kill scores with a two-state HMM:

- states: `human`, `cheat`;
- emissions: per-kill logits calibrated to probabilities;
- transitions: allow toggling (cheat on/off);
- forward-backward gives `P(state_k = cheat | all kills)` and `P(player cheated at any point)`.

This directly answers "single kill vs all kills": one anomalous kill is weak evidence; a contiguous segment of anomalous kills with human kills before/after is strong evidence. Also report a mixture-model summary (mixing weight, component means, max tail) and a CUSUM change-point statistic.

### 6.4 Whole-match mouse trajectory vs kill trajectory

- Build a match-level style stream from non-kill ticks (1 s bins): aim speed distribution, idle jitter, crosshair placement, rotation behavior, movement-stop-fire coordination, shooting rhythm, economy.
- Compare each kill window to the player's own non-kill distribution: Mahalanobis distance / likelihood percentile.
- A cheater who toggles will show a bimodal separation; a cheater always on will shift the whole distribution.
- Use the same-player verification feature families as inspiration (aim/crosshair, mechanics/state, combat, movement, utility, timing, economy, context) but do not reuse their model weights; retrain on our labels.

### 6.5 Attacker vs victim

- Cross-attention over attacker and victim streams.
- Lead-lag and tracking residual: fit a constant-velocity/Kalman predictor to the victim; compare attacker aim to the actual vs predicted trajectory. Humans lag; perfect prediction/zero lag is suspicious.
- Visibility proxy and smoke/flash flags condition the expected reaction; do not penalize a player for pre-aiming an unseen enemy without a visibility model.

### 6.6 Forward and backward

- Offline analysis can use a bidirectional encoder; deployment can use a causal variant.
- Report both. A large gap between bidirectional and causal performance indicates future information is being used and must be removed for deployment.

## 7. Spatiotemporal and External Context

- **Spatial branch**: map embedding, normalized position, distance/bearing/elevation, relative velocity, cover/smoke proxy; optional 2D 64x64 raster with attacker/victim/view cone. Ablate it; do not add a CNN branch unless it earns its compute.
- **Temporal branch**: multi-scale sequences, fractional timing, event tokens (shot, kill, damage, flash, smoke), smoothness metrics.
- **Fusion**: spatial token added to each scale's CLS; cross-attention between attacker/victim; scale attention.
- **External context**: round phase, economy/buy type, score, player count, bomb state, weapon, opponent skill. Use conditioning; exclude `server`, `match_making_type`, and raw match IDs from features (they can leak label collection).
- **Player history**: if available, a same-player style embedding from previous matches; otherwise within-match non-kill style only. Never use future matches.

## 8. Model B: Conditional Trajectory Generator

Goal: given start and end aim points and context, generate human-like intermediate trajectories, and provide a human-likelihood score for detection.

- **Representation**: signed `(dyaw, dpitch)` in target-centric coordinates; variable duration; endpoint constraint.
- **Condition**: start and target view angles, target angular size/distance, victim velocity, weapon, skill, desired time budget.
- **Model**: conditional flow matching (CFM) with a small 1D U-Net (64–128 channels, 4 blocks) or an autoregressive mixture-density Transformer. CVAE as a baseline. Avoid GAN training for the main model; use a GAN only as an evaluation baseline.
- **Endpoint conditioning**: train a stochastic bridge between endpoints; at sampling, project the final point onto the target tolerance or use guided sampling.
- **Training data**: human kill windows from clean players; also human non-kill reach/rotation segments for diversity.
- **Evaluation**:
  - endpoint error <= 1 degree;
  - time-to-target, velocity profile, curvature, submovement count, PSD vs human;
  - discriminator balanced accuracy near 50–60% (train a separate classifier on human vs generated; our detector is one discriminator, but use an independent one for the paper);
  - blind human A/B by experienced CS players (optional, small n).
- **Detection use**: human-likelihood percentile as an auxiliary feature; train the generator only on human data and compute out-of-fold for test players.
- **Responsible release**: the generator is dual-use. Release the detector and evaluation code openly; release generator weights only with a dual-use statement and no game integration, or under controlled access. Do not publish a ready-to-use cheat.

## 9. Training Protocol

- Optimizer: AdamW, lr 1e-4, weight decay 0.01, cosine or StepLR, warmup 5%.
- Batch: 128 for 256-tick sequences; 32–64 for 1024-tick macro; gradient accumulation if needed.
- Epochs: up to 50 with early stopping on validation player-level AUC; patience 10.
- AMP on the RTX 3060; deterministic cuDNN; fixed seeds (at least 3, preferably 5).
- Augmentation (training only): time warp ±5%, mouse delta scale ±2%, small Gaussian noise on relative positions (same noise for attacker and victim), random temporal crop within the macro window. Never augment the test set.
- Class imbalance: weighted BCE or focal loss; calibrate the threshold on validation for the target FPR. Report PR-AUC and recall at fixed FPR.
- Calibration: temperature scaling or isotonic regression on validation; report ECE and Brier.
- Checkpoint selection: validation player-level AUC, not kill-level accuracy.
- Log: config hash, data hash, git commit, seed, split file, metrics, calibration curve, per-group metrics.

## 10. Evaluation Protocol

Primary (player-level):

- ROC-AUC, PR-AUC, F1 at the calibrated threshold.
- FPR at 95% TPR; recall at FPR = 1% and 0.1%.
- Brier score and ECE.
- Grouped bootstrap 95% CI by player; DeLong test vs baselines.

Secondary (kill-level):

- ROC-AUC, PR-AUC, precision/recall, calibration.
- Per-weapon, per-map, per-distance, per-headshot, per-smoke/wallbang subgroup.

Skill stratification:

- Premier rating / CS rank bins: low, mid, high, pro.
- Report FPR per bin. A high FPR on pros is a release blocker.

Cross-dataset:

- Train CS2CD, test Kaggle CS:GO (and vice versa) to measure domain shift.
- If a subset of XGuardian is available, test on it as an independent platform/label source.
- Report both raw transfer and fine-tuned transfer; do not mix them.

Adversarial / humanized:

- Generate synthetic humanized cheats: smoothing, noise injection, reaction delay, intermittent toggling, external mouse injection.
- Report recall at the operating FPR; show which features survive.
- Do not claim robustness to unseen cheats without testing.

Ablations:

- window scales (64/256/1024/match);
- raw mouse deltas and mouse–view residual;
- victim branch / cross-attention;
- style conditioning and InfoNCE;
- HMM aggregation;
- smoothness features;
- signed vs absolute deltas;
- relative vs absolute coordinates;
- map one-hot / metadata features;
- multi-task auxiliary heads.

Leakage audits:

- Shuffle labels: metrics must collapse to chance.
- Metadata-only model (map, rank, server, matchmaking type): if it performs well, the dataset has a shortcut; remove the leaking feature.
- Map-holdout test: train on some maps, test on unseen maps.
- Player-overlap assertion: fail the build if any player appears in two splits.
- Feature importance inspection: check for match ID, server, coordinate artifacts.
- Duplicate/near-duplicate window detection.

## 11. Reproducibility

- `pyproject.toml` + lock file; Python 3.11; PyTorch, polars/pandas, pyarrow, scikit-learn, Hugging Face `datasets`.
- Config system (Hydra or plain YAML) with every hyperparameter.
- Data download script using HF `snapshot_download` with allow-patterns; verify file counts and SHA256.
- Split files committed (player IDs are anonymized, so safe) and hashed.
- One-command pipeline: download -> parse -> windows -> features -> train -> evaluate -> figures.
- Unit tests: angle wrap, window alignment, residual fit, no-leakage assertions, shape tests, deterministic split.
- Model card: intended use, training data, labels, metrics, limitations, ethical considerations.
- Artifact: code, configs, split hashes, model weights, calibration parameters, figure scripts.

## 12. Compute Plan (RTX 3060 6 GB)

- Detector: 3–6 M params, 256-tick batch 128 with AMP; 1024-tick macro batch 32. Training on ~90k windows: a few hours per run; 3–5 seeds feasible.
- Generator: 1–2 M params, 64–128 step trajectories; a few hours per run.
- Storage: raw CS2CD ~49 GB; processed windows ~10–25 GB; checkpoints < 1 GB. Current free space is ~104 GB; keep at least 30 GB headroom. Skip the 585 GB XGuardian raw data.
- If 6 GB is tight: freeze the macro branch, use patch size 16, gradient checkpointing, and mixed precision.

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Label noise (55.6% precision for non-cheaters in cheater matches) | Exclude those negatives; use clean matches only; sensitivity analysis |
| VAC ban recall low; subtle cheats missed | Report limitation; use manual labels; external dataset validation |
| Player leakage across matches | CS2CD IDs are per-match; use match-disjoint splits and state that person-disjoint is unverifiable. Assert no match overlap; use an external stable-ID dataset for a true person-disjoint check. |
| Map/server/metadata shortcuts | Relative coordinates; remove metadata from features; metadata-only audit |
| Absolute-coordinate overfitting | Relative and map-normalized coordinates; augmentation; ablation |
| Tick-rate/raw-mouse limitations | Use `usercmd_mouse_dx/dy`; state that it is 64 Hz, not 1000 Hz |
| Target visibility unavailable | Geometry/FOV/smoke proxy; do not claim wallhack detection beyond aim artifacts |
| Pro false positives | Skill-conditioned scoring; high-rank FPR reporting; human review |
| Humanized cheats | Adversarial tests; joint features; mouse–view residual |
| Generator dual-use | Responsible release; no game integration; dual-use statement |
| Overfitting to one dataset | Cross-dataset transfer; external validation |
| Threshold tuning on test | Calibrate on validation only; lock test |

## 14. Detector-First Lean Execution Plan (current)

The generator was split into `AimForge` on 2026-10-20. The first milestone here is a trustworthy detector on the 9.4 GB precomputed windows. Every step below has a stated reason, a cost, and a go/no-go check.

### 14.0 Decisions and evidence

| Decision | Evidence | Cost avoided |
|---|---|---|
| Use `Context_window_256` (9.4 GB), not raw CS2CD (48.9 GB) | One parquet per kill: 256 x 44; verified `file_0/Player_3/kill_0` maps to the raw match and row 223 is the kill tick; pitch/yaw recover exactly from the normalized values | ~40 GB download, demo parsing, storage |
| Do not download `.dem` / XGuardian raw (585 GB) | Not needed for v1; the window dataset is already parsed | ~585 GB and hours |
| Match-disjoint split, not person-disjoint | CS2CD `steamid` is a per-match pseudonym (`Player_k`); person identity is not recoverable | Prevents a false claim and reviewer attack |
| GBDT baseline before Transformer | Validates labels, split, and features in minutes; a sequence model cannot fix broken labels | GPU hours |
| Single 256-tick scale first | XGuardian gets 0.889 acc with 96 ticks; AntiCheatPT gets 0.9336 AUC with 256. Longer context is an ablation, not a prerequisite | Multi-scale implementation and tuning |
| No map one-hot / server / matchmaking features | They can encode label-collection shortcuts; AntiCheatPT needed coordinate augmentation because absolute positions leaked | Avoids a silent shortcut |
| Drop the published `abs` deltas and the mis-signed pitch-head delta | Verified against the raw match: inverting CS2 pitch reduces mean absolute target error from 6.64 deg to 2.59 deg over the sample window, and the kill-tick origin error from 6.38 deg to 0.24 deg | Removes a known feature bug instead of learning around it |
| No raw mouse residual in v1 | It requires raw parquet; defer to a 100-match subset after the baseline | Keeps v1 small |
| No synthetic cheat generation in v1 | Public labels are enough for a first detector; synthetic data can create circular evaluation | Avoids an unvalidated data source |

### 14.1 Step-by-step, with verification gates

**Step 1 — Cache and audit (no GPU).**
- Stream the tar into a float16 memmap (~2 GB for 90k unaugmented windows) plus `index.parquet`.
- Audit: label counts, source counts, match counts, windows per match/player, mixed-label matches, duplicate paths, non-finite values.
- Gate G1: the unaugmented count is close to 90,707; cheater/non-cheater counts are close to 18,423 / 72,284; no match has mixed labels; no non-finite windows.

**Step 2 — Feature sanity (no GPU).**
- Reconstruct degrees/world positions from the 44 columns; verify against a raw match sample (already done for one window: exact pitch/yaw, distance 1470.8).
- Compute target error, time-to-target, overshoot, settling, speed, jerk, curvature, SPARC, LDLJ, PSD slope, entropy, lead-lag.
- Gate G2: kill-tick origin error is small (< 2 deg in the sample), features are finite or explicitly missing, and distributions are plausible by weapon/distance.

**Step 3 — GBDT baseline (CPU, minutes).**
- `HistGradientBoostingClassifier` on ~80 summary features; balanced sample weights; match-disjoint 70/15/15.
- Report AUC, AP, F1, FPR@95TPR, recall@FPR=1%, calibration, player-level AUC.
- Gate G3: test AUC clearly above chance (target >= 0.85). If not, stop and debug labels/features before any neural model.

**Step 4 — Small Transformer (3060, one run).**
- Input: 256 x ~47 channels (raw non-map features + signed deltas + corrected target errors + motion + event flags).
- Model: 4-layer Transformer, d=64/128, 4 heads, FF=256, CLS token, ~1M params; batch 32–64; AMP; early stopping.
- Same split and metrics as the GBDT; no test-set tuning.
- Gate G4: Transformer beats the GBDT on validation and test player-level AUC; otherwise keep the simpler model and analyze.

**Step 5 — Minimal ablations (only if G4 passes).**
- Signed vs raw deltas; corrected vs published target errors; smoothness features; context/kill flags; player-level mean vs max.
- Do not add multi-scale, HMM, or skill conditioning until the single-scale model is stable.

**Step 6 — Only then: multi-scale and skill.**
- Crop 64-tick micro windows from the same 256 data; add 1024 windows only if the 256 model leaves clear macro errors.
- Add skill stratification using `rank` (requires per-match JSON; ~1 GB) and the one-sided likelihood ratio.
- Add HMM aggregation after per-kill scores are calibrated.

### 14.2 Cost estimates

| Item | Estimate |
|---|---|
| Tar download | 9.4 GB, ~1–2 h on the current proxy |
| Memmap cache | ~2 GB, 10–30 min |
| Summary features | 90k windows, ~10–20 min single-process |
| GBDT | < 10 min CPU |
| Transformer | 30–90 min per run on the 3060 6 GB |
| Disk headroom | ~95 GB free now; cache + features + checkpoints < 10 GB |

### 14.3 What is explicitly out of scope for v1

- Raw mouse deltas / mouse–view residual (needs a raw subset).
- True person-disjoint evaluation (needs a stable-ID dataset).
- 1024-tick macro branch (needs the 1024 dataset or raw matches).
- Map raycast / true visibility.
- Cross-dataset transfer (CS2CD vs Kaggle/CS:GO) — after the in-domain baseline is stable.

## 15. Implementation Status (2026-10-17)

- Environment: project `.venv` with torch 2.6.0+cu124, CUDA available on RTX 3060 Laptop 6 GB.
- Data: 9.4 GB `context_windows_256.tar` downloading; 1.5 GB at last check.
- Code: cache/index, match-disjoint split, geometry reconstruction, handcrafted features, GBDT baseline, audit script, tests.
- Verified: one window maps exactly to the raw match (row 223 = kill tick; pitch/yaw/distance match); pitch sign correction reduces kill-tick target error from 6.38 to 0.24 deg.
- Tests: 4 basic tests pass (geometry, wrap, summary/channels, split disjointness).
- Not yet done: full cache, full audit, GBDT metrics, Transformer.

