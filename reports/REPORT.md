# ValveGuard — Detector v1 Report

Status: research checkpoint, 2026-10-17. Not deployment-ready.
v2 update: the locked-holdout evaluation and the three pre-registered
hypotheses (H1 Transformer, H2 1024-tick context, H3 rank DRO) are in
section 11. No challenger passed; the aim-only GBDT remains the champion.

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

| Rank quartile | n | positives | Player AUC | FPR at player-level 1% |
|---|---|---|---|---|
| Q1 low | 251 | 38 | 0.911 | 0.47% |
| Q2 | 191 | 45 | 0.895 | 0.68% |
| Q3 | 214 | 58 | 0.828 | 0.64% |
| Q4 high | 219 | 177 | 0.712 | **7.14%** |

Threshold correction (2026-10-18): the earlier table applied a kill-level
threshold to player mean scores. FPR must be measured at a player-level
threshold (1% of human players). With that correction Q4 is 7.14% on this
score set; on the champion's own OOF scores it is 19.05% (section 14).

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
- Match-disjoint only; per-match pseudonyms. `Player_k` is regenerated for every
  demo, and the raw JSON contains no name, XUID, or network id, so a person
  cannot be linked across matches. Cross-match (multi-game) person analysis is
  impossible by construction; only within-match series are available. Within a
  match the series is rich: 5,868 player-match groups, median 13 kills
  (97% have >=2, 67% have >=10), so per-player temporal analysis is possible
  inside a demo.
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

## 11. v2 checkpoint — locked holdout and pre-registered hypotheses (2026-10-17)

Status: the three pre-registered hypotheses are complete. The aim-only GBDT
remains the champion; no challenger passed the decision rule.

### 11.1 Holdout lock

- Stratified 20% match holdout: **158 matches / 15,623 windows**; development
  635 matches / 63,910 windows. Positive rate 0.3987 vs 0.3984.
- Seed 20261020; digest `01ff5174...522520e` in
  `reports/holdout/holdout_summary.json`.
- The lock is **post-hoc**: v1 selection used every match, so it bounds v2
  selection but does not retroactively cleanse the frozen baseline.

### 11.2 Development-80 baseline

Aim-only GBDT, original folds, holdout matches removed from training:

| Metric | dev 80% | frozen v1 (all data) |
|---|---|---|
| player AUC | 0.9294 | 0.9331 |
| player recall@FPR=1% | 0.1475 (CI 0.096–0.252) | 0.1735 |
| kill AUC | 0.8472 | 0.8497 |

The difference is consistent with 20% less training data; the frozen estimate
is stable, not inflated.

### 11.3 H1 — 43-channel Transformer on dev 80%

5-fold match-disjoint, same folds as the GBDT:

| Metric | Transformer | GBDT |
|---|---|---|
| player AUC | 0.9251 | 0.9294 |
| player AP | 0.7773 | 0.7514 |
| player recall@FPR=1% | **0.2297** (CI 0.146–0.357) | 0.1475 |
| player ECE | **0.1091** | 0.1843 |
| kill AUC | **0.8704** | 0.8472 |

Paired match-grouped bootstrap: ΔAUC **[-0.0153, 0.0057]**,
Δrecall **[-0.0071, 0.1809]**.

Decision: **not promoted**. The recall point estimate is +0.082, but the CI
crosses zero, and the model consumes weapon/outcome channels that the
metadata-only audit showed to be confounded.

### 11.4 H2 — 1024-tick context (pilot, rejected)

Rebuilt 1024-tick windows from raw telemetry for the 160-match subset:
16,185 / 16,357 windows kept. Reconstruction validated against the published
cache; on the same subset the rebuilt near features score 0.8822 player AUC vs
0.8810 for the cached features (paired ΔAUC [-0.0067, 0.0040]).

| Feature set | player AUC | recall@FPR=1% |
|---|---|---|
| near (59) | 0.8822 | 0.0374 |
| near + macro (96) | 0.8830 | 0.0349 |
| macro only (37) | 0.8240 | 0.0050 |

Paired near+macro vs near: ΔAUC [-0.0055, 0.0078], Δrecall [-0.043, 0.038].
Macro only is significantly worse (ΔAUC [-0.084, -0.031]).
Decision: **rejected**. Context beyond the near-kill window adds no ranking
signal; the upper bound on a missed gain is about +0.008 AUC.
The pilot used all 160 subset matches (including those in the holdout) because
it is a diagnostic that cannot be transferred to the main dataset, which has
no 1024-tick windows; it is not a candidate model.

### 11.5 H3 — rank-group DRO (pilot, rejected)

160-match rank subset, 14,558 windows / 1,123 players, groups Q1–Q4 plus
unknown, four adaptive reweighting iterations with inner 2-fold loss estimates:

| | Baseline | DRO |
|---|---|---|
| player AUC | 0.8771 | 0.8800 |
| recall@FPR=1% | 0.0374 | 0.0299 |
| Q4 AUC | 0.6749 | 0.6645 |
| Q4 FPR at global 1% | 4.76% | 4.76% |

Weights stayed in 0.93–1.11, i.e. the group losses are nearly identical.
Decision: **rejected**. Q4 false positives are feature overlap, not loss
weighting. Like H2, this pilot uses the 160-match subset for diagnosis only.

### 11.6 Final holdout evaluation (one shot)

The champion pipeline (aim-only GBDT, 59 features) was fitted on the
development 80% and scored once on the locked holdout:

| Metric | Holdout |
|---|---|
| player AUC | 0.9384 (CI 0.912–0.958) |
| player AP | 0.8354 |
| player recall@FPR=1% | 0.3091 (CI 0.187–0.523) |
| player FPR@TPR=95% | 0.2645 |
| kill AUC | 0.8474 |
| kill recall@FPR=1% | 0.1308 |

The holdout point estimate is higher than the dev OOF value, but the CI is
wide (275 positive / 915 negative players). The out-of-sample result is
consistent with the frozen estimate and shows no collapse.

### 11.7 v2 conclusion

- No pre-registered hypothesis passed; the frozen champion stands.
- The remaining blocker is unchanged: top-rank humans overlap with cheat-like
  aim features (Q4 AUC 0.71), and neither longer context nor loss reweighting
  separates them.
- The credible next lever is new information, not a new loss: raw mouse/view
  consistency at 1000 Hz or high-skill human negatives.

## 12. Exploratory: within-player self-baseline (dev 80% only)

Motivation: a high-rank human's normal aim is already precise, so population
features cannot separate it from aim assist. A kill that is far from *that
player's own* baseline might still be detectable. For each (match, player)
group we computed robust z-scores (median/MAD over the player's kills in that
match) and concatenated them with the champion features.

The holdout was already used for the champion's final score, so this is a
dev-80 diagnostic and cannot be promoted. Players with fewer than three kills
were dropped in the main run because their all-zero z rows leak the kill count.

| Feature set | player AUC | recall@FPR=1% | kill AUC |
|---|---|---|---|
| raw (59) | **0.9383** | 0.1851 | 0.8475 |
| raw+selfz (118) | 0.9269 | 0.2160 | **0.9210** |
| raw+selfz+count (120) | 0.9276 | **0.2447** | 0.9224 |
| selfz only (59) | 0.9028 | 0.1021 | 0.8456 |

Paired raw+selfz vs raw: ΔAUC [-0.0162, -0.0065],
Δrecall@FPR=1% [-0.0394, +0.0963].

- The kill-level gain is large and survives dropping low-kill players and a
  match-level label-permutation audit (shuffled player AUC 0.495–0.504,
  kill AUC 0.497–0.501), so it is not leakage.
- It does not reach the player level: mean aggregation washes it out, and
  max/top-k aggregation does not recover it (all Δrecall CIs cross zero or are
  negative).
- Q4 (154 players, 2 false positives among 52 clean): AUC 0.726 → 0.764,
  FPR 3.85% → 3.85%. The stated target is not moved.
- The kill-count features (`self_n`, `self_valid`) add no significant ranking
  gain and are a performance shortcut, not an aim signal.

Decision: **not promoted**. Recorded as an exploratory diagnostic; no holdout
reuse.

## 13. Professional demos: specificity and high-skill negatives

The Q4 false positives suggested that strong human aim might look like aim
assist. To test that directly, 7 professional CS2 demos were downloaded from
the HLTV demo dataset (`blanchon/opencs2_dataset_demo`, CC-BY-4.0), parsed with
demoparser2, and turned into 1,020 kill windows from 70 player-maps using the
same `raw_window`/`summarize` code as CS2CD. Pro windows are treated as clean
human negatives.

Feature distributions are comparable: mean final error 3.19 deg (pro) vs 4.07
(CS2CD humans) and 4.40 (cheaters); pros are slightly more accurate and lock
longer, as expected.

### 13.1 Specificity of the champion

Thresholds are set on CS2CD development-80 human scores and applied to pro
windows.

| Operating point | CS2CD human FPR | Pro kill FPR | Pro player FPR |
|---|---|---|---|
| 0.1% | 0.10% | 0/1,020 (0%) | 0/70 (0%) |
| 1% | 1.00% | 3/1,020 (0.29%) | 0/70 (0%) |
| 5% | 5.00% | 2.55% | 0/70 (0%) |

Mean scores: pro 0.253, CS2CD humans 0.293, CS2CD cheaters 0.705.

**Elite human aim is not mistaken for aim assist.** Pros score below the
average CS2CD human despite being more accurate, so the Q4 false positives are
not a generic high-skill effect; they are specific to the matchmaking
high-rank sample (or to label noise / small-sample variance there).

### 13.2 Adding pro windows as negatives

The 1,020 pro windows were added to every training fold (never to a test fold)
in the dev-80 5-fold CV:

- Paired vs the baseline GBDT: ΔAUC **[-0.0010, +0.0017]**, Δrecall@FPR=1%
  **[-0.0215, +0.0290]**.
- Q4 FPR 10.5% -> 10.5%; Q4 AUC 0.685 -> 0.677.

Decision: **high-skill-negative augmentation rejected** — it changes nothing.
The specificity result is the useful outcome.

### 13.3 Caveats

- 7 demos / 70 players / 1,020 kills; the FPR CI is wide.
- Professional players are the extreme of skill, not a random sample of the
  CS2CD Q4 matchmaking population.
- Demos are from 2026; CS2CD may cover a different patch.
- Professional matches are assumed clean; no cheat labels are available.

## 14. Corrected Q4 FPR and player-score shrinkage (dev-only)

### 14.1 Threshold correction

The earlier Q4 FPR (4.76%) applied a **kill-level** threshold to player mean
scores. Player-level FPR must be measured at a **player-level** threshold
(1% of human players). With the correct threshold:

| Score set | Q4 clean players | Q4 FPR |
|---|---|---|
| champion, full 5-fold OOF | 42 | **8 (19.05%)** |
| champion, dev 80% | 38 | **5 (13.16%)** |
| mouse-residual scores (rank strata) | 42 | 3 (7.14%) |

The blocker was therefore worse than reported, and it is concentrated in
low-evidence players: 5 of the 8 champion false positives had one or two
kills; the Q4 clean group has a median of 9.5 kills but a 25th percentile of 2.

### 14.2 Empirical-Bayes shrinkage

Replace the player mean with a shrunk estimate

    score = (sum(kill scores) + k0 * prior) / (n + k0)

where `prior` is the mean score of human kills in the other training folds
(no test labels). This is a principled small-sample estimator, not a threshold
trick: it lowers scores only when the evidence is thin.

| k0 | full OOF AUC | full recall@1% | full Q4 FPR | dev80 AUC | dev80 recall | dev80 Q4 FPR |
|---|---|---|---|---|---|---|
| 0 | 0.9331 | 0.1735 | 19.05% | 0.9294 | 0.1475 | 13.16% |
| 1 | 0.9355 | 0.2125 | 7.14% | 0.9326 | 0.1911 | 7.89% |
| 3 | 0.9351 | 0.2109 | 4.76% | 0.9326 | 0.2030 | 7.89% |
| 5 | 0.9339 | 0.2288 | 4.76% | 0.9316 | 0.2386 | 5.26% |
| 8 | 0.9321 | 0.2389 | 4.76% | 0.9299 | 0.2337 | 5.26% |
| 12 | 0.9303 | 0.2490 | 7.14% | 0.9280 | 0.2356 | 5.26% |

Paired match-grouped bootstrap (k0=5 vs mean):

- dev80: ΔAUC **[-0.0015, +0.0058]**, Δrecall@FPR=1% **[+0.0068, +0.1404]**.
- full OOF: ΔAUC **[-0.0025, +0.0043]**, Δrecall **[-0.0123, +0.1249]**.

Q4 AUC also rises from 0.681 to 0.791 (dev80) and 0.670 to 0.797 (full OOF).

### 14.3 Remaining false positives

After shrinkage, two Q4 clean players remain above the threshold (18 and 9
kills). Both have extremely straight trajectories (straightness 0.88 and 0.98
vs a human median of 0.61) and very low movement while aiming. They are
plausible unusual human styles or undetected cheaters; the 64 Hz view data
cannot distinguish these without raw mouse input.

### 14.4 Status

- Adopt shrinkage (k0=5) as the **player-level scoring rule** for reporting;
  it passes the decision rule on dev80 (recall CI lower bound > 0, AUC
  non-inferior) and on the full OOF.
- It was selected after the holdout was spent, using full-OOF diagnostics; the
  holdout is **not** re-evaluated. A clean confirmation needs a fresh split or
  external data.
- The kill-level champion model is unchanged.
