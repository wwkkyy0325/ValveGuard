# Player-Level Fusion and Behavioral Signals

Status: active (2026-10-20)
Base: aim-only 29-channel wide CNN, 3-seed ensemble, player AUC 0.9480,
recall@FPR1 0.3284, Q4 FPR 3.11%.

## Goal

Raise player-level low-FPR recall without giving up Q4 FPR, using the
existing windows first. Only then add new behavioral channels.

## Steps

- [x] Record the roadmap and decision order.
- [x] Stable-ID / person-disjoint feasibility audit.
      Result: CS2CD `player_name` is `Player_1..10` per match, not a stable
      SteamID. Person-disjoint evaluation is impossible on this dataset;
      match-disjoint is the strongest available split. A longitudinal cohort
      is required for person-level generalization.
- [x] Post-hoc fusion of player views: mean, top-5, dispersion, match-relative.
      Result: negative. Match-relative (AUC 0.793) and dispersion (0.781)
      collapse; top-5 and rank fusion gain at most +0.018 recall with CIs
      crossing zero and lose 0.6-2.3 pp AUC. Mean stays the default.
- [x] Fixed-fold audit. GBDT uses stratified match folds; the window CNN used
      a fresh GroupKFold, so the two were not paired. Added `--folds-from` to
      both trainers and retraining on the `cv_predictions` fold map.
- [x] Player-level MIL / attention pooling over windows.
      Result: negative. Frozen encoder gives recall +0.0148 (CI crosses zero)
      but Q4 FPR +0.0232 [0.0023, 0.0455]; unfrozen loses AUC 0.0172.
      Window training + mean aggregation stays.
- [x] Low-FPR loss (focal / pAUC / hard-negative weighting).
      Result: hard-negative BCE alpha=5 is the new champion. 3-seed ensemble
      recall 0.3619, AUC 0.9502, Q4 FPR 3.11%; vs the mean ensemble
      Δrecall +0.0459 [0.0048, 0.0883], ΔAUC +0.0002.
- [x] Visibility-onset reaction-time channels.
      Result: negative at player level. 34ch vs 29ch gives kill AUC
      +0.0074 and player AUC +0.0040, but recall -0.0019 [-0.0394, 0.0366].
      Not adopted.
- [x] Fuse the best two views and decide the champion.
      Result: post-hoc view fusion was negative, so no fusion is used.
      Champion is the 3-seed hard-negative (alpha=5) aim-only CNN:
      recall 0.3619, AUC 0.9502, Q4 FPR 3.11%. The dataset demo (`cdata:0`)
      flags the labelled cheater and leaves the clean match unflagged.
