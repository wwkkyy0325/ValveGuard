# Longitudinal Player Dataset

Status: proposal (2026-10-20)

## Why the current data is not enough

- The champion scores 256-tick kill windows and aggregates them per player.
  OOF: player AUC 0.9480, recall@FPR1 0.3284, Q4 FPR 3.11%.
- Remaining errors are player-level: cheaters who toggle and have only a few
  blatant kills, and elite humans whose aim looks synthetic in short windows.
- CS2CD provides roughly one match per player and only kill windows. There is
  no stable cross-match player id, no ban timestamp, and no non-kill windows,
  so a temporal model cannot be trained or evaluated on it.

## What a temporal model needs

- Stable SteamID64 across matches, at least 10 matches per player, with match
  timestamps and map/weapon context.
- Player-level labels with time: banned and ban date, matches before/after the
  ban, and never-banned controls. Match-level soft labels are not enough.
- Dense telemetry: full demos or sliding windows over the match, not only
  windows that end in a kill. Toggling is visible in the quiet stretches.

## Cohort design

| Cohort | Players | Matches/player | Purpose |
|---|---|---|---|
| banned cheaters | 100 | 20+ | positives, including toggle cases |
| never-banned, high rank | 100 | 20+ | hard negatives (Q4-like) |
| never-banned, general | 300 | 10+ | base-rate and calibration |

- Person-disjoint split; no player appears in both train and test.
- Time-forward split inside each player: first 70% of matches for training,
  last 30% for testing, to measure detection after the model has seen history.
- Cluster bootstrap by player, not by match, for confidence intervals.

## 5E pilot

- One player's full history is useful for a **case study and demo**, not for
  training: a temporal model needs cohort-level variance.
- If 5E access is available, the smallest informative pilot is 20 banned and
  40 never-banned players with 10 matches each (600 demos). At 100 MB per
  demo this is about 60 GB, well inside the training box capacity.
- Record the platform ban status and date at download time; a later ban list
  cannot reconstruct what was known when a match was played.
- Keep the download manual and local; do not automate platform login.

## Features

- Per-match window score distribution: mean, max, top-5 mean, variance, share
  above the 1% threshold.
- Temporal: change-point or switching model over match scores; time since
  last flagged match; session and weapon-role drift.
- Keep the aim-only window model as the fixed base scorer so temporal gains
  are separable from representation changes.

## Success criteria

- Primary: player-level recall@FPR1 on the held-out last 30% of matches.
- Secondary: time-to-detection (matches until the first flag), Q4 FPR, and
  calibration of the per-player cheating probability.
- Promote only if the recall gain survives player-cluster bootstrap and does
  not raise Q4 FPR beyond the current 3.11%.
