# Next Session — Monday 2026-10-20

Status: the v2 protocol below was executed early on the RTX 3060 (2026-10-17).
No challenger passed; see "v2 execution record" at the end. Monday is now
optional follow-up, not a required run.

Original plan (kept for the record): experiments paused after the 2026-10-17
run. The repository is frozen at the detector-v1 baseline plus the completed
audit/figures/experiment grid.

## Frozen state

Champion: **aim-only GBDT**, 5-fold match-disjoint OOF:
- player AUC **0.9331**
- player recall@FPR=1% **0.1735**
- kill AUC **0.8497**
- High-rank Q4 FPR **4.76%** (release blocker; corrected on 2026-10-18 to
  19.05% at the player-level threshold, see the changelog)

Full 43-channel Transformer 5-fold (exploratory, not promoted):
- player AUC 0.9288, recall@FPR=1% 0.2350, AP 0.7882.
- Paired vs GBDT: ΔAUC [-0.0125, 0.0039], Δrecall [-0.0146, 0.1847].
- The recall gain is not significant and the model uses context/outcome channels;
  keep it as H1 for Monday under the locked-holdout protocol.

Do not promote a challenger without a paired match-grouped bootstrap whose
Δrecall@FPR=1% 95% CI lower bound is > 0 and whose ΔAUC is non-inferior.
Batch 1 (28 variants) and batch 2 motor features produced no such challenger.

## First actions on Monday

1. **Lock a 20% match holdout** (stratified by cheater presence) and never use it
   for model selection. All development happens on the remaining 80%.
2. Re-run the GBDT baseline on the development 80% with nested CV; confirm the
   frozen metric is stable.
3. Run only three pre-registered hypotheses, in this order:
   - H1: full 43-channel Transformer, 5-fold/nested CV.
   - H2: 1024-tick context (GBDT first; only then Transformer).
   - H3: high-rank human negatives / group DRO (pilot on the 160-match subset).
4. Apply Holm correction across H1–H3; promote only by the decision rule above.
5. Evaluate the final candidate **once** on the locked holdout; write the final
   report and stop.

## Hardware

- Monday: RTX 5070 Ti, 16 GB VRAM. This is enough for 1024-token Transformer
  training at d_model 128–192 with batch 32–64, and for the 256-token model at a
  larger batch. Keep the same 5-fold/nested protocol; more VRAM does not justify
  a larger search space.

## Timebox

- Lock holdout + baseline: 30–45 min.
- H1 (43-channel Transformer, dev set): 1–1.5 h.
- H2 (1024 GBDT; optional 1024 Transformer): 1–1.5 h (Transformer adds 2.5–3 h).
- H3 (high-rank pilot): 30–45 min.
- Final holdout + report/figures: 30–45 min.
- Total target: **3–4 h** for a defensible v2; do not exceed 5 h.

## Stop rules

- If no hypothesis clears the paired CI + Holm threshold, freeze the GBDT
  champion and write the report.
- Do not add post-hoc calibration, rank thresholds, or residual features to chase
  the last 1–2 percentage points; previous audits showed they trade away ranking.
- Report all attempted variants, not only the winner.

## Commands

```bash
# Environment
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu124 torch

# Current 5-fold baseline
python -m valveguard.scripts.run_cv_gbdt \
  --cache data/processed/cw256 \
  --features reports/baseline_clean/summary_features.parquet \
  --out reports/cv_gbdt

# Full 43-channel Transformer 5-fold
python -m valveguard.scripts.run_cv_transformer \
  --cache data/processed/cw256 \
  --channels data/processed/cw256_channels/channels.npy \
  --out reports/cv_transformer_full --folds 5 --epochs 25 --batch-size 64 \
  --drop-groups ""

# Figures / report
python -m valveguard.scripts.make_figures
```

## Artifacts to read first

- `reports/REPORT.md` — detector-v1 results and limitations.
- `docs/llmwiki/experiments.md` — protocol, queue, and negative results.
- `reports/figures/` — 10 figures.
- `reports/cv_gbdt/` — OOF predictions and metrics.
- `reports/cv_transformer_full/` — full 43-channel 5-fold run.
- `reports/cascade/` — calibration, aggregation, residual audits.
- `reports/skill/` — rank-quartile analysis.

## Freeze record

- Commit: `ddcabfb`
- Tag: `v0.1.0-detector-baseline`
- Branch: `master`
- Working tree clean at freeze.
- Resume from this commit on Monday; do not rewrite or force-push the tag.

## v2 execution record (2026-10-17, RTX 3060)

- [x] Locked 20% match holdout: 158 matches / 15,623 windows (`reports/holdout`).
- [x] GBDT on dev 80%: player AUC 0.9294, recall@FPR=1% 0.1475.
- [x] H1 Transformer on dev 80%: 0.9251 / 0.2297; ΔAUC [-0.0153, 0.0057],
      Δrecall [-0.0071, 0.1809] -> not promoted.
- [x] H2 1024-tick context: ΔAUC [-0.0055, 0.0078] -> rejected. The macro-only
      arm is significantly worse. Reconstruction validated against the cache.
- [x] H3 rank-group DRO: Q4 FPR unchanged at 4.76% -> rejected.
- [x] Final holdout evaluation: player AUC 0.9384, recall@FPR=1% 0.3091
      (CI 0.187-0.523), kill AUC 0.8474.

What remains is not a model-search problem:

1. New information: 1000 Hz raw mouse/view consistency or high-skill human
   negatives (pro/FACEIT demos). Everything else has been tried on this data.
2. If either becomes available, re-run the locked-holdout protocol above with
   one pre-registered hypothesis at a time.
3. Do not re-tune the current feature/model space against the holdout.

New artifacts: `reports/cv_gbdt_dev80`, `reports/cv_transformer_full_dev80`,
`reports/h2_pilot`, `reports/h3_dro`, `reports/holdout_eval`.
