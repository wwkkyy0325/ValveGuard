# ValveGuard Knowledge Base

Start here when resuming work.

| File | Purpose |
|---|---|
| `design.md` | Living project design and current state; identity scope, decisions, references. |
| `model-design.md` | Detailed v2 model specification (multi-scale, channels, protocols). |
| `experiments.md` | Champion/challenger protocol, experiment queue, and all batch results. |
| `label-noise-audit.md` | Audit of the excluded cdata label-0 windows and the residual Q4 false positives. |
| `rank-replication.md` | Rank-stratified FPR replication on all dev-80 matches (Q4 3.37%). |
| `model-card.md` | Deployment model card: input/output contract, metrics, limitations, retraining policy. |
| `next-session.md` | Frozen-state record, executed v2 protocol, and remaining work. |
| `changelog.md` | Append-only dated record of changes, results, and corrections. |
| `tasks/aimbot-detection-and-synthesis.md` | Task breakdown with checkboxes and verify steps. |

## Conventions

- Living documents are updated in place; `changelog.md` is append-only.
- Numbers that were later corrected keep their original changelog entry and
  gain a correction entry; living docs carry the corrected number.
- Every experiment writes its artifacts under `reports/<name>/` and is
  registered in `experiments.md`.

## Current headline

Aim-only GBDT, 5-fold match-disjoint OOF: player AUC 0.9331,
recall@FPR=1% 0.1735. Player-score shrinkage (k0=5) raises recall to 0.2288
and cuts the replicated Q4 FPR from 3.37% to 2.59%; it is dev-validated only
because the holdout was already spent.
