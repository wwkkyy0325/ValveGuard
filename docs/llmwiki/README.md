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
| `onnx-distribution.md` | ONNX bundle layout, export, parity check, runtime, and distribution. |
| `../paper/valveguard-paper-zh.md` | Final bilingual paper (Chinese). |
| `../paper/valveguard-paper-en.md` | Final bilingual paper (English). |
| `io-contract.md` | Frozen tensor/feature/score shapes and the application boundary. |
| `windows-app.md` | Windows/Tauri app plan, demo input handling, and review UX. |
| `remote-training.md` | LAN training box layout, environment, commands, and cleanup. |
| `segment-features.md` | Movement/pre-aim segmentation pilot and its negative result. |
| `next-session.md` | Frozen-state record, executed v2 protocol, and remaining work. |
| `changelog.md` | Append-only dated record of changes, results, and corrections. |
| `tasks/aimbot-detection.md` | Task breakdown with checkboxes and verify steps. |
| `tasks/demo-artifact.md` | Demo report, review queue, packaging, and regression tests. |

## Conventions

- Living documents are updated in place; `changelog.md` is append-only, except
  for privacy and scope cleanups (recorded as their own entry).
- Numbers that were later corrected keep their original changelog entry and
  gain a correction entry; living docs carry the corrected number.
- Every experiment writes its artifacts under `reports/<name>/` and is
  registered in `experiments.md`.

## Current headline

Aim-only 29-channel wide CNN, 3 seeds x 5 fixed match folds, hard-negative
BCE alpha=5: player AUC **0.9502**, recall@FPR=1% **0.3619**, Q4 FPR **3.11%**,
Q4 recall **39.1%**, kill AUC **0.9015**. The packaged artifact is
`artifacts/valveguard_cnn_v1`; the demo path scores CS2CD match windows and
writes a player report and review queue under `reports/demo_dataset/`.

This repository contains the detector only. Demo reports use the CS2CD
dataset; no external demo data is required.
