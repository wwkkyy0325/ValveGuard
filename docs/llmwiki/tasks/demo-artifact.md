# Demo Artifact and Report

Status: active (2026-10-20)
Champion: `artifacts/valveguard_cnn_v1` (aim-only 29-channel wide CNN,
3 seeds x 5 folds, hard-negative BCE alpha=5).

## Steps

- [x] Package the local artifact: manifest, channel names, human reference
      distributions, 15 fold models, per-seed normalization, checksums.
      Verify: loader recomputes checksums and rejects mismatches.
- [x] Artifact loader `valveguard/models/artifact.py`.
      Verify: selects channels by name, scores a batch, aggregates players.
- [x] `demo_report.py`: score one demo or a directory of demos.
      Verify: arbitrary `.dem` input still works with the packaged artifact.
- [x] `demo_report_dataset.py`: score CS2CD match windows without a raw demo.
      Verify: `cdata:0` gives 10 players and 1 FLAG; `ncdata:36` gives 10
      players and 0 FLAG/REVIEW.
- [x] Charts: player-vs-human distribution and per-player kill windows.
      Verify: both PNGs render and the footer explains the axes.
- [x] Review queue: behavior reasons order REVIEW players; flag unchanged.
      Verify: the `cdata:0` cheater ranks first with 4 reasons; the clean
      match has no FLAG/REVIEW.
- [ ] ONNX export and desktop loader.
      Verify: PyTorch vs ONNX max abs difference < 1e-5.
- [ ] `compare_demos.py` human / synthetic / cheater harness.
      Verify: labeled demos produce grouped distributions and a summary table.
- [x] Regression tests for artifact load, schema, and review ordering.
      Verify: `python -m pytest -q tests` passes (11 tests).
