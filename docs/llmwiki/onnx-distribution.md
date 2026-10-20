# ONNX Distribution Bundle

Status: working (2026-10-20). Champion: `valveguard_cnn_v1`.

The ONNX bundle is the portable inference format for the CNN champion. It
contains the full 15-model ensemble (3 seeds x 5 folds), per-seed
normalization, and the sigmoid average in a single batch-dynamic graph.

## Build

```bash
pip install onnx onnxruntime
PYTHONPATH=src python -m valveguard.scripts.export_onnx \
  --artifact artifacts/valveguard_cnn_v1 \
  --out artifacts/valveguard_cnn_v1_onnx \
  --zip
```

The command writes the bundle and a sibling zip, then fails if the ONNX and
PyTorch scores differ by 1e-5 or more.

## Layout

```
valveguard_cnn_v1_onnx/
  model.onnx          31.5 MB, opset 17, dynamic batch
  manifest.json       channel names, aggregation, threshold, checksums
  channel_names.json  the 29 aim channels
  reference.npz       human player/window score distributions
  reference.json      percentile summary
  README.md           runtime notes
```

Input: `channels`, float32 `(N, 256, 29)`, selected by channel name from the
43-channel demo layout. Output: `scores`, float32 `(N,)`.

## Verification

- Random channels: max abs difference vs PyTorch `2.58e-8`.
- `cdata:0` real windows: max abs difference `1.27e-7`.
- Demo report: identical players, tiers, and flags; max per-kill difference
  `1.0e-6`.
- `tests/test_onnx_artifact.py` covers load, parity, aggregation, and
  missing-channel rejection.

CPU timing for 78 windows (cdata:0):

| Backend | ms |
|---|---|
| PyTorch (15 models) | 401 |
| ONNX Runtime | 520 |

ONNX is chosen for portability and a dependency-light runtime, not for speed
on this machine.

## Runtime

```python
from valveguard.models.onnx_artifact import load_onnx_detector

detector = load_onnx_detector("artifacts/valveguard_cnn_v1_onnx")
scores = detector.score_channels(channels, names)
players = detector.aggregate(frame, scores)
```

This import path does not import torch; the runtime needs only
`onnxruntime`, `numpy`, and `pandas`. The higher-level
`valveguard.models.artifact.load_detector` also accepts the bundle when
torch is installed, and the existing demo scripts need only
`--artifact artifacts/valveguard_cnn_v1_onnx`.

```bash
PYTHONPATH=src python -m valveguard.scripts.demo_report_dataset \
  --artifact artifacts/valveguard_cnn_v1_onnx \
  --match cdata:0 --out reports/demo_onnx
```

## Distribution

`artifacts/` is gitignored, so the 28 MB zip is not committed. Attach
`artifacts/valveguard_cnn_v1_onnx.zip` to a GitHub Release, or rebuild it
from the tracked scripts and the local artifact. The manifest records
SHA256 checksums for every bundle file; the loader rejects mismatches.

## Limitations

- The bundle is offline inference only; no live-game integration.
- The input must be 256 ticks and the 29 channels must be supplied by name.
- Player aggregation and thresholds are implemented in the Python loader,
  not in the ONNX graph.
- If the source artifact changes, rebuild the bundle and update the version.
