# ValveGuard CNN v1 - ONNX bundle

Self-contained ONNX export of the aim-only 29-channel wide CNN
ensemble (3 seeds x 5 folds, hard-negative BCE alpha=5).

## Install

```bash
pip install onnxruntime numpy pandas
```

## Run

```python
from valveguard.models.onnx_artifact import load_onnx_detector

detector = load_onnx_detector('path/to/valveguard_cnn_v1_onnx')
names = detector.channel_names + other_channel_names
scores = detector.score_channels(channels_256_by_c, names)
players = detector.aggregate(frame, scores)
```

This path does not import torch; the runtime needs only
`onnxruntime`, `numpy`, and `pandas`.

`channels_256_by_c` is a float32 array of shape
`(N, 256, C)`; channels are selected by name. The ONNX graph
contains the per-seed normalization, the 15 fold models, and the
sigmoid average. Player aggregation is done by the loader with
the manifest's `k0`, prior, and threshold.

The existing demo scripts accept this bundle directly:

```bash
python -m valveguard.scripts.demo_report_dataset \
  --artifact path/to/valveguard_cnn_v1_onnx --match cdata:0
```

## Contents

- `model.onnx` - batch-dynamic ensemble graph (input `channels`,
  output `scores`).
- `manifest.json` - channel list, normalization, aggregation,
  threshold, metrics, reference files, checksums.
- `channel_names.json` - the 29 aim channels.
- `reference.npz` / `reference.json` - human score distributions.

Offline demo analysis only. Not for live-game integration.
