import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valveguard.models.artifact import load_detector

ROOT = Path(__file__).resolve().parents[1]
TORCH = ROOT / "artifacts" / "valveguard_cnn_v1"
ONNX = ROOT / "artifacts" / "valveguard_cnn_v1_onnx"
pytestmark = pytest.mark.skipif(
    not (TORCH / "manifest.json").exists() or not (ONNX / "model.onnx").exists(),
    reason="packaged CNN artifacts are not present",
)


def test_onnx_loader_and_parity():
    torch_detector = load_detector(TORCH)
    onnx_detector = load_detector(ONNX)
    assert type(onnx_detector).__name__ == "ONNXDetector"
    assert onnx_detector.manifest["format"] == "onnx"
    names = json.loads((ROOT / "artifacts" / "valveguard_cnn_v1" /
                        "channel_names.json").read_text())
    names += [f"unused_{i}" for i in range(14)]
    rng = np.random.default_rng(7)
    channels = rng.normal(size=(3, 256, len(names))).astype(np.float32)
    expected = torch_detector.score_channels(channels, names)
    actual = onnx_detector.score_channels(channels, names)
    assert np.max(np.abs(expected - actual)) < 1e-5
    assert np.allclose(
        actual, onnx_detector.score_channels(channels, names), atol=1e-7
    )


def test_onnx_aggregate_matches_torch():
    torch_detector = load_detector(TORCH)
    onnx_detector = load_detector(ONNX)
    names = json.loads((ROOT / "artifacts" / "valveguard_cnn_v1" /
                        "channel_names.json").read_text())
    channels = np.zeros((4, 256, len(names)), dtype=np.float32)
    scores = onnx_detector.score_channels(channels, names)
    frame = pd.DataFrame({
        "player_name": ["a", "a", "b", "b"],
        "kill_tick": [100, 200, 300, 400],
    })
    left = torch_detector.aggregate(frame, scores)
    right = onnx_detector.aggregate(frame, scores)
    assert np.allclose(left["shrunk_score"], right["shrunk_score"], atol=1e-9)
    assert left["flagged"].tolist() == right["flagged"].tolist()


def test_onnx_missing_channel_rejected():
    detector = load_detector(ONNX)
    with pytest.raises(ValueError):
        detector.score_channels(
            np.zeros((1, 256, 43), dtype=np.float32),
            [f"other_{i}" for i in range(43)],
        )
