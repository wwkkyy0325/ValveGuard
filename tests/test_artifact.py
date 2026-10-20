import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valveguard.models.artifact import CNNDetector, load_detector

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "valveguard_cnn_v1"
pytestmark = pytest.mark.skipif(
    not (ARTIFACT / "manifest.json").exists(),
    reason="packaged CNN artifact is not present",
)


def test_load_and_score():
    detector = load_detector(ARTIFACT)
    assert detector.manifest["channels"] == 29
    assert detector.manifest["ensemble"]["seeds"] == [42, 43, 44]
    names = detector.channel_names + [f"unused_{i}" for i in range(14)]
    channels = np.zeros((2, 256, len(names)), dtype=np.float32)
    first = detector.score_channels(channels, names)
    second = detector.score_channels(channels, names)
    assert first.shape == (2,)
    assert np.isfinite(first).all()
    assert np.allclose(first, second, atol=1e-7)


def test_missing_channel_rejected():
    detector = load_detector(ARTIFACT)
    names = [f"other_{i}" for i in range(43)]
    channels = np.zeros((1, 256, 43), dtype=np.float32)
    with pytest.raises(ValueError):
        detector.score_channels(channels, names)


def test_checksum_mismatch_rejected(tmp_path):
    names = [f"channel_{i}" for i in range(29)]
    (tmp_path / "channel_names.json").write_text(json.dumps(names))
    (tmp_path / "dummy.txt").write_text("tampered")
    manifest = {
        "version": "test",
        "channels": 29,
        "channel_names_file": "channel_names.json",
        "aggregation": {"k0": 5.0, "prior": 0.3},
        "threshold": {"value": 0.6},
        "ensemble": {"seeds": [], "models_per_seed": 0},
        "reference": {"file": "reference.npz"},
        "checksums": {"dummy.txt": "0" * 64},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        CNNDetector(tmp_path)
