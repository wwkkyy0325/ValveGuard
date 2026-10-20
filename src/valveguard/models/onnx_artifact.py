"""Load and run the packaged ONNX detector bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.models.aggregation import aggregate_players


class ONNXDetector:
    def __init__(self, path: str | Path, verify: bool = True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.channel_names = json.loads(
            (self.path / self.manifest["channel_names_file"]).read_text()
        )
        self.aggregation = self.manifest["aggregation"]
        self.threshold = float(self.manifest["threshold"]["value"])
        self.onnx = self.manifest["onnx"]
        if verify:
            self._verify()
        import onnxruntime as ort

        model_path = self.path / self.onnx.get("file", "model.onnx")
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.onnx.get("input_name", "channels")
        self.output_name = self.onnx.get("output_name", "scores")
        reference = np.load(self.path / self.manifest["reference"]["file"])
        self.human_player_scores = reference["human_player_scores"]
        self.human_window_scores = reference["human_window_scores"]

    def _verify(self) -> None:
        for relative, expected in self.manifest.get("checksums", {}).items():
            actual = hashlib.sha256((self.path / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        if len(self.channel_names) != self.manifest["channels"]:
            raise ValueError("channel name count does not match the manifest")

    def score_channels(self, channels: np.ndarray, names: list[str]) -> np.ndarray:
        """Score (N, 256, C) windows; names describe the C input channels."""
        if channels.ndim != 3:
            raise ValueError(f"expected (N, T, C), got {channels.shape}")
        missing = [name for name in self.channel_names if name not in names]
        if missing:
            raise ValueError(f"missing channels: {missing[:5]}")
        index = [names.index(name) for name in self.channel_names]
        x = np.asarray(channels[:, :, index], dtype=np.float32)
        if x.shape[1] != 256:
            raise ValueError(f"expected 256 ticks, got {x.shape[1]}")
        scores = self.session.run(
            [self.output_name], {self.input_name: x}
        )[0]
        return np.asarray(scores, dtype=np.float64).reshape(-1)

    def aggregate(self, frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        return aggregate_players(
            frame, scores, self.aggregation, self.human_player_scores,
            self.threshold,
        )


def load_onnx_detector(path: str | Path, verify: bool = True) -> ONNXDetector:
    """Load an ONNX bundle without importing torch."""
    return ONNXDetector(path, verify=verify)
