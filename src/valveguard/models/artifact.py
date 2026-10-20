"""Load and run a packaged ValveGuard detector artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from valveguard.scripts.train_channel_cnn import CNN, WideCNN


class CNNDetector:
    def __init__(self, path: str | Path, verify: bool = True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.channel_names = json.loads(
            (self.path / self.manifest["channel_names_file"]).read_text()
        )
        self.aggregation = self.manifest["aggregation"]
        self.threshold = float(self.manifest["threshold"]["value"])
        if verify:
            self._verify()
        self.seeds = {}
        for seed in self.manifest["ensemble"]["seeds"]:
            norm = np.load(self.path / "seeds" / str(seed) / "norm.npz")
            models = []
            for fold in range(self.manifest["ensemble"]["models_per_seed"]):
                payload = torch.load(
                    self.path / "seeds" / str(seed) / f"fold{fold}.pt",
                    map_location="cpu", weights_only=True,
                )
                net = (WideCNN if payload["arch"] == "wide" else CNN)(
                    payload["channels"]
                )
                net.load_state_dict(payload["state_dict"])
                net.eval()
                models.append(net)
            self.seeds[seed] = (models, norm["center"], norm["scale"])
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
        x = channels[:, :, index].astype(np.float32)
        scores = np.zeros(len(x), dtype=np.float64)
        total = 0
        with torch.no_grad():
            for models, center, scale in self.seeds.values():
                z = (x - center.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
                z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10, 10)
                tensor = torch.from_numpy(z)
                for net in models:
                    batch = []
                    for start in range(0, len(tensor), 2048):
                        logits = net(tensor[start:start + 2048])
                        batch.append(torch.sigmoid(logits.float()).numpy())
                    scores += np.concatenate(batch)
                    total += 1
        return scores / total

    def aggregate(self, frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        k0 = float(self.aggregation["k0"])
        prior = float(self.aggregation["prior"])
        out = frame.assign(score=scores)
        grouped = out.groupby("player_name", sort=False).agg(
            kills=("score", "size"),
            total=("score", "sum"),
            mean=("score", "mean"),
            max_kill_score=("score", "max"),
        ).reset_index()
        grouped["shrunk_score"] = (
            grouped["total"] + k0 * prior
        ) / (grouped["kills"] + k0)
        grouped["percentile"] = [
            float((self.human_player_scores < value).mean() * 100.0)
            for value in grouped["shrunk_score"]
        ]
        grouped["flagged"] = grouped["shrunk_score"] >= self.threshold
        grouped["reliable"] = grouped["kills"] >= 3
        return grouped.sort_values("shrunk_score", ascending=False).reset_index(drop=True)


def load_detector(path: str | Path, verify: bool = True) -> CNNDetector:
    return CNNDetector(path, verify=verify)
