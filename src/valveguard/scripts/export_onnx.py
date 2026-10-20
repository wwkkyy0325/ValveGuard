"""Export the packaged CNN ensemble to a self-contained ONNX bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn

from valveguard.models.artifact import CNNDetector

KILL_ROW = 223


class _SeedModel(nn.Module):
    def __init__(self, net: nn.Module, center: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.net = net
        self.register_buffer(
            "center", torch.tensor(center, dtype=torch.float32).reshape(1, 1, -1)
        )
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float32).reshape(1, 1, -1)
        )

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        z = (channels - self.center) / self.scale
        z = torch.where(torch.isnan(z), torch.zeros_like(z), z)
        z = torch.clamp(z, -10.0, 10.0)
        return torch.sigmoid(self.net(z))


class _Ensemble(nn.Module):
    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        total = self.members[0](channels)
        for member in self.members[1:]:
            total = total + member(channels)
        return total / len(self.members)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_ensemble(detector: CNNDetector) -> _Ensemble:
    members = []
    for models, center, scale in detector.seeds.values():
        for net in models:
            net.eval()
            members.append(_SeedModel(net, center, scale))
    return _Ensemble(members).eval()


def _check_parity(detector, export_path: Path) -> float:
    import onnxruntime as ort

    names = detector.channel_names + [f"unused_{i}" for i in range(14)]
    rng = np.random.default_rng(42)
    channels = rng.normal(size=(4, 256, len(names))).astype(np.float32)
    expected = detector.score_channels(channels, names)
    session = ort.InferenceSession(
        str(export_path), providers=["CPUExecutionProvider"]
    )
    index = [names.index(name) for name in detector.channel_names]
    actual = session.run(
        ["scores"], {"channels": channels[:, :, index]}
    )[0]
    return float(np.max(np.abs(expected - np.asarray(actual).reshape(-1))))


def _write_readme(out: Path) -> None:
    (out / "README.md").write_text(
        "# ValveGuard CNN v1 - ONNX bundle\n\n"
        "Self-contained ONNX export of the aim-only 29-channel wide CNN\n"
        "ensemble (3 seeds x 5 folds, hard-negative BCE alpha=5).\n\n"
        "## Install\n\n"
        "```bash\n"
        "pip install onnxruntime numpy pandas\n"
        "```\n\n"
        "## Run\n\n"
        "```python\n"
        "from valveguard.models.onnx_artifact import load_onnx_detector\n\n"
        "detector = load_onnx_detector('path/to/valveguard_cnn_v1_onnx')\n"
        "names = detector.channel_names + other_channel_names\n"
        "scores = detector.score_channels(channels_256_by_c, names)\n"
        "players = detector.aggregate(frame, scores)\n"
        "```\n\n"
        "This path does not import torch; the runtime needs only\n"
        "`onnxruntime`, `numpy`, and `pandas`.\n\n"
        "`channels_256_by_c` is a float32 array of shape\n"
        "`(N, 256, C)`; channels are selected by name. The ONNX graph\n"
        "contains the per-seed normalization, the 15 fold models, and the\n"
        "sigmoid average. Player aggregation is done by the loader with\n"
        "the manifest's `k0`, prior, and threshold.\n\n"
        "The existing demo scripts accept this bundle directly:\n\n"
        "```bash\n"
        "python -m valveguard.scripts.demo_report_dataset \\\n"
        "  --artifact path/to/valveguard_cnn_v1_onnx --match cdata:0\n"
        "```\n\n"
        "## Contents\n\n"
        "- `model.onnx` - batch-dynamic ensemble graph (input `channels`,\n"
        "  output `scores`).\n"
        "- `manifest.json` - channel list, normalization, aggregation,\n"
        "  threshold, metrics, reference files, checksums.\n"
        "- `channel_names.json` - the 29 aim channels.\n"
        "- `reference.npz` / `reference.json` - human score distributions.\n\n"
        "Offline demo analysis only. Not for live-game integration.\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/valveguard_cnn_v1")
    ap.add_argument("--out", default="artifacts/valveguard_cnn_v1_onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--zip", action="store_true",
                    help="also write a sibling .zip archive")
    args = ap.parse_args()

    source = Path(args.artifact)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = CNNDetector(source)
    model = _build_ensemble(detector)
    dummy = torch.zeros(2, 256, detector.manifest["channels"], dtype=torch.float32)
    export_path = out / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        export_path,
        input_names=["channels"],
        output_names=["scores"],
        dynamic_axes={"channels": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx.checker.check_model(str(export_path))
    max_diff = _check_parity(detector, export_path)
    if max_diff >= 1e-5:
        raise SystemExit(f"ONNX parity check failed: max abs diff {max_diff}")

    for name in ("channel_names.json", "reference.npz", "reference.json"):
        shutil.copy2(source / name, out / name)
    manifest = dict(detector.manifest)
    manifest.pop("checksums", None)
    manifest["version"] = f"{detector.manifest['version']}_onnx"
    manifest["format"] = "onnx"
    manifest["created"] = date.today().isoformat()
    manifest["source_version"] = detector.manifest["version"]
    manifest["input"] = {
        **detector.manifest.get("input", {}),
        "tensor": "(N, 256, 29) float32, selected by channel name",
    }
    manifest["normalization"] = {
        "method": detector.manifest.get("normalization", {}).get(
            "method", "median_iqr"
        ),
        "embedded": True,
        "note": "per-seed normalization is inside model.onnx",
    }
    manifest["onnx"] = {
        "file": "model.onnx",
        "opset": args.opset,
        "input_name": "channels",
        "output_name": "scores",
        "input_shape": [None, 256, detector.manifest["channels"]],
        "output_shape": [None],
        "size_bytes": export_path.stat().st_size,
        "max_abs_diff_vs_torch": max_diff,
    }
    manifest["reference"] = {
        "file": "reference.npz",
        "summary": "reference.json",
    }
    files = ("model.onnx", "channel_names.json", "reference.npz",
             "reference.json")
    manifest["checksums"] = {name: _sha256(out / name) for name in files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_readme(out)

    if args.zip:
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(out))
        print("zip", archive, archive.stat().st_size, flush=True)

    print(json.dumps({
        "out": str(out),
        "model_bytes": export_path.stat().st_size,
        "models": len(model.members),
        "opset": args.opset,
        "max_abs_diff_vs_torch": max_diff,
    }, indent=2))


if __name__ == "__main__":
    main()
