"""Concatenate two row-aligned per-tick channel caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    base = Path(args.base)
    extra = Path(args.extra)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a = np.load(base / "channels.npy", mmap_mode="r")
    b = np.load(extra / "channels_raw.npy", mmap_mode="r")
    if a.shape[:2] != b.shape[:2]:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")

    names = json.loads((base / "channel_names.json").read_text())
    names += json.loads((extra / "channel_names_raw.json").read_text())
    shape = (a.shape[0], a.shape[1], a.shape[2] + b.shape[2])
    target = np.lib.format.open_memmap(
        out / "channels.npy", mode="w+", dtype=np.float16, shape=shape
    )
    for start in range(0, shape[0], args.chunk):
        stop = min(start + args.chunk, shape[0])
        target[start:stop, :, :a.shape[2]] = a[start:stop]
        target[start:stop, :, a.shape[2]:] = b[start:stop]
    target.flush()

    (out / "channel_names.json").write_text(json.dumps(names, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "shape": list(shape),
        "base": str(base),
        "extra": str(extra),
    }, indent=2))
    print(f"wrote {shape} with {len(names)} channels", flush=True)


if __name__ == "__main__":
    main()
