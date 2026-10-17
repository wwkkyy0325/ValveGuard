"""Export the GBDT to a portable JSON tree ensemble.

The Rust application cannot load a sklearn pickle, and the ONNX converter
fails on this HistGradientBoosting model.  The trees are therefore exported as
plain arrays and evaluated by a small Rust traversal.  The traversal was
verified to reproduce sklearn's `predict_proba` exactly.

Usage:
    python -m valveguard.scripts.export_model_json \
        --model artifacts/valveguard_gbdt_v1.joblib \
        --out artifacts/valveguard_gbdt_v1.model.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    art = joblib.load(args.model)
    model = art["model"]
    trees = []
    for iteration in model._predictors:
        predictor = iteration[0]
        nodes = predictor.nodes
        thr = nodes["num_threshold"].astype(float)
        if np.isnan(thr).any():
            raise ValueError("NaN threshold in GBDT node")
        # +inf means "all finite values go left"; JSON has no Infinity literal,
        # so use a sentinel that preserves the comparison for finite features.
        thr = np.where(np.isinf(thr), np.sign(thr) * 1e308, thr)
        value = nodes["value"].astype(float)
        if not np.isfinite(value).all():
            raise ValueError("non-finite leaf value in GBDT node")
        trees.append({
            "feature": nodes["feature_idx"].astype(int).tolist(),
            "threshold": thr.tolist(),
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "leaf": nodes["is_leaf"].astype(bool).tolist(),
            "missing_left": nodes["missing_go_to_left"].astype(bool).tolist(),
            "value": value.tolist(),
        })

    out = {
        "version": art["version"],
        "features": art["features"],
        "threshold": float(art["threshold"]),
        "prior": float(art["prior"]),
        "k0": float(art["k0"]),
        "baseline": float(np.ravel(model._baseline_prediction)[0]),
        "trees": trees,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(json.dumps({
        "out": str(path),
        "trees": len(trees),
        "features": len(out["features"]),
        "bytes": path.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
