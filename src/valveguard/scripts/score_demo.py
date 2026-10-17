"""Score one CS2 demo with a saved artifact.

Usage:
    python -m valveguard.scripts.score_demo \
        --demo match.dem --model artifacts/valveguard_gbdt_v1.joblib \
        --out result.json

Outputs per-player evidence: kill count, mean and shrunk score, max kill
score, a reliability flag (>=3 kills), and a flag at the stored player-level
1% threshold.  The score is a risk score, not a calibrated probability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from valveguard.scripts.build_pro_windows import demo_windows


def player_names(demo: Path) -> dict[str, str]:
    try:
        from demoparser2 import DemoParser

        info = DemoParser(str(demo)).parse_player_info()
    except Exception:
        return {}
    if "steamid" not in info.columns or "name" not in info.columns:
        return {}
    return {
        str(int(s)): str(n)
        for s, n in zip(info["steamid"], info["name"])
        if pd.notna(s)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-kills", action="store_true")
    args = ap.parse_args()

    art = joblib.load(args.model)
    cols = art["features"]
    demo = Path(args.demo)
    feats = demo_windows(demo)
    if len(feats) == 0:
        raise SystemExit(f"no valid kill windows in {demo}")
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        raise SystemExit(f"artifact expects missing features: {missing[:5]}")

    X = feats[cols].replace([np.inf, -np.inf], np.nan)
    feats = feats.assign(score=art["model"].predict_proba(X)[:, 1])

    names = player_names(demo)
    grouped = feats.groupby("player_name", sort=False).agg(
        kills=("score", "size"),
        total=("score", "sum"),
        mean=("score", "mean"),
        max=("score", "max"),
    ).reset_index()
    k0 = float(art["k0"])
    grouped["shrunk"] = (grouped["total"] + k0 * float(art["prior"])) / (grouped["kills"] + k0)
    grouped["flagged"] = grouped["shrunk"] >= float(art["threshold"])
    grouped["reliable"] = grouped["kills"] >= 3
    grouped["name"] = grouped["player_name"].map(names).fillna(grouped["player_name"])

    players = []
    for _, r in grouped.sort_values("shrunk", ascending=False).iterrows():
        players.append({
            "steamid": str(r["player_name"]),
            "name": str(r["name"]),
            "kills": int(r["kills"]),
            "mean_score": round(float(r["mean"]), 6),
            "max_kill_score": round(float(r["max"]), 6),
            "shrunk_score": round(float(r["shrunk"]), 6),
            "flagged": bool(r["flagged"]),
            "reliable": bool(r["reliable"]),
        })

    result = {
        "demo": str(demo),
        "map": str(feats["map"].iloc[0]),
        "model_version": art["version"],
        "threshold": float(art["threshold"]),
        "prior": float(art["prior"]),
        "k0": k0,
        "players": players,
    }
    if args.include_kills:
        result["kills"] = [
            {
                "steamid": str(r["player_name"]),
                "kill_tick": int(r["kill_tick"]),
                "score": round(float(r["score"]), 6),
            }
            for _, r in feats.sort_values(["player_name", "kill_tick"]).iterrows()
        ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"demo": result["demo"], "map": result["map"],
                      "players": len(players),
                      "flagged": sum(p["flagged"] for p in players)}, indent=2))


if __name__ == "__main__":
    main()
