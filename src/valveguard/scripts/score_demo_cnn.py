"""Score one CS2 demo with a saved fold ensemble of the channel CNN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from valveguard.features.raw_window import build_window, weapon_class
from valveguard.features.summarize import per_tick_channels
from valveguard.scripts.build_pro_windows import KILL_ROW, parse_demo
from valveguard.scripts.train_channel_cnn import CNN, WideCNN

K0 = 5.0
PRIOR = 0.3018746014732038


def _flag(value) -> float:
    return 0.0 if pd.isna(value) else float(bool(value))


def _number(value) -> float:
    return 0.0 if pd.isna(value) else float(value)


def demo_channels(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    map_name, ticks, events = parse_demo(path)
    by_player = {
        sid: d.sort_values("tick").set_index("tick", drop=False)
        for sid, d in ticks.groupby("steamid", sort=False)
    }
    rows, channels, names = [], [], []
    for _, ev in events.sort_values("tick").iterrows():
        attacker_id = ev["attacker_steamid"]
        victim_id = ev["user_steamid"]
        if pd.isna(attacker_id) or pd.isna(victim_id) or attacker_id == victim_id:
            continue
        if ev["attacker_steamid"] not in by_player or ev["user_steamid"] not in by_player:
            continue
        kill_tick = int(ev["tick"])
        if kill_tick < KILL_ROW:
            continue
        kill_row = ticks[
            (ticks["steamid"] == ev["attacker_steamid"]) & (ticks["tick"] == kill_tick)
        ]
        if len(kill_row) == 0:
            continue
        warm = kill_row["is_warmup_period"].iloc[0]
        if pd.notna(warm) and bool(warm):
            continue
        tick_range = np.arange(kill_tick - KILL_ROW, kill_tick + 33)
        attacker = by_player[ev["attacker_steamid"]]
        victim = by_player[ev["user_steamid"]]
        if attacker.reindex(tick_range)[["X", "pitch", "yaw"]].isna().any().any():
            continue
        if victim.reindex(tick_range)[["X", "health"]].isna().any().any():
            continue
        window = build_window(
            attacker, victim, tick_range, map_name, kill_tick, ev.to_dict()
        )
        if window is None:
            continue
        matrix, window_names = per_tick_channels(window)
        rows.append({
            "player_name": str(int(ev["attacker_steamid"])),
            "kill_tick": kill_tick,
            "map": map_name,
            "headshot": _flag(ev.get("headshot")),
            "weapon_class": weapon_class(str(ev.get("weapon", ""))),
            "shots": float(window["attacker_shot"].iloc[:KILL_ROW].sum()),
            "through_smoke": _flag(ev.get("thrusmoke")),
            "wallbang": float(_number(ev.get("penetrated")) > 0),
            "midair": _flag(ev.get("attackerinair")),
        })
        channels.append(matrix)
        names = window_names
    if not rows:
        return pd.DataFrame(), np.zeros((0, 256, 0), dtype=np.float32), []
    return pd.DataFrame(rows), np.stack(channels).astype(np.float32), names


def load_models(model_dir: Path):
    meta = json.loads((model_dir / "model_meta.json").read_text())
    norm = np.load(model_dir / "norm.npz")
    keep = (
        [int(x) for x in meta["channel_index"].split(",")]
        if meta.get("channel_index") else None
    )
    models = []
    for path in sorted(model_dir.glob("fold*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        net = (WideCNN if payload["arch"] == "wide" else CNN)(payload["channels"])
        net.load_state_dict(payload["state_dict"])
        net.eval()
        models.append(net)
    if not models:
        raise SystemExit(f"no fold models in {model_dir}")
    return meta, keep, norm["center"], norm["scale"], models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--model-dir", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--channel-index", default=None,
                    help="indices into the 43-channel layout when the model meta has none")
    ap.add_argument("--include-kills", action="store_true")
    args = ap.parse_args()

    groups = [load_models(Path(p)) for p in args.model_dir]
    meta, keep = groups[0][0], groups[0][1]
    if keep is None and args.channel_index:
        keep = [int(x) for x in args.channel_index.split(",")]
    demo = Path(args.demo)
    frame, channels, names = demo_channels(demo)
    if len(frame) == 0:
        raise SystemExit(f"no valid kill windows in {demo}")
    if keep is not None:
        channels = channels[:, :, keep]

    scores = np.zeros(len(frame), dtype=float)
    total_models = 0
    with torch.no_grad():
        for _, group_keep, center, scale, models in groups:
            if group_keep is not None and group_keep != keep:
                raise SystemExit("model directories use different channel subsets")
            x = (channels - center.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
            x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10, 10)
            tensor = torch.from_numpy(x.astype(np.float32))
            for net in models:
                batch = []
                for start in range(0, len(tensor), 2048):
                    logits = net(tensor[start:start + 2048])
                    batch.append(torch.sigmoid(logits.float()).numpy())
                scores += np.concatenate(batch)
                total_models += 1
    scores /= total_models
    frame["score"] = scores

    grouped = frame.groupby("player_name", sort=False).agg(
        kills=("score", "size"),
        total=("score", "sum"),
        mean=("score", "mean"),
        max=("score", "max"),
    ).reset_index()
    grouped["shrunk"] = (grouped["total"] + K0 * PRIOR) / (grouped["kills"] + K0)
    grouped["flagged"] = (
        grouped["shrunk"] >= args.threshold if args.threshold is not None else False
    )
    grouped["reliable"] = grouped["kills"] >= 3

    result = {
        "demo": str(demo),
        "map": str(frame["map"].iloc[0]),
        "models": total_models,
        "channels": int(channels.shape[2]),
        "threshold": args.threshold,
        "prior": PRIOR,
        "k0": K0,
        "players": [
            {
                "steamid": str(r["player_name"]),
                "kills": int(r["kills"]),
                "mean_score": round(float(r["mean"]), 6),
                "max_kill_score": round(float(r["max"]), 6),
                "shrunk_score": round(float(r["shrunk"]), 6),
                "flagged": bool(r["flagged"]),
                "reliable": bool(r["reliable"]),
            }
            for _, r in grouped.sort_values("shrunk", ascending=False).iterrows()
        ],
    }
    if args.include_kills:
        result["kills"] = [
            {
                "steamid": r["player_name"],
                "kill_tick": int(r["kill_tick"]),
                "score": round(float(r["score"]), 6),
            }
            for _, r in frame.sort_values(["player_name", "kill_tick"]).iterrows()
        ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "demo": result["demo"],
        "map": result["map"],
        "players": len(result["players"]),
        "flagged": sum(p["flagged"] for p in result["players"]),
    }, indent=2))


if __name__ == "__main__":
    main()
