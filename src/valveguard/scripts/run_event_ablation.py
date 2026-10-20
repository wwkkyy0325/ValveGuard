"""Ablate raw event-feature groups on the pilot subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from valveguard.models.baseline import feature_columns
from valveguard.scripts.run_event_pilot import run

GROUPS = {
    "recoil": ("ev_punch_", "ev_view_punch_", "ev_recoil_"),
    "fire": ("ev_shot_", "ev_first_shot_", "ev_last_shot_", "ev_spray_", "ev_bursts"),
    "mouse": ("ev_mouse_",),
    "acquire": (
        "ev_err_", "ev_acquire_", "ev_first_shot_after_acquire_",
        "ev_time_on_target_", "ev_post_acquire_", "ev_error_at_first_shot",
        "ev_error_drop_before_shot", "ev_first_shot_to_kill",
    ),
    "context": (
        "ev_scoped_", "ev_walking_", "ev_ducking_", "ev_airborne_",
        "ev_spotted_", "ev_velocity_", "ev_alive_",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="run one arm only")
    args = parser.parse_args()

    index = pd.read_parquet(args.index)[
        ["row", "label", "match_key", "player_name", "source"]
    ]
    base = pd.read_parquet(args.features)
    events = pd.read_parquet(args.events)
    events = events[["row"] + [c for c in events.columns if c.startswith("ev_")]]
    data = index.merge(base, on="row", how="inner").merge(events, on="row", how="inner")
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()

    base_columns = [c for c in feature_columns(data) if not c.startswith("ev_")]
    event_columns = [c for c in data.columns if c.startswith("ev_")]
    arms = {"base": base_columns}
    for name, prefixes in GROUPS.items():
        arms[name] = base_columns + [
            column for column in event_columns if column.startswith(prefixes)
        ]
    aim_prefixes = GROUPS["recoil"] + GROUPS["fire"] + GROUPS["mouse"]
    arms["aim"] = base_columns + [
        column for column in event_columns if column.startswith(aim_prefixes)
    ]
    aim_all_prefixes = aim_prefixes + GROUPS["acquire"]
    arms["aim_all"] = base_columns + [
        column for column in event_columns if column.startswith(aim_all_prefixes)
    ]
    arms["all"] = base_columns + event_columns
    if args.only:
        if args.only not in arms:
            raise SystemExit(f"unknown arm: {args.only}")
        arms = {args.only: arms[args.only]}

    results = {}
    for name, columns in arms.items():
        result, _, _ = run(data, columns, args.seed)
        result["n_features"] = len(columns)
        results[name] = result
        print(name, json.dumps(result), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filename = f"ablation_{args.only}.json" if args.only else "ablation.json"
    (out / filename).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
