"""Map cached window rows to their absolute kill tick using raw match events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = pd.read_parquet(Path(args.cache) / "index.parquet")
    raw = Path(args.raw)
    rows = []
    for match, group in index.groupby("match_key", sort=False):
        source, idx = match.split(":")
        folder = "with_cheater_present" if source == "cdata" else "no_cheater_present"
        path = raw / folder / f"{int(idx)}.json"
        if not path.exists():
            continue
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
            deaths = sorted(events["player_death"], key=lambda e: int(e["tick"]))
        except Exception:
            continue
        kills: dict[str, list[int]] = {}
        for event in deaths:
            attacker = str(event.get("attacker_steamid"))
            kills.setdefault(attacker, []).append(int(event["tick"]))
        for player, kill_idx, row in zip(
            group["player_name"], group["kill_idx"], group["row"]
        ):
            ticks = kills.get(str(player))
            if ticks is None or int(kill_idx) >= len(ticks):
                continue
            rows.append({
                "row": int(row),
                "match_key": match,
                "player_name": str(player),
                "kill_idx": int(kill_idx),
                "kill_tick": int(ticks[int(kill_idx)]),
            })
        if len(rows) % 10000 < 100:
            print("rows", len(rows), flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print("wrote", len(out), "rows to", args.out)


if __name__ == "__main__":
    main()
