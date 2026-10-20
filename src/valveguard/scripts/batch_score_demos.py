"""Batch-score demos with a packaged artifact and join processed labels."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.models.artifact import load_detector
from valveguard.scripts.score_demo_cnn import demo_channels


def match_id(path: Path) -> str:
    found = re.match(r"^(.+?)_de_", path.stem)
    return found.group(1) if found else path.stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    detector = load_detector(args.artifact)
    labels = {
        row["match_id"]: row["players"]
        for row in json.loads(Path(args.labels).read_text())
    }
    demos = sorted(Path(args.demos).glob("*.dem"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, per_demo = [], {}
    for demo in demos:
        mid = match_id(demo)
        try:
            frame, channels, names = demo_channels(demo)
        except Exception as exc:
            print("skip", demo.name, repr(exc), flush=True)
            continue
        if len(frame) == 0:
            print("empty", demo.name, flush=True)
            continue
        scores = detector.score_channels(channels, names)
        frame["score"] = scores
        players = detector.aggregate(frame, scores)
        players = players.rename(columns={"player_name": "steamid"})
        players["rank"] = players["shrunk_score"].rank(ascending=False, method="first")
        per_demo[mid] = players.to_dict(orient="records")
        print(mid, "players", len(players), "top", round(float(players["shrunk_score"].max()), 3),
              flush=True)
        for labeled in labels.get(mid, []):
            target = float(labeled["steamid"])
            demo_ids = players["steamid"].astype(str).to_numpy()
            distance = np.array([abs(float(sid) - target) for sid in demo_ids])
            index = int(np.argmin(distance))
            if distance[index] > 1000:
                rows.append({"match_id": mid, "isCheater": labeled["isCheater"],
                             "matched": False, "distance": float(distance[index])})
                continue
            player = players.iloc[index]
            rows.append({
                "match_id": mid,
                "isCheater": bool(labeled["isCheater"]),
                "matched": True,
                "distance": float(distance[index]),
                "steamid": str(player["steamid"]),
                "kills": int(player["kills"]),
                "score": float(player["shrunk_score"]),
                "percentile": float(player["percentile"]),
                "flagged": bool(player["flagged"]),
                "rank": int(player["rank"]),
                "players": int(len(players)),
                "top_score": float(players["shrunk_score"].max()),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "labeled_players.csv", index=False)
    (out / "per_demo.json").write_text(json.dumps(per_demo, indent=2))
    print(frame.to_string())


if __name__ == "__main__":
    main()
