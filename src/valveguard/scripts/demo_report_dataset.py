"""Score one CS2CD match with a packaged artifact and draw a review report.

The dataset is the only demo source: windows come from the CS2CD cache and
player names are the per-match pseudonyms stored in the dataset.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.models.artifact import load_detector
from valveguard.scripts.demo_report import (
    _add_review_order,
    _behavior_context,
    _behavior_table,
    _combine_review_queues,
    _player_panels,
    _summary_plot,
)

KILL_ROW = 223
WEAPON_CHANNELS = [
    ("attacker_weapon_knife", 0),
    ("attacker_weapon_auto_rifle", 1),
    ("attacker_weapon_semi_rifle", 2),
    ("attacker_weapon_pistol", 3),
    ("attacker_weapon_grenade", 4),
    ("attacker_weapon_smg", 5),
    ("attacker_weapon_shotgun", 6),
]


def _map_name(tar_path: Path, member: str) -> str:
    try:
        with tarfile.open(tar_path) as handle:
            payload = handle.extractfile(member)
            if payload is None:
                return "unknown"
            window = pd.read_parquet(payload)
    except Exception:
        return "unknown"
    for column in window.columns:
        if column.startswith("map_") and float(window[column].iloc[0]) > 0.5:
            return column.removeprefix("map_")
    return "unknown"


def build_frame(rows: pd.DataFrame, channels: np.ndarray,
                names: list[str]) -> pd.DataFrame:
    position = {name: i for i, name in enumerate(names)}
    kill_row = channels[:, KILL_ROW, :]
    weapon = np.full(len(channels), 3, dtype=int)
    for column, code in WEAPON_CHANNELS:
        weapon[kill_row[:, position[column]] > 0.5] = code
    frame = pd.DataFrame({
        "player_name": rows["player_name"].to_numpy(),
        "kill_tick": rows["kill_tick"].to_numpy(),
        "label": rows["label"].to_numpy(),
        "headshot": kill_row[:, position["is_kill_headshot"]],
        "weapon_class": weapon,
        "shots": channels[:, :KILL_ROW, position["attacker_shot"]].sum(axis=1),
        "through_smoke": kill_row[:, position["is_kill_through_smoke"]],
        "wallbang": kill_row[:, position["is_kill_wallbang"]],
        "midair": kill_row[:, position["attacker_midair"]],
    })
    return frame


def run_match(match: str, detector, cache: Path, channels_path: Path,
              kill_ticks: Path, tar_path: Path, out: Path,
              threshold: float | None) -> dict:
    index = pd.read_parquet(cache / "index.parquet")
    index = index[index["match_key"] == match].copy()
    kills = pd.read_parquet(kill_ticks)
    merged = index.merge(
        kills[["row", "kill_tick"]], on="row", how="inner"
    ).reset_index(drop=True)
    if merged.empty:
        raise SystemExit(f"no kill ticks for {match}")

    names = json.loads((channels_path / "channel_names.json").read_text())
    source = np.load(channels_path / "channels.npy", mmap_mode="r")
    channels = np.asarray(source[merged["row"].to_numpy()], dtype=np.float32)
    scores = detector.score_channels(channels, names)
    frame = build_frame(merged, channels, names)
    frame["score"] = scores

    players = detector.aggregate(frame, scores)
    players = players.merge(_behavior_context(frame), on="player_name", how="left")
    players = players.rename(columns={"player_name": "steamid"})
    labels = frame.groupby("player_name")["label"].max().to_dict()
    players["dataset_label"] = players["steamid"].map(labels).astype(int)
    players["name"] = players["steamid"]

    threshold = threshold if threshold is not None else detector.threshold
    p95 = float(np.quantile(detector.human_player_scores, 0.95))
    players["flagged"] = players["shrunk_score"] >= threshold
    players["percentile"] = [
        float((detector.human_player_scores < value).mean() * 100.0)
        for value in players["shrunk_score"]
    ]
    players = _add_review_order(players, threshold, p95)

    out.mkdir(parents=True, exist_ok=True)
    map_name = _map_name(tar_path, str(merged["path"].iloc[0]))
    reference = {
        "human_player_scores": detector.human_player_scores,
        "human_window_scores": detector.human_window_scores,
    }
    result = {
        "source": "CS2CD",
        "match": match,
        "map": map_name,
        "artifact": detector.manifest["version"],
        "threshold": threshold,
        "human_reference": {
            "players": int(len(detector.human_player_scores)),
            "p95": float(np.quantile(detector.human_player_scores, 0.95)),
            "p99": float(np.quantile(detector.human_player_scores, 0.99)),
        },
        "players": [
            {
                "player": row.steamid,
                "dataset_label": int(row.dataset_label),
                "tier": row.tier,
                "review_rank": int(row.review_rank),
                "review_reasons": list(row.review_reasons),
                "review_reason_count": int(row.review_reason_count),
                "kills": int(row.kills),
                "headshot_rate": round(float(row.headshot_rate), 4),
                "headshot_kills": int(row.headshot_kills),
                "shots_per_kill": round(float(row.shots_per_kill), 3),
                "kill_per_shot": round(float(row.kill_per_shot), 3),
                "max_multikill_4s": int(row.max_multikill_4s),
                "burst_fraction": round(float(row.burst_fraction), 4),
                "median_kill_gap_s": None
                if not np.isfinite(row.median_kill_gap_s)
                else round(float(row.median_kill_gap_s), 3),
                "sniper_share": round(float(row.sniper_share), 4),
                "wallbang_rate": round(float(row.wallbang_rate), 4),
                "behavior_context": row.behavior_context,
                "mean_score": round(float(row.mean), 6),
                "max_kill_score": round(float(row.max_kill_score), 6),
                "shrunk_score": round(float(row.shrunk_score), 6),
                "percentile": round(float(row.percentile), 1),
                "flagged": bool(row.flagged),
                "reliable": bool(row.reliable),
            }
            for row in players.itertuples()
        ],
        "kills": [
            {"player": str(row.player_name),
             "kill_tick": int(row.kill_tick),
             "score": round(float(row.score), 6)}
            for row in frame.sort_values(["player_name", "kill_tick"]).itertuples()
        ],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    queue = players[[
        "review_rank", "steamid", "dataset_label", "tier", "shrunk_score",
        "percentile", "kills", "review_reason_count", "review_reasons",
    ]].rename(columns={"steamid": "player"})
    queue["review_reasons"] = queue["review_reasons"].apply(
        lambda values: "; ".join(values)
    )
    queue.to_csv(out / "review_queue.csv", index=False)
    _summary_plot(players, reference, threshold, p95,
                  {value: value for value in players["steamid"]},
                  out / "players.png")
    _player_panels(players, frame, reference,
                   {value: value for value in players["steamid"]},
                   threshold, p95, out / "kill_windows.png")
    _behavior_table(players,
                    {value: value for value in players["steamid"]},
                    p95, out / "behavior.png")
    return {
        "match": match,
        "map": map_name,
        "players": len(result["players"]),
        "flagged": sum(player["flagged"] for player in result["players"]),
        "top": round(float(players["shrunk_score"].max()), 6),
        "out": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", action="append", required=True,
                    help="CS2CD match key; repeat for a combined queue")
    ap.add_argument("--artifact", default="artifacts/valveguard_cnn_v1")
    ap.add_argument("--cache", default="data/processed/cw256")
    ap.add_argument("--channels", default="data/processed/cw256_channels")
    ap.add_argument("--kill-ticks", default="data/processed/kill_ticks.parquet")
    ap.add_argument("--tar", default="data/raw/context_windows_256.tar")
    ap.add_argument("--out", default="reports/demo_dataset")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    detector = load_detector(args.artifact)
    root = Path(args.out)
    summaries = []
    queues = []
    for match in args.match:
        name = match.replace(":", "_")
        summary = run_match(
            match, detector, Path(args.cache), Path(args.channels),
            Path(args.kill_ticks), Path(args.tar), root / name, args.threshold,
        )
        summaries.append(summary)
        queue = pd.read_csv(root / name / "review_queue.csv")
        queue.insert(0, "demo", name)
        queues.append(queue)
    _combine_review_queues(queues).to_csv(root / "review_queue.csv", index=False)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
