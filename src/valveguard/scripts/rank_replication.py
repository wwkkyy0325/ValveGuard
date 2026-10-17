"""Replicate the rank-stratified false-positive analysis on all dev-80 matches.

The original rank analysis used the 160-match raw subset because it needed the
raw parquet `rank` column.  The per-match JSON carries `rank_update` events
with the same player pseudonyms, so ranks can be recovered for every match
without downloading the full parquet set.  Trajectories still come from the
cached 256-tick windows.

Outputs the player-level 1% FPR by rank quartile, with and without
empirical-Bayes shrinkage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from valveguard.metrics import binary_metrics, player_scores

K0 = 5


def _rank_one(path: Path, match_key: str) -> list[dict]:
    try:
        ev = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    info = ev.get("CSstats_info", [{}])
    mmt = str(info[0].get("match_making_type", "")) if info else ""
    kind = "premier" if "Premier" in mmt else "cs"
    updates = pd.DataFrame(ev.get("rank_update", []))
    if len(updates) == 0:
        return []
    last = (
        updates.sort_values("tick")
        .groupby("user_steamid")["rank_new"]
        .last()
    )
    return [
        {"match_key": match_key, "player_name": str(p), "rank_raw": float(v),
         "rank_type": kind}
        for p, v in last.items() if pd.notna(v) and float(v) > 0
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k0", type=float, default=K0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_dir = Path(args.json_dir)

    rows = []
    for folder in ("with_cheater_present", "no_cheater_present"):
        for path in sorted((json_dir / folder).glob("*.json")):
            src = "cdata" if folder == "with_cheater_present" else "ncdata"
            rows.extend(_rank_one(path, f"{src}:{path.stem}"))
    rank = pd.DataFrame(rows)
    rank = rank.drop_duplicates(["match_key", "player_name"])
    print(f"rank rows {len(rank)} matches {rank.match_key.nunique()}", flush=True)

    pred = pd.read_parquet(args.pred)
    players = player_scores(pred, pred["score"].to_numpy(), method="mean")
    agg = pred.groupby(["match_key", "player_name", "fold"]).agg(
        n=("score", "size"), total=("score", "sum"), label=("label", "max")
    ).reset_index()
    priors = {
        f: float(pred[(pred["label"] == 0) & (pred["fold"] != f)]["score"].mean())
        for f in pred["fold"].unique()
    }
    agg["shrunk"] = (agg["total"] + args.k0 * agg["fold"].map(priors)) / (agg["n"] + args.k0)
    players = players.merge(
        agg[["match_key", "player_name", "n", "shrunk"]],
        on=["match_key", "player_name"], how="left",
    )
    players = players.merge(rank, on=["match_key", "player_name"], how="left")
    players["rank_pct"] = np.nan
    for kind, g in players[players["rank_type"].notna()].groupby("rank_type"):
        players.loc[g.index, "rank_pct"] = g["rank_raw"].rank(pct=True)
    q = pd.qcut(players["rank_pct"], 4, labels=False, duplicates="drop")
    qmap = {0: "Q1 low", 1: "Q2", 2: "Q3", 3: "Q4 high"}
    players["rank_q"] = q.map(qmap)

    def table(score_col: str) -> dict:
        human = players.loc[players["label"] == 0, score_col]
        thr = float(human.quantile(0.99))
        out = {"threshold": thr, "human_fpr": float((human >= thr).mean()), "by_q": {}}
        for name, g in players[players["rank_q"].notna()].groupby("rank_q"):
            h = g[g["label"] == 0]
            entry = {
                "players": int(len(g)),
                "clean": int(len(h)),
                "auc": float(binary_metrics(g["label"], g[score_col])["auc"])
                if g["label"].nunique() > 1 else np.nan,
                "fpr": float((h[score_col] >= thr).mean()) if len(h) else np.nan,
            }
            out["by_q"][name] = entry
        return out

    mean_tab = table("score")
    shrunk_tab = table("shrunk")
    ranked = players[players["rank_q"].notna()]
    result = {
        "players_total": int(len(players)),
        "players_with_rank": int(len(ranked)),
        "coverage": float(len(ranked) / len(players)),
        "mean": mean_tab,
        "shrunk": shrunk_tab,
    }
    with open(out / "rank_replication.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    players[["match_key", "player_name", "label", "n", "score", "shrunk",
             "rank_raw", "rank_type", "rank_pct", "rank_q"]].to_parquet(
        out / "rank_players.parquet", index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
