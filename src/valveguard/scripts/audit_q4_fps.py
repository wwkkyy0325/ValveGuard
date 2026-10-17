"""Per-kill audit of the remaining Q4 false positives.

After empirical-Bayes shrinkage, two Q4 clean players remain above the
player-level 1% threshold.  This lists their kills, feature values, and a
nearest-neighbour vote (closer to a clean human or a cheater window in
standardised feature space).  It is an inspection aid, not a classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from valveguard.scripts.analyze_label_noise import feature_cols

K0 = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet(args.scored)
    pred = pd.read_parquet(args.pred)
    skill = pd.read_parquet(args.skill)
    feats = pd.read_parquet(args.features)
    keys = ["match_key", "player_name"]

    agg = pred.groupby(keys + ["fold"]).agg(
        n=("score", "size"), total=("score", "sum"), label=("label", "max")
    ).reset_index()
    priors = {
        f: float(pred[(pred["label"] == 0) & (pred["fold"] != f)]["score"].mean())
        for f in pred["fold"].unique()
    }
    agg["shrunk"] = (agg["total"] + K0 * agg["fold"].map(priors)) / (agg["n"] + K0)
    agg = agg.merge(
        skill[["match_key", "player_name", "rank_q"]], on=keys, how="left"
    )
    human = agg[agg["label"] == 0]["shrunk"]
    thr = float(human.quantile(0.99))
    fps = agg[(agg["rank_q"] == "Q4 high") & (agg["label"] == 0) & (agg["shrunk"] >= thr)]
    print(f"threshold {thr:.4f}; Q4 false positives {len(fps)}")

    # Nearest-neighbour reference in standardised feature space.
    base = scored.merge(feats, on="row", how="left")
    cols = feature_cols(base)
    clean = base[base["group"] == "clean"].sample(
        n=min(12000, int((base["group"] == "clean").sum())), random_state=args.seed
    )
    cheat = base[base["group"] == "cheater"].sample(
        n=min(8000, int((base["group"] == "cheater").sum())), random_state=args.seed
    )
    mu, sd = clean[cols].mean(), clean[cols].std().replace(0, 1)

    def scale(d):
        return ((d[cols] - mu) / sd).replace([np.inf, -np.inf], 0).fillna(0).to_numpy()

    nn_h = NearestNeighbors(n_neighbors=1).fit(scale(clean))
    nn_c = NearestNeighbors(n_neighbors=1).fit(scale(cheat))

    rows = []
    summary = []
    for _, fp in fps.iterrows():
        kills = pred[
            (pred["match_key"] == fp["match_key"])
            & (pred["player_name"] == fp["player_name"])
        ].merge(feats, on="row", how="left")
        X = scale(kills)
        d_h = nn_h.kneighbors(X)[0][:, 0]
        d_c = nn_c.kneighbors(X)[0][:, 0]
        kills = kills.assign(dist_human=d_h, dist_cheater=d_c,
                             closer=np.where(d_c < d_h, "cheater", "human"))
        for _, k in kills.iterrows():
            rows.append({
                "match_key": fp["match_key"],
                "player_name": fp["player_name"],
                "row": int(k["row"]),
                "kill_score": float(k["score"]),
                "final_err": float(k["final_err"]),
                "straightness": float(k["straightness"]),
                "submovements": float(k["submovements"]),
                "speed_dps_mean": float(k["speed_dps_mean"]),
                "dist_human": float(k["dist_human"]),
                "dist_cheater": float(k["dist_cheater"]),
                "closer": k["closer"],
            })
        summary.append({
            "match_key": fp["match_key"],
            "player_name": fp["player_name"],
            "kills": int(fp["n"]),
            "shrunk_score": float(fp["shrunk"]),
            "mean_kill_score": float(kills["score"].mean()),
            "max_kill_score": float(kills["score"].max()),
            "frac_closer_to_cheater": float((d_c < d_h).mean()),
            "mean_straightness": float(kills["straightness"].mean()),
            "mean_final_err": float(kills["final_err"].mean()),
            "mean_speed": float(kills["speed_dps_mean"].mean()),
        })

    pd.DataFrame(rows).to_csv(out / "fp_kills.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "fp_players.csv", index=False)
    with open(out / "fp_players.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "players": summary}, f, indent=2)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
