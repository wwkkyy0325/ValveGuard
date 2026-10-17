"""Label-noise audit for the excluded cdata label-0 windows.

CS2CD gives every non-cheater player inside a cheater match label 0, but the
dataset card estimates only ~55.6% precision for those labels.  They were
excluded from training and evaluation.  Here we score them out-of-fold and ask
how cheater-like they look:

  clean      -- ncdata label 0 (no cheater present in the match)
  uncertain  -- cdata label 0 (non-cheater player in a cheater match)
  cheater    -- cdata label 1

All scores are out-of-match: for fold k, the model is fitted on the other
folds and scores every window in fold k, including the uncertain rows that
were never used in training.

Three views are reported: score distributions at fixed human thresholds, a
KDE likelihood-ratio mixture estimate, and a nearest-neighbour vote in
standardised feature space.  The estimate is approximate; the confident
groups are used as the two reference distributions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

from valveguard.metrics import player_scores
from valveguard.scripts.ablation_features import META_DROP

OUTCOME = {
    "headshot", "through_smoke", "wallbang", "midair",
    "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
}
GROUPS = ["clean", "uncertain", "cheater"]


def feature_cols(df: pd.DataFrame) -> list[str]:
    weapon = [c for c in df.columns if c.startswith("weapon_")]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    return [
        c for c in df.columns
        if c not in META_DROP and c not in OUTCOME
        and c not in {"fold", "group", "score"}
        and c not in set(weapon + motion) and df[c].notna().any()
    ]


def balanced_weights(y: np.ndarray) -> np.ndarray:
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    idx = pd.read_parquet(Path(args.cache) / "index.parquet")
    meta = idx[["row", "match_key", "player_name", "label", "source", "kill_idx"]]
    data = feats.merge(meta, on="row", how="inner")
    hm = pd.read_parquet(args.holdout)
    dev = data[data["match_key"].isin(set(hm.loc[hm["split"] == "dev", "match_key"]))]
    dev = dev.merge(
        pd.read_parquet(args.folds)[["match_key", "fold"]].drop_duplicates(),
        on="match_key", how="left",
    ).dropna(subset=["fold"]).copy()
    dev["fold"] = dev["fold"].astype(int)
    dev["group"] = np.where(
        (dev["source"] == "cdata") & (dev["label"] == 0), "uncertain",
        np.where(dev["label"] == 1, "cheater", "clean"),
    )
    dev = dev.reset_index(drop=True)
    cols = feature_cols(dev)
    print(dev.groupby("group").size().to_string(), flush=True)

    score = np.full(len(dev), np.nan)
    for fold in sorted(dev["fold"].unique()):
        tr = dev[(dev["fold"] != fold) & (dev["group"] != "uncertain")]
        te_mask = (dev["fold"] == fold).to_numpy()
        y = tr["label"].to_numpy(int)
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=args.seed + int(fold),
        )
        clf.fit(
            tr[cols].replace([np.inf, -np.inf], np.nan), y,
            sample_weight=balanced_weights(y),
        )
        score[te_mask] = clf.predict_proba(
            dev.loc[te_mask, cols].replace([np.inf, -np.inf], np.nan)
        )[:, 1]
        print(f"fold {fold} scored", flush=True)
    dev["score"] = score

    # Human operating points from the clean group.
    clean_kill = dev.loc[dev["group"] == "clean", "score"].to_numpy()
    kill_thr = float(np.quantile(clean_kill, 0.99))
    players = player_scores(dev, dev["score"].to_numpy(), method="mean")
    pg = dev.groupby(["match_key", "player_name"], as_index=False)["group"].first()
    players = players.merge(pg, on=["match_key", "player_name"], how="left")
    clean_player = players.loc[players["group"] == "clean", "score"].to_numpy()
    player_thr = float(np.quantile(clean_player, 0.99))

    rows = []
    for g in GROUPS:
        d = dev[dev["group"] == g]
        p = players[players["group"] == g]
        rows.append({
            "group": g,
            "windows": int(len(d)),
            "players": int(len(p)),
            "kill_mean": float(d["score"].mean()),
            "kill_median": float(d["score"].median()),
            "kill_p95": float(d["score"].quantile(0.95)),
            "frac_kills_above_human1": float((d["score"] >= kill_thr).mean()),
            "player_mean": float(p["score"].mean()) if len(p) else np.nan,
            "frac_players_above_human1": float((p["score"] >= player_thr).mean()) if len(p) else np.nan,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "group_scores.csv", index=False)
    print(table.to_string(index=False))

    # KDE likelihood-ratio mixture estimate on kill scores.
    rng = np.random.default_rng(args.seed)
    def sub(x, n=12000):
        return x if len(x) <= n else rng.choice(x, n, replace=False)
    kh = gaussian_kde(sub(clean_kill))
    kc = gaussian_kde(sub(dev.loc[dev["group"] == "cheater", "score"].to_numpy()))
    unc = dev.loc[dev["group"] == "uncertain", "score"].to_numpy()
    dens_h, dens_c = kh(unc), kc(unc)
    post = dens_c / (dens_h + dens_c + 1e-12)
    mixture = {
        "uncertain_windows": int(len(unc)),
        "estimated_cheater_fraction": float(post.mean()),
        "posterior_quantiles": [
            float(np.quantile(post, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        ],
    }
    print(json.dumps(mixture, indent=2))

    # Nearest-neighbour vote in standardised feature space.
    scaler_mu = dev.loc[dev["group"] == "clean", cols].mean()
    scaler_sd = dev.loc[dev["group"] == "clean", cols].std().replace(0, 1)
    def scale(d):
        return ((d[cols] - scaler_mu) / scaler_sd).replace([np.inf, -np.inf], 0).fillna(0).to_numpy()
    clean_X = scale(dev[dev["group"] == "clean"].sample(n=min(12000, (dev["group"] == "clean").sum()), random_state=args.seed))
    cheat_X = scale(dev[dev["group"] == "cheater"].sample(n=min(8000, (dev["group"] == "cheater").sum()), random_state=args.seed))
    unc_df = dev[dev["group"] == "uncertain"].sample(n=min(3000, (dev["group"] == "uncertain").sum()), random_state=args.seed)
    unc_X = scale(unc_df)
    nn_h = NearestNeighbors(n_neighbors=1).fit(clean_X)
    nn_c = NearestNeighbors(n_neighbors=1).fit(cheat_X)
    d_h = nn_h.kneighbors(unc_X)[0][:, 0]
    d_c = nn_c.kneighbors(unc_X)[0][:, 0]
    nn = {
        "sampled_uncertain": int(len(unc_X)),
        "fraction_closer_to_cheater": float((d_c < d_h).mean()),
        "median_dist_human": float(np.median(d_h)),
        "median_dist_cheater": float(np.median(d_c)),
    }
    print(json.dumps(nn, indent=2))

    # First-moment decomposition: if the uncertain group is a mixture of the
    # confident clean and cheater score distributions, its mean pins the
    # cheater fraction.  This is stable when the two reference distributions
    # overlap; the KDE posterior is reported as a shape-based cross-check.
    m_clean = float(np.mean(clean_kill))
    m_cheat = float(np.mean(dev.loc[dev["group"] == "cheater", "score"]))
    m_unc = float(np.mean(unc))
    mean_pi = (m_unc - m_clean) / (m_cheat - m_clean) if m_cheat != m_clean else np.nan
    mean_decomp = {
        "mean_clean": m_clean,
        "mean_uncertain": m_unc,
        "mean_cheater": m_cheat,
        "estimated_cheater_fraction": float(mean_pi),
    }
    print(json.dumps(mean_decomp, indent=2))

    result = {
        "kill_threshold_clean_1pct": kill_thr,
        "player_threshold_clean_1pct": player_thr,
        "groups": rows,
        "mixture": mixture,
        "mean_decomposition": mean_decomp,
        "nearest_neighbour": nn,
    }
    with open(out / "label_noise.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    dev[["row", "match_key", "player_name", "label", "source", "group",
         "score"]].to_parquet(out / "scored_windows.parquet", index=False)


if __name__ == "__main__":
    main()
