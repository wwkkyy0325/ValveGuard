"""Within-match self-baseline experiment.

For every (match, player) group, normalise each aim feature by that player's
own median and MAD across their kills in the same match.  A kill is then
described by how unusual it is for that specific player, not by how unusual it
is for the population.  This targets the high-rank false positives: a strong
human's normal aim is already precise, so population-level features cannot tell
it apart from aim assist, but a kill that is far from the player's own baseline
still can.

Three feature sets are compared on the same dev-80 folds:
  raw       -- the champion's 59 aim features
  self      -- within-player robust z-scores (plus baseline size/validity)
  raw+self  -- both

All computation is inside a match, and a match never crosses folds, so the
baseline does not leak across train/test.  Players with fewer than three kills
get z=0 and a validity flag.  This is a dev-only diagnostic: the holdout has
already been used for the champion's final evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from valveguard.metrics import binary_metrics, player_scores
from valveguard.scripts.ablation_features import META_DROP

OUTCOME = {
    "headshot", "through_smoke", "wallbang", "midair",
    "shot_count", "victim_noise", "flashed_at_kill", "victim_health",
}
MIN_KILLS = 3


def recall_at_fpr(y, s, f=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def aim_cols(df: pd.DataFrame) -> list[str]:
    weapon = [c for c in df.columns if c.startswith("weapon_")]
    motion = [c for c in df.columns if c.startswith("attacker_vel")]
    return [
        c for c in df.columns
        if c not in META_DROP and c not in OUTCOME and c != "fold"
        and c not in set(weapon + motion) and df[c].notna().any()
    ]


def self_baseline(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keys = ["match_key", "player_name"]
    g = df.groupby(keys, sort=False)
    med = g[cols].transform("median")
    absdev = (df[cols] - med).abs()
    mad = absdev.groupby([df["match_key"], df["player_name"]], sort=False).transform("median")
    scale = 1.4826 * mad
    floor = df[cols].std().replace(0.0, 1.0) * 0.25
    scale = scale.where(scale > 1e-6, floor, axis=1)
    z = ((df[cols] - med) / scale).clip(-10.0, 10.0).fillna(0.0)

    n = g["row"].transform("size")
    valid = (n >= MIN_KILLS).to_numpy()
    z = z.copy()
    z.loc[~valid, :] = 0.0
    z.columns = [f"self_{c}" for c in cols]
    z["self_n_log"] = np.log1p(n.to_numpy())
    z["self_valid"] = valid.astype(float)
    return z


def run_cv(data: pd.DataFrame, cols: list[str], seed: int = 42) -> np.ndarray:
    oof = np.full(len(data), np.nan)
    for fold in sorted(data["fold"].unique()):
        tr = (data["fold"] != fold).to_numpy()
        te = ~tr
        Xtr = data.loc[tr, cols].replace([np.inf, -np.inf], np.nan)
        Xte = data.loc[te, cols].replace([np.inf, -np.inf], np.nan)
        ytr = data.loc[tr, "label"].to_numpy(int)
        pos = max(1, int(ytr.sum()))
        neg = max(1, len(ytr) - pos)
        sw = np.where(ytr == 1, len(ytr) / (2 * pos), len(ytr) / (2 * neg))
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=seed + int(fold),
        )
        clf.fit(Xtr, ytr, sample_weight=sw)
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def score(data: pd.DataFrame, cols: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    oof = run_cv(data, cols)
    d = data.copy()
    d["score"] = oof
    kill = binary_metrics(d["label"].to_numpy(), oof)
    players = player_scores(d, oof, method="mean")
    player = binary_metrics(players["label"].to_numpy(), players["score"].to_numpy())
    return {"kill": kill, "player": player, "n_features": len(cols)}, players, oof


def paired(players_a: pd.DataFrame, players_b: pd.DataFrame, seed: int = 42):
    def add_key(p):
        q = p.copy()
        q["player_key"] = q["match_key"] + ":" + q["player_name"]
        return q

    a = add_key(players_a).set_index("player_key")
    b = add_key(players_b).set_index("player_key")
    common = a.index.intersection(b.index)
    p = a.loc[common]
    y = p["label"].to_numpy()
    matches = p["match_key"].to_numpy()
    sa = a.loc[common, "score"].to_numpy()
    sb = b.loc[common, "score"].to_numpy()

    rng = np.random.default_rng(seed)
    uniq = np.unique(matches)
    aucs, recs = [], []
    for _ in range(1000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(matches == m)[0] for m in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, sb[idx]) - roc_auc_score(yy, sa[idx]))
        recs.append(recall_at_fpr(yy, sb[idx]) - recall_at_fpr(yy, sa[idx]))

    def ci(x):
        x = np.asarray(x)
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {"delta_auc": ci(aucs), "delta_recall_fpr1": ci(recs)}


def q4_table(data: pd.DataFrame, oof: np.ndarray, skill: pd.DataFrame) -> dict:
    d = data.copy()
    d["score"] = oof
    players = player_scores(d, oof, method="mean")
    players = players.merge(
        skill[["match_key", "player_name", "rank_q"]],
        on=["match_key", "player_name"], how="left",
    )
    fpr, _, thr = roc_curve(d["label"], oof)
    ok = np.where(fpr <= 0.01)[0]
    cut = float(thr[ok[-1]]) if len(ok) else 1.0
    res = {}
    for g in ["Q1 low", "Q2", "Q3", "Q4 high"]:
        p = players[players["rank_q"] == g]
        if len(p) == 0 or p["label"].nunique() < 2:
            continue
        pred = (p["score"] >= cut).to_numpy()
        y = p["label"].to_numpy()
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        res[g] = {
            "n": int(len(p)),
            "auc": float(roc_auc_score(y, p["score"])),
            "fpr": fp / max(fp + tn, 1),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="permute labels at match level as a leakage audit")
    ap.add_argument("--drop-invalid", action="store_true",
                    help="drop players with fewer than three kills; their z=0 "
                         "rows otherwise signal the kill count to the model")
    ap.add_argument("--only-set", default=None,
                    help="train a single feature set (raw, raw+selfz, "
                         "raw+selfz+count, selfz)")
    ap.add_argument("--save-oof", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    index = pd.read_parquet(Path(args.cache) / "index.parquet")
    meta = index[["row", "match_key", "player_name", "label", "source", "kill_idx"]]
    data = feats.merge(meta, on="row", how="inner")
    data = data[~((data["source"] == "cdata") & (data["label"] == 0))].copy()
    data = data.merge(
        pd.read_parquet(args.folds)[["match_key", "fold"]].drop_duplicates(),
        on="match_key", how="left",
    )
    dev = pd.read_parquet(args.holdout)
    dev_keys = set(dev.loc[dev["split"] == "dev", "match_key"])
    data = data[data["match_key"].isin(dev_keys)].dropna(subset=["fold"]).copy()
    data["fold"] = data["fold"].astype(int)
    data = data.sort_values(["match_key", "player_name", "kill_idx"]).reset_index(drop=True)

    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        mlab = data.groupby("match_key")["label"].first()
        perm = pd.Series(rng.permutation(mlab.to_numpy()), index=mlab.index)
        data["label"] = data["match_key"].map(perm).astype(int)

    raw_cols = aim_cols(data)
    sb = self_baseline(data, raw_cols)
    count_cols = ["self_n_log", "self_valid"]
    selfz_cols = [c for c in sb.columns if c not in count_cols]
    data = pd.concat([data, sb], axis=1)
    if args.drop_invalid:
        data = data[data["self_valid"] > 0.5].reset_index(drop=True)

    print(f"rows {len(data)} matches {data.match_key.nunique()} "
          f"players {data.groupby(['match_key','player_name']).ngroups} "
          f"raw {len(raw_cols)} selfz {len(selfz_cols)}", flush=True)

    # Keep the kill-count columns in a separate arm: they are a metadata
    # shortcut (cheaters get more kills), not an aim signal.
    sets = {
        "raw": raw_cols,
        "raw+selfz": raw_cols + selfz_cols,
        "raw+selfz+count": raw_cols + selfz_cols + count_cols,
        "selfz": selfz_cols,
    }
    if args.only_set:
        sets = {args.only_set: sets[args.only_set]}
    rows, players, oofs = [], {}, {}
    for name, cols in sets.items():
        print(f"training {name}: {len(cols)} features", flush=True)
        res, pl, oof = score(data, cols)
        players[name] = pl
        oofs[name] = oof
        rows.append({
            "set": name, "n_features": len(cols),
            "player_auc": res["player"]["auc"],
            "player_ap": res["player"]["ap"],
            "player_recall_fpr1": res["player"]["recall_at_fpr1"],
            "player_fpr_tpr95": res["player"]["fpr_at_tpr95"],
            "player_ece": res["player"]["ece"],
            "kill_auc": res["kill"]["auc"],
            "kill_recall_fpr1": res["kill"]["recall_at_fpr1"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(out / "self_baseline.csv", index=False)
    print(table.to_string(index=False))

    cmp = {}
    if "raw" in players and "raw+selfz" in players:
        cmp["raw+selfz vs raw"] = paired(players["raw"], players["raw+selfz"])
    if "raw+selfz" in players and "raw+selfz+count" in players:
        cmp["raw+selfz+count vs raw+selfz"] = paired(
            players["raw+selfz"], players["raw+selfz+count"]
        )
    if "raw" in players and "selfz" in players:
        cmp["selfz vs raw"] = paired(players["raw"], players["selfz"])

    if args.save_oof:
        for name, oof in oofs.items():
            pd.DataFrame({
                "row": data["row"].to_numpy(),
                "match_key": data["match_key"].to_numpy(),
                "player_name": data["player_name"].to_numpy(),
                "label": data["label"].to_numpy(),
                "fold": data["fold"].to_numpy(),
                "score": oof,
            }).to_parquet(out / f"oof_{name.replace('+', '_')}.parquet", index=False)

    if args.shuffle_labels:
        q4 = {}
    else:
        skill = pd.read_parquet(args.skill)
        q4 = {name: q4_table(data, oofs[name], skill) for name in sets}
    with open(out / "self_paired.json", "w", encoding="utf-8") as f:
        json.dump(cmp, f, indent=2)
    with open(out / "self_q4.json", "w", encoding="utf-8") as f:
        json.dump(q4, f, indent=2)
    print(json.dumps(cmp, indent=2))
    print(json.dumps(q4, indent=2))


if __name__ == "__main__":
    main()
