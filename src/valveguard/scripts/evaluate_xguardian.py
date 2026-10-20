"""External check on the XGuardian processed CSVs (CS2, 96-tick windows)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

KEYS = ["match_id", "steamid", "kill_num"]


def window_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("tick")
    kill = group[group["isKilling"]]
    kill_tick = int(kill["tick"].iloc[0]) if len(kill) else int(group["tick"].iloc[-1])
    rel = (group["tick"] - kill_tick).to_numpy()
    pre = rel < 0
    change = group["angle_change"].to_numpy(dtype=float)
    ac = np.abs(change)
    dist = group["distance"].to_numpy(dtype=float)
    speed = np.hypot(group["velocity_x"], group["velocity_y"]).to_numpy(dtype=float)
    accel = np.hypot(group["acceleration_x"], group["acceleration_y"]).to_numpy(dtype=float)
    fire = group["isFiring"].to_numpy(dtype=bool)
    x = group["x"].to_numpy(dtype=float)
    y = group["y"].to_numpy(dtype=float)
    path = np.hypot(np.diff(x), np.diff(y)).sum()
    fire_rel = rel[fire]
    return {
        "ac_abs_mean": ac[pre].mean(),
        "ac_abs_std": ac[pre].std(),
        "ac_abs_p95": np.quantile(ac[pre], 0.95),
        "ac_abs_max": ac[pre].max(),
        "ac_large_frac": float((ac[pre] > 0.2).mean()),
        "ac_sign_flips": float((np.diff(np.sign(change[pre])) != 0).mean()),
        "speed_mean": speed[pre].mean(),
        "speed_std": speed[pre].std(),
        "speed_p95": np.quantile(speed[pre], 0.95),
        "accel_mean": accel[pre].mean(),
        "accel_p95": np.quantile(accel[pre], 0.95),
        "dist_mean": dist[pre].mean(),
        "dist_std": dist[pre].std(),
        "dist_min": dist[pre].min(),
        "dist_max": dist[pre].max(),
        "dist_kill": dist[~pre].min() if (~pre).any() else dist[-1],
        "dist_slope": float(np.polyfit(rel[pre], dist[pre], 1)[0]) if pre.sum() > 2 else 0.0,
        "fire_count": float(fire.sum()),
        "fire_pre_count": float(fire[pre].sum()),
        "first_fire_rel": float(fire_rel.min()) if len(fire_rel) else 0.0,
        "last_fire_rel": float(fire_rel.max()) if len(fire_rel) else 0.0,
        "crouch_frac": float(group["in_crouch"].to_numpy(dtype=float)[pre].mean()),
        "walk_frac": float(group["is_walking"].to_numpy(dtype=float)[pre].mean()),
        "strafe_frac": float(group["is_strafing"].to_numpy(dtype=float)[pre].mean()),
        "path_len": float(path),
        "x_range": float(x.max() - x.min()),
        "y_range": float(y.max() - y.min()),
        "kill_frac": float(pre.mean()),
    }


def build(path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path, dtype={"steamid": str, "match_id": str})
    rows = []
    labels = []
    meta = []
    for key, group in df.groupby(KEYS, sort=False):
        rows.append(window_features(group))
        labels.append(bool(group["isCheater"].iloc[0]))
        meta.append({"match_id": key[0], "steamid": key[1], "kill_num": key[2]})
    return pd.DataFrame(rows), np.array(labels), pd.DataFrame(meta)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    human = score[y == 0]
    positive = score[y == 1]
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "recall_at_fpr1": float((positive >= threshold).mean()),
        "fpr": float((human >= threshold).mean()),
        "windows": int(len(y)),
        "positive_windows": int(y.sum()),
    }


def player_scores(meta: pd.DataFrame, y: np.ndarray, score: np.ndarray) -> tuple:
    frame = meta.assign(label=y, score=score)
    grouped = frame.groupby(["match_id", "steamid"], sort=False).agg(
        label=("label", "max"), score=("score", "mean"), n=("score", "size")
    ).reset_index()
    return grouped["label"].to_numpy(dtype=int), grouped["score"].to_numpy(), grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/xguardian")
    ap.add_argument("--out", default="reports/xguardian_external")
    args = ap.parse_args()
    root = Path(args.root)

    data = {}
    for name, file in (("train", "train.csv"), ("validation", "validation.csv"),
                       ("test", "test.csv")):
        features, labels, meta = build(root / file)
        data[name] = (features, labels, meta)
        print(name, features.shape, "positive", int(labels.sum()), flush=True)

    result = {}
    for arm, columns in (
        ("all", list(data["train"][0].columns)),
        ("aim_only", [c for c in data["train"][0].columns
                      if c.startswith(("ac_", "speed_", "accel_"))]),
        ("no_distance", [c for c in data["train"][0].columns
                         if not c.startswith("dist_")]),
        ("distance_only", ["dist_mean", "dist_std", "dist_min", "dist_max",
                           "dist_kill", "dist_slope"]),
    ):
        xtr = data["train"][0][columns]
        ytr = data["train"][1]
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42,
        )
        model.fit(xtr, ytr)
        split_result = {}
        val_scores = model.predict_proba(data["validation"][0][columns])[:, 1]
        val_y = data["validation"][1]
        threshold = float(np.quantile(val_scores[val_y == 0], 0.99))
        for split in ("validation", "test"):
            features, labels, meta = data[split]
            scores = model.predict_proba(features[columns])[:, 1]
            window = metrics(labels, scores, threshold)
            py, ps, _ = player_scores(meta, labels, scores)
            p_threshold = float(np.quantile(ps[py == 0], 0.99))
            player = metrics(py, ps, p_threshold)
            split_result[split] = {"window": window, "player": player}
        result[arm] = {
            "splits": split_result,
            "threshold": threshold,
            "features": columns,
        }
        print(arm, json.dumps(split_result["test"], indent=2), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
