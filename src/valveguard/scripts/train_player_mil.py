"""Player-level multiple-instance model over the 256-tick kill windows.

The window CNN is trained with window labels and pooled after the fact.
This model trains the same encoder with attention pooling directly on the
player label, so it can concentrate on the few blatant kills of a toggling
cheater instead of averaging them away.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from valveguard.scripts.train_channel_cnn import WideCNN

K0 = 5.0
PRIOR = 0.3018746014732038


class MILNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.encoder = WideCNN(channels)
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )
        self.window_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        b, k, t, c = x.shape
        emb = self.encoder.embed(x.reshape(b * k, t, c)).reshape(b, k, 256)
        score = self.attention(emb).squeeze(-1)
        score = score.masked_fill(~mask, -1.0e4)
        weight = torch.softmax(score, dim=1)
        pooled = (emb * weight.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(1), self.window_head(emb).squeeze(-1)


def _normalization(channels: np.ndarray, rows: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sample = np.random.default_rng(seed).choice(rows, size=min(5000, len(rows)), replace=False)
    values = channels[sample].astype(np.float32)
    values[~np.isfinite(values)] = np.nan
    center = np.nanmedian(values, axis=(0, 1))
    q25 = np.nanpercentile(values, 25, axis=(0, 1))
    q75 = np.nanpercentile(values, 75, axis=(0, 1))
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    center[~np.isfinite(center)] = 0.0
    return center.astype(np.float32), scale.astype(np.float32)


def _metrics(labels: np.ndarray, scores: np.ndarray, q4: pd.DataFrame,
             players: pd.DataFrame) -> dict:
    labels = labels.astype(int)
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    merged = players.assign(score=scores).merge(q4, on=["match_key", "player_name"], how="inner")
    mh = merged[merged["label"] == 0]
    mp = merged[merged["label"] == 1]
    q4h = mh[mh["rank_q"].astype(str).str.startswith("Q4")]
    q4p = mp[mp["rank_q"].astype(str).str.startswith("Q4")]
    return {
        "player_auc": float(roc_auc_score(labels, scores)),
        "player_ap": float(average_precision_score(labels, scores)),
        "player_recall_at_fpr1": float((positive >= threshold).mean()),
        "player_fpr": float((human >= threshold).mean()),
        "q4_fpr": float((q4h["score"] >= threshold).mean()),
        "q4_recall": float((q4p["score"] >= threshold).mean()),
        "players": int(len(labels)),
        "positive_players": int(labels.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-players", type=int, default=32)
    ap.add_argument("--train-k", type=int, default=24)
    ap.add_argument("--eval-k", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds-from", default=None,
                    help="predictions parquet whose match->fold map is reused")
    ap.add_argument("--init-dir", default=None,
                    help="directory with fold{k}.pt window models to warm-start from")
    ap.add_argument("--freeze-epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is not available")

    channels = np.load(args.channels)
    index = pd.read_parquet(args.index)
    index = index[~((index["source"] == "cdata") & (index["label"] == 0))].copy()
    index = index.reset_index(drop=True)
    rows = index["row"].to_numpy(dtype=int)
    labels = index["label"].to_numpy(dtype=int)
    center, scale = _normalization(channels, rows, args.seed)
    center_t = torch.tensor(center).view(1, 1, -1)
    scale_t = torch.tensor(scale).view(1, 1, -1)

    index["pos"] = np.arange(len(index))
    records = [
        (match, name, int(group["label"].max()), len(group),
         group["pos"].to_numpy(dtype=int))
        for (match, name), group in index.groupby(
            ["match_key", "player_name"], sort=False
        )
    ]
    players = pd.DataFrame(
        records, columns=["match_key", "player_name", "label", "n", "positions"]
    )
    print("windows", len(index), "players", len(players),
          "positives", int(players["label"].sum()), "cuda", torch.cuda.get_device_name(0),
          flush=True)
    q4 = pd.read_parquet(args.ranks)[["match_key", "player_name", "rank_q"]]

    def build(batch: pd.DataFrame, k: int, train: bool):
        b = len(batch)
        x = torch.zeros((b, k, 256, channels.shape[2]), dtype=torch.float32)
        mask = torch.zeros((b, k), dtype=torch.bool)
        y = torch.zeros(b, dtype=torch.float32)
        y_win = torch.zeros((b, k), dtype=torch.float32)
        for i, (_, player) in enumerate(batch.iterrows()):
            pos = player["positions"]
            chosen = pos
            if len(pos) > k:
                chosen = (rng.choice(pos, size=k, replace=False) if train
                          else pos[::max(1, len(pos) // k)][:k])
            flat = rows[chosen]
            values = torch.from_numpy(channels[flat].astype(np.float32))
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            values = (values - center_t) / scale_t
            values = torch.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
            x[i, :len(chosen)] = values
            if len(chosen) < k:
                x[i, len(chosen):] = values[0]
            mask[i, :len(chosen)] = True
            y[i] = float(player["label"])
            y_win[i, :len(chosen)] = torch.from_numpy(labels[chosen].astype(np.float32))
        return x.to(device), mask.to(device), y.to(device), y_win.to(device)

    # Use the window-model folds, not a fresh split on player counts.  The
    # greedy GroupKFold balance depends on group sizes, so splitting players
    # directly would assign different matches to folds and invalidate both the
    # OOF comparison and any warm start from a fold model.
    if args.folds_from:
        reference = pd.read_parquet(args.folds_from)[["match_key", "fold"]].drop_duplicates()
        match_fold = dict(zip(reference["match_key"], reference["fold"].astype(int)))
    else:
        match_fold = {}
        for fold, (_, test_idx) in enumerate(
            GroupKFold(n_splits=args.folds).split(index, groups=index["match_key"])
        ):
            for match in index.iloc[test_idx]["match_key"].unique():
                match_fold[match] = fold
    players["fold"] = players["match_key"].map(match_fold)
    if players["fold"].isna().any():
        raise SystemExit("fold map does not cover every player")
    oof = np.full(len(players), np.nan, dtype=float)
    fold_metrics = []
    for fold in range(args.folds):
        train_idx = players.index[players["fold"] != fold].to_numpy()
        test_idx = players.index[players["fold"] == fold].to_numpy()
        train_players = players.iloc[train_idx].reset_index(drop=True)
        test_players = players.iloc[test_idx].reset_index(drop=True)
        positives = max(1, int(train_players["label"].sum()))
        negatives = max(1, len(train_players) - positives)
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        win_labels = labels[np.concatenate(train_players["positions"].to_numpy())]
        win_positive = max(1, int(win_labels.sum()))
        win_weight = torch.tensor(
            [(len(win_labels) - win_positive) / win_positive],
            dtype=torch.float32, device=device,
        )

        model = MILNet(channels.shape[2]).to(device)
        if args.init_dir:
            payload = torch.load(
                Path(args.init_dir) / f"fold{fold}.pt", map_location="cpu",
                weights_only=True,
            )
            model.encoder.load_state_dict(payload["state_dict"], strict=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        win_criterion = nn.BCEWithLogitsLoss(pos_weight=win_weight)
        scaler = torch.amp.GradScaler("cuda")
        start = time.time()
        for epoch in range(args.epochs):
            frozen = epoch < args.freeze_epochs
            for parameter in model.encoder.parameters():
                parameter.requires_grad = not frozen
            model.train()
            order = rng.permutation(len(train_players))
            total = 0.0
            for start_idx in range(0, len(order), args.batch_players):
                batch = train_players.iloc[order[start_idx:start_idx + args.batch_players]]
                k = min(args.train_k, max(1, int(batch["n"].max())))
                x, mask, y, y_win = build(batch, k, True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logit, win_logit = model(x, mask)
                    loss = criterion(logit, y) + args.aux * win_criterion(
                        win_logit[mask], y_win[mask]
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(batch)
            scheduler.step()
            print(f"fold {fold} epoch {epoch + 1}/{args.epochs} "
                  f"loss {total / len(train_players):.4f} "
                  f"elapsed {time.time() - start:.1f}s", flush=True)

        model.eval()
        scores = np.zeros(len(test_players), dtype=float)
        with torch.no_grad():
            order = np.argsort(-test_players["n"].to_numpy())
            for start_idx in range(0, len(order), max(1, args.batch_players // 2)):
                batch = test_players.iloc[order[start_idx:start_idx + max(1, args.batch_players // 2)]]
                k = min(args.eval_k, max(1, int(batch["n"].max())))
                x, mask, _, _ = build(batch, k, False)
                with torch.amp.autocast("cuda"):
                    logit, _ = model(x, mask)
                scores[order[start_idx:start_idx + len(batch)]] = (
                    torch.sigmoid(logit.float()).cpu().numpy()
                )
        oof[test_idx] = scores
        metrics = _metrics(
            test_players["label"].to_numpy(dtype=int), scores, q4, test_players
        )
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        print("fold", fold, json.dumps(metrics), flush=True)

    valid = np.isfinite(oof)
    result = {
        "folds": fold_metrics,
        "overall": _metrics(
            players.loc[valid, "label"].to_numpy(dtype=int), oof[valid], q4,
            players.loc[valid].reset_index(drop=True),
        ),
        "channels": int(channels.shape[2]),
        "epochs": args.epochs,
        "train_k": args.train_k,
        "init_dir": args.init_dir,
        "players": int(len(players)),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({
        "match_key": players["match_key"],
        "player_name": players["player_name"],
        "label": players["label"],
        "n": players["n"],
        "score": oof,
    }).to_parquet(out / "player_oof.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
