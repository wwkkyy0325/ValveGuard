"""GPU 1D-CNN baseline on the 43-channel 256-tick cache."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

K0 = 5.0


class WindowDataset(Dataset):
    def __init__(self, channels: np.ndarray, rows: np.ndarray, labels: np.ndarray):
        self.channels = channels
        self.rows = rows
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        return torch.from_numpy(self.channels[row]), torch.tensor(
            float(self.labels[index]), dtype=torch.float32
        )


class CNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        h = self.net(x.transpose(1, 2))
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(pooled).squeeze(1)


class WideCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 96, 7, padding=3),
            nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96, 192, 5, padding=2),
            nn.BatchNorm1d(192), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(192, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(1)


def _kill_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    human = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return {
        "kill_auc": float(roc_auc_score(labels, scores)),
        "kill_ap": float(average_precision_score(labels, scores)),
        "kill_recall_at_fpr1": float((positive >= threshold).mean()),
        "kill_fpr": float((human >= threshold).mean()),
        "kills": int(len(labels)),
        "positive_kills": int(len(positive)),
    }


def _player_metrics(frame: pd.DataFrame, scores: np.ndarray, prior: float) -> dict:
    tmp = frame.assign(score=scores)
    grouped = tmp.groupby(["match_key", "player_name"], sort=False).agg(
        label=("label", "max"),
        n=("score", "size"),
        total=("score", "sum"),
    )
    grouped["shrunk"] = (grouped["total"] + K0 * prior) / (grouped["n"] + K0)
    labels = grouped["label"].to_numpy(dtype=int)
    values = grouped["shrunk"].to_numpy(dtype=float)
    human = values[labels == 0]
    positive = values[labels == 1]
    threshold = float(np.quantile(human, 0.99))
    return {
        "player_auc": float(roc_auc_score(labels, values)),
        "player_recall_at_fpr1": float((positive >= threshold).mean()),
        "player_fpr": float((human >= threshold).mean()),
        "players": int(len(grouped)),
        "positive_players": int(len(positive)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--arch", choices=["cnn", "wide"], default="cnn")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    if channels.shape[0] <= rows.max():
        raise SystemExit("channel cache does not cover the index rows")
    print("data", channels.shape, "rows", len(index), "cuda", torch.cuda.get_device_name(0), flush=True)

    # Robust per-channel normalization. Some raw channels contain inf and
    # heavy tails (jerk, speed), so use the median and IQR on finite values.
    sample = np.random.default_rng(args.seed).choice(
        rows, size=min(5000, len(rows)), replace=False
    )
    sample_values = channels[sample].astype(np.float32)
    sample_values[~np.isfinite(sample_values)] = np.nan
    center = np.nanmedian(sample_values, axis=(0, 1))
    q25 = np.nanpercentile(sample_values, 25, axis=(0, 1))
    q75 = np.nanpercentile(sample_values, 75, axis=(0, 1))
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    center[~np.isfinite(center)] = 0.0
    mean_t = torch.tensor(center, dtype=torch.float32, device=device).view(1, -1, 1)
    std_t = torch.tensor(scale, dtype=torch.float32, device=device).view(1, -1, 1)

    folds = GroupKFold(n_splits=args.folds)
    oof = np.full(len(index), np.nan, dtype=float)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(
        folds.split(index, groups=index["match_key"])
    ):
        train_frame = index.iloc[train_idx].reset_index(drop=True)
        test_frame = index.iloc[test_idx].reset_index(drop=True)
        train_labels = train_frame["label"].to_numpy(dtype=int)
        positives = max(1, int(train_labels.sum()))
        negatives = max(1, int(len(train_labels) - train_labels.sum()))
        positive_weight = torch.tensor(
            [negatives / positives], dtype=torch.float32, device=device
        )

        model = (WideCNN if args.arch == "wide" else CNN)(
            channels.shape[2]
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        scaler = torch.amp.GradScaler("cuda")
        loader = DataLoader(
            WindowDataset(channels, rows[train_idx], train_labels),
            batch_size=args.batch_size, shuffle=True, num_workers=0,
            pin_memory=True, drop_last=True,
        )
        start = time.time()
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for x, y in loader:
                x = x.to(device, non_blocking=True).float()
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x = (x.permute(0, 2, 1) - mean_t) / std_t
                x = torch.nan_to_num(
                    x, nan=0.0, posinf=10.0, neginf=-10.0
                ).clamp(-10.0, 10.0)
                x = x.transpose(1, 2)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(y)
            scheduler.step()
            print(
                f"fold {fold} epoch {epoch + 1}/{args.epochs} "
                f"loss {total / len(train_labels):.4f} elapsed {time.time() - start:.1f}s",
                flush=True,
            )

        model.eval()
        scores = []
        with torch.no_grad():
            for start_row in range(0, len(test_idx), args.batch_size):
                batch_rows = test_idx[start_row:start_row + args.batch_size]
                x = torch.from_numpy(
                    channels[rows[batch_rows]].astype(np.float32)
                ).to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x = (x.permute(0, 2, 1) - mean_t) / std_t
                x = torch.nan_to_num(
                    x, nan=0.0, posinf=10.0, neginf=-10.0
                ).clamp(-10.0, 10.0)
                x = x.transpose(1, 2)
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        scores = np.concatenate(scores)
        oof[test_idx] = scores
        prior = float(scores[test_frame["label"].to_numpy(dtype=int) == 0].mean())
        metrics = _kill_metrics(test_frame["label"].to_numpy(dtype=int), scores)
        metrics.update(_player_metrics(test_frame, scores, prior))
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        print("fold", fold, json.dumps(metrics), flush=True)

    valid = np.isfinite(oof)
    result = {
        "folds": fold_metrics,
        "overall": _kill_metrics(labels[valid], oof[valid]),
    }
    # Pooled player scores: use each fold's own prior; for this pilot use the
    # global human score mean as a stable prior.
    prior = float(oof[valid & (labels == 0)].mean())
    result["overall"].update(_player_metrics(index.loc[valid].reset_index(drop=True), oof[valid], prior))
    result["channels"] = int(channels.shape[2])
    result["epochs"] = args.epochs
    result["rows"] = int(len(index))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({
        "row": rows, "match_key": index["match_key"], "player_name": index["player_name"],
        "label": labels, "score": oof,
    }).to_parquet(out / "oof.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
