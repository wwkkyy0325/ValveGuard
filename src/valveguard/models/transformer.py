"""Small Transformer detector on per-tick channels.

This is the first GPU model.  It deliberately stays small (d=128, 4 layers)
so it fits the 6 GB RTX 3060 and can be trained in minutes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from ..data.split import assign_match_splits, player_key
from ..metrics import binary_metrics, player_scores


class WindowDataset(Dataset):
    def __init__(self, channels_path: str | Path, rows: np.ndarray,
                 labels: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 keep_idx: np.ndarray | None = None):
        self.channels = np.load(channels_path, mmap_mode="r")
        self.rows = np.asarray(rows)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.keep_idx = None if keep_idx is None else np.asarray(keep_idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x = np.asarray(self.channels[self.rows[i]], dtype=np.float32)
        # A few jerk values overflowed float16 when the channel cache was
        # written; sanitize before normalization.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.keep_idx is not None:
            x = x[:, self.keep_idx]
        x = (x - self.mean) / self.std
        x = np.clip(x, -10.0, 10.0)
        return torch.from_numpy(x), torch.tensor(self.labels[i])


class AimTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.1, max_len: int = 256,
                 pooling: str = "cls"):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        b = x.size(0)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.encoder(x)
        pooled = x[:, 0] if self.pooling == "cls" else x[:, 1:].mean(dim=1)
        return self.head(pooled).squeeze(1)


def _norm_stats(channels_path: str | Path, rows: np.ndarray,
                max_rows: int = 4000,
                keep_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    mm = np.load(channels_path, mmap_mode="r")
    rng = np.random.default_rng(0)
    take = rows if len(rows) <= max_rows else rng.choice(rows, size=max_rows, replace=False)
    k = mm.shape[2] if keep_idx is None else len(keep_idx)
    total = np.zeros(k, dtype=np.float64)
    total_sq = np.zeros(k, dtype=np.float64)
    n = 0
    for r in take:
        x = np.asarray(mm[r], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if keep_idx is not None:
            x = x[:, keep_idx]
        total += x.sum(axis=0)
        total_sq += (x * x).sum(axis=0)
        n += x.shape[0]
    mean = total / max(n, 1)
    var = total_sq / max(n, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-4))
    return mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _eval_metrics(meta: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    m = binary_metrics(y, p, threshold=0.5)
    ps = player_scores(meta, p, method="mean")
    m["player_auc_mean"] = binary_metrics(
        ps["label"].to_numpy(), ps["score"].to_numpy()
    )["auc"]
    return m


def train_transformer(
    cache_dir: str | Path,
    channels_path: str | Path,
    out_dir: str | Path,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-4,
    d_model: int = 128,
    layers: int = 4,
    nhead: int = 4,
    workers: int = 4,
    limit: int | None = None,
    seed: int = 42,
    drop_groups: list[str] | None = None,
    index_df: pd.DataFrame | None = None,
    dropout: float = 0.1,
    pooling: str = "cls",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if index_df is None:
        index = pd.read_parquet(Path(cache_dir) / "index.parquet")
        index = index[~((index["source"] == "cdata") & (index["label"] == 0))].copy()
        index = assign_match_splits(index, seed=seed)
    else:
        index = index_df.copy()
        if "split" not in index.columns:
            raise ValueError("index_df must contain a split column")
    index["player_key"] = player_key(index)
    if limit is not None:
        rng = np.random.default_rng(seed)
        idx = []
        for _, g in index.groupby(["split", "label"]):
            n = max(1, int(round(limit * len(g) / len(index))))
            idx.extend(rng.choice(g.index.to_numpy(), size=min(n, len(g)), replace=False))
        index = index.loc[sorted(set(idx))].copy()

    with open(Path(channels_path).parent / "channel_names.json", encoding="utf-8") as f:
        all_channel_names = json.load(f)

    drop_groups = set(drop_groups or [])
    drop_names: set[str] = set()
    if "weapon" in drop_groups:
        drop_names |= {c for c in all_channel_names if c.startswith("attacker_weapon_")}
    if "outcome" in drop_groups:
        drop_names |= {
            "attacker_shot", "attacker_kill", "is_kill_headshot",
            "is_kill_through_smoke", "is_kill_wallbang", "attacker_midair",
            "victim_health", "victim_noise", "attacker_flashed",
        }
    if "motion" in drop_groups:
        drop_names |= {c for c in all_channel_names if c.startswith("attacker_vel")}
    if "position" in drop_groups:
        drop_names |= {
            "attacker_X", "attacker_Y", "attacker_Z",
            "victim_X", "victim_Y", "victim_Z",
        }
    keep_idx = np.array(
        [i for i, c in enumerate(all_channel_names) if c not in drop_names],
        dtype=np.int64,
    )
    channel_names = [all_channel_names[i] for i in keep_idx]
    n_features = len(channel_names)
    print(f"dropped channels: {sorted(drop_names)}", flush=True)

    train = index[index.split == "train"]
    val = index[index.split == "val"]
    test = index[index.split == "test"]
    mean, std = _norm_stats(
        channels_path, train["row"].to_numpy(), keep_idx=keep_idx
    )

    def loader(d, shuffle):
        ds = WindowDataset(
            channels_path, d["row"].to_numpy(),
            d["label"].to_numpy(), mean, std, keep_idx=keep_idx,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True,
            persistent_workers=(workers > 0),
        )

    train_loader = loader(train, True)
    val_loader = loader(val, False)
    test_loader = loader(test, False)

    model = AimTransformer(
        n_features=n_features, d_model=d_model, nhead=nhead,
        num_layers=layers, dim_feedforward=d_model * 2,
        dropout=dropout, pooling=pooling,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M n_features={n_features}", flush=True)

    pos = max(1, int(train["label"].sum()))
    neg = max(1, len(train) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best = {"player_auc": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            total_loss += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()

        p_val = _predict(model, val_loader, device)
        m_val = _eval_metrics(val, val["label"].to_numpy(), p_val)
        rec = {
            "epoch": epoch,
            "loss": total_loss / max(n, 1),
            "val_auc": m_val["auc"],
            "val_player_auc": m_val["player_auc_mean"],
            "val_recall_at_fpr1": m_val["recall_at_fpr1"],
            "seconds": time.time() - t0,
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)

        score = m_val["player_auc_mean"]
        if np.isfinite(score) and score > best["player_auc"]:
            best = {
                "player_auc": score,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    p_test = _predict(model, test_loader, device)
    m_test = _eval_metrics(test, test["label"].to_numpy(), p_test)

    results = {
        "device": str(device),
        "params": n_params,
        "n_features": n_features,
        "dropout": dropout,
        "pooling": pooling,
        "drop_groups": sorted(drop_groups),
        "dropped_channels": sorted(drop_names),
        "keep_idx": keep_idx.tolist(),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "best_epoch": best["epoch"],
        "best_val_player_auc": best["player_auc"],
        "test": m_test,
        "history": history,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "channel_names": channel_names,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    torch.save(model.state_dict(), out_dir / "model.pt")

    pred = test[["row", "match_key", "player_name", "kill_idx", "label"]].copy()
    pred["score"] = p_test
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    return results
