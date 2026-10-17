"""Index the CS2CD Context_window tar and cache windows into a memmap.

The tar contains ~218k small parquet files.  Reading 218k files from the
filesystem is slow on Windows, so we stream the tar once and write a
contiguous float16 memmap plus a metadata table.

Filename format:
    cdata-cheater-file_0-Player_3-kill_0.parquet
    ncdata-notcheater-file_0-Player_1-kill_2_aug0.parquet
"""
from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(
    r"^(?P<source>cdata|ncdata)-(?P<label>cheater|notcheater)"
    r"-file_(?P<match>\d+)-(?P<player>Player_\d+)-kill_(?P<kill>\d+)"
    r"(?P<aug>_aug\d+)?\.parquet$"
)


def parse_member(name: str) -> dict | None:
    base = Path(name).name
    m = NAME_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    return {
        "path": name,
        "source": d["source"],
        "label": 1 if d["label"] == "cheater" else 0,
        "match_idx": int(d["match"]),
        "player": int(d["player"].replace("Player_", "")),
        "player_name": d["player"],
        "kill_idx": int(d["kill"]),
        "augmented": d["aug"] is not None,
        "match_key": f"{d['source']}:{d['match']}",
    }


def list_members(tar_path: str | Path) -> pd.DataFrame:
    rows = []
    with tarfile.open(tar_path, "r:*") as tf:
        try:
            for m in tf:
                if not m.isfile():
                    continue
                parsed = parse_member(m.name)
                if parsed is not None:
                    parsed["size"] = m.size
                    rows.append(parsed)
        except (tarfile.ReadError, EOFError) as exc:
            # A partial tar is useful for smoke tests; production always
            # uses the complete file.
            print(f"warning: stopped at truncated tar: {exc}", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No context windows found in {tar_path}")
    return df.sort_values(
        ["source", "match_idx", "player", "kill_idx", "augmented"]
    ).reset_index(drop=True)


def _read_member(tf: tarfile.TarFile,
                 member: tarfile.TarInfo | str) -> pd.DataFrame:
    # Pass the TarInfo object when streaming; passing a name forces
    # tarfile to load the entire member table, which fails on a truncated
    # smoke-test archive and is slower on the full one.
    f = tf.extractfile(member)
    if f is None:
        raise RuntimeError(f"Cannot read {member}")
    return pd.read_parquet(io.BytesIO(f.read()))


def build_cache(
    tar_path: str | Path,
    out_dir: str | Path,
    include_augmented: bool = False,
    limit: int | None = None,
    log_every: int = 5000,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    members = list_members(tar_path)
    members.to_parquet(out_dir / "all_members.parquet", index=False)

    selected = members if include_augmented else members[~members["augmented"]]
    selected = selected.reset_index(drop=True)
    if limit is not None:
        selected = selected.head(int(limit)).reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No windows selected")

    n = len(selected)
    row_of = {p: i for i, p in enumerate(selected["path"])}
    selected_paths = set(row_of)
    columns = None
    shape = None
    mm = None
    filled = 0

    with tarfile.open(tar_path, "r:*") as tf:
        try:
            for m in tf:
                if not m.isfile() or m.name not in selected_paths:
                    continue
                df = _read_member(tf, m)
                if columns is None:
                    columns = list(df.columns)
                    shape = df.shape
                    if shape != (256, len(columns)):
                        raise ValueError(f"Unexpected window shape: {shape}")
                    mm = np.lib.format.open_memmap(
                        out_dir / "windows.npy",
                        mode="w+",
                        dtype=np.float16,
                        shape=(n, shape[0], shape[1]),
                    )
                if list(df.columns) != columns:
                    raise ValueError(f"Column mismatch in {m.name}")
                if df.shape != shape:
                    raise ValueError(f"Shape mismatch in {m.name}: {df.shape}")
                mm[row_of[m.name]] = df.to_numpy(dtype=np.float16)
                filled += 1
                if log_every and filled % log_every == 0:
                    print(f"cached {filled}/{n}", flush=True)
                if filled == n:
                    break
        except (tarfile.ReadError, EOFError) as exc:
            print(f"warning: tar ended early after {filled}/{n}: {exc}", flush=True)
        if mm is not None:
            mm.flush()
            del mm
    if columns is None:
        raise RuntimeError("No selected window could be read")
    if filled != n:
        raise RuntimeError(f"Only filled {filled}/{n} windows; use --limit or a complete tar")

    selected = selected.copy()
    selected.insert(0, "row", np.arange(len(selected)))
    selected.to_parquet(out_dir / "index.parquet", index=False)
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2)
    with open(out_dir / "cache_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": int(len(selected)),
                "shape": [int(shape[0]), int(shape[1])],
                "include_augmented": bool(include_augmented),
                "limit": limit,
            },
            f,
            indent=2,
        )
    return selected


def load_cache(cache_dir: str | Path):
    cache_dir = Path(cache_dir)
    index = pd.read_parquet(cache_dir / "index.parquet")
    with open(cache_dir / "feature_names.json", encoding="utf-8") as f:
        names = json.load(f)
    mm = np.load(cache_dir / "windows.npy", mmap_mode="r")
    if mm.shape != (len(index), 256, len(names)):
        raise ValueError(f"Cache shape {mm.shape} does not match index")
    return mm, index, names
