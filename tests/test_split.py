import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valveguard.data.split import assign_match_splits


def test_match_disjoint():
    rows = []
    for i in range(40):
        label = 1 if i < 12 else 0
        source = "cdata" if label else "ncdata"
        for kill in range(2):
            rows.append({
                "match_key": f"{source}:{i}",
                "label": label,
                "player_name": f"Player_{i % 10 + 1}",
                "kill_idx": kill,
            })
    meta = pd.DataFrame(rows)
    out = assign_match_splits(meta, seed=1)
    assert out["split"].notna().all()
    assert out.groupby("match_key")["split"].nunique().max() == 1
    assert set(out["split"].unique()) == {"train", "val", "test"}
