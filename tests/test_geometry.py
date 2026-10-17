import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valveguard.features.geometry import KILL_ROW, recover, wrap_deg
from valveguard.features.summarize import per_tick_channels, summarize

SAMPLE = Path(__file__).parent / "data" / "sample_window.parquet"


def test_recover_sample():
    w = pd.read_parquet(SAMPLE)
    g = recover(w)
    assert g["map"] == "de_mirage"
    # Values checked against the original CS2CD match file.
    assert abs(float(g["pitch"][0]) - 10.4758) < 1e-3
    assert abs(float(g["yaw"][0]) - 35.288773) < 1e-3
    assert abs(float(g["pitch"][KILL_ROW]) - 3.0675964) < 1e-3
    assert abs(float(g["yaw"][KILL_ROW]) - 9.390228) < 1e-3
    assert abs(float(g["distance"][KILL_ROW]) - 1470.8245) < 0.1
    assert np.isfinite(g["err_head"]).all()


def test_wrap_deg():
    assert abs(float(wrap_deg(190.0)) + 170.0) < 1e-9
    assert abs(float(wrap_deg(-190.0)) - 170.0) < 1e-9


def test_summary_and_channels():
    w = pd.read_parquet(SAMPLE)
    f = summarize(w)
    assert f["map"] == "de_mirage"
    assert "time_to_target" in f
    assert "sparc" in f
    assert "lead_lag_ticks" in f
    X, names = per_tick_channels(w)
    assert X.shape == (256, len(names))
    assert np.isfinite(X).all()
