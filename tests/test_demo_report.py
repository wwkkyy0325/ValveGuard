import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valveguard.scripts.demo_report import (
    _add_review_order,
    _behavior_context,
    _combine_review_queues,
    _review_reasons,
)
from valveguard.scripts.demo_report_dataset import KILL_ROW, build_frame


def test_behavior_context():
    frame = pd.DataFrame([
        {"player_name": "a", "kill_tick": 1000, "headshot": 1, "weapon_class": 2,
         "shots": 0, "through_smoke": 0, "wallbang": 0, "midair": 0},
        {"player_name": "a", "kill_tick": 1050, "headshot": 1, "weapon_class": 2,
         "shots": 0, "through_smoke": 0, "wallbang": 0, "midair": 0},
        {"player_name": "a", "kill_tick": 1100, "headshot": 1, "weapon_class": 2,
         "shots": 0, "through_smoke": 0, "wallbang": 0, "midair": 0},
        {"player_name": "a", "kill_tick": 1150, "headshot": 1, "weapon_class": 2,
         "shots": 0, "through_smoke": 0, "wallbang": 0, "midair": 0},
        {"player_name": "b", "kill_tick": 0, "headshot": 0, "weapon_class": 1,
         "shots": 3, "through_smoke": 0, "wallbang": 0, "midair": 0},
        {"player_name": "b", "kill_tick": 900, "headshot": 0, "weapon_class": 1,
         "shots": 3, "through_smoke": 0, "wallbang": 0, "midair": 0},
    ])
    players = _behavior_context(frame).set_index("player_name")
    a = players.loc["a"]
    assert a["max_multikill_4s"] == 4
    assert a["burst_fraction"] == 1.0
    assert a["headshot_rate"] == 1.0
    assert abs(a["shots_per_kill"] - 0.25) < 1e-9
    assert a["sniper_share"] == 1.0
    b = players.loc["b"]
    assert b["max_multikill_4s"] == 1
    assert b["burst_fraction"] == 0.0
    assert b["headshot_rate"] == 0.0


def test_dataset_frame():
    names = [
        "attacker_weapon_knife", "attacker_weapon_auto_rifle",
        "attacker_weapon_semi_rifle", "attacker_weapon_pistol",
        "attacker_weapon_grenade", "attacker_weapon_smg",
        "attacker_weapon_shotgun", "is_kill_headshot",
        "is_kill_through_smoke", "is_kill_wallbang", "attacker_midair",
        "attacker_shot",
    ]
    position = {name: i for i, name in enumerate(names)}
    channels = np.zeros((2, 256, len(names)), dtype=np.float32)
    channels[0, KILL_ROW, position["is_kill_headshot"]] = 1.0
    channels[0, KILL_ROW, position["attacker_weapon_semi_rifle"]] = 1.0
    channels[0, :KILL_ROW, position["attacker_shot"]] = 1.0
    rows = pd.DataFrame({
        "player_name": ["Player_1", "Player_2"],
        "kill_tick": [1000, 1100],
        "label": [1, 0],
    })
    frame = build_frame(rows, channels, names)
    assert frame.loc[0, "headshot"] == 1.0
    assert frame.loc[0, "weapon_class"] == 2
    assert frame.loc[0, "shots"] == KILL_ROW
    assert frame.loc[1, "headshot"] == 0.0
    assert frame.loc[1, "weapon_class"] == 3


def test_review_reasons():
    row = pd.Series({
        "kills": 4, "headshot_rate": 1.0, "shots_per_kill": 0.25,
        "sniper_share": 1.0, "wallbang_rate": 0.0, "smoke_rate": 0.0,
        "max_multikill_4s": 4, "burst_fraction": 1.0,
    })
    reasons = _review_reasons(row)
    assert "HS>=80%" in reasons
    assert "<=1.5 shots/kill" in reasons
    assert "sniper HS" in reasons
    assert "4s multikill>=3" in reasons
    assert "burst>=30%" in reasons
    quiet = row.copy()
    quiet["headshot_rate"] = 0.2
    quiet["shots_per_kill"] = 4.0
    quiet["sniper_share"] = 0.0
    quiet["max_multikill_4s"] = 1
    quiet["burst_fraction"] = 0.0
    assert _review_reasons(quiet) == []


def _player(score, hs, spk, multi, burst):
    return {
        "shrunk_score": score, "flagged": False, "kills": 4,
        "headshot_rate": hs, "shots_per_kill": spk, "sniper_share": 0.0,
        "wallbang_rate": 0.0, "smoke_rate": 0.0,
        "max_multikill_4s": multi, "burst_fraction": burst,
    }


def test_review_order():
    players = pd.DataFrame([
        _player(0.30, 1.0, 0.5, 4, 1.0),   # below but two reasons
        _player(0.80, 0.2, 4.0, 1, 0.0),   # flag
        _player(0.50, 1.0, 0.5, 1, 0.0),   # review, two reasons
        _player(0.45, 0.9, 4.0, 1, 0.0),   # review, one reason
        _player(0.20, 0.2, 4.0, 1, 0.0),   # below, no reasons
    ])
    ordered = _add_review_order(players, threshold=0.7, p95=0.4)
    assert ordered["tier"].tolist() == ["flag", "review", "review", "below", "below"]
    assert ordered["shrunk_score"].tolist() == [0.8, 0.5, 0.45, 0.3, 0.2]
    assert ordered["review_rank"].tolist() == [1, 2, 3, 4, 5]


def test_combine_review_queues():
    one = pd.DataFrame([
        {"demo": "a", "tier": "below", "review_reason_count": 0, "shrunk_score": 0.3},
        {"demo": "a", "tier": "review", "review_reason_count": 2, "shrunk_score": 0.5},
        {"demo": "a", "tier": "flag", "review_reason_count": 0, "shrunk_score": 0.8},
    ])
    two = pd.DataFrame([
        {"demo": "b", "tier": "review", "review_reason_count": 1, "shrunk_score": 0.6},
        {"demo": "b", "tier": "below", "review_reason_count": 2, "shrunk_score": 0.2},
    ])
    combined = _combine_review_queues([one, two])
    assert combined["queue_rank"].tolist() == [1, 2, 3, 4, 5]
    assert combined["tier"].tolist() == ["flag", "review", "review", "below", "below"]
    assert combined["demo"].tolist() == ["a", "a", "b", "a", "b"]
    assert combined["shrunk_score"].tolist() == [0.8, 0.5, 0.6, 0.3, 0.2]
