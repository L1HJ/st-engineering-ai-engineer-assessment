import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from drift_monitor import (  # noqa: E402
    active_component_windows,
    categorical_psi,
    drift_status,
    monthly_alert_burden,
    numeric_psi,
)


def test_numeric_psi_is_zero_for_identical_distribution():
    values = pd.Series(np.random.default_rng(7).normal(size=2000))
    assert numeric_psi(values, values.copy(), bins=10) < 1e-6


def test_numeric_psi_detects_mean_shift():
    rng = np.random.default_rng(7)
    reference = pd.Series(rng.normal(0.0, 1.0, size=3000))
    shifted = pd.Series(rng.normal(1.5, 1.0, size=3000))
    psi = numeric_psi(reference, shifted, bins=10)
    assert psi is not None and psi > 0.25
    assert drift_status(psi, {"watch": 0.10, "action": 0.25}) == "action"


def test_numeric_psi_treats_missing_as_bucket():
    reference = pd.Series([1.0] * 100)
    current = pd.Series([np.nan] * 100)
    psi = numeric_psi(reference, current, bins=10)
    assert psi is not None and psi > 1.0


def test_categorical_psi_handles_missing_and_unseen_categories():
    reference = pd.Series(["a", "b", None] * 200)
    current = pd.Series(["c"] * 600)
    psi = categorical_psi(reference, current)
    assert psi is not None and psi > 1.0


def test_active_component_windows_deduplicates_within_cycle_window():
    frame = pd.DataFrame({
        "aircraft_id": ["A", "A", "A", "A"],
        "component_id": ["C1", "C1", "C1", "C2"],
        "snapshot_cycle": [1, 2, 40, 41],
    })
    assert active_component_windows(frame, cooldown_cycles=30) == 3


def test_monthly_alert_burden_counts_episode_in_first_alert_month():
    frame = pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4"],
        "aircraft_id": ["A"] * 4,
        "component_id": ["C1"] * 4,
        "snapshot_date": ["2024-01-05", "2024-01-20", "2024-02-05", "2024-02-10"],
        "snapshot_cycle": [1, 2, 40, 41],
        "risk_score": [0.9, 0.8, 0.7, 0.6],
        "score_above_threshold": [True, True, True, False],
        "alert_episode_id": ["E1", "E1", "E1", None],
    })
    rows = monthly_alert_burden(frame, max_alerts_per_100=5.0, cooldown_cycles=30)
    assert [row["month"] for row in rows] == ["2024-01", "2024-02"]
    assert rows[0]["alert_episodes"] == 1
    assert rows[1]["alert_episodes"] == 0
