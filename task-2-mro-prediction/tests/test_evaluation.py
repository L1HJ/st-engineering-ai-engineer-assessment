"""Hand-crafted example tests for the strict evaluator."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evaluation import active_component_window_count, assign_alert_episodes, evaluate_predictions


def fixture_data():
    snapshots = pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "aircraft_id": ["AC1"] * 5 + ["AC2"],
        "component_id": ["CP1"] * 4 + ["CP2", "CP3"],
        "snapshot_cycle": [0, 10, 35, 66, 0, 0],
        "snapshot_date": ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-08", "2024-01-01", "2024-01-01"],
        "target": [0, 1, 0, 0, 1, 0],
    })
    events = pd.DataFrame({
        "removal_event_id": ["e1", "e2"],
        "component_id": ["CP1", "CP2"],
        "event_temporal_split": ["val", "val"],
        "removal_type": ["unscheduled", "unscheduled"],
    })
    links = pd.DataFrame({
        "removal_event_id": ["e1", "e1", "e2"],
        "record_id": ["r1", "r2", "r5"],
        "same_temporal_split": [True, True, True],
        "cycles_before_event": [20.0, 10.0, 5.0],
        "event_temporal_split": ["val"] * 3,
        "snapshot_temporal_split": ["val"] * 3,
    })
    return snapshots, events, links


def test_active_window_deduplicates_component_within_aircraft_window():
    snapshots, _, _ = fixture_data()
    assert active_component_window_count(snapshots) == 5


def test_episode_cooling_across_windows_and_new_episode_after_30_cycles():
    snapshots, _, _ = fixture_data()
    predictions = assign_alert_episodes(snapshots, [0.9, 0.9, 0.9, 0.9, 0.0, 0.0], 0.5)
    cp1 = predictions[predictions["component_id"] == "CP1"]
    assert cp1["alert_episode_id"].nunique() == 2
    assert cp1.iloc[0]["alert_episode_id"] == cp1.iloc[2]["alert_episode_id"]
    assert cp1.iloc[2]["alert_episode_id"] != cp1.iloc[3]["alert_episode_id"]
    assert predictions["alert_episode_id"].nunique() == 2


def test_each_evaluation_resets_cooling_state():
    snapshots, _, _ = fixture_data()
    val = snapshots.iloc[[0]].copy()
    test = snapshots.iloc[[1]].copy()
    val_result = assign_alert_episodes(val, [0.9], 0.5)
    test_result = assign_alert_episodes(test, [0.9], 0.5)
    assert val_result.iloc[0]["alert_episode_id"] == "CP1_E0001"
    assert test_result.iloc[0]["alert_episode_id"] == "CP1_E0001"


def test_multiple_event_links_detect_only_once_and_lead_time_is_earliest():
    snapshots, events, links = fixture_data()
    result = evaluate_predictions(snapshots, events, links, [0.9] * 6, 0.5, "val")
    assert result.metrics["detected_event_count"] == 2
    assert result.metrics["actual_event_count"] == 2
    assert result.metrics["event_recall"] == 1.0
    e1 = result.events.set_index("removal_event_id").loc["e1"]
    assert e1["first_warning_cycles"] == 20.0


def test_cross_split_link_cannot_detect_event():
    snapshots, events, links = fixture_data()
    links.loc[links["removal_event_id"] == "e1", "same_temporal_split"] = False
    links.loc[links["removal_event_id"] == "e2", "snapshot_temporal_split"] = "test"
    result = evaluate_predictions(snapshots, events, links, [0.9] * 6, 0.5, "val")
    assert result.metrics["detected_event_count"] == 0


def test_links_outside_30_cycles_cannot_detect_event():
    snapshots, events, links = fixture_data()
    links["cycles_before_event"] = [30.1, -0.1, 31.0]
    result = evaluate_predictions(snapshots, events, links, [0.9] * 6, 0.5, "val")
    assert result.metrics["event_recall"] == 0.0


def test_zero_active_window_denominator_raises():
    snapshots, events, links = fixture_data()
    empty = snapshots.iloc[0:0]
    with pytest.raises(ValueError, match="denominator is 0"):
        evaluate_predictions(empty, events.iloc[0:0], links.iloc[0:0], [], 0.5, "val")


def test_zero_scores_have_zero_event_recall():
    snapshots, events, links = fixture_data()
    result = evaluate_predictions(snapshots, events, links, np.zeros(len(snapshots)), 0.5, "val")
    assert result.metrics["event_recall"] == 0.0
    assert result.metrics["alert_episode_count"] == 0


def test_oracle_linked_snapshot_alerts_recall_all_events():
    snapshots, events, links = fixture_data()
    linked = set(links["record_id"])
    scores = snapshots["record_id"].map(lambda record_id: float(record_id in linked)).to_numpy()
    result = evaluate_predictions(snapshots, events, links, scores, 0.5, "val")
    assert result.metrics["event_recall"] == 1.0
    assert result.metrics["detected_event_count"] == len(events)
