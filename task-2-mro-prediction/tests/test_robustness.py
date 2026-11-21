"""Synthetic-data tests for the entity robustness flow; no real splits or trained models are read."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import robustness
from evaluation import evaluate_predictions


def test_internal_train_val_uses_frozen_dates_only():
    frame = pd.DataFrame({
        "record_id": ["r1", "r2", "r3", "r4"],
        "snapshot_date": ["2023-12-31", "2024-01-01", "2024-03-31", "2024-04-01"],
        "aircraft_id": ["AC1"] * 4,
        "component_id": ["CP1"] * 4,
        "snapshot_cycle": [1, 2, 3, 4],
        "target": [0, 0, 1, 1],
    })
    train, val = robustness.split_internal_train_val(frame)
    assert train["record_id"].tolist() == ["r1"]
    assert val["record_id"].tolist() == ["r2", "r3"]


def test_entity_context_filters_denominator_and_relabels_links():
    snapshots = pd.DataFrame({
        "record_id": ["r1", "r2"],
        "aircraft_id": ["AC_TEST"] * 2,
        "component_id": ["CP_TEST"] * 2,
        "snapshot_cycle": [10, 20],
        "snapshot_date": ["2024-05-01", "2024-05-02"],
        "target": [1, 0],
    })
    events = pd.DataFrame({
        "removal_event_id": ["e_ok", "e_warmup", "e_scheduled", "e_other"],
        "aircraft_id": ["AC_TEST", "AC_TEST", "AC_TEST", "AC_TRAIN"],
        "component_id": ["CP_TEST", "CP_TEST", "CP_TEST", "CP_TRAIN"],
        "event_date": ["2024-05-02"] * 4,
        "event_temporal_split": ["test", "warmup", "test", "test"],
        "removal_type": ["unscheduled", "unscheduled", "scheduled", "unscheduled"],
    })
    links = pd.DataFrame({
        "removal_event_id": ["e_ok", "e_ok", "e_ok", "e_other", "e_ok"],
        "record_id": ["r1", "r2", "r2", "r1", "r_external"],
        "cycles_before_event": [30.0, 30.1, -0.1, 5.0, 5.0],
        "event_temporal_split": ["train"] * 5,
        "snapshot_temporal_split": ["val"] * 5,
        "same_temporal_split": [False] * 5,
    })
    current_events, current_links = robustness.prepare_evaluation_context(
        snapshots, events, links, "aircraft_id", "entity_test"
    )
    assert current_events["removal_event_id"].tolist() == ["e_ok"]
    assert current_links[["removal_event_id", "record_id"]].values.tolist() == [["e_ok", "r1"]]
    assert current_links.iloc[0]["event_temporal_split"] == "entity_test"
    assert current_links.iloc[0]["snapshot_temporal_split"] == "entity_test"
    assert current_links.iloc[0]["same_temporal_split"] is True or current_links.iloc[0]["same_temporal_split"]

    result = evaluate_predictions(
        snapshots, current_events, current_links, [0.9, 0.0], 0.5, "entity_test"
    )
    assert result.metrics["actual_event_count"] == 1
    assert result.metrics["detected_event_count"] == 1


def test_internal_val_event_dates_are_independent_of_original_entity_membership():
    snapshots = pd.DataFrame({
        "record_id": ["r_val"], "aircraft_id": ["AC1"], "component_id": ["CP1"]
    })
    events = pd.DataFrame({
        "removal_event_id": ["e_train", "e_val", "e_test"],
        "aircraft_id": ["AC1"] * 3,
        "component_id": ["CP1"] * 3,
        "event_date": ["2023-12-31", "2024-02-01", "2024-04-01"],
        "event_temporal_split": ["train", "val", "test"],
        "removal_type": ["unscheduled"] * 3,
    })
    links = pd.DataFrame({
        "removal_event_id": ["e_train", "e_val", "e_test"],
        "record_id": ["r_val"] * 3,
        "cycles_before_event": [1.0] * 3,
    })
    current_events, current_links = robustness.prepare_evaluation_context(
        snapshots, events, links, "aircraft_id", "internal_val",
        robustness.INTERNAL_VAL_START, robustness.INTERNAL_VAL_END,
    )
    assert current_events["removal_event_id"].tolist() == ["e_val"]
    assert current_links["removal_event_id"].tolist() == ["e_val"]


def test_run_guards_stop_before_loading_entity_data(tmp_path, monkeypatch):
    monkeypatch.setattr(robustness, "path", lambda relative: str(tmp_path / relative))
    (tmp_path / "models").mkdir()
    manifest = tmp_path / robustness.MANIFEST_PATH
    manifest.write_text(json.dumps({"status": "started"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not completed"):
        robustness.run()

    manifest.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / robustness.OUTPUT_PATH).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exists"):
        robustness.run()
