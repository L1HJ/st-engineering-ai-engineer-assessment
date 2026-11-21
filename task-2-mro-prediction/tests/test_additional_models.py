"""Determinism and scope tests for the KNN/RF supplementary experiments."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config as C
from additional_models import balanced_training_indices, load_experiment_config, make_dense_preprocessor


def test_balanced_training_indices_are_deterministic_and_use_all_positives():
    target = np.array([0, 1, 0, 0, 1, 0, 0, 0])
    first = balanced_training_indices(target, 42)
    second = balanced_training_indices(target, 42)
    np.testing.assert_array_equal(first, second)
    assert set(np.flatnonzero(target == 1)).issubset(set(first))
    assert np.sum(target[first] == 1) == np.sum(target[first] == 0)


def test_dense_preprocessor_uses_only_model_features_and_train_medians():
    train = pd.DataFrame({
        "record_id": ["r1", "r2"], "aircraft_id": ["a1", "a2"],
        "component_id": ["c1", "c2"], "snapshot_date": ["2022-01-01"] * 2,
        "target": [0, 1], "aircraft_model": ["A", "B"],
        "component_type": ["x", "y"], "operator": ["o1", "o2"],
        "fault_code_last": ["NONE", "FC1"], "fault_severity_last": ["low", "high"],
        "last_maint_type": ["scheduled", "scheduled"],
    })
    for feature in C.MODEL_FEATURES:
        if feature not in train:
            train[feature] = [1.0, 3.0]
    preprocessor = make_dense_preprocessor()
    transformed = preprocessor.fit_transform(train)
    assert transformed.ndim == 2
    assert preprocessor.named_steps["clean"].features == C.MODEL_FEATURES
    imputer = preprocessor.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 2.0


def test_additional_experiment_config_forbids_temporal_test():
    config = load_experiment_config()
    assert config["experiment"]["temporal_test_used"] is False
    assert set(config["data"]) == {"train", "val", "removal_events", "snapshot_event_links"}
