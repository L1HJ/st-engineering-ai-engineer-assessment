"""Tests for model preprocessing, reproducible inference, and contribution decomposition."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config as C
from modeling import (
    LightGBMPreprocessor,
    explain_bundle,
    fit_lightgbm,
    fit_logistic,
    load_bundle,
    predict_scores,
    save_bundle,
)


def make_snapshots(n=240, seed=7):
    rng = np.random.default_rng(seed)
    data = {
        "record_id": [f"r{i}" for i in range(n)],
        "snapshot_date": pd.date_range("2022-01-01", periods=n).astype(str),
        "snapshot_cycle": np.arange(n) * 1.5,
        "aircraft_id": [f"AC{i % 8}" for i in range(n)],
        "component_id": [f"CP{i % 20}" for i in range(n)],
        "aircraft_model": rng.choice(["A320", "B737"], n),
        "component_type": rng.choice(["fuel_pump", "starter"], n),
        "operator": rng.choice(["Op_A", "Op_B"], n),
        "fault_code_last": rng.choice(["NONE", "FC001", None], n),
        "fault_severity_last": rng.choice(["low", "high", None], n),
        "last_maint_type": rng.choice(["scheduled", "unscheduled"], n),
        "target": np.arange(n) % 2,
    }
    for field in C.MODEL_FEATURES:
        if field not in data:
            data[field] = rng.normal(10, 2, n)
    data["sensor_rpm_deviation"] = rng.normal(0, 1, n)
    data["sensor_exhaust_temp"] = rng.normal(500, 20, n)
    return pd.DataFrame(data)


def lightgbm_params():
    return {
        "objective": "binary",
        "n_estimators": 20,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 5,
        "random_state": 42,
        "n_jobs": 1,
        "deterministic": True,
        "verbosity": -1,
    }


def logistic_params():
    return {
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 500,
        "random_state": 42,
    }


def test_error_markers_missing_indicators_and_signed_rpm():
    train = make_snapshots()
    test = train.iloc[:1].copy()
    test["sensor_vibration"] = -1
    test["sensor_temp"] = -999
    test["sensor_pressure"] = 9999
    test["sensor_rpm_deviation"] = -1
    test["aircraft_model"] = "UNSEEN"
    preprocessor = LightGBMPreprocessor(C.MODEL_FEATURES).fit(train)
    transformed = preprocessor.transform(test)
    for sensor in ["sensor_vibration", "sensor_temp", "sensor_pressure"]:
        assert np.isnan(transformed.iloc[0][sensor])
        assert transformed.iloc[0][f"{sensor}_is_missing"] == 1
    assert transformed.iloc[0]["sensor_rpm_deviation"] == -1
    assert transformed.iloc[0]["sensor_rpm_deviation_is_missing"] == 0
    assert transformed.iloc[0]["aircraft_model"] == 0


def test_logistic_medians_are_train_only_and_unknown_categories_predict():
    train = make_snapshots()
    test = train.iloc[:3].copy()
    test["component_age_cycles"] = 1e9
    test["aircraft_model"] = "UNSEEN"
    bundle = fit_logistic(train, logistic_params())
    pipeline = bundle["model"]
    imputer = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
    numeric = pipeline.named_steps["preprocess"].transformers_[0][2]
    cleaned = pipeline.named_steps["clean"].transform(train)
    expected = cleaned[numeric].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    assert np.isfinite(predict_scores(bundle, test)).all()


def test_column_order_and_joblib_round_trip_preserve_predictions(tmp_path):
    train = make_snapshots()
    test = train.iloc[:8].copy()
    for name, bundle in [
        ("lr", fit_logistic(train, logistic_params())),
        ("lgbm", fit_lightgbm(train, train.iloc[:60], lightgbm_params(), early_stopping_rounds=5)),
    ]:
        expected = predict_scores(bundle, test)
        actual = predict_scores(bundle, test[list(reversed(test.columns))])
        np.testing.assert_allclose(actual, expected)
        path = tmp_path / f"{name}.joblib"
        save_bundle(bundle, path)
        np.testing.assert_allclose(predict_scores(load_bundle(path), test), expected)


def test_repeated_lightgbm_training_is_reproducible():
    train = make_snapshots()
    val = train.iloc[:60]
    first = fit_lightgbm(train, val, lightgbm_params(), early_stopping_rounds=5)
    second = fit_lightgbm(train, val, lightgbm_params(), early_stopping_rounds=5)
    np.testing.assert_allclose(predict_scores(first, val), predict_scores(second, val))


def test_contributions_reconstruct_raw_scores_for_both_models():
    train = make_snapshots()
    test = train.iloc[:5].copy()
    bundles = [
        fit_logistic(train, logistic_params()),
        fit_lightgbm(train, train.iloc[:60], lightgbm_params(), early_stopping_rounds=5),
    ]
    for bundle in bundles:
        explanations = explain_bundle(bundle, test)
        assert len(explanations) == len(test)
        for explanation in explanations:
            assert np.isclose(explanation["raw_score"], explanation["reconstructed_raw_score"], atol=1e-8)
            assert "increase" in explanation["reasons"]
            assert "decrease" in explanation["reasons"]
