"""Generate the evaluation report from frozen artifacts only; no training, threshold search, or alert-episode rebuilding."""
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from modeling import explain_bundle, load_bundle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
MODEL_ORDER = ("lightgbm", "logistic_regression")
MODEL_LABELS = {"lightgbm": "LightGBM", "logistic_regression": "Logistic Regression"}
SENSORS = [
    "sensor_vibration", "sensor_temp", "sensor_pressure", "sensor_oil_debris",
    "sensor_rpm_deviation", "sensor_exhaust_temp",
]
PREDICTION_FIELDS = [
    "record_id", "aircraft_id", "component_id", "snapshot_date", "snapshot_cycle",
    "risk_score", "threshold", "score_above_threshold", "alert_episode_id",
    "model_name", "model_version", "config_hash",
]
COMPARISON_FIELDS = [
    "model", "split", "threshold", "roc_auc", "pr_auc", "snapshot_precision",
    "snapshot_recall", "snapshot_f1", "alert_episode_count",
    "active_component_window_count", "alerts_per_100_active", "detected_event_count",
    "actual_event_count", "detected_actual", "event_recall", "lead_time_median_cycles",
]
EXAMPLE_LABELS = {
    "correct_early_warning": "correct early warning",
    "false_positive_episode_without_event_link": "false-positive episode without an event link",
    "missed_event": "missed event",
    "high_sensor_missing": "high sensor missingness",
    "new_aircraft_or_new_component": "new aircraft or new component",
    "new_aircraft_or_new_component_pending": "new aircraft or new component (pending)",
    "cooldown_merged_episode": "high scores merged by cooldown within the same episode",
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def relative(path):
    return str(path.relative_to(ROOT))


def as_boolean(series):
    return series.map(lambda value: value is True or str(value).strip().lower() == "true")


def add_sensor_context(frame):
    result = frame.copy()
    missing_columns = []
    for sensor in SENSORS:
        values = pd.to_numeric(result[sensor], errors="coerce")
        invalid = values.isna() | values.isin([-999, 9999])
        if sensor != "sensor_rpm_deviation":
            invalid |= values < 0
        column = f"{sensor}_is_missing_context"
        result[column] = invalid
        missing_columns.append(column)
    flags = result[missing_columns].to_numpy(dtype=bool)
    result["sensor_missing_count"] = flags.sum(axis=1).astype(int)
    sensor_names = np.asarray(SENSORS)
    result["sensor_missing_pattern"] = [
        "none" if not row.any() else "|".join(sensor_names[row]) for row in flags
    ]
    result["snapshot_month"] = result["snapshot_date"].astype(str).str.slice(0, 7)
    return result.drop(columns=missing_columns)


def merge_prediction_snapshots(prediction_path, snapshot_path, model_key=None):
    predictions = pd.read_csv(prediction_path)
    if model_key is not None and "model_name" in predictions:
        selected = predictions["model_name"].astype(str).str.lower().eq(model_key)
        if selected.any() and not selected.all():
            predictions = predictions.loc[selected].copy()
    snapshots = pd.read_csv(snapshot_path)
    if predictions["record_id"].duplicated().any() or snapshots["record_id"].duplicated().any():
        raise ValueError(f"record_id must be unique: {relative(prediction_path)}")
    if set(predictions["record_id"]) != set(snapshots["record_id"]):
        raise ValueError(f"Prediction and snapshot record_id mismatch: {relative(prediction_path)}")

    identifiers = ["aircraft_id", "component_id", "snapshot_date", "snapshot_cycle"]
    check = predictions[["record_id"] + identifiers].merge(
        snapshots[["record_id"] + identifiers], on="record_id", suffixes=("_pred", "_snapshot"),
        validate="one_to_one",
    )
    for column in identifiers:
        left = check[f"{column}_pred"]
        right = check[f"{column}_snapshot"]
        if column == "snapshot_cycle":
            matches = np.isclose(pd.to_numeric(left), pd.to_numeric(right), rtol=0, atol=0)
        else:
            matches = left.astype(str).eq(right.astype(str))
        if not bool(np.all(matches)):
            raise ValueError(f"Prediction and snapshot {column} mismatch: {relative(prediction_path)}")

    extra = snapshots.drop(columns=identifiers)
    work = predictions.merge(extra, on="record_id", how="left", validate="one_to_one")
    work["score_above_threshold"] = as_boolean(work["score_above_threshold"])
    work["alert_episode_id"] = work["alert_episode_id"].astype("string")
    return add_sensor_context(work)


def active_window_count(frame):
    if frame.empty:
        return 0
    windows = frame[["aircraft_id", "component_id", "snapshot_cycle"]].copy()
    windows["window_index"] = np.floor(pd.to_numeric(windows["snapshot_cycle"]) / 30)
    return int(len(windows.drop_duplicates(["aircraft_id", "window_index", "component_id"])))


def prepare_events(work, events, links):
    current_events = events.loc[
        events["event_temporal_split"].eq("test") & events["removal_type"].eq("unscheduled")
    ].copy()
    components = work[["component_id", "component_type"]].drop_duplicates()
    aircraft = work[["aircraft_id", "aircraft_model"]].drop_duplicates()
    if components["component_id"].duplicated().any() or aircraft["aircraft_id"].duplicated().any():
        raise ValueError("Entity attributes in test snapshots are not unique")
    current_events = current_events.merge(
        components, on="component_id", how="left", validate="many_to_one",
    ).merge(
        aircraft, on="aircraft_id", how="left", validate="many_to_one",
    )
    if current_events[["component_type", "aircraft_model"]].isna().any().any():
        raise ValueError("Test events are missing component type or aircraft model registration")
    current_events["event_month"] = current_events["event_date"].astype(str).str.slice(0, 7)

    cycles = pd.to_numeric(links["cycles_before_event"], errors="coerce")
    valid = links.loc[
        as_boolean(links["same_temporal_split"])
        & cycles.between(0, 30, inclusive="both")
        & links["event_temporal_split"].eq("test")
        & links["snapshot_temporal_split"].eq("test")
        & links["removal_event_id"].isin(set(current_events["removal_event_id"]))
        & links["record_id"].isin(set(work["record_id"]))
    ].copy()
    link_context = valid.merge(
        work[[
            "record_id", "risk_score", "score_above_threshold", "alert_episode_id",
            "sensor_missing_pattern", "snapshot_month", "target",
        ]],
        on="record_id", how="inner", validate="many_to_one",
    )
    alert_links = link_context.loc[link_context["alert_episode_id"].notna()]
    detected = set(alert_links["removal_event_id"])
    current_events["detected"] = current_events["removal_event_id"].isin(detected)
    first_warning = alert_links.groupby("removal_event_id")["cycles_before_event"].max()
    current_events["first_warning_cycles"] = current_events["removal_event_id"].map(first_warning)
    return current_events, link_context


def snapshot_group_metrics(subset, actual_ids, detected_ids, threshold, event_basis):
    y_true = subset["target"].to_numpy(dtype=int)
    y_alert = subset["score_above_threshold"].to_numpy(dtype=bool)
    y_score = subset["risk_score"].to_numpy(dtype=float)
    active = active_window_count(subset)
    episodes = int(subset["alert_episode_id"].nunique(dropna=True))
    actual = len(actual_ids)
    detected = len(actual_ids & detected_ids)
    reasons = []
    if len(subset) < 100:
        reasons.append("fewer than 100 snapshots")
    if actual < 30:
        reasons.append("fewer than 30 events")
    return {
        "threshold": float(threshold),
        "n_snapshots": int(len(subset)),
        "n_positive_snapshots": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if y_true.sum() else None,
        "snapshot_precision": float(precision_score(y_true, y_alert, zero_division=0)),
        "snapshot_recall": float(recall_score(y_true, y_alert, zero_division=0)),
        "snapshot_f1": float(f1_score(y_true, y_alert, zero_division=0)),
        "snapshot_alert_count": int(y_alert.sum()),
        "snapshot_alert_rate": float(y_alert.mean()),
        "active_component_window_count": active,
        "alert_episode_count": episodes,
        "alerts_per_100_active": float(100 * episodes / active) if active else None,
        "detected_event_count": detected,
        "actual_event_count": actual,
        "event_recall": float(detected / actual) if actual else None,
        "event_detection_scope": event_basis,
        "small_sample_unstable": bool(reasons),
        "stability_note": "; ".join(reasons) + "; metrics unstable" if reasons else None,
    }


def build_temporal_groups(work, events, link_context, expected_metrics):
    threshold = float(expected_metrics["threshold"])
    global_detected = set(events.loc[events["detected"], "removal_event_id"])
    integrity = {
        "alert_episode_count": int(work["alert_episode_id"].nunique(dropna=True)),
        "active_component_window_count": active_window_count(work),
        "detected_event_count": len(global_detected),
        "actual_event_count": int(len(events)),
    }
    for key, value in integrity.items():
        if value != int(expected_metrics[key]):
            raise ValueError(f"Saved predictions do not match frozen metrics: {key}: {value} != {expected_metrics[key]}")

    dimensions = {
        "component_type": ("component_type", "component_type"),
        "aircraft_model": ("aircraft_model", "aircraft_model"),
        "event_month": ("snapshot_month", "event_month"),
    }
    output = {}
    for dimension, (snapshot_column, event_column) in dimensions.items():
        rows = []
        for value, subset in work.groupby(snapshot_column, sort=True, observed=True):
            value = str(value)
            actual_ids = set(events.loc[events[event_column].astype(str).eq(value), "removal_event_id"])
            metrics = snapshot_group_metrics(
                subset, actual_ids, global_detected, threshold, "whole_test_saved_episodes",
            )
            rows.append({"group": value, **metrics})
        output[dimension] = rows

    pattern_rows = []
    for value, subset in work.groupby("sensor_missing_pattern", sort=True, observed=True):
        pattern_links = link_context.loc[link_context["sensor_missing_pattern"].eq(value)]
        actual_ids = set(pattern_links["removal_event_id"])
        pattern_detected = set(
            pattern_links.loc[pattern_links["alert_episode_id"].notna(), "removal_event_id"]
        )
        metrics = snapshot_group_metrics(
            subset, actual_ids, pattern_detected, threshold, "same_missing_pattern_saved_episode_links",
        )
        pattern_rows.append({
            "group": str(value),
            "sensor_missing_count": int(subset["sensor_missing_count"].iloc[0]),
            **metrics,
        })
    output["sensor_missing_pattern"] = pattern_rows
    return output, integrity


def build_model_comparison(val_metrics, test_metrics, metadata):
    rows = []
    for model in MODEL_ORDER:
        for split, source in (("val", val_metrics), ("test", test_metrics)):
            metrics = source["models"][model]
            rows.append({
                "model": model,
                "split": split,
                "threshold": metrics["threshold"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "snapshot_precision": metrics["snapshot_precision"],
                "snapshot_recall": metrics["snapshot_recall"],
                "snapshot_f1": metrics["snapshot_f1"],
                "alert_episode_count": metrics["alert_episode_count"],
                "active_component_window_count": metrics["active_component_window_count"],
                "alerts_per_100_active": metrics["alerts_per_100_active"],
                "detected_event_count": metrics["detected_event_count"],
                "actual_event_count": metrics["actual_event_count"],
                "detected_actual": f"{metrics['detected_event_count']}/{metrics['actual_event_count']}",
                "event_recall": metrics["event_recall"],
                "lead_time_median_cycles": metrics["first_warning_cycles"]["median"],
                "model_version": metadata["models"][model]["candidate"],
            })
    return rows


def entity_prediction_files():
    candidates = []
    for path in REPORTS.glob("*.csv"):
        name = path.name.lower()
        if "prediction" not in name or "lightgbm" not in name or "internal_val" in name:
            continue
        split = "by_aircraft" if "by_aircraft" in name else "by_component" if "by_component" in name else None
        if split:
            candidates.append((0 if split == "by_aircraft" else 1, path.name, split, path))
    return [(split, path) for _, _, split, path in sorted(candidates)]


def entity_explanation(work, row, split):
    model_paths = sorted(
        MODELS.rglob("*lightgbm*.joblib"),
        key=lambda path: (split not in str(path), str(path.relative_to(ROOT))),
    )
    for model_path in model_paths:
        try:
            explanation = explain_bundle(load_bundle(model_path), row.to_frame().T)[0]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        probability = float(1 / (1 + np.exp(-explanation["raw_score"])))
        if np.isclose(probability, float(row["risk_score"]), rtol=1e-8, atol=1e-10):
            return explanation, model_path
    return {
        "status": "pending_entity_model_bundle",
        "reason": f"Entity model bundle for {split} is not saved; cannot generate explain_bundle with the same model for the saved risk scores. The temporal model was not used as a substitute.",
        "reasons": {"increase": [], "decrease": []},
    }, None


def example_record(row, example_type, reason, sequence, source_file, explanation, context=None):
    context = context or {}
    result = {
        "example_type": example_type,
        "example_sequence": sequence,
        "example_reason": reason,
        "source_split": context.get("source_split", "temporal_test"),
        "source_prediction_file": relative(source_file) if source_file else None,
    }
    for field in PREDICTION_FIELDS:
        result[field] = row.get(field) if row is not None else ("lightgbm" if field == "model_name" else None)
    for field in [
        "target", "aircraft_model", "component_type", "sensor_missing_count",
        "sensor_missing_pattern", "removal_event_id", "event_date", "cycles_before_event",
        "episode_snapshot_count",
    ]:
        result[field] = context.get(field, row.get(field) if row is not None else None)
    result["explain_bundle"] = json.dumps(
        json_safe(explanation), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return result


def select_examples(work, events, link_context, metadata):
    prediction_path = REPORTS / "predictions_test_lightgbm.csv"
    model_path = MODELS / "lightgbm.joblib"
    bundle = load_bundle(model_path)
    examples = []

    def add(row, kind, reason, sequence=1, context=None, source=prediction_path, explanation=None):
        if explanation is None:
            explanation = explain_bundle(bundle, row.to_frame().T)[0]
            probability = float(1 / (1 + np.exp(-explanation["raw_score"])))
            if not np.isclose(probability, float(row["risk_score"]), rtol=1e-8, atol=1e-10):
                raise ValueError("Explanation score does not match the saved prediction")
        examples.append(example_record(row, kind, reason, sequence, source, explanation, context))

    correct = link_context.loc[
        link_context["alert_episode_id"].notna()
        & pd.to_numeric(link_context["cycles_before_event"]).gt(0)
    ].copy()
    if correct.empty:
        raise ValueError("No detected event is available for a correct early-warning example")
    correct = correct.sort_values(
        ["cycles_before_event", "risk_score", "record_id", "removal_event_id"],
        ascending=[False, False, True, True], kind="mergesort",
    ).iloc[0]
    row = work.loc[work["record_id"].eq(correct["record_id"])].iloc[0]
    add(row, "correct_early_warning", "A saved episode hits a same-split event link, and the warning precedes the event.", context={
        "removal_event_id": correct["removal_event_id"], "event_date": correct["event_date"],
        "cycles_before_event": correct["cycles_before_event"],
    })

    high = work.loc[work["alert_episode_id"].notna()].copy()
    episode_sizes = high.groupby("alert_episode_id", observed=True).size()
    linked_records = set(link_context["record_id"])
    false_stats = []
    for episode_id, subset in high.groupby("alert_episode_id", observed=True):
        if subset["record_id"].isin(linked_records).any() or int(subset["target"].sum()) != 0:
            continue
        false_stats.append({
            "alert_episode_id": str(episode_id), "episode_snapshot_count": int(len(subset)),
            "max_risk_score": float(subset["risk_score"].max()),
        })
    if not false_stats:
        raise ValueError("No false-positive episode without a valid event link is available")
    false_stats = pd.DataFrame(false_stats).sort_values(
        ["episode_snapshot_count", "max_risk_score", "alert_episode_id"],
        ascending=[False, False, True], kind="mergesort",
    ).iloc[0]
    false_episode = false_stats["alert_episode_id"]
    false_rows = high.loc[high["alert_episode_id"].eq(false_episode)]
    row = false_rows.sort_values(["risk_score", "record_id"], ascending=[False, True], kind="mergesort").iloc[0]
    add(row, "false_positive_episode_without_event_link", "All high-score snapshots in this episode have no valid test event link, and all targets are 0.", context={
        "episode_snapshot_count": int(false_stats["episode_snapshot_count"]),
    })

    missed_ids = set(events.loc[~events["detected"], "removal_event_id"])
    missed = link_context.loc[link_context["removal_event_id"].isin(missed_ids)].copy()
    if missed.empty:
        raise ValueError("No event scoring opportunity is available for a missed-event example")
    best_missed = missed.sort_values(
        ["removal_event_id", "risk_score", "cycles_before_event", "record_id"],
        ascending=[True, False, False, True], kind="mergesort",
    ).drop_duplicates("removal_event_id")
    selected_missed = best_missed.sort_values(
        ["risk_score", "removal_event_id"], ascending=[False, True], kind="mergesort",
    ).iloc[0]
    row = work.loc[work["record_id"].eq(selected_missed["record_id"])].iloc[0]
    add(row, "missed_event", "None of the snapshots linked to this actual event entered a saved alert episode; the highest-scoring opportunity is shown.", context={
        "removal_event_id": selected_missed["removal_event_id"],
        "event_date": selected_missed["event_date"],
        "cycles_before_event": selected_missed["cycles_before_event"],
    })

    row = work.sort_values(
        ["sensor_missing_count", "risk_score", "record_id"],
        ascending=[False, False, True], kind="mergesort",
    ).iloc[0]
    add(row, "high_sensor_missing", "Selected from the group with the highest cleaned missing count across the six sensors; the original score and contribution explanation are preserved.")

    entity_files = entity_prediction_files()
    if not entity_files:
        pending_reason = "No saved LightGBM predictions for by_aircraft/by_component yet; rerun this script after generating the supplementary predictions to update."
        examples.append(example_record(
            None, "new_aircraft_or_new_component_pending", pending_reason, 1, None,
            {"status": "pending", "reason": pending_reason},
            {"source_split": "by_aircraft_or_by_component"},
        ))
    else:
        for split, source in entity_files:
            entity_work = merge_prediction_snapshots(
                source, ROOT / f"data/splits/{split}/test.csv", model_key="lightgbm",
            )
            row = entity_work.sort_values(["risk_score", "record_id"], ascending=[False, True], kind="mergesort").iloc[0]
            explanation, matched_model = entity_explanation(entity_work, row, split)
            reason = (
                "Saved LightGBM prediction from the entity supplementary split; the explanation model and score are verified to match."
                if matched_model is not None else
                "Real saved prediction from the entity supplementary split; the entity model is not saved, so explain_bundle is explicitly marked as pending and is not replaced by a temporal explanation."
            )
            examples.append(example_record(
                row, "new_aircraft_or_new_component", reason,
                1, source, explanation, {"source_split": split},
            ))

    merged_candidates = []
    for episode_id, subset in high.groupby("alert_episode_id", observed=True):
        ordered = subset.sort_values(["snapshot_cycle", "snapshot_date", "record_id"], kind="mergesort")
        if len(ordered) < 2:
            continue
        cycles = pd.to_numeric(ordered["snapshot_cycle"]).to_numpy(dtype=float)
        gaps = np.diff(cycles)
        valid_positions = np.flatnonzero((gaps > 0) & (gaps <= 30))
        if valid_positions.size:
            merged_candidates.append((str(episode_id), len(ordered), int(valid_positions[0])))
    if not merged_candidates:
        raise ValueError("No episode with multiple consecutive high-score snapshots is available")
    merged_candidates.sort(key=lambda item: (-item[1], item[0], item[2]))
    episode_id, _, position = merged_candidates[0]
    ordered = high.loc[high["alert_episode_id"].eq(episode_id)].sort_values(
        ["snapshot_cycle", "snapshot_date", "record_id"], kind="mergesort",
    )
    pair = ordered.iloc[position:position + 2]
    for sequence, (_, row) in enumerate(pair.iterrows(), start=1):
        add(row, "cooldown_merged_episode", "Adjacent high-score snapshots of the same component are at most 30 cycles apart; the same saved episode is retained.", sequence=sequence, context={
            "episode_snapshot_count": int(len(ordered)),
        })

    example_columns = [
        "example_type", "example_sequence", "example_reason", "source_split",
        "source_prediction_file", *PREDICTION_FIELDS, "target", "aircraft_model",
        "component_type", "sensor_missing_count", "sensor_missing_pattern", "removal_event_id",
        "event_date", "cycles_before_event", "episode_snapshot_count", "explain_bundle",
    ]
    frame = pd.DataFrame(examples).loc[:, example_columns]
    return frame, metadata


def fmt(value, digits=6):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def pct(value, digits=4):
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def integer(value):
    if value is None or pd.isna(value):
        return "—"
    return str(int(value))


def comparison_markdown(comparison, split):
    lines = [
        "| Model | Threshold | ROC-AUC | PR-AUC | Snapshot P/R/F1 | Episodes | Active windows | Alerts/100 | Detected/Actual | Event recall | Median lead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        if row["split"] != split:
            continue
        snapshot = "/".join(fmt(row[key], 4) for key in ["snapshot_precision", "snapshot_recall", "snapshot_f1"])
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {fmt(row['threshold'], 8)} | {fmt(row['roc_auc'], 6)} | "
            f"{fmt(row['pr_auc'], 6)} | {snapshot} | {row['alert_episode_count']} | "
            f"{row['active_component_window_count']} | {fmt(row['alerts_per_100_active'], 5)} | "
            f"{row['detected_actual']} | {pct(row['event_recall'])} | {fmt(row['lead_time_median_cycles'], 4)} |"
        )
    return lines


def test_definitions():
    rows = []
    for filename in ["test_evaluation.py", "test_modeling.py"]:
        path = ROOT / "tests" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                rows.append((filename, node.name))
    return rows


def entity_metric_rows(value, path=()):
    rows = []
    if isinstance(value, dict):
        if "event_recall" in value and "alert_episode_count" in value:
            rows.append(("/".join(path) or "root", value))
        for key, item in value.items():
            if key != "first_warning_cycles":
                rows.extend(entity_metric_rows(item, path + (str(key),)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(entity_metric_rows(item, path + (str(index),)))
    return rows


def build_docs(metadata, manifest, comparison, temporal, examples, entity, generation, val_data, test_data):
    tests = test_definitions()
    test_counts = {
        filename: sum(row_filename == filename for row_filename, _ in tests)
        for filename in ["test_evaluation.py", "test_modeling.py"]
    }
    sanity = read_json(REPORTS / "sanity_checks.json")
    candidates = read_json(REPORTS / "val_candidate_metrics.json")
    threshold_search = pd.read_csv(REPORTS / "val_threshold_search.csv")
    threshold_counts = threshold_search.groupby(["model", "candidate"], sort=True).size().to_dict()
    lgbm_test = test_data["models"]["lightgbm"]
    lr_test = test_data["models"]["logistic_regression"]
    lgbm_val = val_data["models"]["lightgbm"]
    lines = [
        "# MRO Model Evaluation Report", "",
        "> This report only consolidates frozen artifacts; it does not retrain, search thresholds, rebuild cooldown episodes, or rerun the temporal test.", "",
        "## 1. Data version, splits, and frozen configuration", "",
        f"- Data version: `{metadata['dataset_version']}`; config SHA-256: `{metadata['config_hash']}`.",
        f"- 30-cycle prediction window; 30-cycle cooldown and active window. Random seed: `{metadata['random_seed']}`.",
        f"- One-shot test manifest: `models/test_evaluation_manifest.json`, status `{manifest['status']}`; evaluation code SHA-256: `{manifest['evaluation_code_sha256']}`.", "",
        "| Split | Snapshots | Positive snapshots | Distinct actual events | File SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    split_values = {
        "train": (metadata["train_rows"], metadata["train_positive_rows"], generation["event_coverage_by_temporal_split"]["train"]["n_events"]),
        "val": (val_data["models"]["lightgbm"]["n_snapshots"], val_data["models"]["lightgbm"]["n_positive_snapshots"], val_data["models"]["lightgbm"]["actual_event_count"]),
        "test": (lgbm_test["n_snapshots"], lgbm_test["n_positive_snapshots"], lgbm_test["actual_event_count"]),
    }
    for split in ["train", "val", "test"]:
        rows, positives, events_count = split_values[split]
        digest = metadata["input_hashes"][f"data/splits/temporal/{split}.csv"]
        lines.append(f"| {split} | {rows} | {positives} | {events_count} | `{digest}` |")
    lines.extend(["", "Other frozen input hashes (all from `models/model_metadata.json`):", "", "| File | SHA-256 |", "|---|---|"])
    for path, digest in sorted(metadata["input_hashes"].items()):
        if not path.startswith("data/splits/temporal/"):
            lines.append(f"| `{path}` | `{digest}` |")
    lines.extend(["", "Frozen models:", "", "| Model | Candidate | Model SHA-256 | Frozen threshold |", "|---|---|---|---:|"])
    for model in MODEL_ORDER:
        lines.append(
            f"| {MODEL_LABELS[model]} | `{metadata['models'][model]['candidate']}` | "
            f"`{metadata['models'][model]['model_sha256']}` | {fmt(metadata['models'][model]['threshold'], 8)} |"
        )

    lines.extend(["", "## 2. Strict metric definitions and evaluator validation", "",
        "Active windows count distinct components per aircraft within `floor(snapshot_cycle/30)`; episodes merge only alerts for the same component whose adjacent high-score gap is at most 30 cycles; events are detected only by saved episodes through valid same-split links within 0-30 cycles. The event denominator is distinct events, not positive snapshots.", "",
        f"Static inspection of the test files covers **{len(tests)} relevant unit tests**: `tests/test_evaluation.py`: {test_counts['test_evaluation.py']}; `tests/test_modeling.py`: {test_counts['test_modeling.py']}. This report does not rerun pytest; the results are not presented as a new test evaluation.", "",
        "| File | Unit test function |", "|---|---|",
    ])
    for filename, name in tests:
        lines.append(f"| `tests/{filename}` | `{name}` |")
    single_feature = max(sanity["single_feature"]["features"], key=lambda row: row["val_roc_auc"])
    lines.extend(["", "Saved sanity results (`reports/sanity_checks.json`):", "",
        "| Check | Actual result | Pass flag |", "|---|---|---|",
        f"| All-zero scores | Event recall {pct(sanity['zero_scores']['event_recall'])}, episodes {sanity['zero_scores']['alert_episode_count']} | - |",
        f"| Oracle linked alerts | {sanity['oracle_linked_alerts']['detected_event_count']}/{sanity['oracle_linked_alerts']['actual_event_count']}, recall {pct(sanity['oracle_linked_alerts']['event_recall'])} | - |",
        f"| Shuffled labels | Val ROC-AUC {fmt(sanity['shuffled_labels']['roc_auc'])}, event recall {pct(sanity['shuffled_labels']['event_recall'])} | `{sanity['shuffled_labels']['passed']}` |",
        f"| Single-feature ceiling | Highest is `{single_feature['feature']}`, Val ROC-AUC {fmt(single_feature['val_roc_auc'])} | `{sanity['single_feature']['passed']}` |",
        f"| Simple baseline | Event recall {pct(sanity['simple_baseline']['event_recall'])}, alerts/100={fmt(sanity['simple_baseline']['alerts_per_100_active'], 5)} | - |",
        f"| Overall | Saved overall flag | `{sanity['passed']}` |",
    ])

    lines.extend(["", "## 3. Val results and threshold selection curves for both models", "",
        "The full val/test comparison is in `reports/model_comparison.csv`; threshold units, P/R/F1, and lead time are probability, ratio, and cycles, respectively.", "",
        *comparison_markdown(comparison, "val"), "",
        "Thresholds are searched on val only: maximize event recall under alerts/100 <= 4.5, breaking ties by fewer alerts and then a higher threshold. Full curve points are in `reports/val_threshold_search.csv`:", "",
        "| Model | Candidate | Curve points | Threshold | Event recall | Alerts/100 | Selected |", "|---|---|---:|---:|---:|---:|---|",
    ])
    for model in MODEL_ORDER:
        for row in candidates:
            if row["model"] != model:
                continue
            count = int(threshold_counts[(model, row["candidate"])])
            lines.append(
                f"| {MODEL_LABELS[model]} | `{row['candidate']}` | {count} | {fmt(row['threshold'], 8)} | "
                f"{pct(row['event_recall'])} | {fmt(row['alerts_per_100_active'], 5)} | `{row['selected']}` |"
            )

    gate = val_data["internal_gate"]
    lines.extend(["", "## 4. LightGBM internal safety gates", "",
        f"Val detected **{lgbm_val['detected_event_count']}/{lgbm_val['actual_event_count']}**, event recall **{pct(lgbm_val['event_recall'])}** (requirement >=85%); alerts/100 active = **{fmt(lgbm_val['alerts_per_100_active'], 5)}** (requirement <=4.5). Recall gate `{gate['recall_passed']}`, alert gate `{gate['alerts_passed']}`, so the saved `test_permitted={gate['test_permitted']}`.", "",
        "## 5. One-shot temporal test results", "",
        *comparison_markdown(comparison, "test"), "",
        f"**LightGBM: {lgbm_test['detected_event_count']}/{lgbm_test['actual_event_count']} = {pct(lgbm_test['event_recall'])}**, with **{lgbm_test['alert_episode_count']} episodes**, **{lgbm_test['active_component_window_count']} active windows**, and **{fmt(lgbm_test['alerts_per_100_active'], 5)} alerts/100**. The formal targets are event recall >=80% and alerts/100 <=5; the saved conclusion is **{test_data['formal_target']['conclusion']}** (recall `{test_data['formal_target']['recall_passed']}`, alerts `{test_data['formal_target']['alerts_passed']}`).", "",
        f"Logistic Regression: {lr_test['detected_event_count']}/{lr_test['actual_event_count']} = {pct(lr_test['event_recall'])}, {lr_test['alert_episode_count']} episodes, {fmt(lr_test['alerts_per_100_active'], 5)} alerts/100; the active-window denominator is also {lr_test['active_component_window_count']}.", "",
        "These results come from `reports/temporal_test_metrics.json` and the same batch of saved predictions; the test must not be rerun, tuned, re-thresholded, or re-cooled based on group or example findings.", "",
        "## 6. New aircraft, new components, and group robustness", "",
    ])
    if entity is None:
        lines.extend([
            "**Entity supplementary robustness: pending.** `reports/entity_robustness_metrics.json` does not exist yet, so this report does not fabricate new-aircraft/new-component conclusions; rerun `src/reporting.py` after generating that file to update this section and the merged JSON.", "",
        ])
    else:
        lines.extend(["Entity supplementary results come from `reports/entity_robustness_metrics.json` and are merged as-is into `reports/group_robustness_metrics.json`; the table below only excerpts existing strict metrics from the file:", ""])
        entity_rows = entity_metric_rows(entity)
        if entity_rows:
            lines.extend(["| Metric path | Snapshots | Episodes | Active windows | Detected/Actual | Event recall | Alerts/100 |", "|---|---:|---:|---:|---:|---:|---:|"])
            for path, row in entity_rows:
                detected = f"{integer(row.get('detected_event_count'))}/{integer(row.get('actual_event_count'))}"
                lines.append(
                    f"| `{path}` | {integer(row.get('n_snapshots'))} | {integer(row.get('alert_episode_count'))} | "
                    f"{integer(row.get('active_component_window_count'))} | {detected} | {pct(row.get('event_recall'))} | {fmt(row.get('alerts_per_100_active'), 5)} |"
                )
        else:
            lines.append("The file exists, but no standard metric node containing `event_recall` and `alert_episode_count` was found; inspect the original file directly.")
        lines.append("")

    lines.extend([
        "Full temporal group results for both models are in `reports/temporal_group_metrics.json`. LightGBM component-type results are listed below:", "",
        "| Component type | Snapshots | Detected/Actual | Event recall | Episodes | Active windows | Alerts/100 | Small sample |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in temporal["models"]["lightgbm"]["component_type"]:
        lines.append(
            f"| `{row['group']}` | {row['n_snapshots']} | {row['detected_event_count']}/{row['actual_event_count']} | "
            f"{pct(row['event_recall'])} | {row['alert_episode_count']} | {row['active_component_window_count']} | "
            f"{fmt(row['alerts_per_100_active'], 5)} | `{row['small_sample_unstable']}` |"
        )
    group_counts = {key: len(value) for key, value in temporal["models"]["lightgbm"].items()}
    lines.extend(["",
        f"Other groups: {group_counts['aircraft_model']} aircraft models, {group_counts['event_month']} event/snapshot months, and {group_counts['sensor_missing_pattern']} sensor missing patterns. Ordinary-dimension events are grouped by their own attributes or event month; missing-pattern events use the subset with a scoring opportunity of that pattern and count as detected only if a same-pattern link enters a saved episode. Month episodes and missing-pattern events may overlap across groups and must not be summed; groups with fewer than 100 snapshots or fewer than 30 events are flagged `small_sample_unstable=true` and are descriptive only.", "",
        "## 7. False positives, misses, and explanation examples", "",
        "Examples are in `reports/predictions_example.csv`, including the original prediction fields and the `explain_bundle` JSON. LightGBM contributions correspond to raw score/log-odds, not probability percentages, and are not causal effects.", "",
        "| Type | Record | Risk score | Episode / Event | Top increasing contributions (raw score) |", "|---|---|---:|---|---|",
    ])
    for _, example in examples.iterrows():
        kind = example["example_type"]
        explanation = json.loads(example["explain_bundle"])
        increases = explanation.get("reasons", {}).get("increase", [])
        reasons = "; ".join(f"{item['feature']}={item['contribution']:.6g}" for item in increases[:2]) or "-"
        episode = example["alert_episode_id"] if pd.notna(example["alert_episode_id"]) else "—"
        event = example["removal_event_id"] if pd.notna(example["removal_event_id"]) else "—"
        risk = fmt(example["risk_score"], 6) if pd.notna(example["risk_score"]) else "—"
        record = example["record_id"] if pd.notna(example["record_id"]) else "—"
        lines.append(f"| {EXAMPLE_LABELS[kind]} | `{record}` | {risk} | `{episode}` / `{event}` | {reasons} |")
    entity_explanation_pending = any(
        example["example_type"] == "new_aircraft_or_new_component"
        and json.loads(example["explain_bundle"]).get("status") == "pending_entity_model_bundle"
        for _, example in examples.iterrows()
    )
    if entity_explanation_pending:
        lines.extend(["", "The entity prediction examples are included, but `src/robustness.py` did not save an entity model bundle; the corresponding `explain_bundle` is marked `pending_entity_model_bundle`. This report does not substitute temporal-model contributions for entity-prediction explanations.", ""])

    lines.extend(["", "## 8. Limitations, failures, and scope of applicability", "",
        f"- The data are synthetic: `data/audit/generation_summary.json` records {generation['n_rows']} rows, {generation['n_aircraft']} aircraft, and {generation['n_simulated_instances']} simulated component instances; metrics must not be interpreted as real airline operational performance.",
        f"- LightGBM still misses {lgbm_test['actual_event_count'] - lgbm_test['detected_event_count']} events and Logistic Regression misses {lr_test['actual_event_count'] - lr_test['detected_event_count']}; snapshot precision is {fmt(lgbm_test['snapshot_precision'], 6)} and {fmt(lr_test['snapshot_precision'], 6)}, respectively, so false positives and small-group instability are not hidden.",
        "- Feature contributions explain model scores only and do not imply causality; results must not be used for real airworthiness, maintenance release, or maintenance decisions.",
        "- The temporal test was completed exactly once; groups, entity supplements, and explanations must not retroactively change the model, thresholds, evaluator, or data, and the test must not be rerun to chase metrics.",
        "- This script only reads frozen files and generates reports; rerunning it solely to include later entity results does not constitute a new evaluation of the temporal test.", "",
    ])
    return "\n".join(lines)


def main():
    metadata = read_json(MODELS / "model_metadata.json")
    manifest = read_json(MODELS / "test_evaluation_manifest.json")
    val_data = read_json(REPORTS / "val_metrics.json")
    test_data = read_json(REPORTS / "temporal_test_metrics.json")
    generation = read_json(ROOT / "data/audit/generation_summary.json")
    events = pd.read_csv(ROOT / "data/audit/removal_events.csv")
    links = pd.read_csv(ROOT / "data/audit/snapshot_event_links.csv")
    test_snapshots = ROOT / "data/splits/temporal/test.csv"

    comparison = build_model_comparison(val_data, test_data, metadata)
    pd.DataFrame([{key: row[key] for key in COMPARISON_FIELDS} for row in comparison]).to_csv(
        REPORTS / "model_comparison.csv", index=False, lineterminator="\n",
    )

    temporal_models = {}
    integrities = {}
    contexts = {}
    for model in MODEL_ORDER:
        prediction_path = REPORTS / f"predictions_test_{model}.csv"
        work = merge_prediction_snapshots(prediction_path, test_snapshots, model_key=model)
        current_events, link_context = prepare_events(work, events, links)
        groups, integrity = build_temporal_groups(
            work, current_events, link_context, test_data["models"][model],
        )
        temporal_models[model] = groups
        integrities[model] = integrity
        contexts[model] = (work, current_events, link_context)

    temporal = {
        "dataset_version": metadata["dataset_version"],
        "split": "test",
        "methodology": {
            "source": "Saved test risk scores and alert_episode_id",
            "uses_saved_alert_episode_ids": True,
            "recomputes_cooldown": False,
            "valid_event_links": "same_temporal_split=true, event/snapshot split=test, 0<=cycles_before_event<=30",
            "snapshot_metrics": "Computed on each group's snapshot subset; active windows are also deduplicated within the subset",
            "ordinary_event_groups": "component_type/aircraft_model use event entity attributes, event_month uses event_date; snapshot month uses snapshot_date; detection uses saved episodes from the whole test split",
            "missing_pattern_event_groups": "An event must have at least one valid scoring link with this missing pattern; the event counts as detected in this group only if a link of this pattern enters a saved episode; events may belong to multiple pattern groups",
            "groups_may_overlap": "Months may share episodes; missing-pattern scoring opportunities and events may overlap; episode/event counts across groups are not directly additive",
            "small_sample_unstable_rule": "n_snapshots<100 or actual_event_count<30; descriptive stability flag only, not used for selection or tuning",
        },
        "overall_saved_prediction_integrity": integrities,
        "models": temporal_models,
    }
    write_json(REPORTS / "temporal_group_metrics.json", temporal)

    examples, _ = select_examples(*contexts["lightgbm"], metadata)
    examples.to_csv(
        REPORTS / "predictions_example.csv", index=False, lineterminator="\n", float_format="%.17g",
    )

    entity_path = REPORTS / "entity_robustness_metrics.json"
    entity = read_json(entity_path) if entity_path.exists() else None
    merged = {
        "dataset_version": metadata["dataset_version"],
        "temporal_group_metrics": temporal,
        "entity_robustness_metrics": entity if entity is not None else {
            "status": "pending",
            "expected_file": "reports/entity_robustness_metrics.json",
            "message": "Entity results do not exist yet; generate them and rerun src/reporting.py to update, without re-evaluating the temporal test.",
        },
        "entity_status": "available" if entity is not None else "pending",
    }
    write_json(REPORTS / "group_robustness_metrics.json", merged)

    document = build_docs(
        metadata, manifest, comparison, temporal, examples, entity, generation, val_data, test_data,
    )
    (DOCS / "model_evaluation.md").write_text(document, encoding="utf-8")

    for path in [
        REPORTS / "model_comparison.csv", REPORTS / "temporal_group_metrics.json",
        REPORTS / "group_robustness_metrics.json", REPORTS / "predictions_example.csv",
        DOCS / "model_evaluation.md",
    ]:
        print(relative(path))


if __name__ == "__main__":
    main()
