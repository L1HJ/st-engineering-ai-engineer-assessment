"""Post-test entity robustness evaluation for new aircraft and new components."""
import os

import pandas as pd

from evaluation import evaluate_predictions, search_threshold
from modeling import predict_scores
from predict import save_prediction_frame
from train import (
    MODEL_NAMES,
    config_hash,
    fit_candidate,
    load_config,
    path,
    read_json,
    sorted_snapshots,
    write_json,
)

ENTITY_ID_COLUMNS = {
    "by_aircraft": "aircraft_id",
    "by_component": "component_id",
}
INTERNAL_TRAIN_END = pd.Timestamp("2023-12-31")
INTERNAL_VAL_START = pd.Timestamp("2024-01-01")
INTERNAL_VAL_END = pd.Timestamp("2024-03-31")
MANIFEST_PATH = "models/test_evaluation_manifest.json"
OUTPUT_PATH = "reports/entity_robustness_metrics.json"


def split_internal_train_val(entity_train):
    """Split entity train into internal train and validation using the frozen dates."""
    dates = pd.to_datetime(entity_train["snapshot_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("entity train contains invalid snapshot_date values")
    internal_train = sorted_snapshots(entity_train.loc[dates <= INTERNAL_TRAIN_END].copy())
    internal_val = sorted_snapshots(entity_train.loc[
        dates.between(INTERNAL_VAL_START, INTERNAL_VAL_END, inclusive="both")
    ].copy())
    if internal_train.empty or internal_val.empty:
        raise ValueError("entity train internal train or internal val is empty")
    return internal_train, internal_val


def prepare_evaluation_context(
    snapshots,
    events,
    links,
    entity_column,
    split_name,
    start_date=None,
    end_date=None,
):
    """Rebuild events and links from the current entity set; never borrow records from outside the set."""
    entity_ids = set(snapshots[entity_column].dropna())
    record_ids = set(snapshots["record_id"])
    current_events = events.loc[
        events[entity_column].isin(entity_ids)
        & events["removal_type"].eq("unscheduled")
        & events["event_temporal_split"].notna()
        & events["event_temporal_split"].ne("warmup")
    ].copy()
    if start_date is not None or end_date is not None:
        event_dates = pd.to_datetime(current_events["event_date"], errors="coerce")
        date_mask = event_dates.notna()
        if start_date is not None:
            date_mask &= event_dates >= pd.Timestamp(start_date)
        if end_date is not None:
            date_mask &= event_dates <= pd.Timestamp(end_date)
        current_events = current_events.loc[date_mask].copy()
    current_events["event_temporal_split"] = split_name
    current_events = current_events.sort_values(
        "removal_event_id", kind="mergesort"
    ).reset_index(drop=True)

    cycles = pd.to_numeric(links["cycles_before_event"], errors="coerce")
    link_mask = (
        links["removal_event_id"].isin(set(current_events["removal_event_id"]))
        & links["record_id"].isin(record_ids)
        & cycles.between(0, 30, inclusive="both")
    )
    current_links = links.loc[link_mask].copy()
    current_links["cycles_before_event"] = cycles.loc[link_mask].astype(float)
    current_links["event_temporal_split"] = split_name
    current_links["snapshot_temporal_split"] = split_name
    current_links["same_temporal_split"] = True
    current_links = current_links.sort_values(
        ["removal_event_id", "record_id"], kind="mergesort"
    ).reset_index(drop=True)
    return current_events, current_links


def selected_candidate(config, metadata, model_name):
    """Reuse only candidates frozen on temporal val; do not run a new model search."""
    candidate_name = metadata["models"][model_name]["candidate"]
    matches = [
        dict(candidate)
        for candidate in config["models"][model_name]["candidates"]
        if candidate.get("name") == candidate_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Frozen candidate {candidate_name} for {model_name} is missing or not unique in the config")
    return matches[0]


def evaluate_entity_split(config, metadata, split_name, events, links, reports_dir):
    entity_column = ENTITY_ID_COLUMNS[split_name]
    entity_train = sorted_snapshots(pd.read_csv(path(config["data"][f"{split_name}_train"])))
    entity_test = sorted_snapshots(pd.read_csv(path(config["data"][f"{split_name}_test"])))
    train_ids = set(entity_train[entity_column].dropna())
    test_ids = set(entity_test[entity_column].dropna())
    if not train_ids or not test_ids:
        raise ValueError(f"{split_name} train/test entity sets must not be empty")
    if train_ids & test_ids:
        raise RuntimeError(f"{split_name} train/test overlap on {entity_column}")

    internal_train, internal_val = split_internal_train_val(entity_train)
    val_events, val_links = prepare_evaluation_context(
        internal_val, events, links, entity_column, "internal_val",
        INTERNAL_VAL_START, INTERNAL_VAL_END,
    )
    test_events, test_links = prepare_evaluation_context(
        entity_test, events, links, entity_column, "entity_test"
    )
    model_results = {}
    for model_name in MODEL_NAMES:
        candidate = selected_candidate(config, metadata, model_name)
        bundle = fit_candidate(
            config, model_name, internal_train, internal_val, candidate,
            features=metadata["features"],
        )
        val_scores = predict_scores(bundle, internal_val)
        _, best = search_threshold(
            internal_val, val_events, val_links, val_scores, "internal_val",
            max_alerts_per_100=config["threshold_search"]["val_max_alerts_per_100"],
            n_quantiles=config["threshold_search"]["n_quantiles"],
            refinement_points=config["threshold_search"]["refinement_points"],
            cooldown_cycles=config["prediction"]["cooldown_cycles"],
        )
        if best is None:
            raise RuntimeError(f"{split_name}/{model_name} has no feasible threshold on internal val")
        threshold = float(best["threshold"])
        val_result = evaluate_predictions(
            internal_val, val_events, val_links, val_scores, threshold,
            "internal_val", config["prediction"]["cooldown_cycles"],
        )

        test_scores = predict_scores(bundle, entity_test)
        test_result = evaluate_predictions(
            entity_test, test_events, test_links, test_scores, threshold,
            "entity_test", config["prediction"]["cooldown_cycles"],
        )
        for stage, result in (("internal_val", val_result), ("entity_test", test_result)):
            save_prediction_frame(
                result.predictions, model_name, bundle["candidate_name"],
                metadata["config_hash"],
                os.path.join(
                    reports_dir,
                    f"predictions_{split_name}_{stage}_{model_name}.csv",
                ),
            )
        model_results[model_name] = {
            "candidate": bundle["candidate_name"],
            "threshold": threshold,
            "internal_val": val_result.metrics,
            "entity_test": test_result.metrics,
        }

    return {
        "entity_column": entity_column,
        "train_entity_count": len(train_ids),
        "test_entity_count": len(test_ids),
        "internal_train_rows": len(internal_train),
        "internal_val_rows": len(internal_val),
        "entity_test_rows": len(entity_test),
        "models": model_results,
    }


def run():
    output_path = path(OUTPUT_PATH)
    if os.path.exists(output_path):
        raise RuntimeError("Entity robustness report already exists; refusing to overwrite")

    manifest_path = path(MANIFEST_PATH)
    if not os.path.exists(manifest_path):
        raise RuntimeError("Temporal test is not complete; entity robustness evaluation is blocked")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise RuntimeError("Temporal test manifest status is not completed; entity robustness evaluation is blocked")

    config = load_config()
    metadata = read_json(path("models/model_metadata.json"))
    if metadata.get("config_hash") != config_hash(config):
        raise RuntimeError("Config hash changed; refusing to reuse frozen candidates for entity robustness")
    events = pd.read_csv(path(config["data"]["removal_events"]))
    links = pd.read_csv(path(config["data"]["snapshot_event_links"]))
    reports_dir = path(config["outputs"]["reports_dir"])
    entity_splits = {
        split_name: evaluate_entity_split(
            config, metadata, split_name, events, links, reports_dir
        )
        for split_name in ENTITY_ID_COLUMNS
    }
    output = {
        "dataset_version": metadata["dataset_version"],
        "evaluation_batch": "entity_robustness",
        "temporal_test_status": "completed",
        "config_hash": metadata["config_hash"],
        "internal_train_end": str(INTERNAL_TRAIN_END.date()),
        "internal_val_start": str(INTERNAL_VAL_START.date()),
        "internal_val_end": str(INTERNAL_VAL_END.date()),
        "val_max_alerts_per_100": config["threshold_search"]["val_max_alerts_per_100"],
        "entity_splits": entity_splits,
    }
    write_json(output, output_path)
    print(f"Entity robustness evaluation complete; report saved to {OUTPUT_PATH}", flush=True)
    return output


if __name__ == "__main__":
    run()
