"""Standalone batch prediction and single-record explanation; never reads labels or event audit files."""
import argparse
import json
import os

import pandas as pd

import config as C
from evaluation import assign_alert_episodes
from modeling import explain_bundle, load_bundle, predict_scores

PREDICTION_COLUMNS = [
    "record_id", "aircraft_id", "component_id", "snapshot_date", "snapshot_cycle",
    "risk_score", "threshold", "score_above_threshold", "alert_episode_id",
    "model_name", "model_version", "config_hash",
]


def predict_dataframe(bundle, snapshots, threshold, cooldown_cycles=30):
    scores = predict_scores(bundle, snapshots)
    return assign_alert_episodes(snapshots, scores, threshold, cooldown_cycles)


def save_prediction_frame(predictions, model_name, model_version, config_hash, path):
    output = predictions.copy()
    output["model_name"] = model_name
    output["model_version"] = model_version
    output["config_hash"] = config_hash
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output.loc[:, PREDICTION_COLUMNS].to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Batch MRO component risk prediction")
    parser.add_argument("--model", required=True, help="Path to the model joblib file")
    parser.add_argument("--input", required=True, help="Snapshot CSV; target is not required")
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output", required=True, help="Output CSV path for predictions")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--record-id", help="Optional: explain this record")
    parser.add_argument("--explain-output", help="Output JSON path for the explanation")
    args = parser.parse_args()

    snapshots = pd.read_csv(args.input)
    bundle = load_bundle(args.model)
    predictions = predict_dataframe(bundle, snapshots, args.threshold)
    save_prediction_frame(
        predictions, args.model_name, args.model_version, args.config_hash, args.output
    )
    if args.record_id:
        selected = snapshots[snapshots["record_id"] == args.record_id]
        if len(selected) != 1:
            raise ValueError("--record-id must match exactly one record")
        explanation = explain_bundle(bundle, selected)[0]
        text = json.dumps(explanation, ensure_ascii=False, indent=2)
        if args.explain_output:
            os.makedirs(os.path.dirname(args.explain_output), exist_ok=True)
            with open(args.explain_output, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        else:
            print(text)


if __name__ == "__main__":
    main()
