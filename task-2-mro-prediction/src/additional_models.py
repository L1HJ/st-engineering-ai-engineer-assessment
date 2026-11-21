"""KNN and Random Forest supplementary experiments: train/val only, no temporal test reads."""
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from evaluation import evaluate_predictions, search_threshold
from modeling import FeatureCleaner, categorical_fields, numeric_fields
from train import config_hash, path, read_json, sorted_snapshots, write_json

CONFIG_PATH = "configs/additional_models.yaml"


def load_experiment_config():
    with open(path(CONFIG_PATH), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_dense_preprocessor():
    features = list(C.MODEL_FEATURES)
    categorical = categorical_fields(features)
    numeric = numeric_fields(features)
    cleaner = FeatureCleaner(features, categorical, C.SENSOR_FIELDS)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical),
    ], sparse_threshold=0.0)
    return Pipeline([("clean", cleaner), ("preprocess", preprocess)])


def balanced_training_indices(target, random_seed):
    """Build equal-sized positive and negative positions from train labels only."""
    target = np.asarray(target, dtype=int)
    positives = np.flatnonzero(target == 1)
    negatives = np.flatnonzero(target == 0)
    if len(positives) == 0 or len(negatives) < len(positives):
        raise ValueError("KNN balanced sampling requires enough train positives and negatives")
    rng = np.random.default_rng(random_seed)
    sampled_negatives = np.sort(rng.choice(negatives, size=len(positives), replace=False))
    return np.sort(np.concatenate([positives, sampled_negatives]))


def build_classifiers(config):
    knn_params = {k: v for k, v in config["knn"].items() if k != "sampling"}
    rf_params = dict(config["random_forest"])
    return {
        "knn": KNeighborsClassifier(**knn_params),
        "random_forest": RandomForestClassifier(**rf_params),
    }


def file_sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run():
    config = load_experiment_config()
    metrics_path = path(config["experiment"]["metrics_path"])
    if os.path.exists(metrics_path):
        raise RuntimeError("Supplementary results already exist; refusing to overwrite. Use a new config and output directory for new experiments")
    if config["experiment"]["temporal_test_used"] is not False:
        raise RuntimeError("Temporal test is not allowed in supplementary experiments")
    if any("test" in key for key in config["data"] if key not in {"removal_events", "snapshot_event_links"}):
        raise RuntimeError("Supplementary data config must not include a test split")

    manifest_path = path("models/test_evaluation_manifest.json")
    frozen_manifest_hash = file_sha256(manifest_path)
    frozen_code_hashes = {
        name: file_sha256(path(f"src/{name}.py")) for name in ("modeling", "evaluation")
    }

    train = sorted_snapshots(pd.read_csv(path(config["data"]["train"])))
    val = sorted_snapshots(pd.read_csv(path(config["data"]["val"])))
    events = pd.read_csv(path(config["data"]["removal_events"]))
    links = pd.read_csv(path(config["data"]["snapshot_event_links"]))
    y_train = train["target"].to_numpy(dtype=int)

    preprocessor = make_dense_preprocessor()
    X_train = preprocessor.fit_transform(train).astype(np.float32)
    X_val = preprocessor.transform(val).astype(np.float32)
    knn_indices = balanced_training_indices(y_train, config["experiment"]["random_seed"])
    classifiers = build_classifiers(config)

    output_dir = path(config["experiment"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}
    searches = []
    for model_name, classifier in classifiers.items():
        if model_name == "knn":
            classifier.fit(X_train[knn_indices], y_train[knn_indices])
            train_rows = int(len(knn_indices))
            positive_rows = int(y_train[knn_indices].sum())
        else:
            classifier.fit(X_train, y_train)
            train_rows = int(len(train))
            positive_rows = int(y_train.sum())
        scores = classifier.predict_proba(X_val)[:, 1]
        result, best = search_threshold(
            val, events, links, scores, "val",
            max_alerts_per_100=config["threshold_search"]["max_alerts_per_100"],
            n_quantiles=config["threshold_search"]["n_quantiles"],
            refinement_points=config["threshold_search"]["refinement_points"],
            cooldown_cycles=config["threshold_search"]["cooldown_cycles"],
        )
        if best is None:
            raise RuntimeError(f"{model_name} has no feasible val threshold")
        bundle = {
            "model_type": model_name,
            "features": C.MODEL_FEATURES,
            "preprocessor": preprocessor,
            "model": classifier,
            "threshold": float(best["threshold"]),
            "experiment_scope": "train_val_only",
        }
        joblib.dump(bundle, os.path.join(output_dir, f"{model_name}.joblib"))
        metrics[model_name] = {
            **best,
            "train_rows_used": train_rows,
            "train_positive_rows_used": positive_rows,
            "train_sampling": "all_positive_and_equal_negative" if model_name == "knn" else "full_train",
            "model_file": f"{output_dir}/{model_name}.joblib",
            "temporal_test_metrics": None,
        }
        result.insert(0, "model", model_name)
        searches.append(result)
        print(json.dumps({
            "model": model_name, "val_event_recall": best["event_recall"],
            "val_alerts_per_100": best["alerts_per_100_active"],
            "threshold": best["threshold"], "val_pr_auc": best["pr_auc"],
        }, ensure_ascii=False), flush=True)

    search_frame = pd.concat(searches, ignore_index=True)
    search_path = path(config["experiment"]["threshold_search_path"])
    os.makedirs(os.path.dirname(search_path), exist_ok=True)
    search_frame.to_csv(search_path, index=False)

    if file_sha256(manifest_path) != frozen_manifest_hash:
        raise RuntimeError("Frozen test manifest changed during supplementary experiments")
    for name, expected in frozen_code_hashes.items():
        if file_sha256(path(f"src/{name}.py")) != expected:
            raise RuntimeError(f"Frozen code src/{name}.py changed during supplementary experiments")

    output = {
        "experiment": config["experiment"],
        "config_hash": config_hash(config),
        "scope": {
            "train_rows": int(len(train)), "val_rows": int(len(val)),
            "temporal_test_read": False, "temporal_test_evaluated": False,
            "original_frozen_models_modified": False,
            "selection_rule": "maximize event recall under val alerts<=4.5; ties break toward fewer alerts and a higher threshold",
        },
        "models": metrics,
        "conclusion_scope": "KNN/RF are val-only exploratory results and do not change the frozen formal conclusions",
    }
    write_json(output, metrics_path)
    with open(os.path.join(output_dir, "resolved_additional_models.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return output


if __name__ == "__main__":
    run()
