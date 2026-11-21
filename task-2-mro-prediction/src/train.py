"""MRO dual-model train/val flow, freezing, and one-shot temporal test."""
import argparse
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

import config as C
from evaluation import evaluate_predictions, search_threshold
from modeling import (
    FeatureCleaner,
    explain_bundle,
    fit_lightgbm,
    fit_logistic,
    load_bundle,
    predict_scores,
    save_bundle,
)
from predict import save_prediction_frame

MODEL_NAMES = ("logistic_regression", "lightgbm")
SIMPLE_FEATURES = [
    "component_type", "component_age_cycles", "component_age_days",
    "tsr_cycles", "operating_hours_since_install",
]


def path(relative):
    return os.path.join(C.ROOT, relative)


def load_config():
    with open(path("configs/modeling.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_hashes(config):
    names = [
        config["data"]["raw"], config["data"]["temporal_train"],
        config["data"]["temporal_val"], config["data"]["temporal_test"],
        config["data"]["removal_events"], config["data"]["snapshot_event_links"],
        config["data"]["schema"], "configs/modeling.yaml",
    ]
    return {name: sha256_file(path(name)) for name in names}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(value, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sorted_snapshots(frame):
    return frame.sort_values(
        ["snapshot_date", "aircraft_id", "component_id", "record_id"], kind="mergesort"
    ).reset_index(drop=True)


def load_train_val(config):
    train = sorted_snapshots(pd.read_csv(path(config["data"]["temporal_train"])))
    val = sorted_snapshots(pd.read_csv(path(config["data"]["temporal_val"])))
    events = pd.read_csv(path(config["data"]["removal_events"]))
    links = pd.read_csv(path(config["data"]["snapshot_event_links"]))
    return train, val, events, links


def common_params(config, model_name):
    params = {k: v for k, v in config["models"][model_name].items() if k != "candidates"}
    if model_name == "lightgbm":
        params.pop("early_stopping_rounds", None)
        params.pop("eval_metric", None)
    return params


def fit_candidate(config, model_name, train, val, candidate, features=None):
    params = common_params(config, model_name)
    candidate = dict(candidate)
    name = candidate.pop("name")
    params.update(candidate)
    if model_name == "logistic_regression":
        bundle = fit_logistic(train, params, features)
    else:
        bundle = fit_lightgbm(
            train, val, params,
            early_stopping_rounds=config["models"][model_name]["early_stopping_rounds"],
            eval_metric=config["models"][model_name]["eval_metric"],
            features=features,
        )
    bundle["candidate_name"] = name
    return bundle


def candidate_key(record):
    metrics = record["best"]
    return (
        -float(metrics["event_recall"]),
        float(metrics["alerts_per_100_active"]),
        -float(metrics["pr_auc"] or 0.0),
        record["bundle"]["candidate_name"],
    )


def select_on_val(config, model_name, train, val, events, links, features=None):
    candidates = []
    all_thresholds = []
    for candidate in config["models"][model_name]["candidates"]:
        bundle = fit_candidate(config, model_name, train, val, candidate, features)
        scores = predict_scores(bundle, val)
        results, best = search_threshold(
            val, events, links, scores, "val",
            max_alerts_per_100=config["threshold_search"]["val_max_alerts_per_100"],
            n_quantiles=config["threshold_search"]["n_quantiles"],
            refinement_points=config["threshold_search"]["refinement_points"],
            cooldown_cycles=config["prediction"]["cooldown_cycles"],
        )
        if best is None:
            raise RuntimeError(f"{bundle['candidate_name']} has no feasible threshold")
        results.insert(0, "candidate", bundle["candidate_name"])
        results.insert(0, "model", model_name)
        all_thresholds.append(results)
        record = {"bundle": bundle, "scores": scores, "best": best, "search": results}
        candidates.append(record)
        print(json.dumps({
            "stage": "val_candidate", "model": model_name,
            "candidate": bundle["candidate_name"],
            "event_recall": best["event_recall"],
            "alerts_per_100_active": best["alerts_per_100_active"],
            "threshold": best["threshold"],
        }, ensure_ascii=False), flush=True)
    selected = sorted(candidates, key=candidate_key)[0]
    comparison = [{
        "model": model_name,
        "candidate": r["bundle"]["candidate_name"],
        "selected": r is selected,
        **{k: v for k, v in r["best"].items() if k != "first_warning_cycles"},
    } for r in candidates]
    return selected, pd.concat(all_thresholds, ignore_index=True), comparison


def oracle_scores(val, links):
    same = links["same_temporal_split"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    cycles = pd.to_numeric(links["cycles_before_event"], errors="coerce")
    valid = links[
        same & cycles.between(0, 30, inclusive="both")
        & (links["event_temporal_split"] == "val")
        & (links["snapshot_temporal_split"] == "val")
    ]
    linked_records = set(valid["record_id"])
    return val["record_id"].map(lambda record_id: float(record_id in linked_records)).to_numpy()


def single_feature_check(train, val, max_auc):
    cleaner = FeatureCleaner(C.MODEL_FEATURES, C.CATEGORICAL, C.SENSOR_FIELDS)
    train_clean = cleaner.transform(train)
    val_clean = cleaner.transform(val)
    y_train = train["target"].to_numpy(dtype=int)
    y_val = val["target"].to_numpy(dtype=int)
    global_rate = float(y_train.mean())
    results = []
    for feature in C.MODEL_FEATURES:
        if feature in C.CATEGORICAL:
            mapping = pd.DataFrame({"value": train_clean[feature], "target": y_train}).groupby("value")["target"].mean()
            scores = val_clean[feature].map(mapping).fillna(global_rate).to_numpy()
            auc = roc_auc_score(y_val, scores) if np.unique(scores).size > 1 else 0.5
            auc = max(auc, 1 - auc)
        else:
            fill = train_clean[feature].median()
            scores = val_clean[feature].fillna(fill).to_numpy(dtype=float)
            auc = roc_auc_score(y_val, scores) if np.unique(scores).size > 1 else 0.5
            auc = max(auc, 1 - auc)
        results.append({"feature": feature, "val_roc_auc": float(auc)})
    results.sort(key=lambda row: row["val_roc_auc"], reverse=True)
    return {"features": results, "passed": results[0]["val_roc_auc"] < max_auc}


def run_sanity_checks(config, train, val, events, links):
    zero = evaluate_predictions(
        val, events, links, np.zeros(len(val)), 0.5, "val",
        config["prediction"]["cooldown_cycles"],
    ).metrics
    oracle = evaluate_predictions(
        val, events, links, oracle_scores(val, links), 0.5, "val",
        config["prediction"]["cooldown_cycles"],
    ).metrics
    if zero["event_recall"] != 0 or oracle["event_recall"] != 1:
        raise RuntimeError("Evaluator zero/oracle sanity check failed")

    single = single_feature_check(
        train, val, config["sanity_checks"]["single_feature_max_roc_auc"]
    )

    simple_candidate = {"name": "simple_age_baseline", "C": 1.0}
    simple = fit_candidate(config, "logistic_regression", train, val, simple_candidate, SIMPLE_FEATURES)
    simple_scores = predict_scores(simple, val)
    _, simple_best = search_threshold(
        val, events, links, simple_scores, "val",
        config["threshold_search"]["val_max_alerts_per_100"],
        n_quantiles=50, refinement_points=20,
        cooldown_cycles=config["prediction"]["cooldown_cycles"],
    )

    n_sample = min(config["sanity_checks"]["shuffled_train_rows"], len(train))
    shuffled = train.sample(n=n_sample, random_state=config["runtime"]["random_seed"]).sort_index().copy()
    rng = np.random.default_rng(config["runtime"]["random_seed"])
    shuffled["target"] = shuffled["target"].to_numpy()[rng.permutation(len(shuffled))]
    params = common_params(config, "lightgbm")
    params["n_estimators"] = config["sanity_checks"]["shuffled_n_estimators"]
    shuffled_bundle = fit_lightgbm(shuffled, val, params, early_stopping_rounds=20)
    shuffled_scores = predict_scores(shuffled_bundle, val)
    _, shuffled_best = search_threshold(
        val, events, links, shuffled_scores, "val",
        config["threshold_search"]["val_max_alerts_per_100"],
        n_quantiles=50, refinement_points=20,
        cooldown_cycles=config["prediction"]["cooldown_cycles"],
    )
    shuffled_auc = roc_auc_score(val["target"], shuffled_scores)
    shuffled_passed = (
        shuffled_auc <= config["sanity_checks"]["max_shuffled_roc_auc"]
        and shuffled_best["event_recall"] <= config["sanity_checks"]["max_shuffled_event_recall"]
    )
    return {
        "zero_scores": zero,
        "oracle_linked_alerts": oracle,
        "single_feature": single,
        "simple_baseline": simple_best,
        "shuffled_labels": {
            **shuffled_best,
            "roc_auc": float(shuffled_auc),
            "passed": bool(shuffled_passed),
        },
        "passed": bool(single["passed"] and shuffled_passed),
    }


def dependency_versions():
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__, "joblib": joblib.__version__,
    }


def freeze_val_models(config, selected, val_events, val_links, sanity, comparison, thresholds, hashes):
    models_dir = path(config["outputs"]["models_dir"])
    reports_dir = path(config["outputs"]["reports_dir"])
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    cfg_hash = config_hash(config)
    val_metrics = {}
    for model_name in MODEL_NAMES:
        bundle = selected[model_name]["bundle"]
        best = selected[model_name]["best"]
        save_bundle(bundle, os.path.join(models_dir, f"{model_name}.joblib"))
        result = evaluate_predictions(
            val_events["snapshots"], val_events["events"], val_links,
            selected[model_name]["scores"], best["threshold"], "val",
            config["prediction"]["cooldown_cycles"],
        )
        save_prediction_frame(
            result.predictions, model_name, bundle["candidate_name"], cfg_hash,
            os.path.join(reports_dir, f"predictions_val_{model_name}.csv"),
        )
        val_metrics[model_name] = {**best, "candidate": bundle["candidate_name"]}

    lightgbm_metrics = val_metrics["lightgbm"]
    gate = config["validation_gates"]
    recall_passed = lightgbm_metrics["event_recall"] >= gate["lightgbm_val_min_event_recall"]
    alerts_passed = lightgbm_metrics["alerts_per_100_active"] <= gate["lightgbm_val_max_alerts_per_100"]
    gate_result = {
        "recall_passed": bool(recall_passed), "alerts_passed": bool(alerts_passed),
        "passed": bool(recall_passed and alerts_passed), "test_permitted": bool(recall_passed and alerts_passed),
    }
    threshold_payload = {
        "config_hash": cfg_hash,
        "selection": "maximize event recall within the val alert budget; ties break toward fewer alerts and a higher threshold",
        "models": {name: {"threshold": val_metrics[name]["threshold"]} for name in MODEL_NAMES},
    }
    write_json(threshold_payload, os.path.join(models_dir, "frozen_thresholds.json"))

    preprocessors = {
        "lightgbm": selected["lightgbm"]["bundle"]["preprocessor"],
        "logistic_regression": selected["logistic_regression"]["bundle"]["model"][:-1],
    }
    joblib.dump(preprocessors, os.path.join(models_dir, "preprocessors.joblib"))
    with open(os.path.join(models_dir, "resolved_modeling.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    model_metadata = {
        "dataset_version": "v3", "config_hash": cfg_hash, "input_hashes": hashes,
        "features": C.MODEL_FEATURES,
        "excluded_fields": ["record_id", "aircraft_id", "component_id", "snapshot_date", "target"],
        "train_rows": int(val_events["train_rows"]),
        "train_positive_rows": int(val_events["train_positive_rows"]),
        "random_seed": config["runtime"]["random_seed"],
        "dependency_versions": dependency_versions(),
        "models": {},
        "sanity_checks_passed": sanity["passed"],
        "val_internal_gate": gate_result,
    }
    for model_name in MODEL_NAMES:
        bundle = selected[model_name]["bundle"]
        model = bundle["model"]
        details = {
            "candidate": bundle["candidate_name"], "threshold": val_metrics[model_name]["threshold"],
            "val_metrics": val_metrics[model_name],
            "model_sha256": sha256_file(os.path.join(models_dir, f"{model_name}.joblib")),
        }
        if model_name == "lightgbm":
            details["scale_pos_weight"] = float(model.scale_pos_weight)
            details["best_iteration"] = int(getattr(model, "best_iteration", 0) or model.n_estimators)
        else:
            details["n_iter"] = int(model.named_steps["classifier"].n_iter_[0])
        model_metadata["models"][model_name] = details
    write_json(model_metadata, os.path.join(models_dir, "model_metadata.json"))
    write_json({"models": val_metrics, "sanity_checks": sanity, "internal_gate": gate_result},
               os.path.join(reports_dir, "val_metrics.json"))
    write_json(comparison, os.path.join(reports_dir, "val_candidate_metrics.json"))
    thresholds.to_csv(os.path.join(reports_dir, "val_threshold_search.csv"), index=False)
    return gate_result


def run_val():
    config = load_config()
    hashes = input_hashes(config)
    train, val, events, links = load_train_val(config)
    print(json.dumps({"stage": "sanity", "train_rows": len(train), "val_rows": len(val)}), flush=True)
    sanity = run_sanity_checks(config, train, val, events, links)
    write_json(sanity, path("reports/sanity_checks.json"))
    if not sanity["passed"]:
        write_json({"models": {}, "sanity_checks": sanity, "internal_gate": {"passed": False, "test_permitted": False}},
                   path("reports/val_metrics.json"))
        print("Sanity check failed; stopping formal training.", flush=True)
        return False

    selected = {}
    all_searches = []
    comparisons = []
    for model_name in MODEL_NAMES:
        selected[model_name], searches, comparison = select_on_val(
            config, model_name, train, val, events, links
        )
        all_searches.append(searches)
        comparisons.extend(comparison)
    val_context = {
        "snapshots": val, "events": events, "train_rows": len(train),
        "train_positive_rows": int(train["target"].sum()),
    }
    gate = freeze_val_models(
        config, selected, val_context, links, sanity, comparisons,
        pd.concat(all_searches, ignore_index=True), hashes,
    )
    print(json.dumps({"stage": "val_complete", "internal_gate": gate}, ensure_ascii=False), flush=True)
    if not gate["passed"]:
        print("LightGBM internal val gate failed; temporal test will not be read.", flush=True)
    return gate["passed"]


def run_smoke():
    config = load_config()
    train, val, events, links = load_train_val(config)
    train_small = train.sample(n=min(5000, len(train)), random_state=42).sort_index()
    val_small = val.sample(n=min(2000, len(val)), random_state=42).sort_index()
    with tempfile.TemporaryDirectory(prefix="mro_model_smoke_") as tmp:
        for model_name in MODEL_NAMES:
            candidate = dict(config["models"][model_name]["candidates"][0])
            smoke_config = yaml.safe_load(yaml.safe_dump(config))
            if model_name == "lightgbm":
                smoke_config["models"][model_name]["early_stopping_rounds"] = 5
                candidate.update({"n_estimators": 20, "min_child_samples": 5})
            bundle = fit_candidate(smoke_config, model_name, train_small, val_small, candidate)
            scores = predict_scores(bundle, val_small)
            _, best = search_threshold(
                val_small, events, links, scores, "val", 4.5,
                n_quantiles=10, refinement_points=5,
            )
            model_path = os.path.join(tmp, f"{model_name}.joblib")
            save_bundle(bundle, model_path)
            loaded = load_bundle(model_path)
            np.testing.assert_allclose(predict_scores(loaded, val_small), scores)
            explanation = explain_bundle(loaded, val_small.iloc[:1])[0]
            assert np.isclose(explanation["raw_score"], explanation["reconstructed_raw_score"], atol=1e-8)
            assert best is not None
    print(json.dumps({"stage": "smoke_complete", "passed": True}), flush=True)


def run_test():
    config = load_config()
    models_dir = path(config["outputs"]["models_dir"])
    reports_dir = path(config["outputs"]["reports_dir"])
    manifest_path = path(config["outputs"]["test_manifest"])
    if os.path.exists(manifest_path):
        raise RuntimeError("A one-shot test manifest already exists; refusing to run temporal test again")
    val_metrics = read_json(os.path.join(reports_dir, "val_metrics.json"))
    if not val_metrics["internal_gate"].get("test_permitted", False):
        raise RuntimeError("LightGBM internal val gate failed; refusing to read temporal test")

    metadata = read_json(os.path.join(models_dir, "model_metadata.json"))
    if metadata["config_hash"] != config_hash(config):
        raise RuntimeError("Config hash changed; refusing to run test with frozen models")
    current_hashes = input_hashes(config)
    if current_hashes != metadata["input_hashes"]:
        raise RuntimeError("Input file hash changed; refusing to run test with frozen results")
    thresholds = read_json(os.path.join(models_dir, "frozen_thresholds.json"))
    bundles = {name: load_bundle(os.path.join(models_dir, f"{name}.joblib")) for name in MODEL_NAMES}
    model_hashes = {
        name: sha256_file(os.path.join(models_dir, f"{name}.joblib")) for name in MODEL_NAMES
    }
    for name in MODEL_NAMES:
        if model_hashes[name] != metadata["models"][name]["model_sha256"]:
            raise RuntimeError(f"{name} model file hash does not match the val freeze record")

    manifest = {
        "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": metadata["config_hash"], "input_hashes": current_hashes,
        "model_hashes": model_hashes,
        "thresholds": thresholds["models"],
        "evaluation_code_sha256": sha256_file(path("src/evaluation.py")),
        "modeling_code_sha256": sha256_file(path("src/modeling.py")),
        "val_internal_gate": val_metrics["internal_gate"],
        "val_metrics": val_metrics["models"],
        "cooldown_cycles": config["prediction"]["cooldown_cycles"],
        "active_window_cycles": config["prediction"]["active_window_cycles"],
    }
    write_json(manifest, manifest_path)

    test = sorted_snapshots(pd.read_csv(path(config["data"]["temporal_test"])))
    events = pd.read_csv(path(config["data"]["removal_events"]))
    links = pd.read_csv(path(config["data"]["snapshot_event_links"]))
    results = {}
    for name in MODEL_NAMES:
        scores = predict_scores(bundles[name], test)
        threshold = thresholds["models"][name]["threshold"]
        result = evaluate_predictions(
            test, events, links, scores, threshold, "test",
            config["prediction"]["cooldown_cycles"],
        )
        results[name] = result.metrics
        save_prediction_frame(
            result.predictions, name, bundles[name]["candidate_name"], metadata["config_hash"],
            os.path.join(reports_dir, f"predictions_test_{name}.csv"),
        )

    lgbm_metrics = results["lightgbm"]
    gate = config["validation_gates"]
    recall_passed = lgbm_metrics["event_recall"] >= gate["lightgbm_test_min_event_recall"]
    alerts_passed = lgbm_metrics["alerts_per_100_active"] <= gate["lightgbm_test_max_alerts_per_100"]
    output = {
        "dataset_version": "v3", "evaluation_batch": "single_frozen_temporal_test",
        "models": results,
        "formal_target": {
            "recall_passed": bool(recall_passed), "alerts_passed": bool(alerts_passed),
            "passed": bool(recall_passed and alerts_passed),
            "conclusion": "meets expectations" if recall_passed and alerts_passed else "does not meet expectations",
        },
    }
    write_json(output, os.path.join(reports_dir, "temporal_test_metrics.json"))
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["formal_target"] = output["formal_target"]
    write_json(manifest, manifest_path)
    print(json.dumps(output["formal_target"], ensure_ascii=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="MRO dual-model training and strict evaluation")
    parser.add_argument("--stage", choices=["smoke", "val", "test"], required=True)
    args = parser.parse_args()
    if args.stage == "smoke":
        run_smoke()
    elif args.stage == "val":
        run_val()
    else:
        run_test()


if __name__ == "__main__":
    main()
