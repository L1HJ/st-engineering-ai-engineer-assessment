"""Model drift monitoring: measure feature and score distribution drift against the training period.

Governance boundaries:
- Reads only saved split and prediction files; never retrains or re-scores.
- Temporal test is used for distribution monitoring only, not for model selection or threshold tuning.
- Outputs monitoring metrics only; no recall or other effectiveness metrics.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

from config import MODEL_FEATURES, ROOT, SCHEMA

CONFIGS = os.path.join(ROOT, "configs")
SPLITS = os.path.join(ROOT, "data", "splits", "temporal")
MISSING_TOKENS = (-999, 9999)
SENSOR_FIELDS = {f["name"] for f in SCHEMA["fields"] if f.get("role") == "sensor"}
SIGNED_SENSORS = {"sensor_rpm_deviation"}
FEATURE_TYPES = {f["name"]: f["type"] for f in SCHEMA["fields"] if f.get("model_feature")}
NUMERIC_TYPES = {"number", "float", "integer"}
SNAPSHOT_COLUMNS = ["record_id", "aircraft_id", "component_id", "snapshot_date", "snapshot_cycle"]


def load_config(path=None):
    with open(path or os.path.join(CONFIGS, "drift_monitoring.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(split, columns=None):
    usecols = None
    if columns is not None:
        usecols = lambda c: c in columns  # noqa: E731
    return pd.read_csv(os.path.join(SPLITS, f"{split}.csv"), usecols=usecols, low_memory=False)


def clean_features(frame):
    """Apply the training-time cleaning convention: error markers and invalid negative sensor values become missing."""
    out = frame.copy()
    for name in MODEL_FEATURES:
        if FEATURE_TYPES.get(name) not in NUMERIC_TYPES or name not in out.columns:
            continue
        values = pd.to_numeric(out[name], errors="coerce")
        values = values.mask(values.isin(MISSING_TOKENS))
        if name in SENSOR_FIELDS and name not in SIGNED_SENSORS:
            values = values.mask(values < 0)
        out[name] = values
    return out


def _shares_numeric(reference, current, bins, epsilon):
    ref = pd.to_numeric(reference, errors="coerce")
    cur = pd.to_numeric(current, errors="coerce")
    ref_valid = ref.dropna()
    cur_valid = cur.dropna()
    if ref_valid.empty:
        return None
    edges = np.unique(np.nanquantile(ref_valid, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        edges = np.array([ref_valid.min() - 0.5, ref_valid.max() + 0.5])
    ref_counts = np.histogram(ref_valid, bins=edges)[0].astype(float)
    cur_clipped = np.clip(cur_valid, edges[0], edges[-1]) if not cur_valid.empty else cur_valid
    cur_counts = np.histogram(cur_clipped, bins=edges)[0].astype(float)
    ref_shares = np.append(ref_counts / max(ref_counts.sum(), 1), (ref.size - ref_valid.size) / max(ref.size, 1))
    cur_shares = np.append(cur_counts / max(cur_counts.sum(), 1), (cur.size - cur_valid.size) / max(cur.size, 1))
    ref_shares = np.clip(ref_shares, epsilon, None)
    cur_shares = np.clip(cur_shares, epsilon, None)
    return ref_shares, cur_shares


def _psi_from_shares(ref_shares, cur_shares):
    return float(np.sum((cur_shares - ref_shares) * np.log(cur_shares / ref_shares)))


def numeric_psi(reference, current, bins=10, epsilon=1e-4):
    shares = _shares_numeric(reference, current, bins, epsilon)
    if shares is None:
        return None
    return _psi_from_shares(*shares)


def categorical_psi(reference, current, epsilon=1e-4):
    ref = reference.astype("object").where(reference.notna(), "__missing__").astype(str)
    cur = current.astype("object").where(current.notna(), "__missing__").astype(str)
    categories = sorted(set(ref.unique()) | {"__missing__"})
    current_in_reference = cur.isin(categories)
    ref_shares = np.array([(ref == c).mean() for c in categories] + [0.0], dtype=float)
    cur_shares = np.array(
        [(cur == c).mean() for c in categories] + [float((~current_in_reference).mean())], dtype=float
    )
    ref_shares = np.clip(ref_shares, epsilon, None)
    cur_shares = np.clip(cur_shares, epsilon, None)
    return _psi_from_shares(ref_shares, cur_shares)


def ks_statistic(reference, current):
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return None
    return float(ks_2samp(ref, cur).statistic)


def drift_status(psi, thresholds):
    if psi is None:
        return "insufficient_data"
    if psi >= thresholds["action"]:
        return "action"
    if psi >= thresholds["watch"]:
        return "watch"
    return "stable"


def feature_drift(reference, current, thresholds, bins=10, epsilon=1e-4, structural=()):
    """Compute PSI, KS, and missing-rate change for each feature."""
    structural = set(structural)
    rows = []
    for name in MODEL_FEATURES:
        if name not in reference.columns or name not in current.columns:
            continue
        kind = FEATURE_TYPES.get(name)
        if kind in NUMERIC_TYPES:
            psi = numeric_psi(reference[name], current[name], bins=bins, epsilon=epsilon)
            ks = ks_statistic(reference[name], current[name])
        else:
            psi = categorical_psi(reference[name], current[name], epsilon=epsilon)
            ks = None
        rows.append({
            "feature": name,
            "type": kind or "unknown",
            "group": "structural_time" if name in structural else "operational",
            "psi": None if psi is None else round(float(psi), 6),
            "ks": None if ks is None else round(float(ks), 6),
            "status": drift_status(psi, thresholds),
            "reference_missing_rate": round(float(reference[name].isna().mean()), 6),
            "current_missing_rate": round(float(current[name].isna().mean()), 6),
        })
    return rows


def _month(frame):
    return pd.to_datetime(frame["snapshot_date"]).dt.to_period("M").astype(str)


def active_component_windows(frame, cooldown_cycles=30):
    windows = frame[["aircraft_id", "component_id", "snapshot_cycle"]].copy()
    windows["window_index"] = np.floor(pd.to_numeric(windows["snapshot_cycle"], errors="coerce") / cooldown_cycles)
    return int(windows.drop_duplicates(["aircraft_id", "window_index", "component_id"]).shape[0])


def monthly_alert_burden(predictions, max_alerts_per_100=5.0, cooldown_cycles=30):
    """Summarize score distribution and alert burden by calendar month; episodes are assigned to their first alert month."""
    frame = predictions.copy()
    frame["month"] = _month(frame)
    alerts = frame[frame["score_above_threshold"].astype(bool)].copy()
    if not alerts.empty:
        first_month = alerts.groupby("alert_episode_id")["snapshot_date"].min().map(
            lambda value: pd.to_datetime(value).to_period("M").strftime("%Y-%m")
        )
        episode_month = first_month.to_dict()
    else:
        episode_month = {}
    rows = []
    for month, group in frame.groupby("month"):
        month_alerts = group[group["score_above_threshold"].astype(bool)]
        episodes = {e for e in month_alerts["alert_episode_id"].dropna().unique() if episode_month.get(e) == month}
        active = active_component_windows(group, cooldown_cycles)
        rows.append({
            "month": month,
            "snapshots": int(len(group)),
            "alert_snapshots": int(len(month_alerts)),
            "alert_episodes": int(len(episodes)),
            "active_component_windows": active,
            "alerts_per_100_active": round(100 * len(episodes) / active, 6) if active else None,
            "score_mean": round(float(group["risk_score"].mean()), 6),
            "score_p95": round(float(group["risk_score"].quantile(0.95)), 6),
            "within_budget": bool(active and 100 * len(episodes) / active <= max_alerts_per_100),
        })
    return sorted(rows, key=lambda row: row["month"])


def score_summary(predictions):
    scores = pd.to_numeric(predictions["risk_score"], errors="coerce").dropna()
    return {
        "n_snapshots": int(len(predictions)),
        "score_mean": round(float(scores.mean()), 6),
        "score_median": round(float(scores.median()), 6),
        "score_p95": round(float(scores.quantile(0.95)), 6),
        "score_max": round(float(scores.max()), 6),
        "alert_snapshot_rate": round(float(predictions["score_above_threshold"].astype(bool).mean()), 6),
    }


def run(config_path=None, output_dir=None):
    cfg = load_config(config_path)
    thresholds = cfg["thresholds"]
    bins = int(cfg["psi"]["bins"])
    epsilon = float(cfg["psi"]["epsilon"])
    columns = set(SNAPSHOT_COLUMNS + MODEL_FEATURES)

    reference_cfg = cfg["reference"]
    reference_raw = load_split(reference_cfg["split"], columns)
    reference_raw["snapshot_date"] = pd.to_datetime(reference_raw["snapshot_date"])
    reference_raw = reference_raw[
        (reference_raw["snapshot_date"] >= pd.Timestamp(reference_cfg["date_start"]))
        & (reference_raw["snapshot_date"] <= pd.Timestamp(reference_cfg["date_end"]))
    ]
    reference = clean_features(reference_raw)

    window_results = []
    feature_rows = []
    structural = cfg.get("structural_features", [])
    for spec in cfg["windows"]:
        raw = load_split(spec["split"], columns)
        raw["snapshot_date"] = pd.to_datetime(raw["snapshot_date"])
        raw = raw[
            (raw["snapshot_date"] >= pd.Timestamp(spec["date_start"]))
            & (raw["snapshot_date"] <= pd.Timestamp(spec["date_end"]))
        ]
        current = clean_features(raw)
        rows = feature_drift(reference, current, thresholds, bins=bins, epsilon=epsilon, structural=structural)
        for row in rows:
            feature_rows.append({"window": spec["name"], **row})
        actionable = [r for r in rows if r["status"] == "action"]
        watching = [r for r in rows if r["status"] == "watch"]
        operational = [r for r in rows if r["group"] == "operational" and r["psi"] is not None]
        ranked = sorted([r for r in rows if r["psi"] is not None], key=lambda r: r["psi"], reverse=True)
        ranked_operational = sorted(operational, key=lambda r: r["psi"], reverse=True)
        window_results.append({
            "window": spec["name"],
            "role": spec["role"],
            "split": spec["split"],
            "date_start": spec["date_start"],
            "date_end": spec["date_end"],
            "rows": int(len(current)),
            "action_features": len(actionable),
            "watch_features": len(watching),
            "operational_action_features": len([r for r in actionable if r["group"] == "operational"]),
            "operational_watch_features": len([r for r in watching if r["group"] == "operational"]),
            "max_psi": ranked[0]["psi"] if ranked else None,
            "max_psi_feature": ranked[0]["feature"] if ranked else None,
            "operational_max_psi": ranked_operational[0]["psi"] if ranked_operational else None,
            "operational_max_psi_feature": ranked_operational[0]["feature"] if ranked_operational else None,
            "top_features": [{"feature": r["feature"], "psi": r["psi"], "status": r["status"], "group": r["group"]}
                             for r in ranked[:5]],
        })

    score_reference_cfg = next(w for w in cfg["windows"] if w["name"] == cfg["score_reference"])
    score_predictions = {}
    for name, relative in cfg["score_prediction_files"].items():
        frame = pd.read_csv(os.path.join(ROOT, relative))
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        score_predictions[name] = frame

    reference_scores = score_predictions[cfg["score_reference"]]
    score_comparisons = {}
    monthly = {}
    for name, frame in score_predictions.items():
        is_reference = name == cfg["score_reference"]
        if not is_reference:
            score_psi = numeric_psi(reference_scores["risk_score"], frame["risk_score"], bins=bins, epsilon=epsilon)
            score_comparisons[name] = {
                "reference": cfg["score_reference"],
                "psi": None if score_psi is None else round(float(score_psi), 6),
                "reference_summary": score_summary(reference_scores),
                "current_summary": score_summary(frame),
            }
        bucket = monthly_alert_burden(
            frame,
            max_alerts_per_100=cfg["alert_burden"]["max_alerts_per_100"],
            cooldown_cycles=cfg["alert_burden"]["cooldown_cycles"],
        )
        month_labels = _month(frame)
        for row in bucket:
            if is_reference:
                row["score_psi_vs_reference"] = 0.0
                continue
            subset = frame.loc[month_labels == row["month"], "risk_score"]
            row["score_psi_vs_reference"] = round(float(numeric_psi(
                reference_scores["risk_score"], subset, bins=bins, epsilon=epsilon
            )), 6)
        monthly[name] = bucket

    result = {
        "dataset_version": cfg["dataset_version"],
        "reference_window": reference_cfg,
        "psi_settings": {"bins": bins, "epsilon": epsilon, "watch": thresholds["watch"], "action": thresholds["action"]},
        "scope": {
            "purpose": "monitoring_only",
            "temporal_test_read": True,
            "temporal_test_rescored": False,
            "temporal_test_used_for_selection": False,
            "performance_metrics_computed": False,
        },
        "windows": window_results,
        "features": feature_rows,
        "score_drift": {
            "comparisons": score_comparisons,
            "monthly": monthly,
        },
    }

    output_dir = output_dir or os.path.join(ROOT, "reports")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "drift_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.DataFrame(feature_rows).to_csv(os.path.join(output_dir, "drift_feature_psi.csv"), index=False)
    pd.DataFrame([
        {
            "window": item["window"],
            "rows": item["rows"],
            "action_features": item["action_features"],
            "watch_features": item["watch_features"],
            "operational_action_features": item["operational_action_features"],
            "operational_watch_features": item["operational_watch_features"],
            "max_psi": item["max_psi"],
            "max_psi_feature": item["max_psi_feature"],
            "operational_max_psi": item["operational_max_psi"],
            "operational_max_psi_feature": item["operational_max_psi_feature"],
        }
        for item in window_results
    ]).to_csv(os.path.join(output_dir, "drift_window_summary.csv"), index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Model drift monitoring")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    for item in result["windows"]:
        print(f"{item['window']:<18} rows={item['rows']:>7} max_psi={item['max_psi']} ({item['max_psi_feature']}) "
              f"op_action={item['operational_action_features']} "
              f"op_max={item['operational_max_psi']} ({item['operational_max_psi_feature']})")
    for name, comparison in result["score_drift"]["comparisons"].items():
        print(f"score PSI {name} vs {comparison['reference']}: {comparison['psi']}")


if __name__ == "__main__":
    main()
