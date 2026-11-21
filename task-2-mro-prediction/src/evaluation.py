"""Strict business evaluation: active component windows, alert episodes, and independent event recall."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

SORT_COLUMNS = ["component_id", "snapshot_cycle", "snapshot_date", "record_id"]
SNAPSHOT_COLUMNS = ["record_id", "aircraft_id", "component_id", "snapshot_cycle", "snapshot_date", "target"]


@dataclass
class EvaluationResult:
    metrics: dict
    predictions: pd.DataFrame
    events: pd.DataFrame


def active_component_window_count(snapshots: pd.DataFrame) -> int:
    """Count each unique active component once per aircraft per 30-cycle window."""
    if snapshots.empty:
        raise ValueError("Active component window denominator is 0; cannot evaluate")
    windows = snapshots[["aircraft_id", "component_id", "snapshot_cycle"]].copy()
    if windows["snapshot_cycle"].isna().any():
        raise ValueError("snapshot_cycle contains missing values; cannot compute active windows")
    windows["window_index"] = np.floor(windows["snapshot_cycle"].astype(float) / 30)
    return int(windows.drop_duplicates(["aircraft_id", "window_index", "component_id"]).shape[0])


def assign_alert_episodes(
    snapshots: pd.DataFrame,
    scores,
    threshold: float,
    cooldown_cycles: float = 30,
) -> pd.DataFrame:
    """Assign episodes to high-score snapshots; each call starts from a blank cooldown state."""
    required = [c for c in SNAPSHOT_COLUMNS if c != "target"]
    missing = set(required) - set(snapshots.columns)
    if missing:
        raise ValueError(f"Snapshots missing fields: {sorted(missing)}")
    if snapshots["record_id"].duplicated().any():
        raise ValueError("record_id must be unique")

    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(snapshots) or not np.isfinite(scores).all():
        raise ValueError("scores must have the same length as snapshots and contain only finite values")

    work = snapshots.copy()
    work["risk_score"] = scores
    work = work.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)
    work["threshold"] = float(threshold)
    above_threshold = (work["risk_score"] >= threshold).to_numpy()
    work["score_above_threshold"] = above_threshold
    episode_ids = np.full(len(work), None, dtype=object)
    cycles = work["snapshot_cycle"].to_numpy(dtype=float)

    for component_id, indices in work.groupby("component_id", sort=False).groups.items():
        previous_high_cycle = None
        episode_number = 0
        for idx in indices:
            if not above_threshold[idx]:
                continue
            cycle = cycles[idx]
            if previous_high_cycle is None or cycle - previous_high_cycle > cooldown_cycles:
                episode_number += 1
            episode_ids[idx] = f"{component_id}_E{episode_number:04d}"
            previous_high_cycle = cycle
    work["alert_episode_id"] = pd.array(episode_ids, dtype="string")
    return work


def _filter_events(events: pd.DataFrame, split_name: str) -> pd.DataFrame:
    selected = events.copy()
    if "event_temporal_split" in selected.columns:
        selected = selected[selected["event_temporal_split"] == split_name]
    if "removal_type" in selected.columns:
        selected = selected[selected["removal_type"] == "unscheduled"]
    if selected["removal_event_id"].duplicated().any():
        raise ValueError("removal_event_id must be unique within the current split")
    return selected


def _filter_links(links: pd.DataFrame, split_name: str) -> pd.DataFrame:
    selected = links.copy()
    same_split = selected["same_temporal_split"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    cycles = pd.to_numeric(selected["cycles_before_event"], errors="coerce")
    selected = selected[same_split & cycles.between(0, 30, inclusive="both")].copy()
    for column in ("event_temporal_split", "snapshot_temporal_split"):
        if column in selected.columns:
            selected = selected[selected[column] == split_name]
    return selected


def evaluate_predictions(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    links: pd.DataFrame,
    scores,
    threshold: float,
    split_name: str,
    cooldown_cycles: float = 30,
) -> EvaluationResult:
    """Evaluate within a single split; never reuse alerts or event links across splits."""
    missing = set(SNAPSHOT_COLUMNS) - set(snapshots.columns)
    if missing:
        raise ValueError(f"Snapshot missing fields: {sorted(missing)}")

    predictions = assign_alert_episodes(snapshots, scores, threshold, cooldown_cycles)
    active_count = active_component_window_count(snapshots)
    current_events = _filter_events(events, split_name)
    current_links = _filter_links(links, split_name)

    event_ids = set(current_events["removal_event_id"])
    record_ids = set(predictions["record_id"])
    current_links = current_links[
        current_links["removal_event_id"].isin(event_ids) & current_links["record_id"].isin(record_ids)
    ]
    alert_links = current_links.merge(
        predictions.loc[predictions["alert_episode_id"].notna(), ["record_id", "alert_episode_id"]],
        on="record_id",
        how="inner",
    )
    detected_ids = set(alert_links["removal_event_id"])
    event_details = current_events.copy()
    event_details["detected"] = event_details["removal_event_id"].isin(detected_ids)
    first_warning = alert_links.groupby("removal_event_id")["cycles_before_event"].max()
    event_details["first_warning_cycles"] = event_details["removal_event_id"].map(first_warning)

    y_true = predictions["target"].to_numpy(dtype=int)
    y_score = predictions["risk_score"].to_numpy(dtype=float)
    y_alert = predictions["score_above_threshold"].to_numpy(dtype=bool)
    episode_count = int(predictions["alert_episode_id"].nunique(dropna=True))
    actual_count = int(len(current_events))
    detected_count = len(detected_ids)
    lead_values = event_details["first_warning_cycles"].dropna().to_numpy(dtype=float)

    metrics = {
        "split": split_name,
        "threshold": float(threshold),
        "n_snapshots": int(len(predictions)),
        "n_positive_snapshots": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else None,
        "snapshot_recall": float(recall_score(y_true, y_alert, zero_division=0)),
        "snapshot_precision": float(precision_score(y_true, y_alert, zero_division=0)),
        "snapshot_f1": float(f1_score(y_true, y_alert, zero_division=0)),
        "snapshot_alert_rate": float(y_alert.mean()) if len(y_alert) else None,
        "alert_episode_count": episode_count,
        "active_component_window_count": active_count,
        "alerts_per_100_active": float(100 * episode_count / active_count),
        "detected_event_count": int(detected_count),
        "actual_event_count": actual_count,
        "event_recall": float(detected_count / actual_count) if actual_count else None,
        "first_warning_cycles": {
            "count": int(len(lead_values)),
            "min": float(np.min(lead_values)) if len(lead_values) else None,
            "p25": float(np.percentile(lead_values, 25)) if len(lead_values) else None,
            "median": float(np.median(lead_values)) if len(lead_values) else None,
            "p75": float(np.percentile(lead_values, 75)) if len(lead_values) else None,
            "max": float(np.max(lead_values)) if len(lead_values) else None,
        },
    }
    return EvaluationResult(metrics, predictions, event_details)


def threshold_candidates(scores, n_quantiles: int = 100) -> np.ndarray:
    """Generate candidates deterministically from current val scores, including 0, 1, extremes, and the zero-alert boundary."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Threshold search requires non-empty finite scores")
    quantiles = np.quantile(scores, np.linspace(0, 1, n_quantiles + 1))
    boundaries = [0.0, 1.0, scores.min(), scores.max(), np.nextafter(scores.max(), np.inf)]
    return np.unique(np.concatenate([quantiles, boundaries]))


def select_threshold(results: pd.DataFrame, max_alerts_per_100: float) -> dict | None:
    """Maximize event recall within the alert budget; ties break toward fewer alerts and then a higher threshold."""
    feasible = results[results["alerts_per_100_active"] <= max_alerts_per_100]
    feasible = feasible[feasible["event_recall"].notna()]
    if feasible.empty:
        return None
    best = feasible.sort_values(
        ["event_recall", "alerts_per_100_active", "threshold"],
        ascending=[False, True, False],
        kind="mergesort",
    ).iloc[0]
    return best.to_dict()


def search_threshold(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    links: pd.DataFrame,
    scores,
    split_name: str,
    max_alerts_per_100: float,
    n_quantiles: int = 100,
    refinement_points: int = 40,
    cooldown_cycles: float = 30,
) -> tuple[pd.DataFrame, dict | None]:
    """Search thresholds only on the given val set, then refine near the best coarse point."""
    scores = np.asarray(scores, dtype=float)
    coarse = threshold_candidates(scores, n_quantiles)
    records = []
    for threshold in coarse:
        result = evaluate_predictions(
            snapshots, events, links, scores, float(threshold), split_name, cooldown_cycles
        )
        records.append(result.metrics)
    coarse_results = pd.DataFrame(records)
    best = select_threshold(coarse_results, max_alerts_per_100)

    refined = np.array([], dtype=float)
    if best is not None:
        threshold = float(best["threshold"])
        lower = coarse[coarse < threshold]
        upper = coarse[coarse > threshold]
        low = lower[-1] if len(lower) else threshold
        high = upper[0] if len(upper) else threshold
        if low < high:
            refined = np.linspace(low, high, refinement_points + 2)[1:-1]

    for threshold in np.setdiff1d(refined, coarse):
        result = evaluate_predictions(
            snapshots, events, links, scores, float(threshold), split_name, cooldown_cycles
        )
        records.append(result.metrics)

    results = pd.DataFrame(records).sort_values("threshold", kind="mergesort").reset_index(drop=True)
    return results, select_threshold(results, max_alerts_per_100)
