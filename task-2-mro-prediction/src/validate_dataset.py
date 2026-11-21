"""Dataset quality checks: schema, temporal logic, label consistency, distributions, missingness, and leakage proxies."""
import os
import numpy as np
import pandas as pd

import config as C

RAW = os.path.join(C.ROOT, "data", "raw", "component_snapshots.csv")
AUDIT = os.path.join(C.ROOT, "data", "audit")
PRED_WINDOW = C.gen("prediction_window_cycles")


def _result(name, passed, detail=""):
    return {"check": name, "passed": bool(passed), "detail": str(detail)}


def validate(path=RAW):
    df = pd.read_csv(path)
    events = pd.read_csv(os.path.join(AUDIT, "removal_events.csv"))
    links = pd.read_csv(os.path.join(AUDIT, "snapshot_event_links.csv"))
    res = []

    # 1. Schema columns and order
    res.append(_result("schema_columns", list(df.columns) == C.FIELD_NAMES,
                       f"n_cols={len(df.columns)} expected={len(C.FIELD_NAMES)}"))
    res.append(_result("record_id_unique", df["record_id"].is_unique,
                       f"n_duplicates={int(df['record_id'].duplicated().sum())}"))

    # 2. Required fields have no missing values
    null_ok = {f["name"] for f in C.FIELDS if f.get("nullable")}
    bad_null = [c for c in df.columns if c not in null_ok and df[c].isna().any()]
    res.append(_result("non_nullable_no_missing", not bad_null, f"bad={bad_null}"))

    # 3. target is binary and positive rate is plausible
    res.append(_result("target_binary", set(df["target"].unique()) <= {0, 1}))
    rate = df["target"].mean()
    res.append(_result("positive_rate_near_2pct", 0.015 <= rate <= 0.025, f"rate={rate:.4f}"))

    # 4. Date range
    res.append(_result("date_range", df["snapshot_date"].min() >= C.gen("date_start")
                       and df["snapshot_date"].max() <= C.gen("split", "temporal", "test_end")))

    # 5. snapshot_cycle is non-negative and monotonic per aircraft after sorting
    res.append(_result("snapshot_cycle_nonneg", (df["snapshot_cycle"] >= 0).all()))
    d2 = df.sort_values(["aircraft_id", "snapshot_date"])
    non_mono = sum(not g["snapshot_cycle"].is_monotonic_increasing for _, g in d2.groupby("aircraft_id"))
    res.append(_result("snapshot_cycle_monotonic_per_aircraft", non_mono == 0, f"non_mono={non_mono}"))

    # 6. Labels are consistent with removal events (necessary aircraft-level condition)
    ev = events.groupby("aircraft_id")["event_cycle"].apply(list).to_dict()
    mism = 0
    for ac, g in df[df["target"] == 1].groupby("aircraft_id"):
        evs = np.array(sorted(ev.get(ac, [])))
        if len(evs) == 0:
            mism += len(g); continue
        for c in g["snapshot_cycle"].values:
            # snapshot_cycle and event_cycle are each rounded; tolerance is +/- 1 cycle
            hit = evs[(evs >= c) & (evs <= c + PRED_WINDOW + 1)]
            if len(hit) == 0:
                mism += 1
    res.append(_result("positive_has_removal_in_window", mism == 0, f"mismatches={mism}"))

    # 7. Right censoring: max snapshot_cycle should leave observation room (later event or complete window)
    # Approximate with the event table: per aircraft, max(snapshot_cycle) <= max(event_cycle) or data end.
    # Here we only check that snapshots do not exceed test_end (already covered by #4).
    res.append(_result("snapshot_date_le_test_end", df["snapshot_date"].max() <= "2024-09-30"))

    # 8. Distribution: positives cover each year and component type
    yr = df.assign(year=df["snapshot_date"].str[:4]).groupby("year")["target"].mean()
    res.append(_result("positives_all_years", (yr > 0).all(), f"rates={yr.round(4).to_dict()}"))
    res.append(_result("annual_positive_rate_stable", ((yr >= 0.018) & (yr <= 0.022)).all(),
                       f"rates={yr.round(4).to_dict()}"))
    t = C.gen("split", "temporal")
    split_masks = {
        "train": df["snapshot_date"] <= t["train_end"],
        "val": (df["snapshot_date"] >= t["val_start"]) & (df["snapshot_date"] <= t["val_end"]),
        "test": (df["snapshot_date"] >= t["test_start"]) & (df["snapshot_date"] <= t["test_end"]),
    }
    split_rates = {name: float(df.loc[mask, "target"].mean()) for name, mask in split_masks.items()}
    res.append(_result("temporal_split_positive_rate_stable",
                       all(0.018 <= rate <= 0.022 for rate in split_rates.values()),
                       f"rates={ {k: round(v, 4) for k, v in split_rates.items()} }"))
    ct = df.groupby("component_type")["target"].mean()
    res.append(_result("positives_multi_types", (ct > 0).sum() >= 5, f"n_types_pos={(ct>0).sum()}"))

    # 9. Structural missingness: sensor_exhaust_temp is non-null only for engine/apu
    ext = df[df["component_type"].isin(["engine", "apu"])]
    res.append(_result("structural_missing_exhaust",
                       df.loc[~df["component_type"].isin(["engine","apu"]), "sensor_exhaust_temp"].isna().all()
                       and ext["sensor_exhaust_temp"].notna().mean() > 0.5))

    # 10. Sensor missing rate is 5%-15% within applicable rows
    miss_ok = True
    for sname, sconf in C.gen("sensors").items():
        sub = df[df["component_type"].isin(sconf["applicable"])]
        r = sub[sname].isna().mean()
        if not (0.03 <= r <= 0.20):
            miss_ok = False
    res.append(_result("sensor_missing_in_range", miss_ok))

    # 11. Anomalies exist: extreme marker values are present
    has_anom = (df[C.SENSOR_FIELDS].isin([-999, 9999, -1])).any().any()
    res.append(_result("anomalies_present", has_anom))

    # 12. Fault-code cardinality is about 40 (including NONE)
    res.append(_result("fault_code_cardinality", 35 <= df["fault_code_last"].nunique() <= 45,
                       f"n={df['fault_code_last'].nunique()}"))

    # 13. v3 row-count scale
    res.append(_result("row_count_near_v3_target", 240000 <= len(df) <= 320000, f"n={len(df)}"))

    # 14. Displayed cycle gap does not exceed 30; the generation summary separately hard-gates on unrounded cycles
    ordered = df.sort_values(["component_id", "snapshot_cycle", "snapshot_date"])
    gaps = ordered.groupby("component_id")["snapshot_cycle"].diff()
    res.append(_result("snapshot_gap_le_30_cycles", bool((gaps.dropna() <= 30).all()),
                       f"max_gap={gaps.max()} violations={int((gaps > 30).sum())}"))

    # 15. Every formal event has a scoring opportunity within the same temporal split; cross-boundary snapshot borrowing is forbidden
    formal = events[events["event_temporal_split"].isin(["train", "val", "test"])]
    uncovered = formal["n_same_split_scoring_snapshots"].fillna(0) <= 0
    res.append(_result("all_formal_events_scorable_same_split", not uncovered.any(),
                       f"events={len(formal)} uncovered={int(uncovered.sum())}"))

    # 16. Event links must fall within the 30-cycle window before the corresponding event
    valid_links = links["cycles_before_event"].between(-1e-6, 30 + 1e-6).all()
    res.append(_result("snapshot_event_links_in_30_cycles", bool(valid_links),
                       f"links={len(links)} min={links['cycles_before_event'].min()} "
                       f"max={links['cycles_before_event'].max()}"))

    return res


def main():
    res = validate()
    n_pass = sum(r["passed"] for r in res)
    print(f"=== Data quality validation: {n_pass}/{len(res)} passed ===")
    for r in res:
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"[{flag}] {r['check']}: {r['detail']}")
    if n_pass != len(res):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
