"""Dataset splitting: temporal holdout (primary) plus aircraft and component groups for leakage checks."""
import os
import json
import numpy as np
import pandas as pd

import config as C

RAW = os.path.join(C.ROOT, "data", "raw", "component_snapshots.csv")
SPLITS = os.path.join(C.ROOT, "data", "splits")


def write_split(df, subdir, name):
    out = os.path.join(SPLITS, subdir)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)


def split_summary(df, name):
    return {
        "split": name,
        "n_rows": int(len(df)),
        "n_positive": int(df["target"].sum()),
        "positive_rate": float(df["target"].mean()),
        "date_min": df["snapshot_date"].min(),
        "date_max": df["snapshot_date"].max(),
        "n_aircraft": int(df["aircraft_id"].nunique()),
        "n_components": int(df["component_id"].nunique()),
    }


def main():
    df = pd.read_csv(RAW)
    summaries = []

    # ---- Temporal holdout (primary evaluation) ----
    t = C.gen("split", "temporal")
    train = df[df["snapshot_date"] <= t["train_end"]]
    val = df[(df["snapshot_date"] >= t["val_start"]) & (df["snapshot_date"] <= t["val_end"])]
    test = df[(df["snapshot_date"] >= t["test_start"]) & (df["snapshot_date"] <= t["test_end"])]
    write_split(train, "temporal", "train")
    write_split(val, "temporal", "val")
    write_split(test, "temporal", "test")
    summaries += [split_summary(train, "temporal/train"),
                  split_summary(val, "temporal/val"),
                  split_summary(test, "temporal/test")]

    # ---- Aircraft groups (20% of aircraft held out) ----
    rng = np.random.default_rng(C.gen("random_seed") + 50)
    acs = np.array(sorted(df["aircraft_id"].unique()))
    n_test = int(round(len(acs) * C.gen("split", "group_aircraft", "test_frac")))
    test_ac = set(rng.choice(acs, size=n_test, replace=False))
    tr_ac = df[~df["aircraft_id"].isin(test_ac)]
    te_ac = df[df["aircraft_id"].isin(test_ac)]
    # Verify that no aircraft overlaps.
    assert set(tr_ac["aircraft_id"]).isdisjoint(set(te_ac["aircraft_id"]))
    write_split(tr_ac, "by_aircraft", "train")
    write_split(te_ac, "by_aircraft", "test")
    summaries += [split_summary(tr_ac, "by_aircraft/train"),
                  split_summary(te_ac, "by_aircraft/test")]

    # ---- Component groups (20% of component instances held out) ----
    cps = np.array(sorted(df["component_id"].unique()))
    n_test_c = int(round(len(cps) * C.gen("split", "group_component", "test_frac")))
    test_cp = set(rng.choice(cps, size=n_test_c, replace=False))
    tr_cp = df[~df["component_id"].isin(test_cp)]
    te_cp = df[df["component_id"].isin(test_cp)]
    assert set(tr_cp["component_id"]).isdisjoint(set(te_cp["component_id"]))
    write_split(tr_cp, "by_component", "train")
    write_split(te_cp, "by_component", "test")
    summaries += [split_summary(tr_cp, "by_component/train"),
                  split_summary(te_cp, "by_component/test")]

    with open(os.path.join(SPLITS, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
