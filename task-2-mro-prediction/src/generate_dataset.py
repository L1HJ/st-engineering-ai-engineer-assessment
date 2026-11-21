"""Mock dataset generation pipeline (warm-up + forward simulation + replacements + period-wise alpha calibration).

Stages: entities -> operating timeline -> per-slot forward simulation (degradation/fault/maintenance/removal/replacement)
        -> snapshot sampling -> labels (with right censoring) -> quality injection -> write-out.
The positive rate is calibrated to about 2% by adjusting the period-wise hazard intercept alpha.
"""
import argparse
import os
import json
import math
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import config as C

ROOT = C.ROOT
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
AUDIT = os.path.join(DATA, "audit")
for d in (RAW, AUDIT):
    os.makedirs(d, exist_ok=True)

SNAPSHOT_END = datetime.strptime(C.gen("split", "temporal", "test_end"), "%Y-%m-%d")
OBS_END = SNAPSHOT_END + timedelta(days=C.gen("split", "temporal", "purge_obs_extra_days"))
DATE_START = datetime.strptime(C.gen("date_start"), "%Y-%m-%d")
SIM_START = DATE_START - timedelta(days=C.gen("warmup_days"))
PRED_WINDOW = C.gen("prediction_window_cycles")
MAX_SNAPSHOT_GAP = C.gen("snapshot", "max_gap_cycles")
RE_STD = C.gen("hazard", "component_re_std")

SENSOR_EFFECT = {
    "sensor_vibration": (1.0, 3.0),
    "sensor_temp": (1.0, 18.0),
    "sensor_pressure": (-1.0, 6.0),
    "sensor_oil_debris": (1.0, 5.0),
    "sensor_rpm_deviation": (1.0, 5.0),
    "sensor_exhaust_temp": (1.0, 50.0),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ----------------------------- Entities -----------------------------
def build_aircraft_registry(n_aircraft=None):
    rng = np.random.default_rng(C.gen("random_seed") + 1)
    n = C.gen("n_aircraft") if n_aircraft is None else n_aircraft
    models, mw = C.weighted_choices(C.gen("aircraft_models"))
    ops, ow = C.weighted_choices(C.gen("operators"))
    model_idx = rng.choice(len(models), size=n, p=mw)
    op_idx = rng.choice(len(ops), size=n, p=ow)
    util_mult = np.array([o["util_multiplier"] for o in C.gen("operators")])
    util = np.array([m["utilization"] for m in C.gen("aircraft_models")])
    avg_hours = np.array([m["avg_flight_hours"] for m in C.gen("aircraft_models")])
    return pd.DataFrame({
        "aircraft_id": [f"AC{i:04d}" for i in range(n)],
        "aircraft_model": [models[i] for i in model_idx],
        "operator": [ops[i] for i in op_idx],
        "daily_cycle_rate": util[model_idx] * util_mult[op_idx] * rng.uniform(0.85, 1.15, n),
        "avg_flight_hours": avg_hours[model_idx],
    })


def build_slot_components(aircraft):
    rng = np.random.default_rng(C.gen("random_seed") + 2)
    rng2 = np.random.default_rng(C.gen("random_seed") + 3)
    ctypes, cw = C.weighted_choices(C.gen("component_types"))
    ctype_rate = {ct["name"]: ct["degradation_rate"] for ct in C.gen("component_types")}
    n_slots_per_ac = 10
    slot_rows, comp_rows = [], []
    comp_counter = 0
    for _, ac in aircraft.iterrows():
        ct_idx = rng.choice(len(ctypes), size=n_slots_per_ac, p=cw)
        for s in range(n_slots_per_ac):
            ct = ctypes[ct_idx[s]]
            slot_rows.append({
                "slot_id": f"{ac['aircraft_id']}|{ct}|{s:02d}",
                "aircraft_id": ac["aircraft_id"],
                "component_type": ct,
                "degradation_rate": ctype_rate[ct],
            })
            comp_id = f"CP{comp_counter:05d}"
            comp_counter += 1
            comp_rows.append({
                "component_id": comp_id,
                "slot_id": f"{ac['aircraft_id']}|{ct}|{s:02d}",
                "aircraft_id": ac["aircraft_id"],
                "component_type": ct,
                "initial_age_cycles": int(rng2.integers(0, 1500)),
                "individual_re": float(rng2.normal(0, RE_STD)),
            })
    return pd.DataFrame(slot_rows), pd.DataFrame(comp_rows)


# ----------------------------- Operating timeline -----------------------------
def build_aircraft_timelines(aircraft):
    rng = np.random.default_rng(C.gen("random_seed") + 4)
    days = pd.date_range(SIM_START, OBS_END, freq="D")
    n_days = len(days)
    out = {}
    for _, ac in aircraft.iterrows():
        rate = ac["daily_cycle_rate"]
        daily = rng.poisson(rate, size=n_days).astype(float)
        daily[rng.random(n_days) < 0.05] = 0.0
        daily *= rng.uniform(0.7, 1.3, size=n_days)
        cum = np.concatenate([[0.0], np.cumsum(daily)])
        out[ac["aircraft_id"]] = (days, cum, daily)
    return out


def sampling_anchor_dates():
    """Start every formal evaluation period at a scorable anchor so alerts are not borrowed across boundaries."""
    t = C.gen("split", "temporal")
    return [DATE_START,
            datetime.strptime(t["val_start"], "%Y-%m-%d"),
            datetime.strptime(t["test_start"], "%Y-%m-%d")]


def finalize_snapshot_positions(positions, global_cycle, boundary_pos=None):
    """Fill actual cycle gaps; boundary_pos is a removal day, ensuring snapshots exist in its prior 30 cycles."""
    upper = len(global_cycle) - 1 if boundary_pos is None else boundary_pos
    pos = np.asarray(positions, dtype=int)
    pos = pos[(pos >= 0) & (pos <= upper)]
    if boundary_pos is not None:
        pos = np.append(pos, boundary_pos)
    pos = np.unique(pos)
    if len(pos) <= 1:
        return pos
    filled = [int(pos[0])]
    anchor = int(pos[0])
    for original in pos[1:]:
        original = int(original)
        cursor = original
        chain = []
        while global_cycle[cursor] - global_cycle[anchor] > MAX_SNAPSHOT_GAP + 1e-9:
            target = global_cycle[cursor] - MAX_SNAPSHOT_GAP
            p = int(np.searchsorted(global_cycle, target, side="left"))
            if p <= anchor or p >= cursor:
                raise RuntimeError(
                    f"Cannot fill snapshots between actual cycles {global_cycle[anchor]:.3f} and "
                    f"{global_cycle[cursor]:.3f} to cover {MAX_SNAPSHOT_GAP} cycles"
                )
            chain.append(p)
            cursor = p
        filled.extend(reversed(chain))
        filled.append(original)
        anchor = original
    return np.asarray(filled, dtype=int)


# ----------------------------- Single-instance simulation (from a given start date) -----------------------------
def simulate_instance_from(seed, slot_row, start_idx, initial_age, individual_re,
                           days, cum, daily, snapshot_interval, avg_hours, component_id):
    n = len(days)
    if start_idx >= n:
        return None
    s = start_idx
    L = n - s
    global_cycle = cum[s + 1:s + 1 + L] - cum[s]
    age_cycle = initial_age + global_cycle
    age_days = (days[s:s + L] - SIM_START).days.values.astype(float) + initial_age / 3.0
    rng = np.random.default_rng(seed)
    rate = slot_row["degradation_rate"]
    mult = math.exp(individual_re)
    drift = rate * mult * age_cycle
    noise_increments = (C.gen("degradation", "noise_std") * np.sqrt(daily[s:s + L])
                        * rng.standard_normal(L))
    noise = np.cumsum(noise_increments)
    raw_D = np.clip(drift + noise, 0.0, None)
    # Maintenance events
    age_max = age_cycle[-1] if L > 0 else 0
    sched = np.arange(200, int(age_max) + 1, 200)
    overhaul = np.arange(1000, int(age_max) + 1, 1000)
    maint_events = [(m, C.gen("maintenance", "scheduled_recovery")) for m in sched] + \
                   [(m, C.gen("maintenance", "overhaul_recovery")) for m in overhaul]
    maint_events.sort(key=lambda x: x[0])
    D = raw_D.copy()
    maint_at_cycle = []
    for (m, rec) in maint_events:
        p = int(np.searchsorted(age_cycle, m, side="left"))
        if p < L:
            D[p:] = D[p:] * (1.0 - rec)
            maint_at_cycle.append(m)
    maint_at_cycle = np.array(maint_at_cycle) if maint_at_cycle else np.array([-1e9])
    # Faults
    base = C.gen("fault", "base_fault_rate_per_cycle")
    fault_prob = np.clip(base * (1.0 + 3.0 * np.clip(D, 0, 2)) * daily[s:s + L], 0, 0.9)
    fault_today = rng.random(L) < fault_prob
    n_codes = C.gen("fault", "n_codes")
    head = C.gen("fault", "common_code_head")
    code_w = np.concatenate([np.linspace(1.0, 0.4, head), 0.4 * np.power(0.92, np.arange(n_codes - head))])
    code_w /= code_w.sum()
    sev_w = np.array(C.gen("fault", "severity_weights"))
    fault_idx = np.where(fault_today)[0]
    fault_codes = np.full(L, "NONE", dtype=object)
    fault_sevs = np.full(L, "", dtype=object)
    if len(fault_idx) > 0:
        codes = rng.choice([f"FC{i:03d}" for i in range(n_codes)], size=len(fault_idx), p=code_w)
        d_clip = np.clip(D[fault_idx], 0, 2.0)
        p_high = np.clip(sev_w[2] * (1.0 + 2.0 * d_clip), 0, 0.95)
        u1 = rng.random(len(fault_idx))
        is_high = u1 < p_high
        p_med = sev_w[1] / (sev_w[0] + sev_w[1])
        is_med = (~is_high) & (rng.random(len(fault_idx)) < p_med)
        fault_codes[fault_idx] = codes
        fault_sevs[fault_idx] = np.where(is_high, "high", np.where(is_med, "medium", "low"))
    # Hazard excludes alpha
    f_power = C.gen("hazard", "f_degradation_power")
    fault_trailing = np.convolve(fault_today.astype(float), np.ones(8, dtype=float), mode="full")[:L]
    usage_norm = daily[s:s + L] / 4.0
    logit_h_base = (np.power(np.clip(D, 0, C.gen("hazard", "degradation_clip")), f_power)
                    + C.gen("hazard", "usage_coef") * usage_norm
                    + C.gen("hazard", "fault_coef") * np.minimum(fault_trailing, 5.0))
    # Snapshot sampling: install start point + each evaluation-period start + base-interval candidates.
    # Date filtering happens when building the wide table; keep the full sequence here to repair actual cycle gaps.
    snap_pos = [0] if L > 0 else []
    if L > 0:
        gc = global_cycle
        for anchor_date in sampling_anchor_dates():
            anchor_pos = int(days.searchsorted(anchor_date, side="left")) - s
            if 0 <= anchor_pos < L:
                snap_pos.append(anchor_pos)
        next_c = snapshot_interval * rng.uniform(0.85, 1.05)
        while True:
            p = int(np.searchsorted(gc, next_c, side="left"))
            if p >= L:
                break
            snap_pos.append(p)
            next_c += snapshot_interval * rng.uniform(0.85, 1.05)
    snap_pos = np.unique(np.asarray(snap_pos, dtype=int))
    return {
        "component_id": component_id,
        "seed": seed,
        "aircraft_id": slot_row["aircraft_id"],
        "component_type": slot_row["component_type"],
        "slot_id": slot_row["slot_id"],
        "avg_flight_hours": avg_hours,
        "L": L, "s": s, "days_full": days, "cum": cum,
        "global_cycle": global_cycle, "age_cycle": age_cycle, "age_days": age_days,
        "D": D, "logit_h_base": logit_h_base, "daily_slice": daily[s:s + L],
        "fault_today": fault_today, "fault_codes": fault_codes, "fault_sevs": fault_sevs,
        "maint_at_cycle": maint_at_cycle, "snap_pos": snap_pos,
    }


# ----------------------------- Per-slot forward simulation (with replacements) -----------------------------
def simulate_slot_forward(slot_ordinal, slot_row, initial_comp, timelines, alpha, snapshot_interval, avg_hours_map):
    days, cum, daily = timelines[slot_row["aircraft_id"]]
    n = len(days)
    instances = []
    removals = []  # (abs_cycle, component_id, day_idx)
    seq = 0
    cur_comp_id = initial_comp["component_id"]
    start_idx = 0
    initial_age = initial_comp["initial_age_cycles"]
    individual_re = initial_comp["individual_re"]
    while start_idx < n:
        seed = 1000000 + slot_ordinal * 1000 + seq
        inst = simulate_instance_from(seed, slot_row, start_idx, initial_age, individual_re,
                                      days, cum, daily, snapshot_interval,
                                      avg_hours_map[slot_row["aircraft_id"]], cur_comp_id)
        if inst is None or inst["L"] == 0:
            break
        L = inst["L"]
        # Removal sampling (independent random stream from degradation)
        rrng = np.random.default_rng(2000000 + slot_ordinal * 1000 + seq)
        alpha_slice = alpha if np.isscalar(alpha) else alpha[start_idx:start_idx + L]
        logit = alpha_slice + inst["logit_h_base"]
        h_cycle = sigmoid(logit)
        p_day = 1.0 - np.power(np.clip(1.0 - h_cycle, 1e-9, 1.0), inst["daily_slice"])
        draws = rrng.random(L) < p_day
        ev = np.where(draws)[0]
        if len(ev) > 0:
            rpos = int(ev[0])
            abs_c = cum[start_idx] + inst["global_cycle"][rpos]
            inst["retired_pos"] = rpos
            inst["has_removal"] = True
            inst["snap_pos"] = finalize_snapshot_positions(
                inst["snap_pos"], inst["global_cycle"], boundary_pos=rpos)
            instances.append(inst)
            removals.append((abs_c, cur_comp_id, start_idx + rpos))
            # Replacement: starts next day with age 0 and a new individual effect
            seq += 1
            if seq > 100:
                raise RuntimeError(f"Slot {slot_row['slot_id']} exceeded 100 replacements; generation parameters are invalid")
            cur_comp_id = f"R{slot_ordinal:04d}_{seq:02d}"
            start_idx = start_idx + rpos + 1
            initial_age = 0
            individual_re = float(np.random.default_rng(3000000 + slot_ordinal * 1000 + seq).normal(0, RE_STD))
        else:
            inst["retired_pos"] = L - 1
            inst["has_removal"] = False
            inst["snap_pos"] = finalize_snapshot_positions(inst["snap_pos"], inst["global_cycle"])
            instances.append(inst)
            break
    return instances, removals


def run_all_slots(slots, components, timelines, alpha, snapshot_interval, avg_hours_map):
    all_instances = []
    slot_removals = {}
    comp_init = {c["slot_id"]: c for _, c in components.iterrows()}
    for i, (_, sl) in enumerate(slots.iterrows()):
        insts, rems = simulate_slot_forward(i, sl, comp_init[sl["slot_id"]], timelines, alpha,
                                            snapshot_interval, avg_hours_map)
        all_instances.extend(insts)
        slot_removals[sl["slot_id"]] = sorted(rems, key=lambda x: x[0])
        if (i + 1) % 500 == 0 or i + 1 == len(slots):
            print(f"[simulate] slots={i + 1}/{len(slots)} instances={len(all_instances)}", flush=True)
    return all_instances, slot_removals


# ----------------------------- Build snapshot rows + labels (vectorized) -----------------------------
def _instance_frame(inst, rem_abs, max_obs, ac_model, ac_operator):
    s = inst["s"]; cum = inst["cum"]; cum0 = cum[s]
    days_full = inst["days_full"]
    snap = inst["snap_pos"][inst["snap_pos"] <= inst["retired_pos"]]
    if len(snap) == 0:
        return None
    snap_dates = days_full[s + snap]
    snap = snap[(snap_dates >= DATE_START) & (snap_dates <= SNAPSHOT_END)]
    if len(snap) == 0:
        return None
    abs_c = cum0 + inst["global_cycle"][snap]
    keep = abs_c + PRED_WINDOW <= max_obs + 1e-6
    if not keep.any():
        return None
    snap = snap[keep]; abs_c = abs_c[keep]
    age = inst["age_cycle"][snap]
    # Labels and historical removal counts
    if len(rem_abs):
        cnt_le_c = np.searchsorted(rem_abs, abs_c, side="right")
        cnt_le_c30 = np.searchsorted(rem_abs, abs_c + PRED_WINDOW, side="right")
        target = (cnt_le_c30 > cnt_le_c).astype(int)
        rem_hist = np.searchsorted(rem_abs, abs_c, side="left")
    else:
        target = np.zeros(len(snap), dtype=int)
        rem_hist = np.zeros(len(snap), dtype=int)
    # tsr
    mc = inst["maint_at_cycle"]
    lmi = np.searchsorted(mc, age, side="right") - 1
    last_maint_cycle = np.where(lmi >= 0, mc[np.clip(lmi, 0, None)], 0.0)
    tsr = age - last_maint_cycle
    # Cumulative faults and faults in the last 30 cycles
    ft = inst["fault_today"]
    cum_faults_arr = np.concatenate([[0], np.cumsum(ft)])
    cum_faults = cum_faults_arr[snap + 1]
    gc = inst["global_cycle"]
    idx = np.searchsorted(gc, gc[snap] - PRED_WINDOW, side="right")
    f30 = cum_faults_arr[snap + 1] - cum_faults_arr[idx]
    # Most recent fault code/severity (forward-filled)
    fia = np.where(ft)[0]
    L = len(ft)
    if len(fia):
        pos = np.searchsorted(fia, np.arange(L), side="right") - 1
        has = pos >= 0
        pc = np.where(has, pos, 0)
        last_code_all = np.where(has, inst["fault_codes"][fia[pc]], "NONE")
        last_sev_all = np.where(has, inst["fault_sevs"][fia[pc]], None)
    else:
        last_code_all = np.full(L, "NONE", dtype=object)
        last_sev_all = np.full(L, None, dtype=object)
    # Flight cycles in the last 30 days
    cum_d = np.concatenate([[0], np.cumsum(inst["daily_slice"])])
    fc30 = cum_d[snap + 1] - cum_d[np.maximum(snap + 1 - 30, 0)]
    # Sensors (degradation effect + snapshot-level noise)
    rrng = np.random.default_rng(4000000 + inst["seed"])
    Dp = inst["D"][snap]
    n = len(snap)
    sensors = {}
    for sname, sconf in C.gen("sensors").items():
        if inst["component_type"] not in sconf["applicable"]:
            sensors[sname] = np.full(n, np.nan)
            continue
        sign, scale = SENSOR_EFFECT[sname]
        sensors[sname] = (sconf["baseline_mean"] + sign * scale * Dp
                          + sconf["baseline_std"] * rrng.standard_normal(n))
    return pd.DataFrame({
        "snapshot_date": days_full[s + snap].strftime("%Y-%m-%d"),
        "snapshot_cycle": np.round(abs_c).astype(int),
        "aircraft_id": inst["aircraft_id"],
        "component_id": inst["component_id"],
        "aircraft_model": ac_model[inst["aircraft_id"]],
        "component_type": inst["component_type"],
        "operator": ac_operator[inst["aircraft_id"]],
        "component_age_cycles": age,
        "component_age_days": inst["age_days"][snap],
        "tsr_cycles": tsr,
        "operating_hours_since_install": inst["avg_flight_hours"] * age,
        "sensor_vibration": sensors["sensor_vibration"],
        "sensor_temp": sensors["sensor_temp"],
        "sensor_pressure": sensors["sensor_pressure"],
        "sensor_oil_debris": sensors["sensor_oil_debris"],
        "sensor_rpm_deviation": sensors["sensor_rpm_deviation"],
        "sensor_exhaust_temp": sensors["sensor_exhaust_temp"],
        "fault_code_last": last_code_all[snap],
        "fault_count_last_30": f30.astype(int),
        "fault_severity_last": last_sev_all[snap],
        "last_maint_cycles_ago": np.round(tsr).astype(int),
        "last_maint_type": "scheduled",
        "removal_count_history": rem_hist.astype(int),
        "cumulative_faults": cum_faults.astype(int),
        "flight_cycles_last_30": fc30.astype(int),
        "avg_flight_hours": inst["avg_flight_hours"],
        "target": target,
        "snapshot_abs_cycle_exact": abs_c,
    })


def temporal_split_names(dates):
    dates = pd.DatetimeIndex(dates)
    t = C.gen("split", "temporal")
    names = np.full(len(dates), "observation", dtype=object)
    names[dates < pd.Timestamp(DATE_START)] = "warmup"
    names[(dates >= pd.Timestamp(DATE_START)) & (dates <= pd.Timestamp(t["train_end"]))] = "train"
    names[(dates >= pd.Timestamp(t["val_start"])) & (dates <= pd.Timestamp(t["val_end"]))] = "val"
    names[(dates >= pd.Timestamp(t["test_start"])) & (dates <= pd.Timestamp(t["test_end"]))] = "test"
    return names


def build_snapshots(all_instances, slot_removals, timelines, ac_model, ac_operator, slots,
                    with_links=True):
    frames = []
    for inst in all_instances:
        rems = slot_removals.get(inst["slot_id"], [])
        rem_abs = np.array([r[0] for r in rems]) if rems else np.array([])
        f = _instance_frame(inst, rem_abs, inst["cum"][-1], ac_model, ac_operator)
        if f is not None and len(f):
            frames.append(f)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        # Build IDs from unrounded cycles so nearby snapshots are not deduplicated by rounded display fields.
        exact_cycle_id = np.round(df["snapshot_abs_cycle_exact"] * 1000).astype(np.int64).astype(str)
        df["record_id"] = "R" + exact_cycle_id.str.zfill(13) + "_" + df["component_id"].astype(str)
        df = df.drop_duplicates(subset=["record_id"]).reset_index(drop=True)
        df["temporal_split"] = temporal_split_names(df["snapshot_date"])
    slot_ac = dict(zip(slots["slot_id"], slots["aircraft_id"]))
    audit = []
    for slot_id, rems in slot_removals.items():
        ac_id = slot_ac.get(slot_id, "")
        days_full = timelines[ac_id][0] if ac_id in timelines else None
        for (abs_c, cid, day_idx) in rems:
            audit.append({
                "removal_event_id": f"EV_{cid}_{int(round(abs_c*1000)):010d}",
                "component_id": cid,
                "aircraft_id": ac_id,
                "slot_id": slot_id,
                "event_cycle": int(round(abs_c)),
                "event_cycle_exact": float(abs_c),
                "event_date": days_full[day_idx].strftime("%Y-%m-%d") if (days_full is not None and day_idx < len(days_full)) else "",
                "event_temporal_split": temporal_split_names([days_full[day_idx]])[0],
                "removal_type": "unscheduled",
            })
    audit_df = pd.DataFrame(audit).drop_duplicates(subset=["removal_event_id"]).reset_index(drop=True)
    if not with_links:
        return df, audit_df, None

    links = []
    snap_groups = {cid: g.sort_values("snapshot_abs_cycle_exact")
                   for cid, g in df.groupby("component_id", sort=False)}
    for event in audit_df.itertuples(index=False):
        snaps = snap_groups.get(event.component_id)
        if snaps is None or snaps.empty:
            continue
        abs_values = snaps["snapshot_abs_cycle_exact"].to_numpy()
        lo = int(np.searchsorted(abs_values, event.event_cycle_exact - MAX_SNAPSHOT_GAP, side="left"))
        hi = int(np.searchsorted(abs_values, event.event_cycle_exact, side="right"))
        if lo >= hi:
            continue
        matched = snaps.iloc[lo:hi]
        for snap in matched.itertuples(index=False):
            links.append({
                "removal_event_id": event.removal_event_id,
                "record_id": snap.record_id,
                "component_id": event.component_id,
                "aircraft_id": event.aircraft_id,
                "event_date": event.event_date,
                "snapshot_date": snap.snapshot_date,
                "event_temporal_split": event.event_temporal_split,
                "snapshot_temporal_split": snap.temporal_split,
                "same_temporal_split": event.event_temporal_split == snap.temporal_split,
                "cycles_before_event": float(event.event_cycle_exact - snap.snapshot_abs_cycle_exact),
            })
    link_columns = ["removal_event_id", "record_id", "component_id", "aircraft_id",
                    "event_date", "snapshot_date", "event_temporal_split",
                    "snapshot_temporal_split", "same_temporal_split", "cycles_before_event"]
    links_df = pd.DataFrame(links, columns=link_columns)
    all_counts = links_df.groupby("removal_event_id").size() if len(links_df) else pd.Series(dtype=int)
    same_counts = (links_df[links_df["same_temporal_split"]].groupby("removal_event_id").size()
                   if len(links_df) else pd.Series(dtype=int))
    audit_df["n_scoring_snapshots"] = audit_df["removal_event_id"].map(all_counts).fillna(0).astype(int)
    audit_df["n_same_split_scoring_snapshots"] = audit_df["removal_event_id"].map(same_counts).fillna(0).astype(int)
    audit_df["has_scoring_opportunity"] = audit_df["n_scoring_snapshots"] > 0
    audit_df["has_same_split_scoring_opportunity"] = audit_df["n_same_split_scoring_snapshots"] > 0
    return df, audit_df, links_df


# ----------------------------- Data quality injection -----------------------------
def inject_data_quality(df):
    rng = np.random.default_rng(C.gen("random_seed") + 20)
    type_rate = {ct["name"]: ct["degradation_rate"] for ct in C.gen("component_types")}
    deg_proxy = df["component_age_cycles"] * df["component_type"].map(type_rate)
    deg_proxy = (deg_proxy - deg_proxy.mean()) / (deg_proxy.std() + 1e-6)
    mech = C.gen("missing_mechanism")
    for sname, sconf in C.gen("sensors").items():
        base = sconf["missing_rate"]
        applicable = df["component_type"].isin(sconf["applicable"])
        n = len(df)
        mcar = rng.random(n) < base * mech["mcar_frac"]
        mar = (rng.random(n) < base * mech["mar_frac"]) & applicable
        mnar_p = base * mech["mnar_frac"] * (1 + 1.5 * np.clip(deg_proxy, 0, 3))
        mnar = (rng.random(n) < mnar_p) & applicable
        df.loc[(mcar | mar | mnar) & applicable, sname] = np.nan
    anomaly = C.gen("anomaly")
    for sname, sconf in C.gen("sensors").items():
        mask = df[sname].notna()
        idx = df.index[mask]
        n_anom = int(len(idx) * anomaly["cell_frac"])
        if n_anom == 0:
            continue
        chosen = rng.choice(idx.values, size=n_anom, replace=False)
        for c in chosen:
            r = rng.random()
            v = df.at[c, sname]
            if r < anomaly["types"]["degradation_anomaly_frac"]:
                df.at[c, sname] = v * (2 + rng.uniform(0, 1.5))
            elif r < anomaly["types"]["degradation_anomaly_frac"] + anomaly["types"]["measurement_anomaly_frac"]:
                df.at[c, sname] = v + sconf["baseline_std"] * rng.uniform(5, 10) * rng.choice([-1, 1])
            else:
                df.at[c, sname] = float(rng.choice([-999, 9999, -1]))
    return df


# ----------------------------- Main flow -----------------------------
def calibration_periods():
    t = C.gen("split", "temporal")
    return [
        ("2022", None, "2022-12-31"),
        ("2023", "2023-01-01", t["train_end"]),
        ("val", t["val_start"], t["val_end"]),
        ("test", t["test_start"], t["test_end"]),
    ]


def date_mask(dates, start, end):
    dates = pd.DatetimeIndex(dates)
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    return mask


def period_positive_rates(df, periods):
    dates = pd.to_datetime(df["snapshot_date"])
    target = df["target"].to_numpy()
    rates = {}
    for name, start, end in periods:
        mask = date_mask(dates, start, end)
        rates[name] = float(target[mask].mean()) if mask.any() else 0.0
    return rates


def snapshot_coverage_metrics(df, audit):
    gaps = (df.sort_values(["component_id", "snapshot_abs_cycle_exact"])
            .groupby("component_id")["snapshot_abs_cycle_exact"].diff())
    formal = audit[audit["event_temporal_split"].isin(["train", "val", "test"])]
    by_period = {}
    for period in ["train", "val", "test"]:
        events = formal[formal["event_temporal_split"] == period]
        n_events = len(events)
        n_covered = int(events["has_same_split_scoring_opportunity"].sum())
        by_period[period] = {
            "n_events": int(n_events),
            "n_covered_same_split": n_covered,
            "coverage_same_split": float(n_covered / n_events) if n_events else 1.0,
        }
    n_formal = len(formal)
    n_all_covered = int(formal["has_scoring_opportunity"].sum())
    n_same_covered = int(formal["has_same_split_scoring_opportunity"].sum())
    return {
        "n_formal_removal_events": int(n_formal),
        "n_formal_events_with_scoring_opportunity": n_all_covered,
        "formal_event_coverage": float(n_all_covered / n_formal) if n_formal else 1.0,
        "n_formal_events_with_same_split_opportunity": n_same_covered,
        "formal_event_same_split_coverage": float(n_same_covered / n_formal) if n_formal else 1.0,
        "event_coverage_by_temporal_split": by_period,
        "max_snapshot_gap_cycles": float(gaps.max()) if gaps.notna().any() else 0.0,
        "n_snapshot_gap_violations": int((gaps > MAX_SNAPSHOT_GAP + 1e-6).sum()),
    }


def generate(smoke=False):
    output_root = tempfile.mkdtemp(prefix="mro_v3_smoke_") if smoke else DATA
    raw_dir = os.path.join(output_root, "raw")
    audit_dir = os.path.join(output_root, "audit")
    for d in (raw_dir, audit_dir):
        os.makedirs(d, exist_ok=True)
    n_aircraft = 12 if smoke else C.gen("n_aircraft")
    print(f"[mode] {'smoke' if smoke else 'full'} output={output_root}", flush=True)

    aircraft = build_aircraft_registry(n_aircraft)
    slots, components = build_slot_components(aircraft)
    avg_hours_map = dict(zip(aircraft["aircraft_id"], aircraft["avg_flight_hours"]))
    ac_model = dict(zip(aircraft["aircraft_id"], aircraft["aircraft_model"]))
    ac_operator = dict(zip(aircraft["aircraft_id"], aircraft["operator"]))
    timelines = build_aircraft_timelines(aircraft)

    total_slot_cycles = 0.0
    for days, cum, daily in timelines.values():
        active = (days >= DATE_START) & (days <= SNAPSHOT_END)
        total_slot_cycles += daily[active].sum() * 10
    snapshot_target = C.gen("target_snapshots") * n_aircraft / C.gen("n_aircraft")
    snapshot_interval = max(5.0, total_slot_cycles / snapshot_target)
    print(f"[setup] total_slot_cycles={total_slot_cycles:.0f} "
          f"snapshot_interval={snapshot_interval:.3f} target_rows={snapshot_target:.0f}", flush=True)

    periods = calibration_periods()
    target = C.gen("positive_rate_target")

    def eval_alphas(alphas):
        days = next(iter(timelines.values()))[0]
        alpha_by_day = np.full(len(days), alphas[-1], dtype=float)
        for alpha, (name, start, end) in zip(alphas, periods):
            alpha_by_day[date_mask(days, start, end)] = alpha
        all_instances, slot_removals = run_all_slots(slots, components, timelines, alpha_by_day,
                                                     snapshot_interval, avg_hours_map)
        df, audit, _ = build_snapshots(all_instances, slot_removals, timelines, ac_model,
                                       ac_operator, slots, with_links=False)
        if df.empty:
            raise RuntimeError("Calibration produced no valid snapshots")
        return {
            "alphas": alphas.copy(), "df": df, "audit": audit,
            "instances": all_instances, "slot_removals": slot_removals,
            "rates": period_positive_rates(df, periods),
        }

    # Calibrate period by period in chronological order; keep earlier periods fixed to avoid later drift.
    initial_alpha = C.gen("hazard", "alpha_global")
    alphas = np.full(len(periods), initial_alpha, dtype=float)
    result = None
    for p, (name, start, end) in enumerate(periods):
        if p > 0:
            alphas[p] = alphas[p - 1]
        center = alphas[p]
        lo, hi = center - 6.0, center + 6.0
        best = None
        for it in range(10):
            trial = alphas.copy()
            trial[p] = (lo + hi) / 2
            candidate = eval_alphas(trial)
            rate = candidate["rates"][name]
            error = abs(rate - target)
            if best is None or error < best[0]:
                best = (error, candidate)
            rates_text = " ".join(f"{n}={candidate['rates'][n]:.4f}" for n, _, _ in periods)
            print(f"[calibrate {name} {it}] alpha={trial[p]:.4f} rows={len(candidate['df'])} "
                  f"pos={int(candidate['df']['target'].sum())} {rates_text}", flush=True)
            if error < 0.0010:
                break
            if rate > target:
                hi = trial[p]
            else:
                lo = trial[p]
        if best is None or best[0] > 0.0020:
            raise RuntimeError(f"{name} positive-rate calibration failed: best_error={best[0] if best else 'NA'}")
        result = best[1]
        alphas = result["alphas"]

    df, audit, links = build_snapshots(result["instances"], result["slot_removals"], timelines,
                                       ac_model, ac_operator, slots, with_links=True)
    rates = period_positive_rates(df, periods)
    bad_periods = {k: v for k, v in rates.items() if not 0.018 <= v <= 0.022}
    if bad_periods:
        raise RuntimeError(f"Period positive rate outside 1.8%-2.2%: {bad_periods}")
    if not smoke and not 240000 <= len(df) <= 320000:
        raise RuntimeError(f"Unexpected v3 snapshot row count: {len(df)}; expected 240000-320000")
    if len(result["instances"]) > len(slots) * 101:
        raise RuntimeError(f"Unexpected instance count: {len(result['instances'])}")
    coverage = snapshot_coverage_metrics(df, audit)
    if coverage["formal_event_same_split_coverage"] < 1.0 or coverage["n_snapshot_gap_violations"] > 0:
        raise RuntimeError(f"v3 coverage/gap gate failed: {coverage}")

    df = inject_data_quality(df)
    df = df[C.FIELD_NAMES]
    df = df.sort_values(["snapshot_date", "aircraft_id", "component_id"]).reset_index(drop=True)
    df.to_csv(os.path.join(raw_dir, "component_snapshots.csv"), index=False)
    audit.to_csv(os.path.join(audit_dir, "removal_events.csv"), index=False)
    links.to_csv(os.path.join(audit_dir, "snapshot_event_links.csv"), index=False)
    aircraft.to_csv(os.path.join(audit_dir, "aircraft_registry.csv"), index=False)
    components.to_csv(os.path.join(audit_dir, "component_registry.csv"), index=False)
    slots.to_csv(os.path.join(audit_dir, "slot_registry.csv"), index=False)

    applicable_missing = {}
    for sname, sconf in C.gen("sensors").items():
        sub = df[df["component_type"].isin(sconf["applicable"])]
        applicable_missing[sname] = float(sub[sname].isna().mean()) if len(sub) else 0.0
    summary = {
        "dataset_version": "v3-smoke" if smoke else "v3",
        "output_root": output_root,
        "random_seed": C.gen("random_seed"),
        "warmup_days": C.gen("warmup_days"),
        "alpha_by_period": {name: float(a) for a, (name, _, _) in zip(alphas, periods)},
        "snapshot_interval_cycles": float(snapshot_interval),
        "max_snapshot_gap_allowed_cycles": int(MAX_SNAPSHOT_GAP),
        "n_rows": int(len(df)),
        "n_positive": int(df["target"].sum()),
        "positive_rate": float(df["target"].mean()),
        "positive_rate_by_period": rates,
        "positive_rate_by_year": {str(k): float(v) for k, v in df.assign(
            year=df["snapshot_date"].str[:4]).groupby("year")["target"].mean().items()},
        "n_aircraft": int(df["aircraft_id"].nunique()),
        "n_components": int(df["component_id"].nunique()),
        "n_simulated_instances": int(len(result["instances"])),
        "n_removal_events": int(len(audit)),
        "n_snapshot_event_links": int(len(links)),
        "date_range": [df["snapshot_date"].min(), df["snapshot_date"].max()],
        **coverage,
        "sensor_missing_rate_applicable": applicable_missing,
        "sensor_missing_rate_overall": {c: float(df[c].isna().mean()) for c in C.SENSOR_FIELDS},
    }
    with open(os.path.join(audit_dir, "generation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[done]", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the MRO v3 mock dataset")
    parser.add_argument("--smoke", action="store_true", help="validate generation logic with 12 aircraft under /tmp")
    args = parser.parse_args()
    generate(smoke=args.smoke)
