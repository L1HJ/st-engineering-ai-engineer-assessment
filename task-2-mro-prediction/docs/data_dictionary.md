# Data Dictionary: MRO Synthetic Component Removal Dataset v3

This document describes the fields, label, time boundaries, and missingness conventions of the v3 synthetic dataset. The machine-readable definition is in [configs/schema.yaml](../configs/schema.yaml), and the recorded generation statistics are summarized in Section 1. Project-level context is in [README.md](../README.md), the formal model results in [MRO_Model_Formal_Delivery_Report.md](MRO_Model_Formal_Delivery_Report.md), and the serving design in [MRO_Production_Deployment_Design.md](MRO_Production_Deployment_Design.md). The figures below are transcribed from the existing audit outputs; no data was regenerated for this document.

> The data are synthetic and intended for method validation only. They do not represent a real fleet, real maintenance records, or an airworthiness conclusion.

## 1. Dataset card

### 1.1 Basic information

| Item | Content |
|---|---|
| Name | MRO Synthetic Component Removal Dataset |
| Version | v3 |
| Random seed | 42 |
| Generation script | `python src/generate_dataset.py` |
| Time range | 2022-01-01 to 2024-09-30, including a 365-day warm-up and a 120-day label-observation extension |
| Data nature | Synthetic data, not real fleet records |
| License | To be determined; separate licenses must be selected for code and data before release |

### 1.2 Scale

| Item | Value |
|---|---:|
| Snapshot records | 299,474 |
| Fields | 28 |
| Aircraft | 200 |
| Initially installed component instances | 2,000 |
| Final visible component instances | 7,367 |
| Total simulated instances (including warm-up and replacements) | 10,820 |
| Unscheduled removal events | 8,823 (5,371 in the formal period) |
| Positive samples / positive rate | 5,997 / 2.0025% |
| Formal snapshot dates | 2022-01-01 to 2024-09-30 |

The 2.0025% rate is a snapshot-record-level proportion and not an independent-event rate.

### 1.3 File structure

| Path | Size | Content |
|---|---:|---|
| `data/raw/component_snapshots.csv` | 71 MB | Full snapshots and labels |
| `data/splits/temporal/train.csv` | 51 MB | Temporal training split (215,958 rows) |
| `data/splits/temporal/val.csv` | 6.7 MB | Temporal validation split (28,256 rows) |
| `data/splits/temporal/test.csv` | 13 MB | Temporal test split (55,260 rows) |
| `data/splits/split_summary.json` | - | Split statistics |
| `data/audit/snapshot_event_links.csv` | - | Snapshot-event links and same-split flags |
| `data/audit/removal_events.csv` | - | Removal event audit table |
| `data/audit/generation_summary.json` | - | Generation statistics |
| `data/audit/*_registry.csv` | - | Aircraft, component, and slot registries |

The CSV files total about 155 MB. Do not commit this data directly to an ordinary Git repository; publish it through a Hugging Face Dataset, GitHub Release assets, or another object store.

### 1.4 Generation highlights

- 200 aircraft with 10 initial monitored slots each, for 2,000 initial component instances.
- After an unscheduled removal the old instance retires and the replacement receives a new `component_id`.
- Degradation is driven by a continuous random walk plus a risk term; the hazard intercept is calibrated by time period.
- Snapshots are densified by actual flight cycle; the maximum adjacent gap is below 30 cycles.
- A 120-day label-observation window is retained at the end of the period; right-censored snapshots are not labeled negative.
- The base sampling interval was computed from a 260,000-row target at about 26.3369 cycles, then combined with random jitter and additional points. Anchors are placed at the first simulated observation after installation, the formal data start, the val and test start dates, and actual unscheduled-removal boundaries. The maximum observed adjacent interval is 29.9998 cycles using unrounded cycles, with zero violations of the 30-cycle limit.

Actual generation statistics and v3 changes:

- v3 densified the snapshot table from 93,362 to 299,474 rows. In v2, only 342 of the 909 actual test removals (37.62%) had a scorable snapshot; v3 targets observability rather than model tuning or manual positive-sample inflation. A smoke run on 12 aircraft preceded a single full generation, and rows, positives, and labels were not duplicated or hand-edited.
- Snapshot-event links are defined as the same `component_id` with an unrounded cycle difference satisfying `0 <= event_cycle - snapshot_cycle <= 30`. A difference of 0 means the same observation day or cycle; no minimum lead time is enforced. There are 11,368 links in total.
- `record_id` is built from unrounded cycles at thousandth precision, because two nearby snapshots can share the same integer `snapshot_cycle`. All 299,474 IDs are unique.
- The steady-state and risk mechanisms carried over from v2 are the 365-day warm-up, forward simulation of replacements for the 2,000 initial slots, the continuous random-walk degradation perturbation, the `clip(D, 0, 6)^2` unscheduled-removal risk term, and period-wise hazard-intercept calibration instead of direct label edits.

Calibrated hazard intercepts:

| Period | alpha |
|---|---:|
| 2022 | -29.50000 |
| 2023 | -25.00000 |
| Val | -22.00000 |
| Test | -22.75000 |

### 1.5 Quality checks

- All 299,474 `record_id` values are unique.
- Formal-event same-split coverage: train 3,851/3,851, val 548/548, test 972/972 (5,371/5,371 overall, 100%).
- Maximum adjacent snapshot interval is 29.9998 cycles.
- The 28 fields and their order match the schema.
- Every positive sample links to an unscheduled removal within the next 30 cycles.
- Annual and per-split positive rates pass the stability checks.
- Missingness, anomalies, fault-code cardinality, date range, and v3 row-count checks all pass: 22/22 data-quality checks and 3/3 pytest tests.
- The smoke run covered 12 aircraft and 17,744 rows, with 100% same-split formal-event coverage and a maximum observed interval of 29.9993 cycles.
- Generation summary: `data/audit/generation_summary.json`.
- Label-boundary audit: `python src/label_boundary_audit.py` writes `reports/label_boundary_audit.json`.

### 1.6 Citation

If you use this data, cite: MRO Synthetic Component Removal Dataset v3, synthetic data, for method validation only.

## 2. Label definition and time boundary

| Field | Definition |
|---|---|
| `target` | Computed starting after the snapshot time point: whether an unscheduled removal occurs within the **next 30 flight cycles**. 1 if it occurs, otherwise 0. |

Boundary rules:

- A removal that already occurred at or before the snapshot is history and is not counted in the future label.
- An event that occurs exactly on the 30th future cycle counts as a positive.
- Snapshots whose future 30-cycle observation window is incomplete (right-censored) are removed from the supervised wide table and are not labeled negative.
- The evaluation layer counts a snapshot-event link as independent-event recall only when the snapshot and event are in the same split and the event is 0-30 cycles ahead. Multiple links to the same event count once.

Allowed information per field:

| Field | Allowed information range |
|---|---|
| Sensor readings | At or before the snapshot time point |
| `fault_code_last` / `fault_severity_last` | At or before the snapshot time point |
| `fault_count_last_30` | Historical window before the snapshot |
| `last_maint_cycles_ago` / `last_maint_type` | Before the snapshot |
| `removal_count_history` / `cumulative_faults` | Cumulative before the snapshot |
| `target` | The 30 flight cycles after the snapshot |

Cross-split label overlap is covered in Section 3; it is a label-window boundary issue, not feature time leakage.

## 3. Label-window cross-split audit

Temporal splits are hard cuts by date, but a 30-cycle label window can extend past a boundary into the next split, producing a small amount of **label-level overlap**. This is not feature leakage: features use information at or before the snapshot, and preprocessing is fitted on train only. However, some positives at the end of train are driven by val-period events, and the same applies at the end of val.

Audit results (read-only, source `reports/label_boundary_audit.json`):

| Split | Positives | Same-split links | Cross-split links only | Share |
|---|---:|---:|---:|---:|
| Train | 4,291 | 4,211 | 80 (event in val) | 1.8644% |
| Val | 613 | 583 | 30 (event in test) | 4.8940% |
| Test | 1,093 | 1,070 | 23 (event in observation period) | 2.1043% |

There are 11,368 snapshot-event links in total: 11,235 same-split and 133 cross-split (train to val 80, val to test 30, test to observation 23).

Impact and handling:

- The independent-event recall denominator uses same-split events only, so **the event metrics themselves are not contaminated**.
- Val threshold selection uses snapshot labels, and about 4.89% of val positives are driven by test-period events, which may introduce slight optimistic bias.
- The frozen temporal test has been consumed and cannot be corrected retroactively by retraining. This version only quantifies and discloses the issue; the next data version should purge label-window overlap at split boundaries.
- Reproduction command: `python src/label_boundary_audit.py`.

## 4. Splits

Temporal (out-of-time) splits:

| Split | Date range | Rows | Positives | Positive rate | Component instances |
|---|---|---:|---:|---:|---:|
| Train | 2022-01-01 to 2023-12-31 | 215,958 | 4,291 | 1.9870% | 5,846 |
| Val | 2024-01-01 to 2024-03-31 | 28,256 | 613 | 2.1695% | 2,543 |
| Test | 2024-04-01 to 2024-09-30 | 55,260 | 1,093 | 1.9779% | 2,968 |

Annual positive rates:

| Year | Positive rate |
|---|---:|
| 2022 | 1.9939% |
| 2023 | 1.9799% |
| 2024 (through September) | 2.0427% |

All three temporal periods fall inside the 1.8%-2.2% generation acceptance range. Stable label rates only indicate that synthetic event intensity was calibrated; they do not imply that model metrics have met their targets.

Entity-group splits:

| Split | Rows | Positives | Positive rate | Aircraft | Component instances |
|---|---:|---:|---:|---:|---:|
| Aircraft Train | 238,224 | 4,714 | 1.9788% | 160 | 5,815 |
| Aircraft Test | 61,250 | 1,283 | 2.0947% | 40 | 1,552 |
| Component Train | 239,150 | 4,811 | 2.0117% | 200 | 5,894 |
| Component Test | 60,324 | 1,186 | 1.9660% | 200 | 1,473 |

Aircraft entities do not overlap across the aircraft-group split, and component instances do not overlap across the component-group split.

## 5. Field dictionary

Fields with `model_feature=no` are excluded from the model. There are 23 final candidate features.

| Field | Type / Unit | Nullable | Model feature | Definition |
|---|---|---|---|---|
| `record_id` | string | no | no | Unique snapshot record ID |
| `snapshot_date` | date | no | no | Snapshot date (prediction time point) |
| `snapshot_cycle` | integer | no | yes | Cumulative flight cycles of the aircraft up to the snapshot |
| `aircraft_id` | string | no | no | Aircraft instance ID |
| `component_id` | string | no | no | Physical component instance ID; the old instance retires after removal, and the replacement gets a new ID |
| `aircraft_model` | category | no | yes | Aircraft model |
| `component_type` | category | no | yes | Component type |
| `operator` | category | no | yes | Operator |
| `component_age_cycles` | float / cycles | no | yes | Cumulative flight cycles since the component entered service |
| `component_age_days` | float / days | no | yes | Calendar days since the component entered service |
| `tsr_cycles` | float / cycles | no | yes | Flight cycles since the last repair (Time Since Repair) |
| `operating_hours_since_install` | float / hours | no | yes | Cumulative operating hours since the current installation |
| `sensor_vibration` | float / mm/s | yes | yes | Vibration measurement |
| `sensor_temp` | float / celsius | yes | yes | Operating temperature |
| `sensor_pressure` | float / bar | yes | yes | Pressure |
| `sensor_oil_debris` | float / mg/L | yes | yes | Oil debris / wear particle indicator |
| `sensor_rpm_deviation` | float / percent | yes | yes | RPM deviation, which can be negative |
| `sensor_exhaust_temp` | float / celsius | yes | yes | Exhaust temperature, applicable to some component types only |
| `fault_code_last` | category | yes | yes | Most recent fault code before the snapshot; NONE means no fault |
| `fault_count_last_30` | integer | no | yes | Number of faults in the 30 flight cycles before the snapshot |
| `fault_severity_last` | category | yes | yes | Severity of the most recent fault: low/medium/high |
| `last_maint_cycles_ago` | integer | no | yes | Flight cycles since the most recent maintenance |
| `last_maint_type` | category | no | yes | Most recent maintenance type: scheduled/unscheduled |
| `removal_count_history` | integer | no | yes | Historical removal count for this component position before the snapshot |
| `cumulative_faults` | integer | no | yes | Cumulative fault count before the snapshot |
| `flight_cycles_last_30` | integer / cycles | no | yes | Flight cycles in the past 30 days (operating volume) |
| `avg_flight_hours` | float / hours | no | yes | Average flight hours per flight cycle |
| `target` | integer | no | no | Whether an unscheduled removal occurs within the next 30 flight cycles |

## 6. Business definitions

| Field / Concept | Definition |
|---|---|
| `tsr_cycles` | Time Since Repair. It resets after repair depending on the maintenance type; the instance retires after an unscheduled removal. |
| `flight_cycles_last_30` | Flight cycles in the past **30 days**, not the past 30-cycle window. |
| `component_id` | Physical component instance. Version 1 does not simulate transfers between aircraft; the component-aircraft installation relation is static. |
| Unscheduled removal | Only removals caused by component failure or degradation; other non-scheduled reasons such as operational adjustments are excluded. |
| 2% positive rate | Snapshot-record-level target. The independent event count is reported separately, and the two are never mixed. |
| `alerts/100` denominator | Fixed to active component windows (unique aircraft x component x 30-cycle window), not distinct components. |

## 7. Missingness

Sensor missingness is caused by a mix of MCAR, MAR, and MNAR mechanisms together with structural missingness from "component type not applicable". The two must be read separately:

| Sensor | Missing rate within applicable rows | Missing rate over the full table |
|---|---:|---:|
| `sensor_vibration` | 8.76% | 8.76% |
| `sensor_temp` | 7.69% | 7.69% |
| `sensor_pressure` | 9.76% | 43.00% |
| `sensor_oil_debris` | 12.65% | 64.67% |
| `sensor_rpm_deviation` | 10.78% | 55.94% |
| `sensor_exhaust_temp` | 11.66% | 82.15% |

Among applicable rows the random missingness is 7.69%-12.65%, inside the planned 5%-15% per-field range; the full-table maximum is 82.15%, and the difference comes from structural non-applicability. This difference must not be described as "overall missingness of 5%-15%". Reports must state which denominator is used.

Processing rules:

- Audit markers `-999` and `9999` are converted to missing.
- Except for `sensor_rpm_deviation`, negative sensor values are treated as invalid and converted to missing; `-1` is a valid reading for RPM deviation.
- Numeric missing values are imputed with the train median and a missing indicator is generated; categorical missing values are encoded as `Unknown`.
- The preprocessor is fitted on train only; val and test are transformed only.
- Structural missingness is not a data-quality defect, and grouped reporting must state that the applicable-row denominator is used.

## 8. Anomalies

About 1% of sensor observation cells are anomalous, split into true degradation anomalies, measurement anomalies, and data errors. The audit table records the anomaly type; the modeling wide table does not expose anomaly flags, to avoid shortcut learning. Anomaly cleaning only handles illegal markers and illegal negative values; no statistical winsorization is applied.

## 9. Known biases and limitations

1. The synthetic mechanism determines the event rate, degradation shape, and missingness pattern and cannot represent a real operational distribution.
2. Label windows overlap slightly across date splits: train 80/4,291 and val 30/613 positives are driven by the next stage's events; there is no feature leakage, and the event-recall denominator counts same-split links only.
3. Structural missingness leaves some sensors with very low full-table coverage; any cross-component-type summary must distinguish "within applicable rows" from "full table".
4. Replacements create new component instances, so a "new component" in the entity split is not the same as a genuine first-installation component.
5. The data cover only 200 aircraft and 8 component types; fleet-composition changes are not simulated.
6. Model-side caveats, including lower recall on new aircraft and new components and uncalibrated drift warning thresholds, are documented in [README.md](../README.md) and [MRO_Model_Formal_Delivery_Report.md](MRO_Model_Formal_Delivery_Report.md). Model targets are not implied by data coverage; see [MRO_Production_Deployment_Design.md](MRO_Production_Deployment_Design.md) for serving requirements.

## 10. Dataset packaging and SHA-256 verification

```bash
tar -czf mro_v3_synthetic.tar.gz \
  data/raw data/splits data/audit \
  docs/data_dictionary.md \
  configs/schema.yaml

shasum -a 256 mro_v3_synthetic.tar.gz > mro_v3_synthetic.tar.gz.sha256
```

The dataset is also included directly under `data/` in this repository. If a single archive is preferred for a release asset, use the commands above. In either case, publish the following together:

- this dataset card and data dictionary;
- `generation_summary.json`;
- the SHA-256 of the packaged archive, when an archive is used;
- the generation script version or commit.
