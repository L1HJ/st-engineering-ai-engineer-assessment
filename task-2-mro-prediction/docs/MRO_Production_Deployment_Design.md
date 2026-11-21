# MRO Component Failure Risk Model: Production Deployment, Monitoring, and Retraining Design

Formal delivery report annex | Production environment design

| Item | Content |
| --- | --- |
| Document role | Production environment design annex (deployment, monitoring, retraining, MRO workflow integration) |
| Formal model | LightGBM / `lgbm_15_leaves`, frozen threshold `0.05461588446029929` |
| Prediction definition | 30 flight cycles ahead of the snapshot time point; 30-cycle alert cooldown |
| Design basis | Frozen model manifest, metric reports, and entity robustness experiments |
| Data nature | v3 synthetic data; this design is for method validation and does not represent real operational performance |
| Delivery date | 2026-11-21 |

> **Design boundary**
> This document describes a proposed design that has not yet been implemented in a production environment. The monitoring thresholds and retraining trigger lines below are recommended initial values and must be calibrated on new data; the already consumed temporal test must not be used again for repeated validation or tuning.

# 1. Purpose, scope, and design principles

## 1.1 Purpose

This document covers production environment design: deployment, monitoring, retraining, and integration with MRO workflows. It describes the architecture, interfaces, alert strategy, monitoring metrics, retraining gates, and audit requirements needed to move the model from an offline frozen artifact to an operable service.

This document does not redefine the model. The model, features, threshold, and evaluation conventions all follow the frozen delivery: LightGBM / `lgbm_15_leaves`, threshold `0.05461588446029929`, a label defined as the next 30 flight cycles, a 30-cycle alert cooldown, a val threshold search cap of alerts/100 <= 4.5, and formal test targets of event recall >= 80% and alerts/100 <= 5.0.

## 1.2 Design principles

| Principle | Requirement | Rationale |
| --- | --- | --- |
| Point-in-time consistency | Features may only come from the snapshot time point and earlier | Prevents future information from entering online scoring |
| Offline/online consistency | Online serving reuses the same preprocessing, column order, and missing-value rules | Avoids training/serving skew |
| No online threshold adaptation | Thresholds change only through versioned releases | Prevents alert volume from drifting without approval |
| Operable alerts | Alerts are merged by episode and carry reason codes | Lets maintenance staff review and act on them |
| End-to-end auditability | Log model version, input hash, scores, and disposition outcomes | Supports post-hoc review and accountability |
| Gated retraining | A one-shot test is allowed only after passing the val gate | Continues the existing evaluation governance |

## 1.3 Relationship to existing deliverables

This document references rather than duplicates the formal results. Metrics are authoritative in `reports/val_metrics.json`, `reports/temporal_test_metrics.json`, and `reports/entity_robustness_metrics.json`; frozen inputs, model, and code hashes are authoritative in `models/test_evaluation_manifest.json`; the full metrics and conclusions are in `docs/MRO_Model_Formal_Delivery_Report.md`.

| Document | Path | Role |
| --- | --- | --- |
| Repository overview | `README.md` | Project status, metric conventions, layout, and reproduction commands |
| Formal delivery report | `docs/MRO_Model_Formal_Delivery_Report.md` | Frozen model results, evaluation governance, and formal conclusions |
| Data dictionary | `docs/data_dictionary.md` | Field definitions, label definition, missing-value and outlier rules |

Readers must use the data dictionary as the field-level contract for the 23 model fields and the README for environment and metric conventions. Where this design and the frozen artifacts disagree, the frozen artifacts and the formal delivery report prevail.

# 2. Target architecture

The production pipeline is primarily batch scoring, with near-real-time re-scoring as a supplement. MRO removal decisions run on a daily or flight-leg cadence and do not require millisecond online inference; batch scoring reuses the offline feature logic and reduces training/serving skew risk.

![Architecture relationships among data ingestion, feature construction, model serving, alert orchestration, MRO workflow, explanation output, and monitoring/audit](assets/mro_prod_design/image1.png)

Figure 1. Production scoring and alert pipeline

## 2.1 Component responsibilities

| Component | Responsibility | Key constraint |
| --- | --- | --- |
| Data ingestion | Receives component snapshot wide tables or incremental batches; validates fields, types, and snapshot time point | Rejects batches with a missing `snapshot_cycle` or duplicate primary keys |
| Feature construction | Generates the 23 model fields and missing indicators; categorical encoding and median imputation | Statistics must come from training artifacts and are not recomputed on the serving side |
| Model serving | Loads the joblib model and threshold, outputs risk score and model version | Read-only loading of frozen artifacts; records the input file hash |
| Alert orchestration | Threshold decision, 30-cycle cooldown, episode merging, priority ordering | Does not open duplicate alerts for the same component within a window |
| Explanation output | Outputs top contributing features, reason codes, and missing flags | Contributions are in raw score units and are not stated as probabilities or causal effects |
| MRO workflow | Creates or updates work-order recommendations and pushes to the manual review queue | High risk does not auto-approve or auto-remove |
| Monitoring and audit | Stores scoring logs, alerts, disposition outcomes, and label feedback | Retains model and threshold versions and supports traceback |

## 2.2 Operating modes

| Mode | Trigger | Latency target | Applicable scenario |
| --- | --- | --- | --- |
| Batch scoring (primary) | Daily or per-flight-leg batch | Completes within the batch, hourly | Routine fleet risk ranking and work-order scheduling |
| Near-real-time re-scoring | New snapshot or maintenance event arrives | Minutes | Ad-hoc inspections and risk review in AOG situations |
| Replay/shadow mode | New model candidate or threshold candidate | No production impact | Pre-release comparison without changing existing alerts |

This document does not commit to a specific infrastructure choice; batch jobs, message queues, or container services can all host the pipeline. The essential requirement is to keep feature and threshold versions consistent.

# 3. Interfaces and data contracts

## 3.1 Input contract

Each scoring request represents the state of one component instance at one snapshot time point. The input must contain the primary key, the snapshot time point, entity dimensions, the 23 model fields, and missing indicators consistent with training. Records without a snapshot time point or a component instance identifier must not enter scoring.

| Field group | Required | Handling rule |
| --- | --- | --- |
| `record_id` | Required | Unique across the table; duplicates reject the whole batch and raise an alert |
| `aircraft_id` / `component_id` | Required | Used for episode ownership and work-order linkage; not used as model features |
| `snapshot_cycle` / `snapshot_date` | Required | Defines the prediction time point; missing values reject the record |
| 23 model fields | Per schema | Missing values are imputed by training rules and flagged with a missing indicator |
| -999 / 9999 | Cleaning | Converted to missing and excluded from numeric computation |
| Negative sensor values | Cleaning | Treated as missing except for `sensor_rpm_deviation` |
| Future fields / labels | Forbidden | Presence of `target` or post-event fields rejects scoring to prevent leakage |

## 3.2 Output contract

| Output field | Meaning | Use |
| --- | --- | --- |
| `risk_score` | Raw model risk score (0-1) | Ranking and audit |
| `threshold` | Frozen threshold used for this scoring run | Reproducing the decision |
| `is_alert` | `risk_score` >= `threshold` | Triggers alert orchestration |
| `alert_episode_id` | Alert episode identifier after cooldown merging | Deduplication and work-order linkage |
| `model_version` | Model file name + training/frozen version | Version tracking |
| `top_factors` | Top contributing features and their contributions | Manual review and explanation |
| `missing_flags` | Flags for key missing fields | Data-quality signal |
| `scored_at` / `input_hash` | Scoring time and input batch hash | Audit and replay |

# 4. Alert strategy and MRO workflow integration

## 4.1 Alert decision

| Rule | Current value | Note |
| --- | --- | --- |
| Risk threshold | `0.05461588446029929` | Frozen LightGBM threshold; may change only through a versioned release |
| Prediction horizon | Next 30 flight cycles | Not 30 days; counted in cycles |
| Alert cooldown | 30 cycles for the same component | Consecutive high-score snapshots merge into one episode |
| Alert budget | alerts/100 active <= 5.0 | Formal acceptance convention; val search cap is 4.5 |
| Ordering basis | Descending risk score; ties broken by aircraft age/critical component | Used for work-order priority suggestions |
| Deduplication granularity | Component instance + episode | Multiple snapshots of the same event do not open duplicate work orders |

## 4.2 MRO workflow

Alerts do not directly trigger a removal; they enter a manual review queue. The recommended process is:

- The system generates an alert entry with the risk score, episode, top contributing features, missing flags, and the risk trend over the last 30 cycles.
- Maintenance planners rank by risk score and fleet availability and confirm using borescope inspection, test runs, oil analysis, and other checks.
- The confirmation result is recorded as "true positive / false positive / observe further" and written back to the work-order system as one source of labels for later performance monitoring and retraining.
- After a removal or repair, the actual removal time and cause are written back for event-level recall and lead-time statistics.
- High-risk alerts have a review deadline; alerts not dispositioned in time escalate to the duty manager.

## 4.3 Linkage to existing metrics

| Existing evidence | Value | Production implication |
| --- | --- | --- |
| Frozen test event recall | 969/972 = 99.6914% | Offline upper-bound reference; must be recomputed on new windows after go-live |
| Frozen test alert burden | 3.50498 alerts/100 | Below the 5.0 budget, leaving headroom |
| valve-group alert burden | 7.34653 alerts/100 | Higher review workload for this component type; requires separate staffing |
| actuator-group alert burden | 6.02303 alerts/100 | Same as above; allocate review capacity by component type |
| New-aircraft entity recall | 89.6124% | Tighten manual review during initial fleet entry |
| New-component entity recall | 91.2296% | First-installed components require cold-start monitoring |

Within-group alerts/100 values are descriptive slices and cannot serve as the overall acceptance convention; the formal 5.0 target is an overall metric.

The denominator must be fixed to the **active component window** (unique aircraft x component x 30-cycle window). If the same test set were deduplicated by distinct component instead, the 972 events and 2,968 components would put the oracle lower bound at 32.7 alerts/100, making the budget mathematically unreachable; a component-month denominator would give 11.797. Monitoring dashboards, acceptance sheets, and external materials must therefore all state "denominator = 30-cycle active component windows", otherwise the numbers are not comparable.

# 5. Threshold governance and change control

The threshold is the central switch between operating cost and recall and must be versioned like the model.

| Control item | Rule |
| --- | --- |
| Selection data | Thresholds may be selected only on val, maximizing event recall subject to alerts/100 <= 4.5, with ties resolved toward fewer alerts and a higher threshold |
| Test isolation | The consumed temporal test must not be reused for threshold selection or repeated comparison |
| Change approval | A threshold change requires an impact assessment: expected alert volume, review staffing, and historical true-positive rate |
| Staged release | Pilot in shadow mode or on a single aircraft type first, observe for 2-4 weeks, then release fully |
| Rollback | Retain the previous threshold and model versions; revert with one action if anomalies appear |
| Audit | Record who changed what, when, on what basis, the impact scope, and the approval record |

> **Prohibited actions**
> Do not automatically lower or raise the threshold based on production alert volume; do not tune the threshold directly on missed-detection examples from the current test; do not replace the threshold file without recording a version.

When operators want higher recall in exchange for more alerts, they should create a new candidate threshold and evaluate it on a new time window or new data rather than modifying the frozen results.

# 6. Monitoring

## 6.1 Monitoring layers

| Layer | Monitored object | Frequency | Typical problem |
| --- | --- | --- | --- |
| System layer | Job success rate, latency, backlog, model load failures | Every run | Batch failure causing unscored snapshots |
| Data-quality layer | Field missing rates, out-of-range values, duplicate primary keys, abnormal snapshot time points | Every batch | Upstream data change or collection outage |
| Feature layer | Distribution drift of the 23 features (PSI/KS) | Daily/weekly | Fleet composition or sensor behavior change |
| Scoring layer | Risk-score distribution, alert rate, episode count | Daily | Wrong threshold or model version |
| Effectiveness layer | Event recall, lead time, false-positive rate (delayed labels) | Monthly/quarterly | Degradation of real performance |
| Process layer | Alert review timeliness, false-positive confirmation rate, work-order closure rate | Weekly | Alerts not used effectively |

## 6.2 Core monitoring metrics and recommended warning lines

| Metric | Computation | Recommended warning line | Action |
| --- | --- | --- | --- |
| Batch success rate | Successful batches / planned batches | < 99% | Investigate upstream systems and scheduling immediately |
| Scoring coverage | Scored snapshots / snapshots that should be scored | < 99.5% | Locate unscored records; silent gaps are not allowed |
| Key-field missing rate | Missing share per field | Increase > 10 pp over baseline | Data-quality work order |
| Feature PSI | Computed with bins from the training distribution | > 0.10 watch; > 0.25 action | Feature-level investigation and review |
| Risk-score mean shift | Last 7 days mean vs baseline | Shift > 20% | Check model/threshold version |
| alerts/100 active | episodes / active windows x 100 | > 5.0 for 2 consecutive weeks | Review alert strategy and staffing |
| Event recall (delayed) | Share of confirmed events that had been alerted earlier | < 80% | Start model review |
| Median lead time | Event cycles from first alert | Decrease > 20% over baseline | Investigate feature freshness and data latency |
| Review timeliness | Share of alerts reviewed within the deadline | < 90% | Adjust process and staffing |

The warning lines above are recommended initial values and must be calibrated against real fleet size, review capacity, and data cadence; they must not be presented as validated conclusions.

## 6.3 Delayed labels and effectiveness evaluation

Removal events can be confirmed only after they occur and are written back, so effectiveness monitoring is inherently lagged. Maintain a "pending confirmation queue": keep alerted components under observation until 30 cycles have elapsed, backfill after event confirmation, and compute event recall, false-positive rate, and lead time monthly. Alerts still inside the observation period must not be counted as missed detections.

Event ownership is determined by the **actual event time**. For retrospective evaluation, snapshots whose label window crosses the evaluation boundary should be purged or flagged separately so that removals from the next stage do not label snapshots from the current stage as positive. The offline audit shows 80 train-to-val and 30 val-to-test cross-period links in v3; if production backfill does not handle these boundaries, the same type of overlap will enter online evaluation.

## 6.4 Drift detection method

- Numeric features: PSI as the primary measure, KS as a secondary measure; stratify by component type and aircraft model to avoid false drift caused by fleet composition changes.
- Categorical features: change in category share and share of unseen categories; a rising unknown-category share requires checking upstream encoding.
- Sensor missingness: change in the share of structural missingness patterns, with emphasis on component-related missingness such as `sensor_exhaust_temp`.
- Score drift: risk-score distribution, alert rate, and episode length, compared while bound to the threshold version.
- Concept drift: after labels are backfilled, compare actual event rates with model calibration and run a calibration review when needed.

## 6.5 Empirical drift results (v3)

The results in this section come from the empirical drift monitoring study on v3 data. The study only reads saved data splits and prediction files: it does not retrain, does not re-score, does not compute effectiveness metrics, and does not participate in model or threshold selection. The temporal test is used for distribution monitoring only.

### 6.5.1 Method and conventions

**Reference and monitoring windows.** The reference distribution is the training period (2022-01-01 to 2023-12-31). Monitoring windows are the two baseline stability windows (2022 and 2023), the 2024 Q1 validation window, and the 2024 Q2-Q3 monitoring window.

**PSI.** For numeric features, the 10 quantiles of the reference window define bin boundaries, with missing values in a separate bin:

$$\mathrm{PSI} = \sum_i (p_i^{cur} - p_i^{ref}) \ln \frac{p_i^{cur}}{p_i^{ref}}$$

Probabilities are floored at 0.0001 to avoid division by zero. Categorical features use reference-window category shares, with unseen categories merged into an `other` bin. Decision lines: below 0.10 stable, 0.10-0.25 watch, greater than or equal to 0.25 action.

**KS and score monitoring.** A two-sample KS statistic is also computed for numeric features. Score drift uses the saved `risk_score`, with the validation window as reference, and reports the mean, median, P95, and share of alerting snapshots.

**Structural feature split.** `snapshot_cycle`, `component_age_days`, `component_age_cycles`, `tsr_cycles`, and `operating_hours_since_install` grow mechanically with time and are isolated as the `structural_time` group; the remaining 18 features form the `operational` group. Action-feature counts in the window summary cover operational features only.

### 6.5.2 Window summary

| Window | Snapshots | Operational action features | Operational watch features | Max operational PSI | Max PSI across all features |
| --- | --- | --- | --- | --- | --- |
| 2022 | 108,730 | 0 | 5 | 0.209 (`cumulative_faults`) | 1.415 (`snapshot_cycle`) |
| 2023 | 107,228 | 0 | 3 | 0.164 (`cumulative_faults`) | 2.206 (`component_age_days`) |
| 2024 Q1 (validation window) | 28,256 | 5 | 4 | 0.513 (`cumulative_faults`) | 6.853 (`component_age_days`) |
| 2024 Q2-Q3 (monitoring window) | 55,260 | 7 | 2 | 0.743 (`cumulative_faults`) | 6.909 (`component_age_days`) |

In the baseline stability windows (2022 and 2023) no operational feature showed action-level drift, indicating that the 0.25 action line does not raise false alarms on normal year-over-year variation in the synthetic data. In 2024 the number of operational action features rose from 5 to 7, and the maximum PSI rose from 0.513 to 0.743.

### 6.5.3 Operational feature detail

| Feature | PSI (Q1) | KS (Q1) | PSI (Q2-Q3) | KS (Q2-Q3) | Status | Missing rate Q1 / Q2-Q3 |
| --- | --- | --- | --- | --- | --- | --- |
| `cumulative_faults` | 0.513 | 0.263 | 0.743 | 0.313 | action | 0.0% / 0.0% |
| `sensor_oil_debris` | 0.341 | 0.262 | 0.472 | 0.309 | action | 65.0% / 65.4% |
| `sensor_vibration` | 0.311 | 0.238 | 0.418 | 0.284 | action | 9.5% / 9.8% |
| `sensor_exhaust_temp` | 0.299 | 0.231 | 0.377 | 0.254 | action | 82.4% / 82.5% |
| `removal_count_history` | 0.283 | 0.177 | 0.369 | 0.208 | action | 0.0% / 0.0% |
| `sensor_rpm_deviation` | 0.241 | 0.219 | 0.328 | 0.255 | action | 56.4% / 56.5% |
| `sensor_temp` | 0.209 | 0.199 | 0.276 | 0.229 | action | 8.2% / 9.0% |
| `sensor_pressure` | 0.174 | 0.185 | 0.227 | 0.213 | watch | 43.7% / 43.8% |
| `fault_count_last_30` | 0.119 | 0.156 | 0.155 | 0.177 | watch | 0.0% / 0.0% |
| `fault_severity_last` | 0.044 | - | 0.057 | - | stable | 3.2% / 3.0% |
| `fault_code_last` | 0.003 | - | 0.003 | - | stable | 0.0% / 0.0% |
| `flight_cycles_last_30` | 0.005 | 0.023 | 0.001 | 0.009 | stable | 0.0% / 0.0% |
| `last_maint_cycles_ago` | 0.000 | 0.004 | 0.000 | 0.003 | stable | 0.0% / 0.0% |
| `aircraft_model` | 0.000 | - | 0.000 | - | stable | 0.0% / 0.0% |
| `avg_flight_hours` | 0.000 | 0.003 | 0.000 | 0.003 | stable | 0.0% / 0.0% |
| `operator` | 0.000 | - | 0.000 | - | stable | 0.0% / 0.0% |
| `component_type` | 0.000 | - | 0.000 | - | stable | 0.0% / 0.0% |
| `last_maint_type` | 0.000 | - | 0.000 | - | stable | 0.0% / 0.0% |

![PSI of operational features](assets/mro_drift/drift_psi_val_test.png)

![Operational feature PSI: window comparison](assets/mro_drift/drift_psi_heatmap.png)

The main changes are concentrated in cumulative fault count, oil debris, vibration, exhaust temperature, and removal history. These features relate to component degradation, and their rightward shift is consistent with higher fleet utilization in 2024; because the model already learns degradation signals from these features, scores remain stable.

Missingness is structurally stable: key sensor missing rates change by less than 0.5 percentage points between the validation and monitoring windows (for example, `sensor_exhaust_temp` from 82.36% to 82.46%), so the current PSI is driven mainly by the numeric distribution rather than by missing-rate change.

### 6.5.4 Structural features

| Feature | PSI (Q1) | KS (Q1) | PSI (Q2-Q3) | KS (Q2-Q3) | Status |
| --- | --- | --- | --- | --- | --- |
| `component_age_days` | 6.853 | 0.807 | 6.909 | 0.856 | action |
| `snapshot_cycle` | 3.608 | 0.600 | 4.733 | 0.696 | action |
| `component_age_cycles` | 0.553 | 0.278 | 0.842 | 0.348 | action |
| `operating_hours_since_install` | 0.278 | 0.221 | 0.442 | 0.277 | action |
| `tsr_cycles` | 0.000 | 0.004 | 0.000 | 0.003 | stable |

The large PSI values of structural features are not anomalies: `component_age_days` grows with calendar time and `snapshot_cycle` accumulates with flight cycles, so any cross-year comparison yields high values. Production monitoring should instead use "PSI within the same age stratum" or monitor the age distribution trend directly.

### 6.5.5 Score drift

Using the 28,256 scores from 2024 Q1 as reference, the 55,260 scores from 2024 Q2-Q3 compare as follows:

| Metric | 2024 Q1 (reference) | 2024 Q2-Q3 | Change |
| --- | --- | --- | --- |
| Risk-score mean | 0.0379 | 0.0389 | +2.54% |
| Risk-score median | 0.0143 | 0.0143 | 0.00% |
| Risk-score P95 | 0.2916 | 0.3229 | +10.71% |
| Share of alerting snapshots | 9.21% | 9.25% | +0.05 pp |
| Risk-score PSI | Reference distribution | 0.000987 | - |
| Alert episodes | - | 1,526 | - |

The risk-score PSI of 0.000987 is far below the 0.10 watch line. The mean score rose 2.54% and P95 rose 10.71%, but the median and alert share are essentially unchanged, indicating that the change is concentrated in the high-score tail and has not altered the overall alert volume. The alerting-snapshot share moved from 9.205% to 9.254% between the two windows.

### 6.5.6 Monthly alert burden

| Month | Snapshots | Alert episodes | Active component windows | alerts/100 | Mean score | Score PSI | Within budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01 | 10,763 | 413 | 8,258 | 5.00 | 0.0382 | 0.0000 | **No** |
| 2024-02 | 8,574 | 260 | 7,093 | 3.67 | 0.0388 | 0.0000 | Yes |
| 2024-03 | 8,919 | 237 | 7,447 | 3.18 | 0.0366 | 0.0000 | Yes |
| 2024-04 | 10,460 | 394 | 8,064 | 4.89 | 0.0389 | 0.0015 | Yes |
| 2024-05 | 9,199 | 247 | 7,644 | 3.23 | 0.0396 | 0.0008 | Yes |
| 2024-06 | 8,724 | 224 | 7,212 | 3.11 | 0.0395 | 0.0016 | Yes |
| 2024-07 | 9,121 | 228 | 7,554 | 3.02 | 0.0385 | 0.0007 | Yes |
| 2024-08 | 8,973 | 222 | 7,484 | 2.97 | 0.0379 | 0.0011 | Yes |
| 2024-09 | 8,783 | 211 | 7,318 | 2.88 | 0.0386 | 0.0012 | Yes |

![Monthly alert burden and mean score](assets/mro_drift/drift_monthly_burden.png)

At full precision, 2024-01 is 5.0012 alerts/100, slightly above 5.0, and 2024-04 is 4.8859, close to the limit (the table rounds these to 5.00 and 4.89); the other months lie between 2.88 and 3.67. The overall test metric of 3.50498 meets the target, but a monthly average can hide a single-month spike.

The monthly denominator needs a note: active component windows are deduplicated within each calendar month, so a 30-cycle window crossing a month boundary is counted once in each adjacent month. The sum of the nine monthly denominators (45,276) is therefore larger than the overall deduplicated value (43,538). Monthly alerts/100 is used for trend judgment, not as an acceptance convention; episodes are attributed to the month of first alert, and the nine-month total of 1,526 matches the frozen test result.

### 6.5.7 Interpretation and monitoring recommendations

1. **Layered monitoring.** Use PSI/KS for operational features; use age distributions and trends for structural features, which do not participate in action-level alerting.
2. **Prioritize the top five.** `cumulative_faults`, `sensor_oil_debris`, `sensor_vibration`, `sensor_exhaust_temp`, and `removal_count_history` cross the action line in both 2024 windows and should enter weekly review.
3. **Score stability does not mean review can be skipped.** Score PSI is very small, but monthly alert burden can briefly exceed 5.0; compute alerts/100 weekly and check the threshold and review staffing when the line is exceeded for two consecutive weeks.
4. **Monitor missing rates separately.** `sensor_exhaust_temp` missingness exceeds 82% and `sensor_oil_debris` is about 65%; missingness is currently stable, but a rapid rise should first go through the data-quality process.
5. **Retraining trigger.** Enter retraining review only when operational drift stays at action level for two consecutive months and backfilled labels show falling event recall or shortened lead time; do not repeatedly tune on consumed test results.

### 6.5.8 Limitations of the empirical drift results

1. The degradation mechanism in the synthetic data is fixed, so drift patterns are more regular than in a real fleet; PSI values cannot be extrapolated directly to real operations.
2. PSI is sensitive to bin count and sample size; this study fixes 10 bins and a 0.0001 floor, and comparisons across reports must use the same convention.
3. The split between structural and operational features is empirical; in real data `operating_hours_since_install` may also carry utilization-intensity information.
4. Concept drift and calibration drift were not computed; both require event-label backfill before they can be assessed.
5. No re-scoring was performed; score drift reflects only the distribution of the frozen LightGBM over existing prediction files.

## 6.6 Retraining triggers

| Trigger type | Condition (recommended) | Note |
| --- | --- | --- |
| Performance trigger | Rolling event recall < 80% or median lead time down > 20% | Requires sufficient label volume to avoid small-sample noise |
| Drift trigger | Multiple key features with PSI > 0.25, or persistent score-distribution shift | Investigate data problems first, then decide whether to retrain |
| Operational trigger | alerts/100 > 5.0 for 2 consecutive weeks or review backlog over limit | May be a threshold or process problem; retraining is not necessarily required |
| Data trigger | New aircraft model, new component type, or data-source change | Assess cold start and feature coverage |
| Cadence trigger | Routine review every 6-12 months | Check data and assumptions even when metrics are stable |

## 6.7 Retraining process and gates

![Closed loop of daily monitoring, weekly review, trigger decision, retraining, and release or rollback](assets/mro_prod_design/image2.png)

Figure 2. Monitoring, review, and retraining loop

| Stage | Action | Gate |
| --- | --- | --- |
| 1 Initiation | State the trigger reason, data window, and evaluation plan | Must not reuse a consumed test |
| 2 Data | Fix new training/validation windows; repeat leakage checks | Input hashes stored |
| 3 Training | Reuse or explicitly change features and hyperparameters; record versions | Reproducible experiment |
| 4 Validation | val event recall, alerts/100, group robustness | val event recall >= 85%, alerts/100 <= 4.5 |
| 5 One-shot test | Evaluate only once after pre-registration | Enters release only at >= 80% / <= 5.0 |
| 6 Shadow run | Run in parallel with the live model without changing alerts | Differences and alert volume acceptable |
| 7 Release/rollback | Update model and threshold versions; retain the old version | Approval and rollback plan in place |

Retraining gates follow the formal delivery governance: use only train/val for selection, evaluate the test set once, stop if targets are not met, and do not repeatedly try the test set. Without a new time window or new data, current test results must not be used as a basis for retraining.

## 6.8 Versioning and rollback

- Model version: record the file name, training data window, feature version, threshold, and code commit together.
- Release package: model, preprocessor, threshold, schema, and manifest hashes form an inseparable unit.
- Rollback conditions: abnormal alert volume, rising scoring failure rate, or a significant drop in review confirmation rate.
- Rollback action: switch back to the previous release package and retain logs from the problematic version for analysis.

# 7. Failure modes and degradation

## 7.1 Batch and record-level failures

- Batch-level failure: missing fields, duplicate primary keys, or an invalid snapshot time point leads to rejection of the whole batch, with no partial alerts.
- Record-level failure: an out-of-range value or failed encoding marks that record as unscored and sends it to the data-quality queue; records are never silently dropped.
- Model unavailable: return an explicit error code and stop emitting alerts; falling back to an unversioned temporary model is prohibited.
- Monitoring interruption: alerts may continue on the last frozen version, but drift and burden checks must be completed once monitoring is restored.

## 7.2 New components and cold start

Entity experiments show event recall of 89.6124% for new aircraft and 91.2296% for new components, below the 99.6914% temporal test result. New entities therefore require a more conservative operating strategy.

| Scenario | Handling strategy |
| --- | --- |
| First installation of a new component | Score normally but flag `cold_start`; increase the manual review share for the first 90 days |
| New aircraft model | Check aircraft-model category coverage; handle unknown categories per training rules and monitor separately |
| No historical maintenance record | Treat fields such as `removal_count_history` as missing; do not fabricate history |
| Component transfer | Track cross-aircraft installation by `component_id`; episode ownership follows the current installed aircraft |
| New component type | Flag unknown category if the type is unseen; assess whether training data needs expansion |

## 7.3 Missing data

- Follow training rules: train-median imputation, unknown categories encoded as `Unknown`, and missing indicators generated.
- When an entire sensor group is missing, the explanation output must explicitly state "insufficient data" and must not present a high-confidence conclusion.
- When the missing rate rises abnormally, run the data-quality process first and do not trigger model retraining directly.

## 7.4 Anomalous readings

- -999 and 9999 are treated as error markers and converted to missing; negative sensor values are invalid except for `sensor_rpm_deviation`.
- A single anomalous record does not block the whole batch; when anomalies exceed 5% of a batch, pause alert emission and confirm manually.
- Changes to anomaly-handling rules must be versioned and kept in sync with the offline evaluation conventions.

# 8. Security and operations

| Requirement | Implementation |
| --- | --- |
| Traceable model and threshold | Every scoring run records `model_version`, `threshold`, and `input_hash` |
| Data minimization | Retain only the fields needed for scoring; share only necessary explanations with the work-order side |
| Access control | Model artifacts and threshold files are read-only; changes require an approved account |
| Audit log | Record the changer, time, reason, impact assessment, and approval record |
| Reproducibility | Retain preprocessing, column order, dependency versions, and random seeds |
| Synthetic-data disclosure | External materials must state that the data are synthetic and not for airworthiness or maintenance release |
| No overstatement of results | Monitoring dashboards separate offline frozen metrics from online observed metrics |

# 9. Rollout and validation

## 9.1 Rollout stages

| Stage | Objective | Exit condition |
| --- | --- | --- |
| Stage 0: Offline freeze | Complete formal evaluation and reporting | Done; one-shot test evaluation passed |
| Stage 1: Shadow run | Connect batch scoring without generating real work orders | Scoring coverage and alert volume stable for 4 consecutive weeks |
| Stage 2: Pilot | Connect a single aircraft model or component type to the review workflow | Review timeliness >= 90%, false positives explainable |
| Stage 3: Expansion | Extend to the main fleet | Alert budget matches staffing; no major data incidents |
| Stage 4: Steady state | Monitoring, review, and retraining loop | Retraining gates and rollback drills passed |

Before Stage 1, the following must be complete: interface integration testing, feature-consistency validation, threshold and model version freezing, and confirmation of monitoring dashboards and the alert budget. Full rollout at Stage 2 is not recommended because new-entity recall is below the offline time-split results.

## 9.2 Design conclusion

> **Design conclusion**
> This design provides a deployment, alerting, monitoring, and retraining approach consistent with the existing frozen model: batch scoring as the primary mode, risk scores merged into episodes after a 30-cycle cooldown and then routed to manual review; versioned governance of thresholds and models; monitoring covering data quality, feature drift, alert burden, and delayed-label performance; and retraining that follows the val gate and one-shot test rules.

Before implementation, interface integration, feature-consistency validation, calibration of monitoring warning lines, and pilot validation of the review workflow must be completed. The current offline results come only from synthetic data and do not constitute a real operational commitment.

# 10. Limitations

- Synthetic-data limitation: degradation mechanisms and event rates are simulated, so production performance cannot be extrapolated from offline metrics.
- Unvalidated monitoring thresholds: PSI, shift ratios, and other warning lines are common industry starting points and must be calibrated on real data.
- Label lag: effectiveness monitoring depends on removal write-back, so only proxy metrics are available in the short term.
- Within-group differences: valve and actuator alert burdens are higher than the overall figure, so review staffing must be allocated by component type.
- New-entity generalization: recall drops for new aircraft and new components, requiring stricter manual review during cold start.
- Explanation boundary: feature contributions are not causal and cannot be the sole basis for a maintenance decision.
- Not implemented: this is a design document and contains no real deployment, load-testing, or security-certification results.
- Drift-specific caveats (fixed synthetic degradation mechanism, PSI sensitivity to binning and sample size, empirical structural/operational split, no concept or calibration drift, no re-scoring) are listed in Section 6.5.8.

# 11. Reproduction and artifacts

## 11.1 Reproduction commands

The drift study is reproducible with read-only commands; the sealed test stage must not be rerun:

```bash
python src/drift_monitor.py
python -m pytest tests -q
```

## 11.2 Run parameters and citations

| Parameter | Value |
| --- | --- |
| Formal model | LightGBM / `lgbm_15_leaves` |
| Frozen threshold | `0.05461588446029929` |
| Label window | Next 30 flight cycles |
| Alert cooldown | 30 cycles for the same component |
| val threshold search cap | alerts/100 <= 4.5 |
| Formal acceptance targets | Event recall >= 80%; alerts/100 <= 5.0 |
| Frozen test result | 969/972 = 99.6914%; 3.50498 alerts/100 |
| Model artifacts | `models/lightgbm.joblib`; `models/preprocessors.joblib`; `models/frozen_thresholds.json` |
| Governance manifest | `models/test_evaluation_manifest.json` |
| Metric reports | `reports/val_metrics.json`; `reports/temporal_test_metrics.json`; `reports/entity_robustness_metrics.json` |
| Formal report | `docs/MRO_Model_Formal_Delivery_Report.md` |

## 11.3 Artifacts

| Artifact | Path |
| --- | --- |
| Drift monitoring module | `src/drift_monitor.py` |
| Drift configuration | `configs/drift_monitoring.yaml` |
| Drift metrics | `reports/drift_metrics.json` |
| Feature-level detail | `reports/drift_feature_psi.csv` |
| Window summary | `reports/drift_window_summary.csv` |
| Unit tests | `tests/test_drift_monitor.py` (6 tests) |
| Figures | `docs/assets/mro_drift/` |

This document is a production design annex and does not replace the formal delivery report; where the two conflict, the frozen artifacts and the formal delivery report prevail.
