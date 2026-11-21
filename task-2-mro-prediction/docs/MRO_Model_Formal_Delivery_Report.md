# MRO Component Unscheduled Removal Risk Prediction Model

## Formal Delivery Report

Frozen temporal test evaluation with supplementary KNN / Random Forest validation experiments

| Item | Content |
| --- | --- |
| Data version | v3 (synthetic data) |
| Formal model | LightGBM / `lgbm_15_leaves` |
| Frozen evaluation status | completed; the temporal test has been completed and was not rerun |
| Supplementary experiments | KNN, Random Forest; train/val only, `temporal_test_used=false` |
| Delivery date | 2026-11-21 |
| Document status | Formal delivery version (detailed metrics, robustness, error samples, and governance appendix) |

> **Applicability statement**
> All results in this report come from synthetic data and are intended for method validation and delivery review. The metrics must not be interpreted as real airline operating performance and must not be used directly for real airworthiness, maintenance release, or maintenance decisions.

# Executive Summary

> **Formal delivery conclusion: meets expectations**
> The frozen LightGBM model detected 969/972 independent events on the one-shot temporal test, an event recall of 99.6914%; the alert burden was 3.50498 alerts/100 active windows. The two formal targets were event recall >=80% and alert burden <=5.0, and both passed.

Key results at a glance

| Metric | LightGBM frozen test | Target / reference | Verdict |
| --- | --- | --- | --- |
| Independent event recall | 969/972 = 99.6914% | >=80% | Passed |
| Alert burden | 1,526 episodes / 43,538 active = 3.50498 | <=5.0 | Passed |
| First-warning lead time | median 17.7815 cycles; P25-P75: 11.0423-23.9046 | descriptive | - |
| Snapshot P / R / F1 | 0.2083 / 0.9744 / 0.3432 | descriptive | - |
| Random Forest supplementary val | 548/548; 4.30886 alerts/100; median lead 15.4160 | val-only exploration | no test evidence |
| KNN supplementary val | 517/548; 4.48991 alerts/100; median lead 13.8539 | val-only exploration | no recall gain |

- The formal delivery model remains LightGBM. It simultaneously maintains the highest event recall, a lower alert burden, and a longer first-warning lead time on the frozen test.
- On val, Random Forest matches LightGBM at 100% event recall and has a slightly higher PR-AUC (0.335240) and F1 (0.372587); however, it produces 42 more alert episodes, 0.190097 more alerts/100, and a median lead time that is 0.8037 cycles shorter.
- KNN achieves 94.3431% val event recall, identical to Logistic Regression, with an alert burden close to the 4.5 ceiling; it offers no delivery advantage over the formal model.
- Neither KNN nor Random Forest read or evaluated the temporal test; their `temporal_test_metrics` are null. They must not be used to rewrite the frozen test conclusion or to claim superiority over the formal model.
- Entity splits show that LightGBM still outperforms Logistic Regression on new aircraft and new components, but recall drops relative to the temporal test; this should be a key monitoring item for real-world generalization.

# 1. Scope, Evaluation Tracks, and Governance

## 1.1 Task definition

The prediction target is whether an unscheduled removal occurs within the next 30 flight cycles starting from each component snapshot. An event that occurs exactly in the 30th future cycle counts as a positive; right-censored snapshots with an incomplete future window are not treated as negatives. The core of the delivery is not single-snapshot classification but usability measured jointly by independent removal event recall and the alert burden after cooldown merging.

## 1.2 Evaluation tracks and evidence boundaries

| Track | Data and models | Purpose | Temporal test usage |
| --- | --- | --- | --- |
| Formal training and selection | temporal train/val; LR, LightGBM | candidate selection, threshold search, internal safety gate | not used |
| Formal frozen evaluation | frozen LR, LightGBM, and thresholds | one-shot final acceptance | completed once; status completed |
| Entity robustness | independent by_aircraft / by_component training with internal val | assess generalization to new aircraft and new components | temporal test not rerun |
| Grouping and samples | saved test risk scores and episode IDs | descriptive robustness and error analysis | read-only saved artifacts, no rescoring |
| KNN / RF supplementary | temporal train/val only | supplementary model-family exploration | not read, not evaluated; metrics are null |

Note: the supplementary experiment configuration sets `temporal_test_used=false`; report generation only reads saved JSON/CSV files, images, and model metadata.

## 1.3 One-shot test freeze requirements

- One-shot test manifest: `models/test_evaluation_manifest.json`, status `completed`; started at 2026-11-20T21:38:41.571536+00:00 and completed at 2026-11-20T21:38:42.682974+00:00.
- The temporal test is evaluated only once, after the LightGBM val internal safety gate passes. Retuning, changing thresholds, changing cooldown rules, or repeated evaluation based on test, group, or sample results is prohibited.
- The frozen configuration SHA-256 is `dcb063ae1b926e8967f9e3e0670c2bbac3028b18a506ff23b6054c571a32f66d`; the full input, model, and code hashes are listed in Section 10.
- All val/test figures in this report come directly from frozen metric files; report generation and unit tests do not constitute a new temporal test evaluation.

## 1.4 Data-level and model-level acceptance

The dataset layer and the model layer are separate acceptance stages. The v3 generation round established the data layer: approximately 300,000 rows, a stable label rate, same-split scoring opportunities for all formal events, adjacent snapshot gaps of no more than 30 cycles, and all data checks and tests passing. That round did not train or evaluate the formal LightGBM / Logistic Regression models, so the two model targets (event recall >=80% and <=5 alert episodes per 100 active component windows) remained unverified at that point and could not be inferred from data coverage alone. This report closes that gap with the frozen train/val/test evidence described below.

## 1.5 Related documentation

This report is one of a set of English delivery documents and should be read together with:

- `README.md` - repository layout, environment, reproduction commands, and current status.
- `docs/MRO_Production_Deployment_Design.md` - deployment architecture, interfaces, alert workflow, threshold governance, monitoring, retraining, and cold-start handling.
- `docs/data_dictionary.md` - field definitions, label definition, time boundaries, and missingness conventions.

# 2. Data, Splits, and Preprocessing

> **Data nature**
> v3 is a synthetic dataset. Its results validate the modeling, evaluation, and delivery process; they do not represent real fleet performance and do not constitute a basis for airworthiness or maintenance release.

## 2.1 Actual dataset scale

| Item | Value | Item | Value |
| --- | --- | --- | --- |
| Snapshot records | 299,474 | Positive snapshots | 5,997 (2.0025%) |
| Fields | 28 | Candidate model fields | 23 |
| Aircraft | 200 | Visible component instances | 7,367 |
| Simulated component instances | 10,820 | Initial installed component instances | 2,000 |
| Unscheduled removal events (total) | 8,823 | Formal-period removal events | 5,371 |
| Snapshot-event links | 11,368 | Same-split scoring coverage of formal events | 100.00% |
| Snapshot date range | 2022-01-01 to 2024-09-30 | Random seed | 42 |
| Warm-up period | 365 days | Label observation extension | 120 days |
| Base snapshot interval | 26.3369 cycles | Maximum actual snapshot interval | 29.9998 cycles |

The 28 fields match `configs/schema.yaml`. Each aircraft is initialized with 10 monitoring slots; the 7,367 visible components are the unique `component_id` values in formal snapshots and include replacement parts. The 10,820 simulated instances include warm-up instances and instances that never enter formal snapshots. The 8,823 events cover the warm-up period, the formal snapshot period, and the label observation extension; 5,371 of them fall inside the formal train/val/test snapshot period. The prediction window is 30 future flight cycles, not 30 days.

## 2.2 Label rate and temporal splits

Annual positive rates are 1.9939% for 2022, 1.9799% for 2023, and 2.0427% for 2024 through September.

| Split | Date range | Snapshots | Positive snapshots | Independent actual events | Positive rate | Component instances |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2022-01-01 to 2023-12-31 | 215,958 | 4,291 | 3,851 | 1.98696% | 5,846 |
| val | 2024-01-01 to 2024-03-31 | 28,256 | 613 | 548 | 2.16945% | 2,543 |
| test | 2024-04-01 to 2024-09-30 | 55,260 | 1,093 | 972 | 1.97792% | 2,968 |

All 5,371 formal events have a valid same-split scoring opportunity. The event denominator and the positive-snapshot denominator are kept strictly separate: for the test split the formal event denominator is 972, not 1,093. All three periods fall within the 1.8%-2.2% generation acceptance band. A stable label rate only means the synthetic event intensity was calibrated; it does not mean model metrics have passed.

## 2.3 Entity-group splits

| Split | Rows | Positive snapshots | Positive rate | Aircraft | Component instances |
| --- | --- | --- | --- | --- | --- |
| Aircraft Train | 238,224 | 4,714 | 1.9788% | 160 | 5,815 |
| Aircraft Test | 61,250 | 1,283 | 2.0947% | 40 | 1,552 |
| Component Train | 239,150 | 4,811 | 2.0117% | 200 | 5,894 |
| Component Test | 60,324 | 1,186 | 1.9660% | 200 | 1,473 |

Aircraft entities do not overlap between the aircraft train and test splits, and component instances do not overlap between the component train and test splits.

## 2.4 v3 adjustments and rationale

**Snapshot densification from 93,362 to 299,474 rows.** The v2 event audit found that only 342 of the 909 actual test removals had scorable snapshots, a coverage of 37.62%. Even a model capable of recognizing covered events could not compensate for the majority of events having no scoring opportunity at all. The core v3 goal was therefore to repair observability rather than to keep tuning models or manually add positives. A smoke run on 12 aircraft preceded a single full generation. No records, positives, or labels were copied or hand-edited to reach the target scale.

**Snapshot intervals filled using actual flight cycles.** The base sampling interval was computed automatically from a 260,000-row target as approximately 26.3369 cycles. The generator sets anchors at the first simulated observation after component installation, the formal data start, the val and test starts, and each actual unscheduled removal boundary. Because the raw timeline is aggregated by day, direct `searchsorted` could overshoot the target interval when a day contained multiple flight cycles. In v3, after candidate snapshots were generated, additional points were inserted using the unrounded actual cycle sequence until every adjacent snapshot gap was no more than 30 cycles. The final maximum actual interval was 29.9998 cycles with zero violations.

**Snapshot-event links and a same-split coverage gate.** `snapshot_event_links.csv` records each scoring opportunity. A link requires the same `component_id` and an unrounded cycle difference satisfying `0 <= event_cycle - snapshot_cycle <= 30`. A value of 0 denotes a snapshot on the same observation day/cycle; no additional minimum lead in cycles was imposed in this round.

| Temporal split | Actual events | Scoreable events within the same split | Coverage |
| --- | --- | --- | --- |
| Train | 3,851 | 3,851 | 100% |
| Val | 548 | 548 | 100% |
| Test | 972 | 972 | 100% |
| Total | 5,371 | 5,371 | 100% |

Links also record `event_temporal_split`, `snapshot_temporal_split`, and `same_temporal_split`. Formal evaluation may only use links whose event and snapshot belong to the same split, preventing a val-end snapshot alert from hitting a test event or a test snapshot from hitting a val event in reverse. Events in the warm-up and label observation extension periods are excluded from the formal event denominator.

Cross-split links do not enter independent event recall, but because labels are computed over the next 30 cycles, the corresponding snapshots can still be labeled positive. This label-window boundary overlap is quantified in Section 2.7. Reproduction uses `src/label_boundary_audit.py` and `reports/label_boundary_audit.json`.

**Record IDs based on unrounded cycles.** Two nearby snapshots can share the same integer `snapshot_cycle`. v3 generates `record_id` from the unrounded cycle at thousandth precision, avoiding deletion of valid snapshots merely because a display field was rounded. All 299,474 record IDs are currently unique.

**Steady-state and risk mechanisms retained from v2.** Unchanged mechanisms are the 365-day warm-up, forward simulation of replacement parts with 2,000 initial slots mapping to later physical instances, a continuous random-walk degradation perturbation, the `clip(D, 0, 6)^2` unscheduled-removal risk term, and calibration of the hazard intercept by period instead of editing labels directly. The calibrated intercepts are:

| Period | alpha |
| --- | --- |
| 2022 | -29.50000 |
| 2023 | -25.00000 |
| Val | -22.00000 |
| Test | -22.75000 |

## 2.5 Features and leakage prevention

The model uses 23 candidate fields, grouped by business meaning:

- Time and categorical: `snapshot_cycle`, `aircraft_model`, `component_type`, `operator`.
- Life and utilization: `component_age_cycles`, `component_age_days`, `tsr_cycles`, `operating_hours_since_install`, `flight_cycles_last_30`, `avg_flight_hours`.
- Sensors: `sensor_vibration`, `sensor_temp`, `sensor_pressure`, `sensor_oil_debris`, `sensor_rpm_deviation`, `sensor_exhaust_temp`.
- Fault and maintenance history: `fault_code_last`, `fault_count_last_30`, `fault_severity_last`, `last_maint_cycles_ago`, `last_maint_type`, `removal_count_history`, `cumulative_faults`.

`record_id`, `aircraft_id`, `component_id`, `snapshot_date`, `target`, and audit link fields are not model features. All features may only use information available at or before the snapshot time; labels use only the future window after the snapshot. Generator-side latent degradation values, true hazard, future maintenance, and future faults are excluded. All fitting operations - missing-value statistics, categorical vocabularies and encodings, numeric scaling parameters, imbalance weights, any feature selection or target encoding, and model parameters - are fitted on train only; val is used only for the predefined selection flow, and test participates in no fitting. Processing the full dataset before splitting is prohibited.

## 2.6 Missingness, error markers, and preprocessing

- `-999` and `9999` are converted to missing for the six sensors, with missing indicators generated; no clipping is enabled.
- Negative values for `sensor_vibration`, `sensor_temp`, `sensor_pressure`, `sensor_oil_debris`, and `sensor_exhaust_temp` are treated as invalid; `sensor_rpm_deviation=-1` is a legitimate signed reading and is retained.
- Numeric missing values for Logistic Regression are filled with train medians; categorical missing values use `Unknown`, and unknown categories are ignored at prediction time. The preprocessor is fitted on training data only.
- LightGBM retains numeric `NaN` and lets the model choose split directions; `fault_code_last=NONE` is a valid business category, not missingness.
- `sensor_exhaust_temp` and other fields have component-type-related structural missingness; missing-pattern grouping results are descriptive only and do not feed back into the model.
- Measurement spikes are retained by default; outliers are handled by invalid-value rules rather than statistical winsorization, which is not enabled.

| Sensor | Missing rate among applicable rows | Full-table missing rate |
| --- | --- | --- |
| `sensor_vibration` | 8.76% | 8.76% |
| `sensor_temp` | 7.69% | 7.69% |
| `sensor_pressure` | 9.76% | 43.00% |
| `sensor_oil_debris` | 12.65% | 64.67% |
| `sensor_rpm_deviation` | 10.78% | 55.94% |
| `sensor_exhaust_temp` | 11.66% | 82.15% |

The applicable-row missing rates come from a mixture of MCAR, MAR, and MNAR mechanisms. Full-table rates additionally include structural missingness where a sensor does not apply to a component type. The construction plan's "5%-15% per field" refers to random missingness within applicable rows; the actual 7.69%-12.65% range remains inside that plan band. The full-table maximum of 82.15% is structural and must not be described as "overall missingness of 5%-15%". Reports and modeling analyses must state which convention is used.

## 2.7 Label boundary audit

The temporal split is a hard date cut, but the label window spans 30 future flight cycles, so labels near a boundary can reference a removal in the next period. The read-only audit results are below (source `reports/label_boundary_audit.json`, reproduction command `python src/label_boundary_audit.py`):

| Split | Positives | Same-split links | Cross-split-only links | Share |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,291 | 4,211 | 80 (event in val) | 1.8644% |
| Val | 613 | 583 | 30 (event in test) | 4.8940% |
| Test | 1,093 | 1,070 | 23 (event in observation period) | 2.1043% |

There are 11,368 snapshot-event links in total, of which 133 are cross-split (train to val 80, val to test 30, test to observation 23).

This is not feature leakage: all features are no later than the snapshot time, and the preprocessor is fitted on train only; independent event recall counts only same-split links, so the event denominator is not contaminated. However, about 4.89% of val positives are driven by test-period events, so threshold selection may carry a slight optimistic bias. The frozen temporal test has been consumed and this version does not retrain retroactively; it only quantifies and discloses the effect. The next data generation should purge samples with overlapping label windows at split boundaries.

# 3. Models, Thresholds, and Strict Metric Definitions

## 3.1 Model configurations

| Model | Training data / sampling | Key parameters | Role |
| --- | --- | --- | --- |
| Logistic Regression | full train; `class_weight=balanced` | `solver=liblinear`; `C=1`; `max_iter=2000`; `random_state=42` | formal baseline |
| LightGBM | full train; `scale_pos_weight=49.328129` | `lgbm_15_leaves`; `num_leaves=15`; `min_child_samples=100`; `learning_rate=0.05`; `n_estimators=2000`; `reg_lambda=1`; `best_iteration=2000` | formal delivery |
| Random Forest | full train (215,958 rows) | 300 trees; `max_depth=16`; `min_samples_leaf=5`; `max_features=sqrt`; `class_weight=balanced_subsample` | val-only supplementary |
| KNN | all 4,291 positives plus an equal number of randomly sampled negatives, 8,582 rows total | `n_neighbors=51`; distance weighting; brute force; Minkowski `p=2`; standardized without centering | val-only supplementary |

All four models use random seed 42. KNN/RF use train-median imputation, numeric standardization, and one-hot categorical encoding; their configurations and outputs are independent of the frozen models. Logistic Regression uses One-Hot encoding with `handle_unknown="ignore"`, median-filled numeric fields, sparse-safe scaling without mean centering, and a saved full sklearn Pipeline. LightGBM uses a train-fitted categorical vocabulary, maps unseen categories to an explicit unknown category at inference, preserves the original field order, and uses no target encoding.

Candidate grids were deliberately small. Logistic Regression compared `C` values of 0.1, 1.0, and 10.0 and selected `lr_c_1`. LightGBM compared 31-leaf and 15-leaf candidates and selected `lgbm_15_leaves`; the 15-leaf configuration used 100 minimum child samples and 2,000 boosting rounds with `best_iteration=2000`. Model selection considered val event recall under the alert constraint first, then alert volume and PR-AUC; ROC-AUC alone was never the selection criterion.

## 3.2 Class imbalance and auxiliary metrics

- Logistic Regression uses `class_weight=balanced`.
- LightGBM uses `scale_pos_weight = negatives / positives`, approximately 49.33 (recorded as 49.328129).
- Evaluation does not center on Accuracy. The core metrics are independent event recall, PR-AUC, and alert burden; ROC-AUC, snapshot recall/precision/F1, snapshot alert rate, episode count, active-window count, detected/actual events, and the per-event first-warning distribution are reported alongside but do not replace the formal metrics.

## 3.3 Threshold selection method

Thresholds are searched on val only. Deterministic quantile candidates are generated from val scores, including the minimum, maximum, and boundary values; every candidate is scored for event recall, episode count, and alert intensity; a local refinement is then performed around the best feasible region so that coarse quantiles do not miss feasible points. All candidate results are saved, not just the best point. The search never uses the test score distribution.

Selection rule for each model:

```text
first constraint: val alerts_per_100_active <= 4.5
first objective:  maximize val event recall
tie-break:        fewer alerts, then higher threshold
auxiliary:        val PR-AUC
```

The recorded implementation scanned 100 quantile points and then performed a 40-point refinement near the candidates. The LightGBM internal safety gate requires val event recall >=85% and alerts/100 <=4.5. The formal test targets are event recall >=80% and alerts/100 <=5.0. If no threshold satisfies the val constraint, val results are saved and the process stops; test must not be inspected to find a better model or threshold.

Frozen thresholds are LightGBM `0.05461588446029929` and Logistic Regression `0.7953625805980529` (displayed as 0.05461588 and 0.79536258 in the metric tables). Random Forest and KNN each selected thresholds independently on val, at 0.1298756655 and 0.7688021824 respectively, but neither entered formal delivery nor read the temporal test.

## 3.4 Alert episode construction

Only snapshots with a score greater than or equal to the threshold are processed:

1. Group by `component_id` and order by aircraft cycle.
2. At the start of each evaluation set, no component has alert history.
3. If the current high-score snapshot is no more than 30 aircraft cycles from the component's previous high-score alert, merge it into the current episode and update the last alert position.
4. If more than 30 cycles have elapsed, create a new episode.
5. Episodes may span consecutive 30-cycle windows and are not automatically recounted at window boundaries.
6. Cooldown state must be reset between val and test.

All inputs are deterministically ordered by `component_id`, `snapshot_cycle`, `snapshot_date`, and `record_id` so that ties cannot make results irreproducible.

## 3.5 Strict metric definitions

| Metric | Definition and convention |
| --- | --- |
| ROC-AUC / PR-AUC | Ranking metrics based on snapshot risk scores; PR-AUC focuses on positive-class identification at the roughly 2% positive rate. |
| Snapshot Precision / Recall / F1 | Computed per snapshot at the frozen threshold; cannot replace independent event metrics. |
| Active component window | Count of unique components within an aircraft and `floor(snapshot_cycle/30)` window; used as the alert-burden denominator. |
| Alert episode | Adjacent high-score snapshots of the same component separated by <=30 cycles are merged into one episode, avoiding repeated alerts from consecutive high scores. |
| Alerts / 100 active | 100 x alert episode count / active component window count. |
| Independent event recall | An event is detected only if a same-split, 0-30 cycle valid snapshot-event link enters a saved episode; multiple links to the same event count once. |
| First warning cycles | Event cycle minus the snapshot cycle of the earliest matching episode for that event; computed only for detected events. |
| Small-sample instability | Fewer than 100 snapshots or fewer than 30 actual events; a descriptive stability flag only, not used for selection or tuning. |

A zero denominator is not reported as zero alerts: the evaluator raises an error and returns a non-evaluable state. Windows at split boundaries may be incomplete and are never backfilled with records from another split.

## 3.6 Denominator convention for alert burden

The task requires "no more than 5 alerts per 100 active components". This report uses **active component window** (unique aircraft x component x 30-cycle window) as the exposure-normalized denominator, consistent with the 30-cycle alert merging rule. The three denominators on test compare as follows:

| Denominator | Count | Alerts/100 for the same 1,526 episodes |
| --- | ---: | ---: |
| Active component windows (this report) | 43,538 | 3.50498 |
| Component-months | 12,936 | 11.797 |
| Distinct components | 2,968 | 51.415 |

Rationale: the test split has 972 actual events and 2,968 distinct components. Even an oracle that detects every event would score at least 32.7 alerts/100 under component de-duplication, making <=5 mathematically unreachable. "Per 100 active components" can therefore only be interpreted as normalization by exposure windows; otherwise the budget is mathematically infeasible. Component-months are not equivalent to 30-cycle windows (one component-month can span multiple 30-cycle windows), so they are not the primary convention. Any cross-report comparison must confirm that the same denominator is used.

# 4. Evaluator Validation and Reproducibility

## 4.1 Saved sanity checks

| Check | Actual result | Conclusion |
| --- | --- | --- |
| All-zero scores | event recall 0%; 0 episodes; 0 alerts/100 | a zero-score model produces no event detections |
| Oracle linked alerts | 548/548; event recall 100%; 2.48031 alerts/100 | event linking, de-duplication, and lead-time aggregation can hit all events |
| Shuffled labels | val ROC-AUC 0.509128; event recall 0% | passed: below the ROC 0.65 and recall 20% limits |
| Strongest single field | `sensor_temp`; val ROC-AUC 0.947106 | passed: below the 0.98 limit |
| Simple baseline | event recall 38.3212%; 4.49896 alerts/100 | the formal model is clearly better than a simple rule |
| Overall flag | `sanity_checks.passed=true` | passed |

## 4.2 Automated tests

The full pytest suite was executed for the current snapshot: 30 tests passed, with 4 LightGBM dependency deprecation warnings that do not affect results. Of these, 14 tests directly cover the evaluator and modeling pipeline: active-window de-duplication within an aircraft window; episode cooling across windows and a new episode after 30 cycles; cooldown reset per evaluation set; multiple links detecting an event once with earliest lead time; rejection of cross-split links; rejection of links outside 30 cycles; an explicit error on a zero active-window denominator; zero scores yielding zero event recall; oracle linked snapshot alerts recalling all events; error markers, missing indicators, and signed RPM; train-only medians with unknown-category prediction; column-order and joblib round-trip stability; repeated LightGBM training reproducibility; and contribution reconstruction of raw scores for both models. New KNN/RF configuration and val-only governance checks were added on top. Unit tests are not equivalent to a temporal test and did not re-evaluate the frozen test set.

## 4.3 Reproducibility and frozen inputs

With the same inputs and configuration, the following are fixed: data reading and sorting; categorical vocabulary order; One-Hot feature order; model random seeds; threshold candidates and search order; tie handling; and output file naming. Metadata records the actual versions of Python, NumPy, pandas, scikit-learn, and LightGBM. If any input file changes before training, the process must stop and the experimental baseline must be re-confirmed rather than silently reusing old results.

# 5. Validation Results and Model Selection

All four models use the same temporal val: 28,256 snapshots, 613 positive snapshots, 548 independent actual events, and 22,094 active component windows. KNN/RF are supplementary explorations; the formal LightGBM/LR results remain frozen.

Table 5-1. Validation ranking and snapshot metrics

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Snapshot alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM (formal) | 0.05461588 | 0.972393 | 0.314403 | 0.226451 | 0.960848 | 0.366521 | 9.2051% |
| Logistic Regression | 0.79536258 | 0.958054 | 0.240188 | 0.231862 | 0.745514 | 0.353715 | 6.9755% |
| Random Forest (supplementary) | 0.12987567 | 0.974990 | 0.335240 | 0.232064 | 0.944535 | 0.372587 | 8.8300% |
| KNN (supplementary) | 0.76880218 | 0.960265 | 0.270706 | 0.210213 | 0.805873 | 0.333446 | 8.3168% |

Table 5-2. Validation event and alert metrics

| Model | Episodes | Active | Alerts/100 | Detected/actual | Event recall | Median lead |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM (formal) | 910 | 22,094 | 4.11877 | 548/548 | 100.0000% | 16.2197 cycles |
| Logistic Regression | 988 | 22,094 | 4.47180 | 517/548 | 94.3431% | 12.4543 cycles |
| Random Forest (supplementary) | 952 | 22,094 | 4.30886 | 548/548 | 100.0000% | 15.4160 cycles |
| KNN (supplementary) | 992 | 22,094 | 4.48991 | 517/548 | 94.3431% | 13.8539 cycles |

Table 5-3. Validation first-warning lead-time distribution (cycles)

| Model | Detected events | Min | P25 | Median | P75 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM (formal) | 548 | 0.0000 | 9.2904 | 16.2197 | 23.0578 | 29.9689 |
| Logistic Regression | 517 | 0.0000 | 4.1796 | 12.4543 | 21.9569 | 29.9689 |
| Random Forest (supplementary) | 548 | 0.0000 | 8.7752 | 15.4160 | 22.7592 | 29.9689 |
| KNN (supplementary) | 517 | 0.0000 | 6.6182 | 13.8539 | 22.5257 | 29.9689 |

Min=0 means that some events received a valid warning only in the same cycle in which the event occurred; the distribution covers detected events only and must not treat undetected events as zero lead time.

Threshold-selection results for the formal candidates are:

| Model | Candidate | Curve points | Threshold | ROC-AUC | PR-AUC | Event recall | Alerts/100 | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | `lr_c_0_1` | 144 | 0.79893100 | 0.957997 | 0.240610 | 94.1606% | 4.46728 | no |
| Logistic Regression | `lr_c_1` | 144 | 0.79536258 | 0.958054 | 0.240188 | 94.3431% | 4.47180 | yes |
| Logistic Regression | `lr_c_10` | 144 | 0.80129101 | 0.958072 | 0.240084 | 93.7956% | 4.40391 | no |
| LightGBM | `lgbm_31_leaves` | 65 | 0.03154515 | 0.973735 | 0.329013 | 99.8175% | 4.36770 | no |
| LightGBM | `lgbm_15_leaves` | 64 | 0.05461588 | 0.972393 | 0.314403 | 100.0000% | 4.11877 | yes |

## 5.1 LightGBM internal safety gate

Val detected 548/548, event recall 100.0000% against a >=85% requirement, and alerts/100 active = 4.11877 against a <=4.5 requirement. The recall gate is `True`, the alert gate is `True`, and the saved `test_permitted=True`; the temporal test was therefore permitted exactly once.

## 5.2 Random Forest versus LightGBM on val

| Comparison | LightGBM | Random Forest | RF - LightGBM | Interpretation |
| --- | --- | --- | --- | --- |
| ROC-AUC | 0.972393 | 0.974990 | +0.002597 | RF slightly higher |
| PR-AUC | 0.314403 | 0.335240 | +0.020837 | RF slightly higher |
| Snapshot F1 | 0.366521 | 0.372587 | +0.006065 | RF slightly higher |
| Event recall | 100.0000% | 100.0000% | +0.0000 pp | both detect 548/548 |
| Alert episodes | 910 | 952 | +42 | RF issues more alerts |
| Alerts/100 | 4.11877 | 4.30886 | +0.19010 | higher alert burden for RF |
| Median lead | 16.2197 | 15.4160 | -0.8037 | shorter median lead for RF |

Conclusion: Random Forest is a competitive follow-up candidate, but the current difference is only a descriptive comparison on a single split. No significance test was performed and there is no temporal test evidence. Its value must be weighed together with the higher alert burden and shorter lead time; PR-AUC or F1 alone must not be used to replace the formal LightGBM.

## 5.3 KNN interpretation

KNN is trained on 8,582 rows after balanced sampling and achieves val ROC-AUC 0.960265 and PR-AUC 0.270706, but only 94.3431% event recall (517/548), with alerts/100=4.48991 close to the 4.5 ceiling. Its median lead is 13.8539 cycles. Compared with the 100% event recall of LightGBM/RF, KNN offers no delivery advantage; compared with LR it improves PR-AUC, but detects the same 517 events and has a lower snapshot F1.

# 6. One-Shot Temporal Test

> **Evidence boundary**
> This chapter transcribes only `reports/temporal_test_metrics.json` and predictions saved in the same batch. KNN and Random Forest have no temporal test metrics and did not participate in this chapter; val exploration results must not be entered here.

Table 6-1. Frozen test ranking and snapshot metrics

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Snapshot alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.05461588 | 0.975844 | 0.340406 | 0.208252 | 0.974382 | 0.343161 | 9.2544% |
| Logistic Regression | 0.79536258 | 0.961336 | 0.243510 | 0.193005 | 0.817932 | 0.312314 | 8.3822% |

Table 6-2. Frozen test event and alert metrics

| Model | Episodes | Active | Alerts/100 | Detected/actual | Event recall | Median lead |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 1,526 | 43,538 | 3.50498 | 969/972 | 99.6914% | 17.7815 cycles |
| Logistic Regression | 1,821 | 43,538 | 4.18255 | 937/972 | 96.3992% | 15.4215 cycles |

Both models share the same active-window denominator of 43,538; the event denominator is the 972 independent test events, not the 1,093 positive snapshots.

Table 6-3. Frozen test first-warning lead-time distribution (cycles)

| Model | Detected events | Min | P25 | Median | P75 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 969 | 0.0000 | 11.0423 | 17.7815 | 23.9046 | 29.9799 |
| Logistic Regression | 937 | 0.0000 | 7.9867 | 15.4215 | 23.0077 | 29.9751 |

## 6.1 Formal target acceptance

| Acceptance item | Target | LightGBM actual | Margin | Verdict |
| --- | --- | --- | --- | --- |
| Event recall | >=80% | 969/972 = 99.6914% | 19.6914 pp above | passed |
| Alert burden | <=5.0 | 3.50498 alerts/100 | 1.49502 below | passed |
| Overall | both must pass | meets expectations | `recall_passed=true`; `alerts_passed=true` | meets expectations |

If only one of the two targets is met, the overall verdict would still be "does not meet formal expectations". Both were met simultaneously.

## 6.2 Change from val to test

| LightGBM metric | Val | Test | Test - Val |
| --- | --- | --- | --- |
| ROC-AUC | 0.972393 | 0.975844 | +0.003451 |
| PR-AUC | 0.314403 | 0.340406 | +0.026003 |
| Snapshot Precision | 0.226451 | 0.208252 | -0.018200 |
| Event recall | 100.0000% | 99.6914% | -0.3086 pp |
| Alerts/100 | 4.11877 | 3.50498 | -0.61378 |
| Median lead | 16.2197 | 17.7815 | +1.5618 |

LightGBM's event recall falls by only 0.3086 pp from val to test, alert burden falls by 0.61378, ranking metrics improve slightly, and the median first-warning lead increases by 1.5618 cycles. The test period contains 972 independent events; LightGBM misses 3 and Logistic Regression misses 35. Although LightGBM snapshot recall is very high, snapshot precision is 0.208252, so false-alarm cost must still be confronted.

# 7. Robustness and Subgroup Analysis

## 7.1 New aircraft and new components

Entity experiments split by `aircraft_id` or `component_id` and re-partition an internal temporal val inside each training entity to select the threshold. The models and thresholds below are therefore independent entity experiments, not the frozen temporal model applied directly to entity tests; entity results are for generalization research and do not write back to formal thresholds.

| Entity split | Training entities | Test entities | Internal train/val snapshots | Entity test snapshots | LightGBM threshold |
| --- | --- | --- | --- | --- | --- |
| by_aircraft | 160 aircraft | 40 new aircraft | 171,719 / 22,534 | 61,250 | 0.09050872 |
| by_component | 5,894 components | 1,473 new components | 172,291 / 22,599 | 60,324 | 0.06143632 |

Table 7-1. Detailed entity test results

| Split / model | ROC-AUC | PR-AUC | Snapshot P/R/F1 | Detected/actual | Event recall | Alerts/100 | Median lead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| New aircraft / LightGBM | 0.974476 | 0.328339 | 0.1859/0.9790/0.3125 | 1,156/1,290 | 89.6124% | 3.89670 | 17.6044 |
| New aircraft / LR | 0.955020 | 0.230339 | 0.1927/0.7942/0.3102 | 1,112/1,290 | 86.2016% | 4.72543 | 15.4323 |
| New components / LightGBM | 0.974016 | 0.339246 | 0.1663/0.9831/0.2845 | 1,061/1,163 | 91.2296% | 3.89042 | 17.5349 |
| New components / LR | 0.957041 | 0.238495 | 0.1959/0.7968/0.3144 | 1,020/1,163 | 87.7042% | 4.33638 | 14.7813 |

Table 7-2. Entity experiment internal val threshold selection

| Split / model | Threshold | Detected/actual | Event recall | Episodes | Active | Alerts/100 |
| --- | --- | --- | --- | --- | --- | --- |
| New aircraft / LightGBM | 0.09050872 | 428/428 | 100.0000% | 712 | 17,602 | 4.04499 |
| New aircraft / LR | 0.78640915 | 405/428 | 94.6262% | 792 | 17,602 | 4.49949 |
| New components / LightGBM | 0.06143632 | 432/432 | 100.0000% | 770 | 17,698 | 4.35077 |
| New components / LR | 0.79547987 | 409/432 | 94.6759% | 796 | 17,698 | 4.49768 |

LightGBM reaches 89.6124% and 91.2296% event recall on new aircraft and new components, above Logistic Regression at 86.2016% and 87.7042%, with a lower alert burden. However, compared with 99.6914% on the temporal test, new-entity recall drops noticeably, so real deployment should closely monitor performance on new fleets, new components, and components with unseen maintenance histories. The entity model bundles were not saved, so per-record feature contributions for entity predictions are not available (`explain_bundle=pending_entity_model_bundle`); temporal model contributions are not substituted for entity explanations. Entity metrics are reported in `reports/entity_robustness_metrics.json` and merged unchanged into `reports/group_robustness_metrics.json`.

## 7.2 Temporal test grouping method

This chapter slices saved test risk scores and `alert_episode_id` values descriptively without rebuilding cooldown episodes or rerunning the test. Ordinary event groups are classified by event entity attributes or event month; missing-pattern events are judged on the subset of scoring opportunities that carry that pattern. Events are counted only when a same-pattern link enters a saved episode. Month episodes and missing-pattern events can overlap across groups, so they must not be summed across groups. Groups with fewer than 100 snapshots or fewer than 30 events are marked `small_sample_unstable=true` and treated as descriptive only.

## 7.3 Component type

Table 7-3. LightGBM component-type groups

| Component type | Snapshots | Detected/actual | Event recall | Episodes | Active | Alerts/100 | Small sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actuator | 7,173 | 234/234 | 100.0000% | 340 | 5,645 | 6.02303 | no |
| apu | 6,039 | 39/39 | 100.0000% | 76 | 4,754 | 1.59865 | no |
| engine | 5,123 | 0/0 | - | 4 | 4,043 | 0.09894 | yes |
| environmental_control | 4,296 | 8/8 | 100.0000% | 18 | 3,401 | 0.52926 | yes |
| fuel_pump | 8,964 | 206/208 | 99.0385% | 320 | 7,063 | 4.53065 | no |
| generator | 7,125 | 124/125 | 99.2000% | 226 | 5,634 | 4.01136 | no |
| hydraulic_pump | 10,074 | 92/92 | 100.0000% | 171 | 7,948 | 2.15148 | no |
| valve | 6,466 | 266/266 | 100.0000% | 371 | 5,050 | 7.34653 | no |

`engine` has no actual events in test, so recall is not applicable; `environmental_control` has only 8 events. In-group alerts/100 for actuator and valve are 6.02303 and 7.34653, above the overall 3.50498, but the formal 5.0 target applies to the overall convention; in-group descriptions must not be written up as an overall acceptance failure.

## 7.4 Aircraft model

Table 7-4. LightGBM aircraft-model groups

| Aircraft model | Snapshots | Detected/actual | Event recall | Episodes | Active | Alerts/100 |
| --- | --- | --- | --- | --- | --- | --- |
| A320neo | 17,741 | 281/282 | 99.6454% | 429 | 13,920 | 3.08190 |
| A330-300 | 4,056 | 95/95 | 100.0000% | 143 | 3,238 | 4.41631 |
| A350-900 | 3,740 | 75/76 | 98.6842% | 119 | 2,986 | 3.98526 |
| B737-800 | 16,169 | 254/255 | 99.6078% | 394 | 12,600 | 3.12698 |
| B777-300ER | 8,084 | 168/168 | 100.0000% | 275 | 6,435 | 4.27350 |
| B787-9 | 5,470 | 96/96 | 100.0000% | 166 | 4,359 | 3.80821 |

Across the six aircraft groups, event recall ranges from 98.6842% to 100% and alerts/100 ranges from 3.08190 to 4.41631. A350-900 has the lowest recall (75/76=98.6842%); A330-300 and B777-300ER have relatively higher alert burdens.

## 7.5 Event month

Table 7-5. LightGBM event-month groups

| Event month | Snapshots | Detected/actual | Event recall | Episodes | Active | Alerts/100 |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-04 | 10,460 | 161/161 | 100.0000% | 394 | 8,064 | 4.88591 |
| 2024-05 | 9,199 | 171/171 | 100.0000% | 385 | 7,644 | 5.03663 |
| 2024-06 | 8,724 | 156/157 | 99.3631% | 359 | 7,212 | 4.97781 |
| 2024-07 | 9,121 | 176/176 | 100.0000% | 361 | 7,554 | 4.77893 |
| 2024-08 | 8,973 | 154/155 | 99.3548% | 342 | 7,484 | 4.56975 |
| 2024-09 | 8,783 | 151/152 | 99.3421% | 338 | 7,318 | 4.61875 |

Monthly recall ranges from 99.3421% to 100% and alerts/100 ranges from 4.56975 to 5.03663. 2024-05 is 5.03663, a group-level description; months can share episodes, and group episodes and events cannot be summed directly.

## 7.6 Sensor missingness patterns

There are 62 missingness-pattern groups, of which 11 are not small samples and have a valid event recall. Across non-small groups, event recall ranges from 97.3262% to 100.0000% and alerts/100 ranges from 1.90563 to 11.85567. Missing-pattern scoring opportunities and events can overlap across groups and must not be summed; small groups are descriptive only.

| Missing pattern (top 6 by snapshots) | Snapshots | Detected/actual | Event recall | Alerts/100 |
| --- | --- | --- | --- | --- |
| sensor_oil_debris \| sensor_rpm_deviation \| sensor_exhaust_temp | 9,662 | 295/300 | 98.3333% | 5.47749 |
| sensor_pressure \| sensor_oil_debris \| sensor_rpm_deviation \| sensor_exhaust_temp | 7,277 | 289/291 | 99.3127% | 6.84160 |
| sensor_rpm_deviation \| sensor_exhaust_temp | 6,309 | 81/81 | 100.0000% | 2.52055 |
| sensor_oil_debris \| sensor_exhaust_temp | 5,842 | 182/187 | 97.3262% | 5.33147 |
| sensor_pressure \| sensor_exhaust_temp | 4,392 | 113/115 | 98.2609% | 4.79434 |
| sensor_pressure \| sensor_oil_debris | 3,838 | 35/35 | 100.0000% | 1.90563 |

# 8. Error Analysis and Explainability

Samples come from `reports/predictions_example.csv` and preserve the original risk scores, episodes, event links, and explanations. LightGBM feature contributions correspond to the raw score / log-odds, not probability percentages, and do not represent causal effects. LightGBM uses native `pred_contrib=True` (TreeSHAP contributions) without an additional SHAP package; Logistic Regression uses `transformed_feature_value * fitted_coefficient`, mapped back to readable category values, with intercept and total log-odds reconciled to the model's raw output. Outputs include reason codes and missing indicators so that "numeric contribution" and "insufficient data" are distinguishable.

| Type | Record / risk score | Episode / event | Main observation |
| --- | --- | --- | --- |
| Correct early warning | R0000003569450_R1626_09<br>0.363282 | R1626_09_E0002<br>EV_R1626_09_0003599430 | 29.9799 cycles of lead; main risk-increasing contributions are sensor_vibration=1.93532 and sensor_temp=0.852834. |
| False-positive episode without an event link | R0000003823753_R0985_11<br>0.358012 | R0985_11_E0002<br>no valid event | 15 high-score snapshots in the episode; sensor_vibration's erroneous 9999 marker is cleaned to missing and still contributes strongly, indicating that the missingness/false-positive interaction needs review. |
| Missed event | R0000004894433_R0404_11<br>0.028554 | no matching episode<br>EV_R0404_11_0004894433 | Risk score is below 0.05461588; this is the highest-scoring opportunity. component_age_days and sensor_vibration increases are insufficient to cross the threshold. |
| Many missing sensors | R0000003140107_R1195_06<br>0.363282 | R1195_06_E0001<br>no valid event | All 6 sensors are missing; missing indicators for sensor_vibration and sensor_temp contribute strongly, making this a false-positive sample that requires priority review. |
| Cooldown merge within one episode | R0000003191143_R1029_06; R0000003205073_R1029_06 | R1029_06_E0001 | Two high-score snapshots 14 cycles apart belong to one episode; risk scores are 0.068686 and 0.101485. |
| New aircraft / new component | R0000004553865_R0079_15; R0000002477537_R0985_08 | entity-split saved predictions | The entity model bundle was not saved; `explain_bundle=pending_entity_model_bundle`; temporal model contributions are not used as a substitute for entity explanations. |

The complete per-field `explain_bundle` is authoritative in `predictions_example.csv`; the table only excerpts the main observations and does not interpret raw-score contributions as probabilities or causation.

## 8.1 Error conclusions

- LightGBM misses 3 events on the temporal test, so the formal delivery cannot be described as having "no misses"; Logistic Regression misses 35.
- LightGBM snapshot precision is 0.208252, meaning high event recall is accompanied by many snapshot-level false positives; cooldown episodes reduce repeated alerts but do not eliminate inspection review cost.
- Missing-sensor-related contributions can push scores upward; they should be addressed through missing-pattern monitoring and data-quality review, not interpreted as failure causes.
- Priority review targets are actuator and valve alert burden, missing-sensor false positives, and the small number of generator/fuel_pump misses; tuning must not be driven directly by the current test samples.

# 9. Limitations and Applicability

## 9.1 Limitations

- Synthetic-data limitation: degradation mechanisms, event rates, and missingness mechanisms are produced by simulation, so the results cannot prove effectiveness, safety, or airworthiness in real airline operations.
- False-positive cost: formal LightGBM snapshot precision is 0.208252; even though overall alerts/100 passes, in-group alert burdens such as actuator and valve remain high.
- Entity generalization decreases: new-aircraft and new-component recall is below the temporal test, and entity model bundles were not saved, so same-model feature contributions cannot be produced.
- Small samples and overlapping groups: engine, environmental_control, and several missingness-pattern groups are unstable; months and missingness patterns can overlap, so cross-group sums or simple rankings are invalid.
- Statistical uncertainty and causal boundaries: no confidence intervals or significance tests for model differences are provided; feature contributions explain scores only, not failure causation.
- Insufficient supplementary-model evidence: Random Forest's val advantage cannot be extrapolated to test or real operations; KNN/RF cannot replace the frozen formal conclusion.
- Label windows crossing periods: train has 80/4,291 (1.8644%) positives and val has 30/613 (4.8940%) positives driven only by next-period events; the event-recall denominator is unaffected, but val threshold selection carries a slight optimistic bias that cannot be corrected retroactively after the test is frozen.
- Denominator convention: alerts/100 uses active component windows; under distinct components, even the oracle lower bound is 32.7, making the task budget mathematically unreachable, so the denominator definition must accompany every comparison.
- Delivery form: the formal report and design documents are complete, but code repository publication, remote data release, and actual deployment are not completed; drift warning lines have not been calibrated on real data.

## 9.2 Recommendations

- Current delivery: retain LightGBM / `lgbm_15_leaves`, threshold 0.05461588446029929, and the 30-cycle cooldown convention as the formal baseline for the research generation environment.
- Follow-up candidates: if Random Forest is to be compared, pre-register the experimental protocol and obtain a new time window or real data source that did not participate in any selection; do not reuse the current temporal test for repeated model selection.
- Error governance: prioritize reviewing actuator and valve alert burden, missing-sensor false positives, and the small number of generator/fuel_pump misses on train/val or new data; do not tune directly on current test samples.
- Before real application: complete real-data validation, probability calibration, maintenance-cost-sensitive threshold analysis, data-drift monitoring, and manual review process design, and retain an independent final evaluation set.
- Engineering: initialize and publish the code repository with reproducible experiment instructions, release the mock dataset with the data dictionary and generation parameters, and calibrate drift warning lines and alert budgets on real or new-window data.

# 10. Reproduction and Artifacts

## 10.1 Artifact inventory

| Category | Path | Content |
| --- | --- | --- |
| Formal model | `models/lightgbm.joblib` | frozen LightGBM / `lgbm_15_leaves` |
| Formal baseline | `models/logistic_regression.joblib` | frozen Logistic Regression / `lr_c_1` |
| Preprocessing | `models/preprocessors.joblib` | formal preprocessor fitted on train only |
| Thresholds and metadata | `models/frozen_thresholds.json`; `models/model_metadata.json` | frozen thresholds, feature order, dependency versions, input hashes |
| One-shot test governance | `models/test_evaluation_manifest.json` | completed status, input/model/code hashes, val gate, formal conclusion |
| Formal metrics | `reports/val_metrics.json`; `reports/temporal_test_metrics.json` | strict val/test metrics and lead time |
| Comparison and threshold curves | `reports/model_comparison.csv`; `reports/val_threshold_search.csv` | formal model comparison and threshold search points |
| Robustness | `reports/entity_robustness_metrics.json`; `reports/temporal_group_metrics.json` | entity splits and aircraft/component/month/missing-pattern groups |
| Predictions and samples | `reports/predictions_test_lightgbm.csv`; `reports/predictions_example.csv` | saved risk scores, episodes, event links, and explanation samples |
| Supplementary models | `models/experiments/knn.joblib`; `models/experiments/random_forest.joblib` | KNN/RF val-only experiment models |
| Supplementary metrics | `reports/additional_models_val_metrics.json`; `reports/additional_models_val_threshold_search.csv` | supplementary val metrics, threshold curves, and test-unused declaration |
| Implementation and configuration | `src/train.py`; `src/evaluation.py`; `src/robustness.py`; `src/additional_models.py` | training, strict evaluation, robustness, and supplementary experiments |
| Configuration | `configs/modeling.yaml`; `configs/additional_models.yaml`; `configs/schema.yaml` | formal configuration, supplementary configuration, and field definitions |
| Documentation | `docs/data_dictionary.md`; `docs/MRO_Production_Deployment_Design.md` | labels, leakage conventions, dataset card, serving design, and frozen evaluation detail |
| Label boundary audit | `src/label_boundary_audit.py`; `reports/label_boundary_audit.json` | read-only quantification of cross-split label overlap; no retraining or test re-evaluation |
| Formal delivery report | `docs/MRO_Model_Formal_Delivery_Report.md` | this report |

## 10.2 Input file SHA-256

These hashes come from `models/test_evaluation_manifest.json` and `models/model_metadata.json` and are used to verify the frozen inputs, models, and evaluation code referenced by this report. Supplementary experiments use separate configurations and do not modify the files covered by these hashes.

| File | SHA-256 |
| --- | --- |
| `data/raw/component_snapshots.csv` | `22561ae9b1f95dc9d3fa56dfd89fc8d901f919431887ce583d8edeee7cd2fe73` |
| `data/splits/temporal/train.csv` | `9d6c86dba6a4c0aaf828ce4132ae8e16c85b883c2f14c87cf941fef6da18b70c` |
| `data/splits/temporal/val.csv` | `46c766faa7154b398ff5cf1b38d7a92f9baadefd8a5012101741f2c5a81cfe77` |
| `data/splits/temporal/test.csv` | `e02e3a195c525f6decf2cc9754f6a1e735c759b83471f4efc30f9ff626209356` |
| `data/audit/removal_events.csv` | `f82e049c893989e69080b278be85bf9e0b58b89d0c9b018fe926e4a9d1a55123` |
| `data/audit/snapshot_event_links.csv` | `45509c1f224c3d263e1dda60237039727324fa6439fa87a4953dd0e18e690529` |
| `configs/schema.yaml` | `ff6377ba4d23d14b13bacc1f9efd20fe02c9933428e32af97eb0e40101c1eb09` |
| `configs/modeling.yaml` | `524c5d2d3b81008efefca9180ebdda124198c537532950eaa1f464a05b48ef68` |

## 10.3 Model and code SHA-256

| Item | SHA-256 |
| --- | --- |
| `models/lightgbm.joblib` | `5042f926a273b06c94cc13276b1c96f4cfcacc2ede986b800ebd04f9b8530b2e` |
| `models/logistic_regression.joblib` | `ce4a1ef45354f2af5ca55b53cad8764f81e881f57dbfe62dfd39f8933daf4182` |
| `src/evaluation.py` | `1f19ab9714ad93c2464230d6cb029cc1d6420f9fe094cfa0abfeba676d383feb` |
| `src/modeling.py` | `e9b5acf12e77e1d9df06470307db3dca59b2a6e85ca688c77d05187a204aa159` |
| `configs/modeling.yaml` (configuration content hash) | `dcb063ae1b926e8967f9e3e0670c2bbac3028b18a506ff23b6054c571a32f66d` |

## 10.4 Formal dependency versions

| Dependency | Version | Dependency | Version |
| --- | --- | --- | --- |
| Python | 3.11.5 | NumPy | 1.26.4 |
| pandas | 2.0.3 | scikit-learn | 1.3.0 |
| LightGBM | 4.7.0 | joblib | 1.2.0 |
| Random seed | 42 | Data version | v3 |

## 10.5 Supplementary experiment integrity

| Verification item | Random Forest | KNN |
| --- | --- | --- |
| Training rows | 215,958 | 8,582 |
| Training positives | 4,291 | 4,291 |
| Training sample | `full_train` | `all_positive_and_equal_negative` |
| Temporal test read | false | false |
| Temporal test evaluated | false | false |
| `temporal_test_metrics` | null | null |
| Experiment model | `models/experiments/random_forest.joblib` | `models/experiments/knn.joblib` |
| Conclusion scope | val-only; does not change the frozen formal delivery | val-only; does not change the frozen formal delivery |

Supplementary experiment configuration SHA-256: `c6c3c1823950329d8ec48f22e9c6de0442050e21c4537436dffbd48e40186da5`. The experiment manifest explicitly records `original_frozen_models_modified=false`, and the `selection_rule` is "maximum event recall under val alerts <=4.5, with ties broken by fewer alerts and then a higher threshold".

# Appendix A. Design-vs-Implementation Differences

The original plan and design targets are retained in the local archive. The table below records where the implementation differs and why. It is an appendix of differences, not a replacement for the result chapters above.

| # | Design / plan | Actual implementation | Reason | Impact |
| ---: | --- | --- | --- | --- |
| 1 | Approximately 80,000 snapshots | 299,474 | fixed 30-cycle observability and densified snapshots | complete event coverage, higher generation and training cost |
| 2 | 2,000 component instances | 7,367 visible / 10,820 simulated | removals create replacement instances | new-component evaluation is more realistic |
| 3 | Timeline through 2024-12 | through 2024-09-30 | reserved a 120-day label observation period | test covers 6 months |
| 4 | Sensor missingness 5%-15% per field | full-table maximum 82.15% | structural non-applicability by component type | missing indicators and group interpretation are mandatory; applicable-row rates of 7.69%-12.65% remain in the plan band |
| 5 | CatBoost or LightGBM | LightGBM; RF and KNN added | LightGBM already passed the gate; RF covers the non-linear tree baseline role | CatBoost not validated |
| 6 | Time or group split | time + aircraft + component | more comprehensive generalization check | entity metrics are lower than time-split metrics |
| 7 | Explain main factors | TreeSHAP / `pred_contrib` + reason codes + 6 sample types | implemented as planned | entity explanations missing |
| 8 | Production interface suggestions | separate design document + measured offline drift | completed the evaluation dimension | still not actually deployed |
| 9 | Report plus code/data submission | report complete; public repository/data links missing | repository initialized but not published | submission requirement only partially met |
| 10 | Threshold selected on val only | strictly enforced; test run once | governance constraint | features not leaked; small label-boundary overlap remains |
| 11 | Time split isolates future events | train 80 and val 30 positives driven by next-period events | 30-cycle labels cross date boundaries | event denominator clean; val threshold slightly optimistic |
| 12 | <=5 alerts per 100 active components | active component window used; distinct-component oracle lower bound 32.7 | task wording does not specify a denominator | denominator must be explained with every report |
| 13 | Outlier handling | invalid-value rules (`-999`, `9999`, negative sensor values); no winsorization | statistical clipping not enabled by default | limited effect on LightGBM; LR remains a simple baseline |
| 14 | Small predefined candidate grid | LR 3 candidates, LightGBM 2 candidates, plus RF/KNN | avoid an open-ended search | selection remains auditable |
| 15 | Detailed monitoring metrics | offline drift computation and monthly alert burden added in the separate drift report | strengthen the evidence chain | warning lines still uncalibrated on real data |

Additional implementation notes:

- The feature list, label definition, and preprocessing fitting boundary match the design: 28 fields, 23 candidate features, a 30-cycle label, and train-only fitting.
- The v3 endpoint change to 2024-09-30 is driven by the 120-day label observation extension, which prevents right-censored snapshots from being mislabeled as negatives.
- The full-table missingness increase is structural rather than a generation failure: `sensor_exhaust_temp` is missing 82.15% across the full table but only 11.66% among applicable rows.
- Anomaly injection uses error markers and negative-value rules rather than an explicit approximately 1% statistical anomaly share; remaining extreme values are left to tree-model robustness.
- The implementation added lead-time quantiles, entity and group robustness, and sanity checks beyond the original plan, and discloses in-group alert budgets above 5 instead of hiding them behind the overall metric.
- Production design advanced from "interface suggestions" to a reviewable deployment, monitoring, and retraining plan, but deployment, retraining triggers, and drift thresholds remain unexercised on real data.

Report end. All formal figures are governed by the corresponding frozen JSON/CSV artifacts; if any transcription here differs from the machine-readable artifacts, the frozen artifacts and hash verification prevail.
