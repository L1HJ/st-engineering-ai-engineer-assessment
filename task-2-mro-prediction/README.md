# Aircraft MRO Unscheduled Removal Prediction

Machine-learning prototype that predicts whether a component will experience an unscheduled removal within the next 30 flight cycles. The data are a synthetic v3 dataset intended for method validation only and do not represent a real fleet or real maintenance performance.

## Status

| Item | Result |
|---|---|
| Formal model | LightGBM / `lgbm_15_leaves`, frozen threshold `0.05461588446029929` |
| Val (selection and gate) | Event recall 100% (548/548), 4.11877 alerts/100 |
| Temporal test (one-shot, frozen) | Event recall 99.6914% (969/972), 3.50498 alerts/100 |
| Formal targets | Event recall >=80%, alerts/100 <=5.0; both passed |
| Supplementary experiments | Random Forest and KNN, train/val only; test never read |
| Automated tests | 30 passed |

The temporal test has been consumed; **do not rerun the test stage or tune on it**.

## Data and metric conventions

- 299,474 snapshots, 200 aircraft, 7,367 visible component instances, 5,997 positive snapshots (2.0025%).
- Temporal splits: train 2022-01-01 to 2023-12-31, val 2024-01-01 to 2024-03-31, test 2024-04-01 to 2024-09-30.
- Label: whether an unscheduled removal occurs within the next 30 flight cycles after the snapshot.
- The `alerts/100` denominator is fixed to **active component windows** (unique aircraft x component x 30-cycle window), not distinct components.
- Sensor missingness includes structural missingness. Among applicable rows the rate is 7.69%-12.65%; the full-table maximum is 82.15%. Reports must state which denominator is used.
- Label windows cross the date-split boundaries for a small number of positive snapshots (train 80, val 30); features are not leaked. See `reports/label_boundary_audit.json`.

## Layout

```text
src/            dataset generation, splitting, training, evaluation, robustness, drift, audit
configs/        data schema, formal modeling, supplementary models, drift monitoring
data/           synthetic v3 data, splits, and audit tables (included in the repository)
models/         frozen models, thresholds, preprocessors, one-shot test manifest (included)
reports/        metric JSON/CSV, prediction samples, drift and label-boundary audits
tests/          pytest automated tests
docs/           English delivery documents
docs/archive/   local-only archived documents (not submitted)
```

## Dataset and model artifacts

The synthetic v3 dataset (about 299 MB) is included under `data/`, and the frozen models and thresholds (about 32 MB) are included under `models/`. The largest single file is 71 MB, below GitHub's 100 MB per-file limit, so the repository can be pushed without Git LFS. Prediction CSVs remain untracked because they are large and reproducible.

## Environment

Python 3.11. Dependency versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction and checks

Read-only checks (no retraining, no test re-evaluation):

```bash
python -m pytest tests -q
python src/label_boundary_audit.py
python src/drift_monitor.py
```

Full retraining (overwrites local models and reports; the temporal test stage is sealed, so do not run `--stage test`):

```bash
python src/generate_dataset.py
python src/split_dataset.py
python src/train.py --stage val
python src/additional_models.py
```

> These commands rewrite `data/`, `models/`, and some files under `reports/`. To preserve the current frozen evidence, run them in a copy of the project rather than overwriting the v3 data and models.

Supplementary experiments may use train/val only. Any new final evaluation must use a new data window that did not participate in any selection, and a new experiment protocol must be registered in advance.

## Documentation

- [Formal delivery report](docs/MRO_Model_Formal_Delivery_Report.md): frozen model, strict metric definitions, one-shot temporal test, robustness, and limitations.
- [Production deployment design](docs/MRO_Production_Deployment_Design.md): architecture, interfaces, alert workflow, threshold governance, monitoring, and empirical drift results.
- [Data dictionary](docs/data_dictionary.md): dataset card, label definition and boundary audit, splits, 28-field dictionary, missingness, and local packaging.

Earlier design, plan, and draft documents are retained in `docs/archive/` for local traceability only and are excluded from the submission.

## Known limitations

- Synthetic data; results must not be extrapolated to real aviation safety, airworthiness, or maintenance decisions.
- Label windows overlap slightly across splits; the frozen test has already been consumed and cannot be retroactively corrected. The next dataset version should purge at the boundaries.
- Recall on new aircraft and new components is lower than on the temporal split, and no explanation bundle was saved for the entity models.
- Drift warning thresholds are initial suggested values and have not been calibrated on real data.
- The repository has not yet been pushed to a remote; the dataset and frozen models are included in the working tree and are ready to commit.
