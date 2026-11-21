"""Train-only preprocessing, dual-model training, prediction, and local contribution explanations."""
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

ERROR_MARKERS = (-999, 9999)
NONNEGATIVE_SENSORS = [s for s in C.SENSOR_FIELDS if s != "sensor_rpm_deviation"]


def numeric_fields(features):
    return [field for field in features if field not in C.CATEGORICAL]


def categorical_fields(features):
    return [field for field in features if field in C.CATEGORICAL]


def missing_indicators(features):
    return [f"{field}_is_missing" for field in C.SENSOR_FIELDS if field in features]


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Apply deterministic cleaning only; never read val/test statistics."""

    def __init__(self, features, categorical, sensors):
        self.features = features
        self.categorical = categorical
        self.sensors = sensors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model input fields: {sorted(missing)}")
        cleaned = X.loc[:, list(self.features)].copy()
        numeric = [field for field in self.features if field not in self.categorical]
        for field in numeric:
            cleaned[field] = pd.to_numeric(cleaned[field], errors="coerce")
        for sensor in self.sensors:
            if sensor not in cleaned:
                continue
            invalid = cleaned[sensor].isin(ERROR_MARKERS)
            if sensor in NONNEGATIVE_SENSORS:
                invalid |= cleaned[sensor] < 0
            cleaned.loc[invalid, sensor] = np.nan
        for field in self.categorical:
            cleaned[field] = cleaned[field].astype("object").where(cleaned[field].notna(), "Unknown")
            cleaned[field] = cleaned[field].replace("", "Unknown").astype(str)
        return cleaned


class LightGBMPreprocessor:
    """Fit the category vocabulary on train; keep numeric NaN for LightGBM."""

    def __init__(self, features):
        self.features = list(features)
        self.numeric = numeric_fields(self.features)
        self.categorical = categorical_fields(self.features)
        self.missing = missing_indicators(self.features)
        self.output_features = self.numeric + self.categorical + self.missing
        self.cleaner = FeatureCleaner(self.features, self.categorical, C.SENSOR_FIELDS)

    def fit(self, X, y=None):
        cleaned = self.cleaner.fit(X).transform(X)
        self.category_mappings_ = {}
        for field in self.categorical:
            categories = sorted(cleaned[field].unique().tolist())
            self.category_mappings_[field] = {category: i + 1 for i, category in enumerate(categories)}
        return self

    def transform(self, X):
        cleaned = self.cleaner.transform(X)
        output = pd.DataFrame(index=cleaned.index)
        for field in self.numeric:
            output[field] = cleaned[field].astype(float)
        for field in self.categorical:
            mapping = self.category_mappings_[field]
            output[field] = cleaned[field].map(mapping).fillna(0).astype(np.int32)
        for sensor in C.SENSOR_FIELDS:
            indicator = f"{sensor}_is_missing"
            if indicator in self.missing:
                output[indicator] = cleaned[sensor].isna().astype(np.int8)
        return output.loc[:, self.output_features]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def make_logistic_pipeline(model_params, features=None):
    features = list(features or C.MODEL_FEATURES)
    categorical = categorical_fields(features)
    numeric = numeric_fields(features)
    cleaner = FeatureCleaner(features, categorical, C.SENSOR_FIELDS)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical),
    ], sparse_threshold=1.0)
    return Pipeline([
        ("clean", cleaner),
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(**model_params)),
    ])


def fit_logistic(train, model_params, features=None):
    features = list(features or C.MODEL_FEATURES)
    pipeline = make_logistic_pipeline(model_params, features)
    pipeline.fit(train, train["target"].astype(int))
    return {
        "model_type": "logistic_regression",
        "model": pipeline,
        "features": features,
        "preprocessor": None,
    }


def fit_lightgbm(train, val, model_params, early_stopping_rounds=100, eval_metric="binary_logloss", features=None):
    features = list(features or C.MODEL_FEATURES)
    preprocessor = LightGBMPreprocessor(features)
    X_train = preprocessor.fit_transform(train)
    y_train = train["target"].astype(int)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("LightGBM training data must contain both positive and negative samples")
    params = dict(model_params)
    params["scale_pos_weight"] = negatives / positives
    params.setdefault("verbosity", -1)
    model = lgb.LGBMClassifier(**params)
    fit_params = {"categorical_feature": preprocessor.categorical}
    if val is not None:
        X_val = preprocessor.transform(val)
        fit_params.update({
            "eval_set": [(X_val, val["target"].astype(int))],
            "eval_metric": eval_metric,
            "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
        })
    model.fit(X_train, y_train, **fit_params)
    return {
        "model_type": "lightgbm",
        "model": model,
        "features": features,
        "preprocessor": preprocessor,
    }


def predict_scores(bundle, snapshots):
    if bundle["model_type"] == "logistic_regression":
        return bundle["model"].predict_proba(snapshots)[:, 1]
    if bundle["model_type"] == "lightgbm":
        transformed = bundle["preprocessor"].transform(snapshots)
        return bundle["model"].predict_proba(transformed)[:, 1]
    raise ValueError(f"Unknown model type: {bundle['model_type']}")


def save_bundle(bundle, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path):
    return joblib.load(path)


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reason_rows(feature_names, contributions, snapshots, i, categorical, top_n):
    reasons = []
    raw = snapshots.iloc[i]
    for j, contribution in enumerate(contributions):
        if contribution == 0:
            continue
        feature = feature_names[j]
        source_feature = feature
        if feature.endswith("_is_missing"):
            source_feature = feature[:-len("_is_missing")]
            value = bool(pd.isna(raw[source_feature]))
        elif feature in raw.index:
            value = _json_value(raw[feature])
        else:
            value = None
        reasons.append({
            "feature": feature,
            "source_feature": source_feature,
            "value": value,
            "contribution": float(contribution),
        })
    increases = sorted((r for r in reasons if r["contribution"] > 0), key=lambda r: r["contribution"], reverse=True)
    decreases = sorted((r for r in reasons if r["contribution"] < 0), key=lambda r: r["contribution"])
    return {"increase": increases[:top_n], "decrease": decreases[:top_n]}


def explain_lightgbm(bundle, snapshots, top_n=5):
    if bundle["model_type"] != "lightgbm":
        raise ValueError("Not a LightGBM model")
    transformed = bundle["preprocessor"].transform(snapshots)
    model = bundle["model"]
    best_iteration = getattr(model, "best_iteration", None)
    kwargs = {"num_iteration": best_iteration} if best_iteration else {}
    contributions = model.booster_.predict(transformed, pred_contrib=True, **kwargs)
    raw_scores = model.booster_.predict(transformed, raw_score=True, **kwargs)
    explanations = []
    for i in range(len(snapshots)):
        feature_contributions = contributions[i, :-1]
        explanations.append({
            "record_id": snapshots.iloc[i]["record_id"],
            "raw_score": float(raw_scores[i]),
            "intercept": float(contributions[i, -1]),
            "reconstructed_raw_score": float(contributions[i].sum()),
            "reasons": _reason_rows(
                transformed.columns.tolist(), feature_contributions, snapshots, i,
                bundle["preprocessor"].categorical, top_n,
            ),
        })
    return explanations


def _logistic_feature_names(pipeline):
    return pipeline.named_steps["preprocess"].get_feature_names_out().tolist()


def explain_logistic(bundle, snapshots, top_n=5):
    if bundle["model_type"] != "logistic_regression":
        raise ValueError("Not a Logistic Regression model")
    pipeline = bundle["model"]
    transformed = pipeline.named_steps["preprocess"].transform(
        pipeline.named_steps["clean"].transform(snapshots)
    )
    classifier = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    names = _logistic_feature_names(pipeline)
    raw_scores = pipeline.decision_function(snapshots)
    categorical = categorical_fields(bundle["features"])
    explanations = []
    for i in range(len(snapshots)):
        row_contributions = transformed[i].multiply(coefficients).toarray().ravel()
        readable_names = []
        raw = snapshots.iloc[i]
        values = []
        for name in names:
            stripped = name.split("__", 1)[1]
            source = stripped
            if name.startswith("cat__"):
                for field in sorted(categorical, key=len, reverse=True):
                    if stripped.startswith(field + "_"):
                        source = field
                        break
            readable_names.append(source)
            values.append(_json_value(raw[source]) if source in raw.index else None)
        reasons = []
        for feature, value, contribution in zip(readable_names, values, row_contributions):
            if contribution == 0:
                continue
            reasons.append({
                "feature": feature,
                "source_feature": feature,
                "value": value,
                "contribution": float(contribution),
            })
        increases = sorted((r for r in reasons if r["contribution"] > 0), key=lambda r: r["contribution"], reverse=True)
        decreases = sorted((r for r in reasons if r["contribution"] < 0), key=lambda r: r["contribution"])
        explanations.append({
            "record_id": raw["record_id"],
            "raw_score": float(raw_scores[i]),
            "intercept": float(classifier.intercept_[0]),
            "reconstructed_raw_score": float(row_contributions.sum() + classifier.intercept_[0]),
            "reasons": {"increase": increases[:top_n], "decrease": decreases[:top_n]},
        })
    return explanations


def explain_bundle(bundle, snapshots, top_n=5):
    if bundle["model_type"] == "lightgbm":
        return explain_lightgbm(bundle, snapshots, top_n)
    return explain_logistic(bundle, snapshots, top_n)
