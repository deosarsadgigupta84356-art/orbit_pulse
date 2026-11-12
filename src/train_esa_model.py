
"""
ORBIT PULSE - ESA Two-Stage Risk Model
======================================

Trains and evaluates a two-stage model using the ESA Collision
Avoidance Challenge dataset.

Stage 1: Predict whether final risk reaches the -30 risk floor.
Stage 2: Predict final log-risk for events above that floor.

Validation is event-disjoint and stratified.

The private ESA evaluation labels are NEVER loaded here.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.esa_data import load_esa_training_data
from src.esa_features import (
    build_esa_training_dataset,
    DEFAULT_PREDICTION_CUTOFF_DAYS,
    RISK_FLOOR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models" / "esa"
REPORT_DIR = PROJECT_ROOT / "reports" / "esa"

MODEL_PATH = MODEL_DIR / "esa_two_stage_model.joblib"
METRICS_PATH = REPORT_DIR / "esa_validation_metrics.json"
METADATA_PATH = MODEL_DIR / "esa_model_metadata.json"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def _build_preprocessor(numeric_columns, categorical_columns):
    """Create training-fitted numerical and categorical preprocessing."""

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _build_classifier(numeric_columns, categorical_columns):
    """Build the risk-floor classification pipeline."""

    return Pipeline([
        (
            "preprocessor",
            _build_preprocessor(
                numeric_columns,
                categorical_columns,
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def _build_regressor(numeric_columns, categorical_columns):
    """Build the above-floor conditional regression pipeline."""

    return Pipeline([
        (
            "preprocessor",
            _build_preprocessor(
                numeric_columns,
                categorical_columns,
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=400,
                max_features=0.7,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def _regression_metrics(actual, predicted):
    """Calculate regression metrics."""

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def train_and_evaluate_esa_models():
    """Train, validate, compare baselines, and save ESA models."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ESA training partition...")

    raw = load_esa_training_data()

    X, y = build_esa_training_dataset(
        raw,
        prediction_cutoff_days=DEFAULT_PREDICTION_CUTOFF_DAYS,
    )

    del raw

    numeric_columns = list(
        X.select_dtypes(include=[np.number]).columns
    )

    categorical_columns = list(
        X.select_dtypes(
            include=["object", "string", "category"]
        ).columns
    )

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable model features found.")

    if X.isna().all().any():
        raise ValueError("Entirely missing model features detected.")

    if np.isinf(
        X[numeric_columns].to_numpy(dtype=float)
    ).any():
        raise ValueError("Infinite numerical feature values detected.")

    # NaN values are permitted because the preprocessing pipeline
    # handles them using training-fitted median imputation.

    # Split event indices, not individual CDM rows.
    indices = np.arange(len(X))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y["risk_floor"],
    )

    X_train = X.iloc[train_indices]
    X_validation = X.iloc[validation_indices]

    y_train = y.iloc[train_indices]
    y_validation = y.iloc[validation_indices]

    print("\nEVENT-LEVEL SPLIT")
    print("Training events:", len(X_train))
    print("Validation events:", len(X_validation))
    print(
        "Training risk-floor events:",
        int(y_train["risk_floor"].sum()),
    )
    print(
        "Validation risk-floor events:",
        int(y_validation["risk_floor"].sum()),
    )

    # ---------------------------------------------------------
    # STAGE 1: RISK-FLOOR CLASSIFICATION
    # ---------------------------------------------------------

    print("\nTraining risk-floor classifier...")

    classifier = _build_classifier(
        numeric_columns,
        categorical_columns,
    )

    classifier.fit(
        X_train,
        y_train["risk_floor"],
    )

    floor_predictions = classifier.predict(X_validation)

    classes = list(classifier.named_steps["model"].classes_)
    floor_class_index = classes.index(True)

    floor_probabilities = classifier.predict_proba(
        X_validation
    )[:, floor_class_index]

    classifier_baseline = DummyClassifier(
        strategy="most_frequent",
    )

    classifier_baseline.fit(
        X_train,
        y_train["risk_floor"],
    )

    baseline_floor_predictions = classifier_baseline.predict(
        X_validation
    )

    classifier_metrics = {
        "accuracy": float(
            accuracy_score(
                y_validation["risk_floor"],
                floor_predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_validation["risk_floor"],
                floor_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_validation["risk_floor"],
                floor_predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_validation["risk_floor"],
                floor_predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_validation["risk_floor"],
                floor_probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_validation["risk_floor"],
                floor_probabilities,
            )
        ),
        "confusion_matrix_labels": ["above_floor", "risk_floor"],
        "confusion_matrix": confusion_matrix(
            y_validation["risk_floor"],
            floor_predictions,
            labels=[False, True],
        ).tolist(),
    }

    classifier_baseline_metrics = {
        "accuracy": float(
            accuracy_score(
                y_validation["risk_floor"],
                baseline_floor_predictions,
            )
        ),
        "f1": float(
            f1_score(
                y_validation["risk_floor"],
                baseline_floor_predictions,
                zero_division=0,
            )
        ),
    }

    # ---------------------------------------------------------
    # STAGE 2: ABOVE-FLOOR CONDITIONAL REGRESSION
    # ---------------------------------------------------------

    print("\nTraining above-floor risk regressor...")

    regression_train_mask = ~y_train["risk_floor"]
    regression_validation_mask = ~y_validation["risk_floor"]

    X_regression_train = X_train.loc[
        regression_train_mask
    ]

    y_regression_train = y_train.loc[
        regression_train_mask,
        "final_risk",
    ]

    X_regression_validation = X_validation.loc[
        regression_validation_mask
    ]

    y_regression_validation = y_validation.loc[
        regression_validation_mask,
        "final_risk",
    ]

    if len(X_regression_train) < 10:
        raise ValueError(
            "Insufficient above-floor events for regression."
        )

    if len(X_regression_validation) == 0:
        raise ValueError(
            "Validation split contains no above-floor events."
        )

    regressor = _build_regressor(
        numeric_columns,
        categorical_columns,
    )

    regressor.fit(
        X_regression_train,
        y_regression_train,
    )

    regression_predictions = regressor.predict(
        X_regression_validation
    )

    regressor_baseline = DummyRegressor(
        strategy="median",
    )

    regressor_baseline.fit(
        X_regression_train,
        y_regression_train,
    )

    median_predictions = regressor_baseline.predict(
        X_regression_validation
    )

    # Simple persistence baseline: final risk is approximated
    # by the latest observed eligible CDM risk.
    latest_risk_predictions = X_regression_validation[
        "risk"
    ].to_numpy(dtype=float)

    regression_metrics = {
        "evaluation_events": int(
            len(y_regression_validation)
        ),
        "random_forest": _regression_metrics(
            y_regression_validation,
            regression_predictions,
        ),
        "median_baseline": _regression_metrics(
            y_regression_validation,
            median_predictions,
        ),
        "latest_observed_risk_baseline": _regression_metrics(
            y_regression_validation,
            latest_risk_predictions,
        ),
    }

    # ---------------------------------------------------------
    # COMBINED TWO-STAGE MODEL
    # ---------------------------------------------------------

    combined_predictions = np.full(
        len(X_validation),
        RISK_FLOOR,
        dtype=float,
    )

    above_floor_indices = np.flatnonzero(
        np.asarray(floor_predictions) == False
    )

    if len(above_floor_indices):
        combined_predictions[above_floor_indices] = (
            regressor.predict(
                X_validation.iloc[above_floor_indices]
            )
        )

    combined_baseline_predictions = X_validation[
        "risk"
    ].to_numpy(dtype=float)

    combined_metrics = {
        "two_stage_model": _regression_metrics(
            y_validation["final_risk"],
            combined_predictions,
        ),
        "latest_observed_risk_baseline": _regression_metrics(
            y_validation["final_risk"],
            combined_baseline_predictions,
        ),
        "validation_events": int(len(X_validation)),
    }

    # ---------------------------------------------------------
    # SAVE MODEL AND EVALUATION ARTIFACTS
    # ---------------------------------------------------------

    metrics = {
        "dataset": "ESA Collision Avoidance Challenge",
        "prediction_cutoff_days": DEFAULT_PREDICTION_CUTOFF_DAYS,
        "risk_floor": RISK_FLOOR,
        "validation_method": "Stratified event-level holdout",
        "validation_fraction": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "total_eligible_events": int(len(X)),
        "training_events": int(len(X_train)),
        "validation_events": int(len(X_validation)),
        "feature_count": int(len(numeric_columns) + len(categorical_columns)),
        "numeric_feature_count": int(len(numeric_columns)),
        "categorical_features": categorical_columns,
        "classifier": {
            "random_forest": classifier_metrics,
            "majority_class_baseline": classifier_baseline_metrics,
        },
        "conditional_regression": regression_metrics,
        "combined_prediction": combined_metrics,
        "private_evaluation_labels_used": False,
    }

    model_bundle = {
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "prediction_cutoff_days": DEFAULT_PREDICTION_CUTOFF_DAYS,
        "risk_floor": RISK_FLOOR,
        "random_state": RANDOM_STATE,
    }

    metadata = {
        "model_name": "ORBIT PULSE ESA Two-Stage Risk Model",
        "dataset": "ESA Collision Avoidance Challenge",
        "dataset_source": "https://zenodo.org/records/4463683",
        "dataset_license": (
            "Creative Commons Attribution 4.0 International"
        ),
        "prediction_cutoff_days": DEFAULT_PREDICTION_CUTOFF_DAYS,
        "target": "Final event-level base-10 logarithmic risk",
        "risk_floor": RISK_FLOOR,
        "validation_method": "Stratified event-level holdout",
        "random_state": RANDOM_STATE,
        "feature_count": len(X.columns),
        "training_events": len(X_train),
        "validation_events": len(X_validation),
        "private_evaluation_labels_used": False,
        "operational_status": "Research prototype",
    }

    joblib.dump(model_bundle, MODEL_PATH)

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nMODEL TRAINING COMPLETE")
    print("Model:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)
    print("Metadata:", METADATA_PATH)

    print("\nCLASSIFICATION RESULTS")
    print(json.dumps(metrics["classifier"], indent=4))

    print("\nCONDITIONAL REGRESSION RESULTS")
    print(json.dumps(regression_metrics, indent=4))

    print("\nCOMBINED TWO-STAGE RESULTS")
    print(json.dumps(combined_metrics, indent=4))

    return metrics


if __name__ == "__main__":
    train_and_evaluate_esa_models()
