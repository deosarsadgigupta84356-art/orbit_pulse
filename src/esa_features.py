
"""
ORBIT PULSE - ESA Event-Level Feature Engineering
=================================================

Builds leakage-controlled event-level features from ESA CDM histories.

Only CDMs with time_to_tca >= prediction_cutoff_days
are eligible as model inputs.

Final event risk labels are derived separately from the
complete training history.
"""

import numpy as np
import pandas as pd


DEFAULT_PREDICTION_CUTOFF_DAYS = 2.0
RISK_FLOOR = -30.0

REQUIRED_COLUMNS = {
    "event_id",
    "time_to_tca",
    "risk",
}

EXCLUDED_FEATURE_COLUMNS = {
    "event_id",
    "mission_id",
}


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy an ESA CDM dataframe."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("ESA input dataframe is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"ESA input is missing columns: {sorted(missing)}"
        )

    if df["event_id"].isna().any():
        raise ValueError("ESA input contains missing event IDs.")

    result = df.copy()

    result["event_id"] = result["event_id"].astype("string")

    for column in ["time_to_tca", "risk"]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if not np.isfinite(
            result[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"ESA input contains invalid values in {column}."
            )

    return result


def build_esa_event_features(
    df: pd.DataFrame,
    prediction_cutoff_days: float = DEFAULT_PREDICTION_CUTOFF_DAYS,
) -> pd.DataFrame:
    """
    Build one leakage-controlled feature row per eligible event.

    Selects the CDM closest to the prediction boundary while
    remaining at least prediction_cutoff_days before TCA.
    """

    if prediction_cutoff_days <= 0:
        raise ValueError("Prediction cutoff must be positive.")

    data = _validate_input(df)

    eligible = data.loc[
        data["time_to_tca"] >= prediction_cutoff_days
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No ESA CDMs satisfy the prediction cutoff."
        )

    # Smaller time_to_tca means a later observation.
    # Select the latest eligible CDM for each event.
    latest = (
        eligible
        .sort_values(
            ["event_id", "time_to_tca"],
            ascending=[True, True],
            kind="stable",
        )
        .groupby("event_id", sort=False)
        .head(1)
        .copy()
    )

    # Preserve event_id as the dataframe index, not as a model feature.
    features = latest.set_index("event_id")

    features = features.drop(
        columns=list(
            EXCLUDED_FEATURE_COLUMNS.intersection(features.columns)
        )
    )

    # Log-transform large positive covariance determinants.
    # This preserves their ordering while reducing numerical scale.
    covariance_columns = [
        "t_position_covariance_det",
        "c_position_covariance_det",
    ]

    for column in covariance_columns:
        if column in features.columns:
            values = features[column].to_numpy(dtype=np.float64)

            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(
                    f"{column} must contain finite positive values."
                )

            features[f"{column}_log10"] = np.log10(values)
            features = features.drop(columns=[column])

    if features.index.has_duplicates:
        raise ValueError(
            "Feature builder produced duplicate event IDs."
        )

    if (features["time_to_tca"] < prediction_cutoff_days).any():
        raise ValueError(
            "Feature leakage detected: a snapshot violates the cutoff."
        )

    features.attrs["prediction_cutoff_days"] = (
        prediction_cutoff_days
    )
    features.attrs["feature_selection"] = (
        "Latest eligible CDM per event"
    )
    features.attrs["dataset_source"] = (
        "ESA Collision Avoidance Challenge"
    )

    return features


def build_esa_training_dataset(
    df: pd.DataFrame,
    prediction_cutoff_days: float = DEFAULT_PREDICTION_CUTOFF_DAYS,
):
    """
    Build event-level training features and final-risk labels.

    Events without eligible pre-cutoff observations are excluded.

    Returns:
        X: event-level features
        y: final-risk labels and risk-floor indicator
    """

    data = _validate_input(df)

    X = build_esa_event_features(
        data,
        prediction_cutoff_days=prediction_cutoff_days,
    )

    # Final CDM = smallest time_to_tca in the complete event history.
    final_rows = (
        data
        .sort_values(
            ["event_id", "time_to_tca"],
            ascending=[True, True],
            kind="stable",
        )
        .groupby("event_id", sort=False)
        .head(1)
    )

    final_labels = final_rows.set_index("event_id")["risk"]

    y = pd.DataFrame(
        {
            "final_risk": final_labels,
        }
    )

    # -30 is treated as a dataset risk-floor outcome,
    # not as a precisely measured zero probability.
    y["risk_floor"] = np.isclose(
        y["final_risk"],
        RISK_FLOOR,
        atol=1e-9,
    )

    # Keep only events with eligible feature snapshots.
    y = y.loc[X.index]

    if not X.index.equals(y.index):
        raise ValueError(
            "Feature and target event IDs are misaligned."
        )

    if X.index.has_duplicates or y.index.has_duplicates:
        raise ValueError(
            "Duplicate event IDs detected in training dataset."
        )

    X.attrs["prediction_cutoff_days"] = (
        prediction_cutoff_days
    )
    y.attrs["target_definition"] = (
        "Final event risk from the complete CDM history"
    )

    return X, y
