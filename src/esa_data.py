"""
ORBIT PULSE - ESA Collision Avoidance Challenge Data Loader
==========================================================
Loads the ESA historical collision-avoidance challenge dataset.

Training features and targets are handled separately from public
test inputs and private evaluation labels.

Dataset source:
https://zenodo.org/records/4463683

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

from pathlib import Path
from typing import Optional
import io
import zipfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESA_ROOT = PROJECT_ROOT / "data" / "raw" / "esa"

DATASET_SOURCE = "ESA Collision Avoidance Challenge"
DATASET_URL = "https://zenodo.org/records/4463683"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

REQUIRED_TRAIN_COLUMNS = {
    "event_id",
    "time_to_tca",
    "risk",
}

REQUIRED_TEST_COLUMNS = {
    "event_id",
    "time_to_tca",
    "risk",
}

REQUIRED_PRIVATE_COLUMNS = {
    "event_id",
    "true_risk",
}


def _find_unique_file(root: Path, filename: str) -> Path:
    """Find exactly one dataset file beneath the configured root."""
    if not root.exists():
        raise FileNotFoundError(
            f"ESA dataset directory does not exist: {root}"
        )

    matches = list(root.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} beneath {root}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Expected one {filename}, found {len(matches)}: {matches}"
        )

    return matches[0]


def _validate_esa_frame(
    df: pd.DataFrame,
    required_columns: set,
    risk_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Validate essential ESA fields without altering scientific units."""

    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")

    if not df.columns.is_unique:
        raise ValueError(
            f"{dataset_name} contains duplicate column names."
        )

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )

    if df["event_id"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing event IDs."
        )

    df["event_id"] = df["event_id"].astype("string")

    for column in ["time_to_tca", risk_column]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        values = df[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{dataset_name} contains missing or non-finite "
                f"values in {column}."
            )

    # Negative time_to_tca is valid in historical training records
    # and private evaluation observations near or after TCA.
    # Public prediction inputs must respect the 2-day cutoff.
    if dataset_name == "ESA public test data":
        if (df["time_to_tca"] < 2.0).any():
            raise ValueError(
                "ESA public test data contains records inside "
                "the required 2-day prediction cutoff."
            )

    return df


def _attach_provenance(
    df: pd.DataFrame,
    path: str,
    partition: str,
) -> pd.DataFrame:
    """Attach traceable dataset provenance to a loaded dataframe."""

    df.attrs["dataset_name"] = DATASET_SOURCE
    df.attrs["dataset_source"] = DATASET_URL
    df.attrs["dataset_license"] = DATASET_LICENSE
    df.attrs["dataset_partition"] = partition
    df.attrs["dataset_path"] = str(path)
    df.attrs["is_synthetic_fixture"] = False
    df.attrs["provenance_status"] = "SOURCE_IDENTIFIED"
    df.attrs["risk_units"] = "base-10 logarithmic risk"
    df.attrs["prediction_cutoff_days"] = 2.0

    return df


def load_esa_training_data(
    data_root: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the ESA training CSV directly from train_data.zip.

    No extraction of the large CSV is required.
    """
    root = Path(data_root) if data_root else DEFAULT_ESA_ROOT

    archive_path = _find_unique_file(root, "train_data.zip")

    with zipfile.ZipFile(archive_path, "r") as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if Path(name).name == "train_data.csv"
        ]

        if len(csv_members) != 1:
            raise ValueError(
                "Expected exactly one train_data.csv inside "
                f"{archive_path}; found {len(csv_members)}."
            )

        with archive.open(csv_members[0], "r") as csv_file:
            text_file = io.TextIOWrapper(
                csv_file,
                encoding="utf-8-sig",
            )

            df = pd.read_csv(
                text_file,
                dtype={"event_id": "string"},
                low_memory=False,
            )

    df = _validate_esa_frame(
        df,
        REQUIRED_TRAIN_COLUMNS,
        "risk",
        "ESA training data",
    )

    return _attach_provenance(
        df,
        f"{archive_path}!{csv_members[0]}",
        "training",
    )


def load_esa_test_data(
    data_root: Optional[str] = None,
) -> pd.DataFrame:
    """Load public ESA test inputs without accessing private labels."""

    root = Path(data_root) if data_root else DEFAULT_ESA_ROOT
    path = _find_unique_file(root, "test_data.csv")

    df = pd.read_csv(
        path,
        dtype={"event_id": "string"},
        low_memory=False,
    )

    df = _validate_esa_frame(
        df,
        REQUIRED_TEST_COLUMNS,
        "risk",
        "ESA public test data",
    )

    return _attach_provenance(
        df,
        path,
        "public_test_inputs",
    )


def load_esa_private_labels(
    data_root: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load private ESA labels for final evaluation only.

    Do not use this function during training, feature engineering,
    preprocessing fit, model selection, or threshold tuning.
    """
    root = Path(data_root) if data_root else DEFAULT_ESA_ROOT
    path = _find_unique_file(root, "test_data_private.csv")

    df = pd.read_csv(
        path,
        dtype={"event_id": "string"},
        low_memory=False,
    )

    df = _validate_esa_frame(
        df,
        REQUIRED_PRIVATE_COLUMNS,
        "true_risk",
        "ESA private evaluation labels",
    )

    if df["event_id"].duplicated().any():
        raise ValueError(
            "Private evaluation labels must contain exactly "
            "one row per event."
        )

    return _attach_provenance(
        df,
        path,
        "private_evaluation_labels",
    )
