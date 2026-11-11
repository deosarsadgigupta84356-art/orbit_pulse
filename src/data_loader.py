"""
ORBIT PULSE - Data Loader & Config Manager
==========================================
Responsible for loading raw CDM files, detecting data provenance,
and providing centralized configuration parameters.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "synthetic_cdm_fixture.csv")
DEFAULT_DEMO_DATA_PATH = str(PROJECT_ROOT / "data" / "demo" / "demo_cdm_stream.csv")
CONFIG_THRESHOLDS_PATH = str(PROJECT_ROOT / "configs" / "thresholds.yaml")
CONFIG_SCHEMA_PATH = str(PROJECT_ROOT / "configs" / "feature_schema.yaml")

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_thresholds_config() -> Dict[str, Any]:
    """Get decision and evidence thresholds configuration."""
    return load_yaml_config(CONFIG_THRESHOLDS_PATH)

def get_feature_schema_config() -> Dict[str, Any]:
    """Get feature schema and ablation sets configuration."""
    return load_yaml_config(CONFIG_SCHEMA_PATH)

def load_raw_cdm_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw CDM dataset from CSV.
    Determines if dataset is an approved source or a synthetic fixture.
    """
    path = file_path or DEFAULT_RAW_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"CDM data file not found at: {path}")

    df = pd.read_csv(path)

        # Audit dataset provenance conservatively.
    # A missing or false synthetic flag does NOT prove official provenance.

    provenance_status = "UNKNOWN_UNVERIFIED"

    if "is_synthetic_fixture" in df.columns:
        raw_flags = df["is_synthetic_fixture"].dropna()

        normalized_flags = raw_flags.astype(str).str.strip().str.lower()

        true_values = {"true", "1", "yes", "y"}
        false_values = {"false", "0", "no", "n"}

        recognized_flags = set(normalized_flags.unique())

        if recognized_flags and recognized_flags.issubset(true_values):
            provenance_status = "SYNTHETIC"

        elif recognized_flags and recognized_flags.issubset(false_values):
            provenance_status = "UNKNOWN_UNVERIFIED"

        else:
            provenance_status = "UNKNOWN_UNVERIFIED"

    is_synthetic = provenance_status == "SYNTHETIC"

    # Attach audit metadata to dataframe attributes.
    df.attrs["dataset_path"] = path
    df.attrs["is_synthetic_fixture"] = is_synthetic
    df.attrs["provenance_status"] = provenance_status

    if is_synthetic:
        df.attrs["dataset_version"] = "SYNTHETIC_CDM_FIXTURE_v1.0"
        df.attrs["license_notice"] = (
            "SYNTHETIC TEST FIXTURE: UI & Pipeline Validation Only. "
            "Not Real-World Operational Validation."
        )
    else:
        df.attrs["dataset_version"] = "UNKNOWN_UNVERIFIED"
        df.attrs["license_notice"] = (
            "DATA PROVENANCE UNVERIFIED: Official source and usage "
            "authorization have not been established."
        )
    # Attach audit metadata to dataframe attributes
    df.attrs["dataset_path"] = path
    df.attrs["is_synthetic_fixture"] = is_synthetic
    df.attrs["dataset_version"] = "SYNTHETIC_CDM_FIXTURE_v1.0" if is_synthetic else "OFFICIAL_HISTORICAL_CDM_v1.0"
    df.attrs["license_notice"] = (
        "SYNTHETIC TEST FIXTURE: UI & Pipeline Validation Only. Not Real Validation."
        if is_synthetic else "Historical CDM Dataset under Approved Research Terms."
    )

    return df
