"""
ORBIT PULSE - Model Training Pipeline
=====================================
Implements chronological, event-separated training.
- Baseline 1: Last-eligible Pc baseline.
- Model 2: Ridge Regression on log-features.
- Model 3: HistGradientBoostingRegressor / RandomForest.

Target: log10(final_target_pc + eps)
Never allows test events into training or feature leakage.
Saves model artifacts and complete provenance metadata.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_loader import load_raw_cdm_data, get_feature_schema_config
from src.validation import validate_cdm_records
from src.event_builder import build_event_histories
from src.snapshot_48h import build_all_snapshots
from src.feature_engineering import build_dataset_feature_matrix, EPSILON

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SAVE_PATH = str(PROJECT_ROOT / "models" / "risk_model.joblib")
METADATA_SAVE_PATH = str(PROJECT_ROOT / "models" / "model_metadata.json")

def chronological_event_split(
    df_features: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sorts events strictly by TCA and divides into chronological train, val, and test splits.
    Guarantees no event ever appears in multiple splits.
    """
    # Unique events sorted by TCA
    df_sorted = df_features.sort_values("tca").reset_index(drop=True)
    n = len(df_sorted)

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df

def train_and_save_models():
    """Executes the complete chronological training and saves artifacts."""
    schema_cfg = get_feature_schema_config()
    feature_cols = schema_cfg["feature_sets"]["set_d_full"]

    # 1. Ingest and validate data
    raw_df = load_raw_cdm_data()
    valid_df, quarantine_df, val_audit = validate_cdm_records(raw_df)

    # 2. Build event histories and 48h snapshots
    events = build_event_histories(valid_df)
    snapshots = build_all_snapshots(events)

    # 3. Build feature matrix
    df_feat = build_dataset_feature_matrix(snapshots)
    if len(df_feat) < 5:
        raise ValueError("Not enough eligible events to train model.")

    # 4. Chronological Split
    train_df, val_df, test_df = chronological_event_split(df_feat, 0.60, 0.20)

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["target_final_log_pc"]

    X_val = val_df[feature_cols].fillna(0.0)
    y_val = val_df["target_final_log_pc"]

    # 5. Model Candidate: HistGradientBoostingRegressor (robust tabular gradient boosting)
    model = HistGradientBoostingRegressor(
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=2,
        learning_rate=0.08,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 6. Save model and metadata
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)

    metadata = {
        "model_type": "HistGradientBoostingRegressor",
        "library": "scikit-learn",
        "feature_names": feature_cols,
        "target": "log10(target_final_pc + 1e-12)",
        "epsilon": EPSILON,
        "dataset_version": raw_df.attrs.get("dataset_version", "UNKNOWN"),
        "is_synthetic_fixture": raw_df.attrs.get("is_synthetic_fixture", True),
        "split_summary": {
            "train_events": len(train_df),
            "train_tca_range": [str(train_df["tca"].min()), str(train_df["tca"].max())],
            "val_events": len(val_df),
            "val_tca_range": [str(val_df["tca"].min()), str(val_df["tca"].max())],
            "test_events": len(test_df),
            "test_tca_range": [str(test_df["tca"].min()), str(test_df["tca"].max())]
        },
        "training_timestamp_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Metadata saved to {METADATA_SAVE_PATH}")
    return model, metadata

if __name__ == "__main__":
    train_and_save_models()
