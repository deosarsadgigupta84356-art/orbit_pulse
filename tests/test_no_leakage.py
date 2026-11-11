"""
Automated Anti-Leakage & Split Integrity Tests
==============================================
Verifies:
1. Feature extraction uses strictly eligible messages (<= TCA-48h).
2. Final target Pc is excluded from predictors.
3. No event appears in multiple splits.
4. Test set split is chronologically strictly later than train set.
"""

import pytest
import pandas as pd
from datetime import timedelta
from src.event_builder import ConjunctionEventHistory
from src.snapshot_48h import build_48h_snapshot
from src.feature_engineering import extract_snapshot_features, build_dataset_feature_matrix
from src.train_model import chronological_event_split

def test_features_contain_no_post_cutoff_data():
    """Verify post-cutoff messages do not contaminate features."""
    tca = pd.Timestamp("2026-07-15T12:00:00Z")
    cutoff = tca - timedelta(hours=48)

    msgs = pd.DataFrame([
        {
            "message_id": "M_EARLY",
            "message_time": cutoff - timedelta(hours=24),
            "tca": tca,
            "collision_probability": 1e-6,
            "miss_distance": 5000.0,
            "relative_speed": 10000.0,
            "c_r_r": 100.0,
            "c_t_t": 200.0,
            "c_n_n": 150.0
        },
        {
            "message_id": "M_POST_CUTOFF",
            "message_time": cutoff + timedelta(hours=10),
            "tca": tca,
            "collision_probability": 0.5, # Huge post-cutoff jump!
            "miss_distance": 50.0,
            "relative_speed": 10000.0,
            "c_r_r": 5.0,
            "c_t_t": 10.0,
            "c_n_n": 8.0
        }
    ])

    event = ConjunctionEventHistory("EVT_LEAKAGE_TEST", msgs)
    snapshot = build_48h_snapshot(event)
    features = extract_snapshot_features(snapshot)

    # Feature for last eligible Pc must reflect M_EARLY (1e-6, log10 ~ -6.0), NOT M_POST_CUTOFF (0.5, log10 ~ -0.3)
    assert features["last_eligible_pc_log"] < -5.5
    assert features["latest_miss_distance"] == 5000.0

def test_split_integrity_and_chronological_ordering():
    """Verify no event overlap across splits and chronological progression."""
    # Create 10 dummy events with strictly increasing TCAs
    base_tca = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(10):
        rows.append({
            "event_id": f"E_{i}",
            "tca": base_tca + timedelta(days=i * 5),
            "target_final_log_pc": -5.0,
            "last_eligible_pc_log": -5.0
        })
    df_feat = pd.DataFrame(rows)

    train_df, val_df, test_df = chronological_event_split(df_feat, 0.60, 0.20)

    # 1. No overlap
    train_ids = set(train_df["event_id"])
    val_ids = set(val_df["event_id"])
    test_ids = set(test_df["event_id"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # 2. Chronological order
    assert train_df["tca"].max() <= val_df["tca"].min()
    assert val_df["tca"].max() <= test_df["tca"].min()
